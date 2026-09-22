"""Export OpenFOAM fields as compact machine-learning datasets."""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .._progress import finish_progress, start_progress
from ..openfoam.environment import _Toolchain, toolchain
from ..openfoam.io import (
    FieldData,
    OpenFOAMDecodeError,
    collated_blocks,
    read_field,
    read_field_payload,
    read_label_payload,
)
from ..reconstruction import _numeric_directories, reconstruct
from ..resources import available_memory as _available_memory
from ..resources import workers as _workers
from ..selection import TimeSelection
from ..selection import select_times as _select_times
from .derived import gradient_source, materialise_gradients, reconstruction_fields
from .thermodynamics import PerfectGasMixture, perfect_gas_mixture

_PROCESSOR = re.compile(r"processor(\d+)$")
_COLLATED = re.compile(r"processors(\d+)(?:_(\d+)-(\d+))?$")
_FIELD_COMPONENTS = {
    "volScalarField": 1,
    "volSphericalTensorField": 1,
    "volVectorField": 3,
    "volSymmTensorField": 6,
    "volTensorField": 9,
}


@dataclass(frozen=True, slots=True)
class ExportedDataset:
    """Summary of one dataset export."""

    case_dir: Path
    output_dir: Path
    fields: tuple[str, ...]
    times: tuple[float, ...]
    dtype: str
    mode: str
    snapshots: int
    elapsed_seconds: float

    def downsample(self, **kwargs):
        """Downsample this exported dataset with a declared filtering contract."""

        from .downsample import downsample

        return downsample(dataset_dir=self.output_dir, **kwargs)

    def compute(self, **kwargs):
        """Compute and persist declared fields in this dataset."""

        from .compute import compute

        return compute(dataset_dir=self.output_dir, **kwargs)

    def postprocess(self, **kwargs):
        """Materialise supported operations in this Cartesian dataset."""

        from .postprocess import postprocess

        return postprocess(dataset_dir=self.output_dir, **kwargs)


@dataclass(frozen=True, slots=True)
class _Source:
    path: Path
    first_rank: int
    last_rank: int
    collated: bool


@dataclass(frozen=True, slots=True)
class _TimeLayout:
    name: str
    sources: tuple[_Source, ...]


@dataclass(frozen=True, slots=True)
class _AddressingIndex:
    """Disk-backed processor addressing shared by export workers."""

    path: Path
    offsets: tuple[int, ...]
    global_count: int


@dataclass(frozen=True, slots=True)
class _DirectFieldTask:
    """One independently executable decomposed-field export."""

    sources: tuple[_Source, ...]
    time_name: str
    field_name: str
    output_names: tuple[str, ...]
    addressing: _AddressingIndex
    dtype: str
    data_dir: Path
    snapshot_id: int
    density_mixture: PerfectGasMixture | None


def _error(message: str, error_type: type[Exception] = ValueError):
    raise error_type(f"[StackNordic] {message}")


def _release_file_cache(path: Path) -> None:
    """Release clean file-backed pages after a staged array is complete."""

    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _source_groups(case_dir: Path) -> tuple[tuple[_Source, ...], ...]:
    uncollated: list[_Source] = []
    collated: dict[int, list[_Source]] = {}
    for path in case_dir.iterdir():
        if not path.is_dir():
            continue
        if match := _PROCESSOR.fullmatch(path.name):
            rank = int(match.group(1))
            uncollated.append(_Source(path, rank, rank, False))
        elif match := _COLLATED.fullmatch(path.name):
            total = int(match.group(1))
            first = int(match.group(2) or 0)
            last = int(match.group(3) or total - 1)
            collated.setdefault(total, []).append(_Source(path, first, last, True))
    groups: list[tuple[_Source, ...]] = []
    if uncollated:
        groups.append(tuple(sorted(uncollated, key=lambda source: source.first_rank)))
    groups.extend(
        tuple(sorted(sources, key=lambda source: source.first_rank))
        for _, sources in sorted(collated.items())
    )
    if not groups:
        _error(f"no decomposed processor data was found in {case_dir}", FileNotFoundError)
    return tuple(groups)


def _validate_sources(sources: tuple[_Source, ...]) -> None:
    expected = 0
    total = max(source.last_rank for source in sources) + 1
    for source in sources:
        if source.first_rank != expected or source.last_rank < source.first_rank:
            _error("processor ranges are incomplete or overlap")
        expected = source.last_rank + 1
    if expected != total:
        _error("processor ranges do not cover a consecutive decomposition")


def _available_layouts(case_dir: Path) -> dict[float, _TimeLayout]:
    candidates: dict[float, list[_TimeLayout]] = {}
    for original in _source_groups(case_dir):
        per_source = {source: _numeric_directories(source.path) for source in original}
        union = set().union(*(set(values) for values in per_source.values()))
        for value in union:
            present = tuple(source for source in original if value in per_source[source])
            if not present:
                continue
            try:
                _validate_sources(present)
            except ValueError:
                continue
            spellings = {per_source[source][value] for source in present}
            if len(spellings) != 1:
                _error(f"processor data uses inconsistent names for time {value:g}")
            candidates.setdefault(value, []).append(_TimeLayout(spellings.pop(), present))
    layouts: dict[float, _TimeLayout] = {}
    for value, options in candidates.items():
        if len(options) > 1:
            descriptions = ", ".join(
                f"{len(option.sources)} source director{'y' if len(option.sources) == 1 else 'ies'}"
                for option in options
            )
            _error(f"multiple complete decompositions exist at t={value:g}: {descriptions}")
        layouts[value] = options[0]
    if not layouts:
        _error("processor data has no common physical times")
    return dict(sorted(layouts.items()))


def _mesh_file(source: _Source, time_value: float, name: str) -> Path:
    candidates: list[tuple[float, Path]] = []
    constant = source.path / "constant/polyMesh" / name
    if constant.is_file():
        candidates.append((-math.inf, constant))
    for value, spelling in _numeric_directories(source.path).items():
        path = source.path / spelling / "polyMesh" / name
        if value <= time_value and path.is_file():
            candidates.append((value, path))
    if not candidates:
        _error(
            f"{name} is unavailable for {source.path.name} at t={time_value:g}",
            FileNotFoundError,
        )
    return max(candidates, key=lambda item: item[0])[1]


def _processor_payloads(source: _Source, path: Path):
    if source.collated:
        expected = set(range(source.first_rank, source.last_rank + 1))
        seen: set[int] = set()
        rank_offset: int | None = None
        for rank, payload in collated_blocks(path):
            if rank_offset is None:
                if rank == source.first_rank:
                    rank_offset = 0
                elif rank == 0:
                    rank_offset = source.first_rank
                else:
                    _error(f"{path} does not start at its declared processor range")
            rank += rank_offset
            if rank not in expected or rank in seen:
                _error(f"{path} does not contain its declared processor range")
            seen.add(rank)
            yield rank, payload
        if seen != expected:
            _error(f"{path} does not contain its declared processor range")
        return
    yield source.first_rank, path.read_bytes()


def _addressing(sources: tuple[_Source, ...], time_value: float) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for source in sources:
        path = _mesh_file(source, time_value, "cellProcAddressing")
        for rank, payload in _processor_payloads(source, path):
            result[rank] = read_label_payload(payload)
    expected = set(range(max(result) + 1))
    if set(result) != expected:
        _error(f"cellProcAddressing is incomplete at t={time_value:g}")
    return result


def _global_cell_count(addressing: Mapping[int, np.ndarray]) -> int:
    arrays = tuple(addressing[rank] for rank in sorted(addressing))
    if any(np.any(values < 0) for values in arrays):
        _error("cellProcAddressing contains negative global cell indices")
    entries = sum(values.size for values in arrays)
    count = max((int(values.max()) + 1 for values in arrays if values.size), default=0)
    if entries != count:
        _error("cellProcAddressing does not provide one complete global-cell permutation")
    seen = np.zeros(count, dtype=np.bool_)
    for values in arrays:
        seen[values] = True
    if not np.all(seen):
        _error("cellProcAddressing does not provide one complete global-cell permutation")
    return count


def _addressing_key(sources: tuple[_Source, ...], time_value: float) -> tuple[str, ...]:
    """Identify one mesh generation by its exact addressing files."""

    return tuple(str(_mesh_file(source, time_value, "cellProcAddressing").resolve()) for source in sources)


def _store_addressing_index(
    sources: tuple[_Source, ...],
    time_value: float,
    work_dir: Path,
    index: int,
) -> _AddressingIndex:
    addressing = _addressing(sources, time_value)
    global_count = _global_cell_count(addressing)
    ranks = tuple(sorted(addressing))
    if ranks != tuple(range(len(ranks))):
        _error(f"cellProcAddressing is incomplete at t={time_value:g}")
    offsets = [0]
    for rank in ranks:
        offsets.append(offsets[-1] + addressing[rank].size)
    path = work_dir / f"addressing-{index:03d}.dat"
    output = np.memmap(path, dtype=np.int64, mode="w+", shape=(offsets[-1],))
    try:
        for rank in ranks:
            output[offsets[rank] : offsets[rank + 1]] = addressing[rank]
        output.flush()
    finally:
        del output
    for source in sources:
        _release_file_cache(_mesh_file(source, time_value, "cellProcAddressing"))
    return _AddressingIndex(path=path, offsets=tuple(offsets), global_count=global_count)


def _open_addressing(index: _AddressingIndex) -> tuple[dict[int, np.ndarray], np.memmap]:
    storage = np.memmap(index.path, dtype=np.int64, mode="r")
    addressing = {
        rank: storage[index.offsets[rank] : index.offsets[rank + 1]]
        for rank in range(len(index.offsets) - 1)
    }
    return addressing, storage


def _write_direct_field(
    sources: tuple[_Source, ...],
    time_name: str,
    field_name: str,
    output_names: tuple[str, ...],
    addressing: Mapping[int, np.ndarray],
    dtype: np.dtype,
    data_dir: Path,
    snapshot_id: int,
    global_count: int,
) -> dict[str, str]:
    outputs: tuple[np.memmap, ...] | None = None
    field_class: str | None = None
    seen: set[int] = set()
    filenames = tuple(f"{name}_id{snapshot_id:03d}.dat" for name in output_names)
    try:
        for source in sources:
            path = source.path / time_name / field_name
            if not path.is_file():
                source_field = gradient_source(field_name)
                if source_field is not None:
                    _error(
                        f"field {field_name!r} is missing at t={time_name}. "
                        f"Use mode='reconstruct' to generate grad({source_field}) with "
                        "OpenFOAM postProcess in an isolated workspace",
                        FileNotFoundError,
                    )
                _error(
                    f"field {field_name!r} is missing at t={time_name} in {source.path.name}",
                    FileNotFoundError,
                )
            payloads = _processor_payloads(source, path)
            for rank, payload in payloads:
                data = read_field_payload(payload, cell_count=addressing[rank].size)
                _validate_components(field_name, output_names, data)
                if field_class is None:
                    field_class = data.field_class
                    outputs = tuple(
                        np.memmap(data_dir / filename, dtype=dtype, mode="w+", shape=(global_count,))
                        for filename in filenames
                    )
                elif data.field_class != field_class:
                    _error("processor field classes are inconsistent")
                values = np.asarray(data.values)
                if values.shape[0] != addressing[rank].size:
                    _error(f"processor{rank} field length does not match cellProcAddressing")
                columns = (values,) if values.ndim == 1 else tuple(
                    values[:, index] for index in range(values.shape[1])
                )
                assert outputs is not None
                for output, column in zip(outputs, columns, strict=True):
                    output[addressing[rank]] = column
                seen.add(rank)
            _release_file_cache(path)
        if seen != set(addressing):
            _error(f"field {field_name!r} has incomplete processor coverage at t={time_name}")
        assert outputs is not None
        for output in outputs:
            output.flush()
    finally:
        if outputs is not None:
            del outputs
        for filename in filenames:
            path = data_dir / filename
            if path.is_file():
                _release_file_cache(path)
    return {name: f"./data/{filename}" for name, filename in zip(output_names, filenames, strict=True)}


def _write_direct_density(
    sources: tuple[_Source, ...],
    time_name: str,
    output_name: str,
    addressing: Mapping[int, np.ndarray],
    dtype: np.dtype,
    data_dir: Path,
    snapshot_id: int,
    global_count: int,
    mixture: PerfectGasMixture,
) -> dict[str, str]:
    filename = f"{output_name}_id{snapshot_id:03d}.dat"
    output = np.memmap(data_dir / filename, dtype=dtype, mode="w+", shape=(global_count,))
    try:
        seen: set[int] = set()
        field_names = ("p", "T", *mixture.species)
        for source in sources:
            paths = tuple(source.path / time_name / name for name in field_names)
            missing = next((path for path in paths if not path.is_file()), None)
            if missing is not None:
                _error(
                    f"field {missing.name!r} required to derive rho is missing at t={time_name}",
                    FileNotFoundError,
                )
            streams = tuple(_processor_payloads(source, path) for path in paths)
            for blocks in zip(*streams, strict=True):
                ranks = {rank for rank, _ in blocks}
                if len(ranks) != 1:
                    _error(f"thermodynamic processor blocks are misaligned at t={time_name}")
                rank = blocks[0][0]
                fields = {
                    name: np.asarray(
                        read_field_payload(payload, cell_count=addressing[rank].size).values
                    )
                    for name, (_, payload) in zip(field_names, blocks, strict=True)
                }
                output[addressing[rank]] = mixture.density(
                    lambda name, fields=fields: fields[name]
                )
                seen.add(rank)
            for path in paths:
                _release_file_cache(path)
        if seen != set(addressing):
            _error(f"fields required to derive rho have incomplete coverage at t={time_name}")
        output.flush()
    finally:
        del output
        if (data_dir / filename).is_file():
            _release_file_cache(data_dir / filename)
    print(f"[StackNordic] Derived rho at t={time_name} from perfectGas thermodynamics.")
    return {output_name: f"./data/{filename}"}


def _run_direct_field_task(task: _DirectFieldTask) -> tuple[int, dict[str, str]]:
    """Execute one field export in a worker process."""

    addressing, storage = _open_addressing(task.addressing)
    try:
        missing = any(
            not (source.path / task.time_name / task.field_name).is_file()
            for source in task.sources
        )
        if task.field_name == "rho" and missing and task.density_mixture is not None:
            if len(task.output_names) != 1:
                _error("field 'rho' requires one output name")
            filenames = _write_direct_density(
                task.sources,
                task.time_name,
                task.output_names[0],
                addressing,
                np.dtype(task.dtype),
                task.data_dir,
                task.snapshot_id,
                task.addressing.global_count,
                task.density_mixture,
            )
        else:
            filenames = _write_direct_field(
                task.sources,
                task.time_name,
                task.field_name,
                task.output_names,
                addressing,
                np.dtype(task.dtype),
                task.data_dir,
                task.snapshot_id,
                task.addressing.global_count,
            )
        return task.snapshot_id, filenames
    finally:
        del addressing
        del storage


def _normalise_fields(
    fields: Mapping[str, str | Sequence[str]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(fields, Mapping) or not fields:
        _error("fields must be a non-empty mapping")
    result: list[tuple[str, tuple[str, ...]]] = []
    outputs: set[str] = set()
    for field, names in fields.items():
        if not field or Path(field).name != field:
            _error("each input field must be one non-empty path component")
        selected = (names,) if isinstance(names, str) else tuple(names)
        if not selected or any(not name or Path(name).name != name for name in selected):
            _error(f"output names for field {field!r} must be non-empty filenames")
        duplicate = outputs.intersection(selected)
        if duplicate:
            _error(f"output name {min(duplicate)!r} is used more than once")
        outputs.update(selected)
        result.append((field, selected))
    return tuple(result)


def _validate_components(field: str, output_names: tuple[str, ...], data: FieldData) -> None:
    expected = _FIELD_COMPONENTS.get(data.field_class)
    if expected is None:
        _error(f"field {field!r} uses unsupported OpenFOAM class {data.field_class!r}")
    if len(output_names) != expected:
        noun = "name" if expected == 1 else "component names"
        _error(
            f"field {field!r} is {data.field_class} and requires {expected} output {noun}; "
            f"received {len(output_names)}"
        )


def _cartesian_grid(case_dir: Path, cell_count: int, dtype: np.dtype) -> dict[str, np.ndarray] | None:
    path = case_dir / "system/blockMeshDict"
    if not path.is_file():
        return None
    payload = re.sub(r"/\*.*?\*/|//[^\n]*", " ", path.read_text(errors="replace"), flags=re.DOTALL)
    scalars = {
        name: float(value)
        for name, value in re.findall(
            r"(?m)^\s*([A-Za-z_]\w*)\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*;",
            payload,
        )
    }
    scale = scalars.get("scale", scalars.get("convertToMeters", 1.0))
    blocks = re.findall(
        r"hex\s*\(([^)]*)\)\s*\(\s*\$?(\w+)\s+\$?(\w+)\s+\$?(\w+)\s*\)"
        r"\s*simpleGrading\s*\(([^)]*)\)",
        payload,
    )
    if len(blocks) != 1:
        return None
    vertices_text, *dimension_tokens, grading_text = blocks[0]
    try:
        dimensions = tuple(
            int(scalars.get(token, token)) for token in dimension_tokens
        )
        grading = tuple(
            float(scalars.get(token.lstrip("$"), token.lstrip("$")))
            for token in grading_text.split()
        )
        vertex_indices = tuple(int(token) for token in vertices_text.split())
    except ValueError:
        return None
    if len(grading) != 3 or any(not math.isclose(value, 1.0) for value in grading):
        return None
    if len(vertex_indices) != 8:
        return None
    if math.prod(dimensions) != cell_count:
        return None
    names = (("xMin", "xMax"), ("yMin", "yMax"), ("zMin", "zMax"))
    bounds: tuple[tuple[float, float], ...] | None = None
    if all(item in scalars for pair in names for item in pair):
        bounds = tuple(
            (scalars[lower] * scale, scalars[upper] * scale)
            for lower, upper in names
        )
    else:
        section = re.search(r"vertices\s*\((.*?)\)\s*;", payload, flags=re.DOTALL)
        if section is None:
            return None
        vertices: list[tuple[float, float, float]] = []
        for entry in re.findall(r"\(([^()]*)\)", section.group(1)):
            tokens = entry.split()
            if len(tokens) != 3:
                return None
            try:
                vertices.append(
                    tuple(
                        float(scalars.get(token.lstrip("$"), token.lstrip("$")))
                        * scale
                        for token in tokens
                    )
                )
            except ValueError:
                return None
        if max(vertex_indices, default=-1) >= len(vertices):
            return None
        selected = tuple(vertices[index] for index in vertex_indices)
        coordinates = tuple(tuple(sorted({point[axis] for point in selected})) for axis in range(3))
        if any(len(values) != 2 for values in coordinates):
            return None
        corners = {
            (x, y, z)
            for x in coordinates[0]
            for y in coordinates[1]
            for z in coordinates[2]
        }
        if set(selected) != corners:
            return None
        bounds = tuple((values[0], values[1]) for values in coordinates)
    result: dict[str, np.ndarray] = {}
    for axis, size, (lower, upper) in zip("XYZ", dimensions, bounds, strict=True):
        spacing = (upper - lower) / size
        result[axis] = (lower + (np.arange(size) + 0.5) * spacing).astype(dtype)
    return result


def _write_snapshot(
    case_dir: Path,
    sources: tuple[_Source, ...],
    time_value: float,
    time_name: str,
    fields: tuple[tuple[str, tuple[str, ...]], ...],
    dtype: np.dtype,
    data_dir: Path,
    snapshot_id: int,
    density_mixture: PerfectGasMixture | None,
) -> tuple[dict[str, object], int]:
    addressing = _addressing(sources, time_value)
    global_count = _global_cell_count(addressing)
    local: dict[str, object] = {"id": snapshot_id, "time [s]": time_value}
    for field_name, output_names in fields:
        missing = any(not (source.path / time_name / field_name).is_file() for source in sources)
        if field_name == "rho" and missing and density_mixture is not None:
            if len(output_names) != 1:
                _error("field 'rho' requires one output name")
            filenames = _write_direct_density(
                sources,
                time_name,
                output_names[0],
                addressing,
                dtype,
                data_dir,
                snapshot_id,
                global_count,
                density_mixture,
            )
        else:
            filenames = _write_direct_field(
                sources,
                time_name,
                field_name,
                output_names,
                addressing,
                dtype,
                data_dir,
                snapshot_id,
                global_count,
            )
        for output_name, filename in filenames.items():
            local[f"{output_name} filename"] = filename
    local["cells"] = global_count
    return local, global_count


def _write_reconstructed_snapshot(
    case_dir: Path,
    time_value: float,
    time_name: str,
    fields: tuple[tuple[str, tuple[str, ...]], ...],
    dtype: np.dtype,
    data_dir: Path,
    snapshot_id: int,
) -> tuple[dict[str, object], int]:
    local: dict[str, object] = {"id": snapshot_id, "time [s]": time_value}
    global_count: int | None = None
    decoded: list[tuple[tuple[str, ...], FieldData]] = []
    for field_name, output_names in fields:
        path = case_dir / time_name / field_name
        if not path.is_file():
            _error(f"field {field_name!r} is missing at t={time_name}", FileNotFoundError)
        data = read_field(path, cell_count=global_count)
        if data.values.ndim == 0:
            if global_count is None:
                _error(
                    f"uniform field {field_name!r} cannot establish the global cell count; "
                    "place a nonuniform field first"
                )
            data = read_field(path, cell_count=global_count)
        _validate_components(field_name, output_names, data)
        count = data.values.shape[0]
        global_count = count if global_count is None else global_count
        if count != global_count:
            _error(f"field {field_name!r} has an inconsistent global cell count")
        decoded.append((output_names, data))
    assert global_count is not None
    for output_names, data in decoded:
        values = np.asarray(data.values, dtype=dtype)
        columns = (values,) if values.ndim == 1 else tuple(
            values[:, index] for index in range(values.shape[1])
        )
        for output_name, column in zip(output_names, columns, strict=True):
            filename = f"{output_name}_id{snapshot_id:03d}.dat"
            np.asarray(column, dtype=dtype).tofile(data_dir / filename)
            local[f"{output_name} filename"] = f"./data/{filename}"
    local["cells"] = global_count
    return local, global_count


def _mesh_varies(sources: tuple[_Source, ...], selected_times: tuple[float, ...]) -> bool:
    for source in sources:
        names = _numeric_directories(source.path)
        for value in selected_times:
            spelling = names.get(value)
            if spelling is not None and (source.path / spelling / "polyMesh").is_dir():
                return True
    return False


def _estimated_worker_bytes(
    available: Mapping[float, _TimeLayout],
    selected_times: tuple[float, ...],
    fields: tuple[tuple[str, tuple[str, ...]], ...],
    density_mixture: PerfectGasMixture | None,
) -> int:
    largest = 0
    for value in selected_times:
        time_name = available[value]
        sources = time_name.sources
        addressing_bytes = sum(
            _mesh_file(source, value, "cellProcAddressing").stat().st_size
            for source in sources
        )
        sizes: list[int] = []
        for field, _ in fields:
            paths = tuple(source.path / time_name.name / field for source in sources)
            missing = next((path for path in paths if not path.is_file()), None)
            if missing is not None:
                if field == "rho" and density_mixture is not None:
                    dependencies = ("p", "T", *density_mixture.species)
                    by_field = tuple(
                        tuple(source.path / time_name.name / dependency for source in sources)
                        for dependency in dependencies
                    )
                    dependency_paths = tuple(path for paths in by_field for path in paths)
                    absent = next((path for path in dependency_paths if not path.is_file()), None)
                    if absent is not None:
                        _error(
                            f"field {absent.name!r} required to derive rho is missing "
                            f"at t={time_name.name}",
                            FileNotFoundError,
                        )
                    sizes.append(max(sum(path.stat().st_size for path in paths) for paths in by_field))
                    continue
                source_field = gradient_source(field)
                if source_field is not None:
                    _error(
                        f"field {field!r} is missing at t={time_name.name}. "
                        f"Use mode='reconstruct' to generate grad({source_field}) with "
                        "OpenFOAM postProcess in an isolated workspace",
                        FileNotFoundError,
                    )
                _error(f"field {field!r} is missing at t={time_name.name}", FileNotFoundError)
            sizes.append(sum(path.stat().st_size for path in paths))
        field_bytes = max(sizes)
        largest = max(largest, addressing_bytes + field_bytes)
    return max(1, math.ceil(largest * 1.5))


def _export_workers(
    n_jobs: int | None,
    work_count: int,
    estimated_worker_bytes: int | None,
) -> int:
    workers = _workers(n_jobs, work_count)
    if n_jobs != -1 or estimated_worker_bytes is None:
        return workers
    available = _available_memory()
    if available is None:
        return workers
    memory_workers = max(1, int(available * 0.75) // estimated_worker_bytes)
    selected = min(workers, memory_workers)
    if selected < workers:
        print(
            f"[StackNordic] Dataset workers: n_jobs=-1 resolved to {selected} "
            "from the available memory and selected fields."
        )
    return selected


def export(
    *,
    case_dir: str | Path,
    output_dir: str | Path,
    fields: Mapping[str, str | Sequence[str]],
    times: TimeSelection = None,
    dtype: str = "float32",
    mode: str = "direct",
    metadata: Mapping[str, object] | None = None,
    n_jobs: int | None = None,
    progress: bool = False,
    overwrite: bool = False,
    _toolchain_override: _Toolchain | None = None,
) -> ExportedDataset:
    """Export selected OpenFOAM fields without assuming a fixed decomposition."""

    started = perf_counter()
    source_case = Path(case_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if mode not in {"direct", "reconstruct"}:
        _error("mode must be 'direct' or 'reconstruct'")
    if dtype not in {"float32", "float64"}:
        _error("dtype must be 'float32' or 'float64'")
    if not isinstance(progress, bool) or not isinstance(overwrite, bool):
        raise TypeError("progress and overwrite must be booleans")
    selected_fields = _normalise_fields(fields)
    if destination.exists():
        if not destination.is_dir():
            _error(f"output path is not a directory: {destination}", FileExistsError)
        if any(destination.iterdir()) and not overwrite:
            _error(f"output directory is not empty: {destination}", FileExistsError)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".stacknordic-export-", dir=destination.parent))
    reconstructed_stage: Path | None = None
    try:
        export_case = source_case
        layouts: dict[float, _TimeLayout] | None = None
        if mode == "reconstruct":
            reconstructed_stage = Path(
                tempfile.mkdtemp(prefix=".stacknordic-reconstructed-", dir=destination.parent)
            )
            reconstruct(
                case_dir=source_case,
                output_dir=reconstructed_stage,
                times=times,
                fields=reconstruction_fields(name for name, _ in selected_fields),
                n_jobs=n_jobs,
                progress=progress,
                _toolchain_override=_toolchain_override,
            )
            export_case = reconstructed_stage
            available = _numeric_directories(export_case)
            selected_for_derivation = _select_times(tuple(available), times)
            materialise_gradients(
                export_case,
                source_case,
                tuple((value, available[value]) for value in selected_for_derivation),
                (name for name, _ in selected_fields),
                _toolchain_override or toolchain(),
            )
        else:
            root_times = _numeric_directories(export_case)
            try:
                layouts = _available_layouts(export_case)
            except FileNotFoundError:
                layouts = {}
            available = dict(layouts)
            available.update(root_times)
        if not available:
            _error("no exportable physical times were found")
        selected_times = _select_times(tuple(available), times)
        density_mixture = perfect_gas_mixture(source_case)
        reconstructed_times = {
            value
            for value in selected_times
            if value in root_times
            and all((export_case / root_times[value] / name).is_file() for name, _ in selected_fields)
        } if mode == "direct" else set(selected_times)
        estimated_worker_bytes = (
            _estimated_worker_bytes(layouts, selected_times, selected_fields, density_mixture)
            if layouts is not None and not reconstructed_times
            else None
        )
        data_dir = stage / "data"
        grid_dir = stage / "grid"
        data_dir.mkdir()
        local: list[dict[str, object] | None] = [None] * len(selected_times)
        cell_counts: list[int | None] = [None] * len(selected_times)
        if mode == "direct" and not reconstructed_times:
            assert layouts is not None
            work_dir = stage / ".work"
            work_dir.mkdir()
            addressing_cache: dict[tuple[str, ...], _AddressingIndex] = {}
            tasks: list[_DirectFieldTask] = []
            derived_tasks: list[_DirectFieldTask] = []
            for snapshot_id, value in enumerate(selected_times):
                layout = layouts[value]
                key = _addressing_key(layout.sources, value)
                addressing = addressing_cache.get(key)
                if addressing is None:
                    if progress:
                        print(
                            f"[StackNordic] Indexing mesh addressing at t={value:g} "
                            f"from {len(layout.sources)} source director"
                            f"{'y' if len(layout.sources) == 1 else 'ies'}."
                        )
                    addressing = _store_addressing_index(
                        layout.sources,
                        value,
                        work_dir,
                        len(addressing_cache),
                    )
                    addressing_cache[key] = addressing
                local[snapshot_id] = {"id": snapshot_id, "time [s]": value}
                cell_counts[snapshot_id] = addressing.global_count
                for field_name, output_names in selected_fields:
                    field_task = _DirectFieldTask(
                        sources=layout.sources,
                        time_name=layout.name,
                        field_name=field_name,
                        output_names=output_names,
                        addressing=addressing,
                        dtype=dtype,
                        data_dir=data_dir,
                        snapshot_id=snapshot_id,
                        density_mixture=density_mixture,
                    )
                    missing = any(
                        not (source.path / layout.name / field_name).is_file()
                        for source in layout.sources
                    )
                    target = derived_tasks if field_name == "rho" and missing else tasks
                    target.append(field_task)
            tasks.extend(derived_tasks)
            workers = _export_workers(n_jobs, len(tasks), estimated_worker_bytes)
            if progress:
                print(
                    f"[StackNordic] Dataset workers: {workers} for {len(tasks)} "
                    "time-field tasks."
                )
            reporter = start_progress(len(tasks), "StackNordic export") if progress else None
            try:
                if workers == 1:
                    results = map(_run_direct_field_task, tasks)
                    for snapshot_id, filenames in results:
                        assert local[snapshot_id] is not None
                        local[snapshot_id].update(filenames)
                        if reporter is not None:
                            reporter.update()
                else:
                    with ProcessPoolExecutor(
                        max_workers=workers,
                        mp_context=multiprocessing.get_context("fork"),
                    ) as executor:
                        futures = tuple(executor.submit(_run_direct_field_task, task) for task in tasks)
                        for future in as_completed(futures):
                            snapshot_id, filenames = future.result()
                            assert local[snapshot_id] is not None
                            local[snapshot_id].update(filenames)
                            if reporter is not None:
                                reporter.update()
            finally:
                if reporter is not None:
                    finish_progress(reporter)
                shutil.rmtree(work_dir, ignore_errors=True)
        else:
            workers = _export_workers(n_jobs, len(selected_times), estimated_worker_bytes)
            reporter = (
                start_progress(len(selected_times), "StackNordic export")
                if progress
                else None
            )

            def task(index: int, value: float):
                if value in reconstructed_times:
                    result = _write_reconstructed_snapshot(
                        export_case,
                        value,
                        root_times[value] if mode == "direct" else available[value],
                        selected_fields,
                        np.dtype(dtype),
                        data_dir,
                        index,
                    )
                else:
                    if layouts is None or value not in layouts:
                        _error(
                            f"selected fields are incomplete in the reconstructed case and "
                            f"no complete decomposition exists at t={value:g}",
                            FileNotFoundError,
                        )
                    layout = layouts[value]
                    result = _write_snapshot(
                        export_case,
                        layout.sources,
                        value,
                        layout.name,
                        selected_fields,
                        np.dtype(dtype),
                        data_dir,
                        index,
                        density_mixture,
                    )
                return index, result

            try:
                if workers == 1:
                    results = (task(index, value) for index, value in enumerate(selected_times))
                    for index, (record, count) in results:
                        local[index], cell_counts[index] = record, count
                        if reporter is not None:
                            reporter.update()
                else:
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        futures = {
                            executor.submit(task, index, value): index
                            for index, value in enumerate(selected_times)
                        }
                        for future in as_completed(futures):
                            index, (record, count) = future.result()
                            local[index], cell_counts[index] = record, count
                            if reporter is not None:
                                reporter.update()
            finally:
                if reporter is not None:
                    finish_progress(reporter)
        counts = {count for count in cell_counts if count is not None}
        grid: dict[str, np.ndarray] | None = None
        root_mesh_varies = any(
            value in reconstructed_times
            and (
                export_case
                / (root_times[value] if mode == "direct" else available[value])
                / "polyMesh"
            ).is_dir()
            for value in selected_times
        )
        decomposed_mesh_varies = layouts is not None and any(
            value in layouts
            and value not in reconstructed_times
            and _mesh_varies(layouts[value].sources, (value,))
            for value in selected_times
        )
        mesh_varies = root_mesh_varies or decomposed_mesh_varies
        if len(counts) == 1 and not mesh_varies:
            grid = _cartesian_grid(source_case, next(iter(counts)), np.dtype(dtype))
        grid_entries: dict[str, str] = {}
        if grid is not None:
            grid_dir.mkdir()
            for axis, values in grid.items():
                filename = f"{axis}_m.dat"
                values.tofile(grid_dir / filename)
                grid_entries[axis.lower()] = f"./grid/{filename}"
        variables = [name for _, names in selected_fields for name in names]
        info = {
            "global": {
                "snapshots": len(selected_times),
                "variables": variables,
                "dtype": dtype,
                "layout": "cartesian" if grid is not None else "cells",
                "grid": grid_entries or None,
                "fields": {field: list(names) for field, names in selected_fields},
                "metadata": dict(metadata or {}),
            },
            "local": local,
        }
        if grid is not None:
            info["global"]["Nxyz"] = [len(grid[axis]) for axis in "XYZ"]
        (stage / "info.json").write_text(
            json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if destination.exists():
            shutil.rmtree(destination)
        stage.replace(destination)
    except OpenFOAMDecodeError as error:
        shutil.rmtree(stage, ignore_errors=True)
        raise ValueError(f"[StackNordic] {error}") from error
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        if reconstructed_stage is not None:
            shutil.rmtree(reconstructed_stage, ignore_errors=True)
    return ExportedDataset(
        case_dir=source_case,
        output_dir=destination,
        fields=tuple(name for name, _ in selected_fields),
        times=selected_times,
        dtype=dtype,
        mode=mode,
        snapshots=len(selected_times),
        elapsed_seconds=perf_counter() - started,
    )


def __dir__() -> list[str]:
    return ["ExportedDataset", "export"]
