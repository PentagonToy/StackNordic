"""Memory-bounded Cartesian dataset filtering and downsampling."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter

from .._progress import finish_progress, start_progress
from ..filtering import Filtering
from ..resources import memory_bounded_workers
from ._contract import (
    SERIES_SCHEMA,
    SNAPSHOT_SCHEMA,
    SNAPSHOTS_ATTRIBUTE,
    compatible,
    contract,
    normalise_hdf5_path,
    read_contract,
    write_contract,
)
from ._manifest import component_file


@dataclass(frozen=True, slots=True)
class DownsampledDataset:
    """Summary of one completed downsampling operation."""

    input_dir: Path
    output_dir: Path
    kernel: str
    factor: tuple[int, int, int]
    snapshots: int
    n_jobs: int
    elapsed_seconds: float
    output_file: Path | None = None


def _error(message: str, error_type: type[Exception] = ValueError):
    raise error_type(f"[StackNordic] {message}")


def _triplet(values, name: str, value_type):
    try:
        selected = tuple(value_type(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain three values") from error
    if len(selected) != 3 or any(value <= 0 for value in selected):
        raise ValueError(f"{name} must contain three positive values")
    return selected


class _Snapshot:
    def __init__(self, root: Path, info: dict[str, object], record: dict[str, object]):
        global_info = info["global"]
        self.root = root
        self.record = record
        self.dtype = np.dtype(global_info["dtype"])
        self.nxyz = tuple(int(value) for value in global_info["Nxyz"])
        self.shape = tuple(reversed(self.nxyz))
        self.fields = {
            name: tuple(components)
            for name, components in global_info["fields"].items()
        }
        self.hdf_fields = global_info.get("hdf_fields")

    def field(self, name: str) -> np.ndarray:
        if name not in self.fields:
            _error(f"filtering requested unavailable field {name!r}", KeyError)
        components = self.fields[name]
        if self.hdf_fields is not None:
            filename = self.record["__hdf5_file__"]
            group = str(self.record.get("__hdf5_group__") or "").strip("/")
            target = self.hdf_fields[name]
            path = f"{group}/{target}" if group else target
            with h5py.File(filename, "r") as file:
                return np.asarray(file[path])
        arrays: list[np.memmap] = []
        for component in components:
            try:
                filename = component_file(self.record, component)
            except KeyError:
                _error(f"snapshot {self.record['id']} has no file for {component!r}")
            path = self.root / filename
            arrays.append(np.memmap(path, dtype=self.dtype, mode="r", shape=self.shape))
        if len(arrays) == 1:
            return np.asarray(arrays[0], dtype=np.float64)
        result = np.empty((*self.shape, len(arrays)), dtype=np.float64)
        for index, array in enumerate(arrays):
            result[..., index] = array
        return result


def _hdf5_info(source: Path) -> tuple[dict[str, object], Path]:
    paths = (source,) if source.is_file() else tuple(sorted(source.glob("*.h5")))
    if not paths:
        _error(f"dataset manifest or HDF5 packages were not found in {source}")
    records = []
    definitions = []
    grid_reference: dict[str, np.ndarray] | None = None
    for path in paths:
        with h5py.File(path, "r") as file:
            definition = read_contract(file)
            definitions.append(definition)
            current_grid = {
                axis: np.asarray(file[f"grid/{axis}_axis"])
                for axis in "xyz"
            }
            if grid_reference is None:
                grid_reference = current_grid
            elif any(
                not np.array_equal(grid_reference[axis], current_grid[axis])
                for axis in "xyz"
            ):
                _error("HDF5 snapshot packages do not share one Cartesian grid")
            if definition.schema == SNAPSHOT_SCHEMA:
                records.append(
                    {
                        "id": int(file.attrs["snapshot_id"]),
                        "time [s]": float(file.attrs["time_s"]),
                        "__hdf5_file__": str(path),
                        "__hdf5_group__": "",
                    }
                )
            elif definition.schema == SERIES_SCHEMA:
                try:
                    snapshots = json.loads(str(file.attrs[SNAPSHOTS_ATTRIBUTE]))
                except (KeyError, json.JSONDecodeError) as error:
                    raise ValueError(
                        "[StackNordic] compiled HDF5 series has invalid snapshot metadata"
                    ) from error
                records.extend(
                    {
                        "id": int(item["id"]),
                        "time [s]": float(item["time_s"]),
                        "__hdf5_file__": str(path),
                        "__hdf5_group__": str(item["group"]),
                    }
                    for item in snapshots
                )
            else:
                _error(f"unsupported StackNordic HDF5 schema: {definition.schema!r}")
    reference = definitions[0]
    for path, definition in zip(paths[1:], definitions[1:], strict=True):
        if not compatible(reference, definition):
            _error(f"HDF5 package contract differs: {path.name}")
    identifiers = [int(record["id"]) for record in records]
    times = [float(record["time [s]"]) for record in records]
    if len(set(identifiers)) != len(identifiers) or len(set(times)) != len(times):
        _error("HDF5 packages contain duplicate snapshot identifiers or physical times")
    records.sort(key=lambda record: (float(record["time [s]"]), int(record["id"])))
    info = {
        "global": {
            "snapshots": len(records),
            "variables": [
                component
                for components in reference.components.values()
                for component in components
            ],
            "dtype": reference.dtype,
            "layout": "cartesian",
            "grid": {axis: f"grid/{axis}_axis" for axis in "xyz"},
            "fields": {
                name: list(components)
                for name, components in reference.components.items()
            },
            "hdf_fields": dict(reference.fields),
            "hdf_grid_file": str(paths[0]),
            "Nxyz": list(reversed(reference.shape_zyx)),
            "metadata": dict(reference.metadata),
        },
        "local": records,
    }
    return info, source


def _source_info(source: Path) -> tuple[dict[str, object], Path]:
    manifest = source / "info.json" if source.is_dir() else None
    if manifest is not None and manifest.is_file():
        try:
            return json.loads(manifest.read_text()), source
        except json.JSONDecodeError as error:
            raise ValueError(f"[StackNordic] invalid dataset manifest in {source}") from error
    return _hdf5_info(source)


def _grid_axis(source: Path, global_info: dict[str, object], axis: str) -> np.ndarray:
    if "hdf_grid_file" in global_info:
        with h5py.File(global_info["hdf_grid_file"], "r") as file:
            return np.asarray(file[global_info["grid"][axis]])
    return np.memmap(
        source / global_info["grid"][axis],
        dtype=np.dtype(global_info["dtype"]),
        mode="r",
    )


def _box(array: np.ndarray, factor_zyx: tuple[int, int, int]) -> np.ndarray:
    spatial = array.shape[:3]
    trimmed = tuple(size // factor * factor for size, factor in zip(spatial, factor_zyx, strict=True))
    slices = tuple(slice(0, size) for size in trimmed)
    selected = array[slices]
    bz, by, bx = factor_zyx
    nz, ny, nx = trimmed
    tail = array.shape[3:]
    viewed = selected.reshape(nz // bz, bz, ny // by, by, nx // bx, bx, *tail)
    return viewed.mean(axis=(1, 3, 5), dtype=np.float64)


def _gaussian(
    array: np.ndarray,
    factor_zyx: tuple[int, int, int],
    sigma_zyx: tuple[float, float, float],
    boundary: str,
) -> np.ndarray:
    spatial = array.shape[:3]
    output_shape = tuple(size // factor for size, factor in zip(spatial, factor_zyx, strict=True))
    offset = tuple((factor - 1.0) / 2.0 for factor in factor_zyx)

    def component(values: np.ndarray) -> np.ndarray:
        filtered = gaussian_filter(values, sigma=sigma_zyx, mode=boundary)
        return affine_transform(
            filtered,
            matrix=np.diag(factor_zyx),
            offset=offset,
            output_shape=output_shape,
            order=1,
            mode=boundary,
            prefilter=False,
        )

    if array.ndim == 3:
        return component(array)
    return np.stack([component(array[..., index]) for index in range(array.shape[-1])], axis=-1)


def _reducer(
    kernel: str,
    factor_xyz: tuple[int, int, int],
    sigma_xyz: tuple[float, float, float] | None,
    boundary: str,
):
    factor_zyx = tuple(reversed(factor_xyz))
    if kernel == "box":
        return lambda array: _box(array, factor_zyx)
    assert sigma_xyz is not None
    sigma_zyx = tuple(reversed(sigma_xyz))
    return lambda array: _gaussian(array, factor_zyx, sigma_zyx, boundary)


def _filtered_fields(snapshot: _Snapshot, filtering: Filtering, reduce) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    retained: dict[str, np.ndarray] = {}
    reused = set(filtering.variance).union(filtering.stress)

    def field(name: str) -> np.ndarray:
        return retained[name] if name in retained else snapshot.field(name)

    density = field(filtering.density) if filtering.density is not None else None
    density_filtered = reduce(density) if density is not None else None
    for name in filtering.mean:
        values = density if name == filtering.density else field(name)
        assert values is not None
        result[name] = reduce(values)
        if name in reused:
            retained[name] = values
    for name in filtering.favre:
        assert density is not None and density_filtered is not None
        values = field(name)
        weighted = density[..., None] * values if values.ndim == 4 else density * values
        denominator = density_filtered[..., None] if values.ndim == 4 else density_filtered
        if np.any(denominator <= 0.0):
            _error(f"Favre filtering of {name!r} encountered non-positive density")
        result[name] = reduce(weighted) / denominator
        if name in reused:
            retained[name] = values
    for source, target in filtering.variance.items():
        values = field(source)
        if values.ndim != 3:
            _error(f"variance source {source!r} must be scalar")
        if filtering.density is None:
            mean = result.get(source, reduce(values))
            second = reduce(values * values)
        else:
            assert density is not None and density_filtered is not None
            mean = result.get(source)
            if mean is None:
                mean = reduce(density * values) / density_filtered
            second = reduce(density * values * values) / density_filtered
        result[target] = np.maximum(second - mean * mean, 0.0)
        retained.pop(source, None)
    for source, target in filtering.stress.items():
        velocity = field(source)
        if velocity.ndim != 4 or velocity.shape[-1] != 3:
            _error(f"stress source {source!r} must have three components")
        mean = result.get(source)
        if mean is None:
            if filtering.density is None:
                mean = reduce(velocity)
            else:
                assert density is not None and density_filtered is not None
                mean = reduce(density[..., None] * velocity) / density_filtered[..., None]
        pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
        stress = np.empty((*mean.shape[:3], 6), dtype=np.float64)
        for index, (first, second) in enumerate(pairs):
            product = velocity[..., first] * velocity[..., second]
            moment = (
                reduce(product)
                if filtering.density is None
                else reduce(density * product) / density_filtered
            )
            stress[..., index] = moment - mean[..., first] * mean[..., second]
        result[target] = stress
        retained.pop(source, None)
    for stress_name, energy_name in filtering.energy.items():
        stress = result[stress_name]
        result[energy_name] = 0.5 * np.maximum(
            stress[..., 0] + stress[..., 1] + stress[..., 2], 0.0
        )
    for source, target in filtering.specific.items():
        assert density_filtered is not None
        if np.any(density_filtered <= 0.0):
            _error(f"specific source {target!r} encountered non-positive density")
        result[target] = result[source] / density_filtered
    return result


def _component_names(name: str, values: np.ndarray, source_fields: dict[str, tuple[str, ...]]):
    if name in source_fields and len(source_fields[name]) == (1 if values.ndim == 3 else values.shape[-1]):
        return source_fields[name]
    if values.ndim == 3:
        return (name,)
    suffixes = ("x", "y", "z") if values.shape[-1] == 3 else ("xx", "yy", "zz", "xy", "xz", "yz")
    return tuple(f"{name}_{suffix}" for suffix in suffixes)


def _write_snapshot(
    root: Path,
    data_dir: Path,
    info: dict[str, object],
    record: dict[str, object],
    filtering: Filtering,
    reduce,
    dtype: np.dtype,
) -> tuple[dict[str, object], dict[str, tuple[str, ...]]]:
    snapshot = _Snapshot(root, info, record)
    fields = _filtered_fields(snapshot, filtering, reduce)
    output: dict[str, object] = {"id": record["id"], "time [s]": record["time [s]"]}
    mappings: dict[str, tuple[str, ...]] = {}
    for name, values in fields.items():
        names = _component_names(name, values, snapshot.fields)
        mappings[name] = names
        columns = (values,) if values.ndim == 3 else tuple(
            values[..., index] for index in range(values.shape[-1])
        )
        for component, column in zip(names, columns, strict=True):
            filename = f"{component}_id{int(record['id']):03d}.dat"
            np.asarray(column, dtype=dtype).tofile(data_dir / filename)
            output[f"{component} filename"] = f"./data/{filename}"
    first = next(iter(fields.values()))
    output["cells"] = math.prod(first.shape[:3])
    return output, mappings


def _hdf5_options(compression: str | None) -> dict[str, object]:
    if compression not in {None, "lzf", "gzip"}:
        _error("compression must be None, 'lzf', or 'gzip'")
    return {} if compression is None else {
        "compression": compression,
        "shuffle": True,
        "chunks": True,
    }


def _hdf5_fields(fields: Mapping[str, str] | None) -> dict[str, str]:
    if not isinstance(fields, Mapping) or not fields:
        raise TypeError("fields must be a non-empty mapping when output_file is used")
    try:
        selected = {
            str(name): normalise_hdf5_path(target)
            for name, target in fields.items()
        }
    except ValueError as error:
        _error(str(error))
    targets = tuple(selected.values())
    if any(not name for name in selected):
        _error("field names must be non-empty")
    if len(set(targets)) != len(targets):
        _error("HDF5 field paths must be unique")
    for first in targets:
        for second in targets:
            if first != second and second.startswith(first + "/"):
                _error(f"HDF5 field path {first!r} conflicts with {second!r}")
    return selected


def _write_hdf5_snapshot(
    root: Path,
    stage: Path,
    info: dict[str, object],
    record: dict[str, object],
    filtering: Filtering,
    reduce,
    dtype: np.dtype,
    fields: Mapping[str, str],
    compression: str | None,
) -> tuple[Path, dict[str, tuple[str, ...]]]:
    snapshot = _Snapshot(root, info, record)
    filtered = _filtered_fields(snapshot, filtering, reduce)
    unavailable = set(fields).difference(filtered)
    if unavailable:
        _error(f"HDF5 output requested unavailable fields: {sorted(unavailable)}", KeyError)
    mappings = {
        name: _component_names(name, filtered[name], snapshot.fields)
        for name in fields
    }
    path = stage / f"snapshot-{int(record['id']):06d}.h5"
    options = _hdf5_options(compression)
    with h5py.File(path, "w") as file:
        for name, target in fields.items():
            parent_name, _, dataset_name = target.rpartition("/")
            parent = file.require_group(parent_name) if parent_name else file
            parent.create_dataset(
                dataset_name,
                data=np.asarray(filtered[name], dtype=dtype),
                dtype=dtype,
                **options,
            )
    return path, mappings


def _copy_hdf5_snapshot(
    source_path: Path,
    output: h5py.File,
    record: dict[str, object],
    fields: Mapping[str, str],
) -> None:
    group = output.create_group(f"id_{int(record['id']):03d}")
    group.attrs["snapshot_id"] = int(record["id"])
    group.attrs["time_s"] = float(record["time [s]"])
    with h5py.File(source_path, "r") as source:
        for target in fields.values():
            parent_name, _, dataset_name = target.rpartition("/")
            parent = group.require_group(parent_name) if parent_name else group
            source.copy(source[target], parent, name=dataset_name)


def _downsample_hdf5(
    *,
    source: Path,
    destination: Path,
    info: dict[str, object],
    records: tuple[dict[str, object], ...],
    filtering: Filtering,
    reduce,
    factor_xyz: tuple[int, int, int],
    sigma_xyz: tuple[float, float, float] | None,
    boundary: str,
    kernel: str,
    dtype: str,
    fields: Mapping[str, str],
    compression: str | None,
    workers: int,
    progress: bool,
    overwrite: bool,
    started: float,
) -> DownsampledDataset:
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".stacknordic-downsample-", dir=destination.parent))
    temporary = stage / destination.name
    reporter = start_progress(len(records), "StackNordic downsampling") if progress else None
    mappings: dict[str, tuple[str, ...]] | None = None
    snapshots: list[dict[str, object]] = []
    global_info = info["global"]
    try:
        with h5py.File(temporary, "w") as output:
            grid = output.create_group("grid")
            for axis, factor_value in zip("xyz", factor_xyz, strict=True):
                values = _grid_axis(source, global_info, axis)
                count = len(values) // factor_value * factor_value
                reduced = np.asarray(values[:count]).reshape(-1, factor_value).mean(axis=1)
                grid.create_dataset(f"{axis}_axis", data=reduced.astype(dtype))

            def task(index: int):
                path, current_mappings = _write_hdf5_snapshot(
                    source,
                    stage,
                    info,
                    records[index],
                    filtering,
                    reduce,
                    np.dtype(dtype),
                    fields,
                    compression,
                )
                return index, path, current_mappings

            with ThreadPoolExecutor(max_workers=workers) as executor:
                pending = {
                    executor.submit(task, index): index
                    for index in range(min(workers, len(records)))
                }
                next_index = len(pending)
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future)
                        index, path, current_mappings = future.result()
                        if mappings is None:
                            mappings = current_mappings
                        elif mappings != current_mappings:
                            _error("filtered field components changed between snapshots")
                        record = records[index]
                        _copy_hdf5_snapshot(path, output, record, fields)
                        path.unlink()
                        snapshots.append(
                            {
                                "id": int(record["id"]),
                                "time_s": float(record["time [s]"]),
                                "group": f"id_{int(record['id']):03d}",
                            }
                        )
                        if reporter is not None:
                            reporter.update()
                        if next_index < len(records):
                            pending[executor.submit(task, next_index)] = next_index
                            next_index += 1

            assert mappings is not None
            snapshots.sort(key=lambda item: (item["time_s"], item["id"]))
            nxyz = tuple(int(value) for value in global_info["Nxyz"])
            reduced_nxyz = tuple(
                size // factor
                for size, factor in zip(nxyz, factor_xyz, strict=True)
            )
            definition = contract(
                schema=SERIES_SCHEMA,
                name=destination.stem,
                dtype=dtype,
                shape_zyx=tuple(reversed(reduced_nxyz)),
                fields=dict(fields),
                components=mappings,
                metadata={
                    **dict(global_info.get("metadata") or {}),
                    "downsampling": {
                        "kernel": kernel,
                        "factor_xyz": list(factor_xyz),
                        "sigma_xyz": list(sigma_xyz) if sigma_xyz is not None else None,
                        "boundary": boundary if kernel == "gaussian" else None,
                        "source": str(source),
                    },
                },
            )
            write_contract(output, definition)
            output.attrs["schema"] = SERIES_SCHEMA
            output.attrs["snapshots"] = len(snapshots)
            output.attrs[SNAPSHOTS_ATTRIBUTE] = json.dumps(snapshots, separators=(",", ":"))
        if destination.exists():
            if not overwrite:
                _error(f"output file already exists: {destination}", FileExistsError)
            destination.unlink()
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        if reporter is not None:
            finish_progress(reporter)
    stage.rmdir()
    return DownsampledDataset(
        input_dir=source,
        output_dir=destination.parent,
        output_file=destination,
        kernel=kernel,
        factor=factor_xyz,
        snapshots=len(records),
        n_jobs=workers,
        elapsed_seconds=perf_counter() - started,
    )


def downsample(
    *,
    dataset_dir: str | Path,
    factor: tuple[int, int, int],
    filtering: Filtering,
    output_dir: str | Path | None = None,
    output_file: str | Path | None = None,
    fields: Mapping[str, str] | None = None,
    kernel: str = "box",
    sigma: tuple[float, float, float] | None = None,
    boundary: str = "reflect",
    dtype: str = "float32",
    compression: str | None = "lzf",
    n_jobs: int | None = None,
    progress: bool = False,
    overwrite: bool = False,
) -> DownsampledDataset:
    """Downsample a Cartesian exported dataset with explicit physical rules."""

    started = perf_counter()
    source = Path(dataset_dir).expanduser().resolve()
    if (output_dir is None) == (output_file is None):
        _error("provide exactly one of output_dir and output_file")
    destination = Path(output_dir or output_file).expanduser().resolve()
    selected_fields = _hdf5_fields(fields) if output_file is not None else None
    if output_dir is not None and fields is not None:
        _error("fields is only valid with output_file")
    _hdf5_options(compression)
    if kernel not in {"box", "gaussian"}:
        _error("kernel must be 'box' or 'gaussian'")
    factor_xyz = _triplet(factor, "factor", int)
    sigma_xyz = None if sigma is None else _triplet(sigma, "sigma", float)
    if kernel == "box" and sigma is not None:
        _error("sigma is only valid for the Gaussian kernel")
    if kernel == "gaussian" and sigma_xyz is None:
        _error("the Gaussian kernel requires sigma")
    if boundary not in {"reflect", "nearest", "wrap"}:
        _error("boundary must be 'reflect', 'nearest', or 'wrap'")
    if dtype not in {"float32", "float64"}:
        _error("dtype must be 'float32' or 'float64'")
    if not isinstance(filtering, Filtering):
        raise TypeError("filtering must be a StackNordic Filtering contract")
    if destination.exists() and not overwrite:
        kind = "file" if output_file is not None else "directory"
        _error(f"output {kind} already exists: {destination}", FileExistsError)
    info, source = _source_info(source)
    global_info = info.get("global", {})
    if global_info.get("layout") != "cartesian" or "Nxyz" not in global_info:
        _error("downsampling currently requires a verified Cartesian dataset")
    if not isinstance(global_info.get("fields"), dict):
        _error("dataset manifest does not contain logical field mappings")
    records = tuple(info.get("local", ()))
    if not records:
        _error("dataset contains no snapshots")
    first_record = records[0]
    requested = set(filtering.mean).union(filtering.favre, filtering.variance, filtering.stress)
    if filtering.density is not None:
        requested.add(filtering.density)
    unavailable = requested.difference(global_info["fields"])
    if unavailable:
        _error(f"filtering requested unavailable fields: {sorted(unavailable)}", KeyError)
    if "hdf_fields" in global_info:
        cell_count = int(np.prod(tuple(global_info["Nxyz"])))
        input_bytes = sum(
            cell_count
            * np.dtype(global_info["dtype"]).itemsize
            * len(global_info["fields"][name])
            for name in requested
        )
    else:
        input_bytes = sum(
            (source / component_file(first_record, component)).stat().st_size
            for name in requested
            for component in global_info["fields"][name]
        )
    estimated_worker_bytes = max(input_bytes * 6, 256 * 1024**2)
    selected_workers = memory_bounded_workers(n_jobs, len(records), estimated_worker_bytes)
    if n_jobs == -1 and selected_workers < min(len(records), os.cpu_count() or 1):
        print(
            f"[StackNordic] Downsampling workers: n_jobs=-1 resolved to {selected_workers} "
            "from the available memory and selected fields."
        )
    reduce = _reducer(kernel, factor_xyz, sigma_xyz, boundary)
    if output_file is not None:
        assert selected_fields is not None
        return _downsample_hdf5(
            source=source,
            destination=destination,
            info=info,
            records=records,
            filtering=filtering,
            reduce=reduce,
            factor_xyz=factor_xyz,
            sigma_xyz=sigma_xyz,
            boundary=boundary,
            kernel=kernel,
            dtype=dtype,
            fields=selected_fields,
            compression=compression,
            workers=selected_workers,
            progress=progress,
            overwrite=overwrite,
            started=started,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".stacknordic-downsample-", dir=destination.parent))
    data_dir = stage / "data"
    grid_dir = stage / "grid"
    data_dir.mkdir()
    grid_dir.mkdir()
    output_records: list[dict[str, object] | None] = [None] * len(records)
    mappings: dict[str, tuple[str, ...]] = {}
    reporter = (
        start_progress(len(records), "StackNordic downsampling")
        if progress
        else None
    )
    try:
        def task(index: int):
            record, fields = _write_snapshot(
                source,
                data_dir,
                info,
                records[index],
                filtering,
                reduce,
                np.dtype(dtype),
            )
            return index, record, fields

        if selected_workers == 1:
            completed = (task(index) for index in range(len(records)))
            for index, record, current_mappings in completed:
                output_records[index] = record
                mappings.update(current_mappings)
                if reporter is not None:
                    reporter.update()
        else:
            with ThreadPoolExecutor(max_workers=selected_workers) as executor:
                futures = [executor.submit(task, index) for index in range(len(records))]
                for future in as_completed(futures):
                    index, record, current_mappings = future.result()
                    output_records[index] = record
                    mappings.update(current_mappings)
                    if reporter is not None:
                        reporter.update()
        nxyz = tuple(int(value) for value in global_info["Nxyz"])
        reduced_nxyz = tuple(size // factor for size, factor in zip(nxyz, factor_xyz, strict=True))
        grid_entries: dict[str, str] = {}
        for axis, factor_value in zip("xyz", factor_xyz, strict=True):
            values = _grid_axis(source, global_info, axis)
            count = len(values) // factor_value * factor_value
            reduced = np.asarray(values[:count]).reshape(-1, factor_value).mean(axis=1)
            filename = f"{axis.upper()}_m.dat"
            reduced.astype(dtype).tofile(grid_dir / filename)
            grid_entries[axis] = f"./grid/{filename}"
        output_info = {
            "global": {
                "snapshots": len(records),
                "variables": [component for names in mappings.values() for component in names],
                "dtype": dtype,
                "layout": "cartesian",
                "grid": grid_entries,
                "fields": {name: list(names) for name, names in mappings.items()},
                "Nxyz": list(reduced_nxyz),
                "metadata": {
                    **dict(global_info.get("metadata") or {}),
                    "downsampling": {
                        "kernel": kernel,
                        "factor_xyz": list(factor_xyz),
                        "sigma_xyz": list(sigma_xyz) if sigma_xyz is not None else None,
                        "boundary": boundary if kernel == "gaussian" else None,
                        "source": str(source),
                    },
                },
            },
            "local": output_records,
        }
        (stage / "info.json").write_text(json.dumps(output_info, indent=4) + "\n")
        if destination.exists():
            shutil.rmtree(destination)
        stage.replace(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        if reporter is not None:
            finish_progress(reporter)
    return DownsampledDataset(
        input_dir=source,
        output_dir=destination,
        kernel=kernel,
        factor=factor_xyz,
        snapshots=len(records),
        n_jobs=selected_workers,
        elapsed_seconds=perf_counter() - started,
        output_file=None,
    )
