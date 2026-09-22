"""Package StackNordic snapshots into portable analysis files."""

from __future__ import annotations

import json
import os
import queue
import shutil
import tempfile
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from .._progress import finish_progress, start_progress
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
class PackagedDataset:
    """Summary of one completed dataset packaging operation."""

    dataset_dir: Path
    output_dir: Path
    files: tuple[Path, ...]
    format: str
    snapshots: int
    n_jobs: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class CompiledDataset:
    """Summary of one compiled HDF5 time series."""

    dataset_dir: Path
    output_file: Path
    source_files: tuple[Path, ...]
    snapshots: int
    elapsed_seconds: float


def _error(message: str, error_type: type[Exception] = ValueError):
    raise error_type(f"[StackNordic] {message}")


def _normalise_fields(fields: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    if not isinstance(fields, Mapping) or not fields:
        raise TypeError("fields must be a non-empty mapping")
    try:
        selected = tuple(
            (str(name), normalise_hdf5_path(target)) for name, target in fields.items()
        )
    except ValueError as error:
        _error(str(error))
    if any(not name for name, _ in selected):
        _error("field names must be non-empty")
    targets = tuple(target for _, target in selected)
    if len(set(targets)) != len(targets):
        _error("HDF5 field paths must be unique")
    for first in targets:
        for second in targets:
            if first != second and second.startswith(first + "/"):
                _error(f"HDF5 field path {first!r} conflicts with {second!r}")
    return selected


def _dataset_options(compression: str | None) -> dict[str, object]:
    if compression not in {None, "lzf", "gzip"}:
        _error("compression must be None, 'lzf', or 'gzip'")
    if compression is None:
        return {}
    return {"compression": compression, "shuffle": True, "chunks": True}


def _write_array(
    output: h5py.Dataset,
    source: np.memmap,
    *,
    component: int | None = None,
) -> None:
    cells_per_plane = int(np.prod(source.shape[1:]))
    planes = max(1, (64 * 1024**2) // max(cells_per_plane * source.dtype.itemsize, 1))
    for start in range(0, source.shape[0], planes):
        stop = min(start + planes, source.shape[0])
        if component is None:
            output[start:stop] = source[start:stop]
        else:
            output[start:stop, ..., component] = source[start:stop]


def _write_field(
    file: h5py.File,
    root: Path,
    record: Mapping[str, object],
    components: tuple[str, ...],
    target: str,
    shape: tuple[int, int, int],
    dtype: np.dtype,
    options: Mapping[str, object],
) -> None:
    parent_name, _, dataset_name = target.rpartition("/")
    parent = file.require_group(parent_name) if parent_name else file
    output_shape = shape if len(components) == 1 else (*shape, len(components))
    output = parent.create_dataset(dataset_name, shape=output_shape, dtype=dtype, **options)
    for index, component in enumerate(components):
        source = np.memmap(
            root / component_file(record, component),
            dtype=dtype,
            mode="r",
            shape=shape,
        )
        _write_array(output, source, component=None if len(components) == 1 else index)


def _write_snapshot(
    root: Path,
    stage: Path,
    name: str,
    record: Mapping[str, object],
    global_info: Mapping[str, object],
    fields: tuple[tuple[str, str], ...],
    compression: str | None,
    progress_callback=None,
) -> Path:
    snapshot_id = int(record["id"])
    destination = stage / f"{name}_id{snapshot_id:03d}.h5"
    dtype = np.dtype(global_info["dtype"])
    shape = tuple(reversed(tuple(int(value) for value in global_info["Nxyz"])))
    mappings = global_info["fields"]
    options = _dataset_options(compression)
    definition = contract(
        schema=SNAPSHOT_SCHEMA,
        name=name,
        dtype=dtype.name,
        shape_zyx=shape,
        fields=dict(fields),
        components={source: tuple(mappings[source]) for source, _ in fields},
        metadata=dict(global_info.get("metadata") or {}),
    )
    with h5py.File(destination, "w") as file:
        file.attrs["name"] = name
        file.attrs["snapshot_id"] = snapshot_id
        file.attrs["time_s"] = float(record["time [s]"])
        file.attrs["dtype"] = dtype.name
        file.attrs["schema"] = "stacknordic-hdf5-v1"
        metadata = dict(global_info.get("metadata") or {})
        file.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
        if "equivalence_ratio" in metadata:
            file.attrs["phi"] = float(metadata["equivalence_ratio"])
        write_contract(file, definition)
        grid = file.create_group("grid")
        for axis in "xyz":
            values = np.fromfile(root / global_info["grid"][axis], dtype=dtype)
            grid.create_dataset(f"{axis}_axis", data=values, dtype=dtype, **options)
            if progress_callback is not None:
                progress_callback(1)
        for source, target in fields:
            _write_field(
                file,
                root,
                record,
                tuple(mappings[source]),
                target,
                shape,
                dtype,
                options,
            )
            if progress_callback is not None:
                progress_callback(1)
    return destination


def package(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    name: str,
    format: str = "hdf5",
    fields: Mapping[str, str],
    compression: str | None = "lzf",
    n_jobs: int | None = None,
    progress: bool = False,
    overwrite: bool = False,
) -> PackagedDataset:
    """Package each Cartesian snapshot as one HDF5 file."""

    started = perf_counter()
    source = Path(dataset_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if format != "hdf5":
        _error("format must be 'hdf5'")
    if not isinstance(name, str) or not name.strip() or "/" in name:
        _error("name must be a non-empty filename stem")
    selected = _normalise_fields(fields)
    try:
        info = json.loads((source / "info.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        _error(f"dataset manifest is unavailable or invalid in {source}")
    global_info = info.get("global", {})
    if global_info.get("layout") != "cartesian" or "Nxyz" not in global_info:
        _error("HDF5 packaging requires a verified Cartesian dataset")
    if not isinstance(global_info.get("grid"), dict):
        _error("dataset manifest does not contain a Cartesian grid")
    available = set(global_info.get("fields", {}))
    missing = {field for field, _ in selected}.difference(available)
    if missing:
        _error(f"packaging requested unavailable fields: {sorted(missing)}", KeyError)
    records = tuple(info.get("local", ()))
    if not records:
        _error("dataset contains no snapshots")
    if destination.exists():
        if not destination.is_dir():
            _error(f"output path is not a directory: {destination}", FileExistsError)
        if any(destination.iterdir()) and not overwrite:
            _error(f"output directory is not empty: {destination}", FileExistsError)
    destination.parent.mkdir(parents=True, exist_ok=True)
    estimated_worker_bytes = 192 * 1024**2
    selected_workers = memory_bounded_workers(n_jobs, len(records), estimated_worker_bytes)
    stage = Path(tempfile.mkdtemp(prefix=".stacknordic-package-", dir=destination.parent))
    reporter = (
        start_progress(len(records) * (len(selected) + 3), "StackNordic packaging")
        if progress
        else None
    )
    outputs: list[Path | None] = [None] * len(records)
    try:
        updates: queue.SimpleQueue[int] = queue.SimpleQueue()

        def task(index: int, callback=None) -> tuple[int, Path]:
            return index, _write_snapshot(
                source,
                stage,
                name.strip(),
                records[index],
                global_info,
                selected,
                compression,
                callback,
            )

        def report_updates() -> None:
            if reporter is None:
                return
            completed = 0
            while True:
                try:
                    completed += updates.get_nowait()
                except queue.Empty:
                    break
            if completed:
                reporter.update(completed)

        if selected_workers == 1:
            callback = reporter.update if reporter is not None else None
            completed = (task(index, callback) for index in range(len(records)))
            for index, path in completed:
                outputs[index] = path
        else:
            with ThreadPoolExecutor(max_workers=selected_workers) as executor:
                callback = updates.put if reporter is not None else None
                pending = {
                    executor.submit(task, index, callback)
                    for index in range(len(records))
                }
                while pending:
                    done, pending = wait(
                        pending,
                        timeout=0.1,
                        return_when=FIRST_COMPLETED,
                    )
                    report_updates()
                    for future in done:
                        index, path = future.result()
                        outputs[index] = path
                report_updates()
        if destination.exists():
            shutil.rmtree(destination)
        stage.replace(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        if reporter is not None:
            finish_progress(reporter)
    files = tuple(destination / path.name for path in outputs if path is not None)
    return PackagedDataset(
        dataset_dir=source,
        output_dir=destination,
        files=files,
        format=format,
        snapshots=len(records),
        n_jobs=selected_workers,
        elapsed_seconds=perf_counter() - started,
    )


def _copy_dataset(source: h5py.File, target: h5py.Group, path: str) -> None:
    parent_name, _, dataset_name = path.rpartition("/")
    parent = target.require_group(parent_name) if parent_name else target
    source.copy(source[path], parent, name=dataset_name)


def compile(
    *,
    dataset_dir: str | Path,
    output_file: str | Path,
    pattern: str,
    progress: bool = False,
    overwrite: bool = False,
) -> CompiledDataset:
    """Compile self-describing snapshot packages into one HDF5 series."""

    started = perf_counter()
    source = Path(dataset_dir).expanduser().resolve()
    destination = Path(output_file).expanduser().resolve()
    if not source.is_dir():
        _error(f"dataset directory was not found: {source}", FileNotFoundError)
    if not isinstance(pattern, str) or not pattern.strip():
        _error("pattern must be a non-empty glob string")
    candidates = tuple(
        path for path in sorted(source.glob(pattern))
        if path.is_file() and path.resolve() != destination
    )
    if not candidates:
        _error(f"pattern matched no snapshot packages in {source}", FileNotFoundError)
    snapshots: list[tuple[int, float, Path]] = []
    definitions = []
    for path in candidates:
        with h5py.File(path, "r") as file:
            definition = read_contract(file)
            if definition.schema != SNAPSHOT_SCHEMA:
                _error(f"input is not a snapshot package: {path.name}")
            definitions.append(definition)
            snapshots.append((int(file.attrs["snapshot_id"]), float(file.attrs["time_s"]), path))
    reference = definitions[0]
    for path, definition in zip(candidates[1:], definitions[1:], strict=True):
        if not compatible(reference, definition):
            _error(f"snapshot package contract differs: {path.name}")
    snapshot_ids = [item[0] for item in snapshots]
    times = [item[1] for item in snapshots]
    if len(set(snapshot_ids)) != len(snapshot_ids):
        _error("snapshot packages contain duplicate snapshot identifiers")
    if len(set(times)) != len(times):
        _error("snapshot packages contain duplicate physical times")
    snapshots.sort(key=lambda item: (item[1], item[0]))
    if destination.exists() and not overwrite:
        _error(f"output file already exists: {destination}", FileExistsError)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}-",
        suffix=destination.suffix or ".h5",
        dir=destination.parent,
    )
    os.close(descriptor)
    Path(temporary_name).unlink()
    temporary = Path(temporary_name)
    reporter = start_progress(len(snapshots), "StackNordic compiling") if progress else None
    try:
        with h5py.File(temporary, "w") as output:
            write_contract(output, reference.with_schema(SERIES_SCHEMA))
            output.attrs["schema"] = "stacknordic-hdf5-series-v1"
            output.attrs["snapshots"] = len(snapshots)
            output.attrs[SNAPSHOTS_ATTRIBUTE] = json.dumps(
                [
                    {"id": snapshot_id, "time_s": time, "group": f"id_{snapshot_id:03d}"}
                    for snapshot_id, time, _ in snapshots
                ],
                separators=(",", ":"),
            )
            with h5py.File(snapshots[0][2], "r") as first:
                first.copy(first["grid"], output, name="grid")
            for snapshot_id, time, path in snapshots:
                group = output.create_group(f"id_{snapshot_id:03d}")
                group.attrs["snapshot_id"] = snapshot_id
                group.attrs["time_s"] = time
                with h5py.File(path, "r") as input_file:
                    for target in reference.fields.values():
                        _copy_dataset(input_file, group, target)
                if reporter is not None:
                    reporter.update()
        if destination.exists():
            destination.unlink()
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if reporter is not None:
            finish_progress(reporter)
    return CompiledDataset(
        dataset_dir=source,
        output_file=destination,
        source_files=tuple(item[2] for item in snapshots),
        snapshots=len(snapshots),
        elapsed_seconds=perf_counter() - started,
    )


__all__ = ["CompiledDataset", "PackagedDataset", "compile", "package"]
