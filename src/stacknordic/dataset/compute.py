"""Persistent derived-field computation for exported datasets."""

from __future__ import annotations

import json
import os
import queue
import shutil
import tempfile
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .._progress import finish_progress, start_progress
from ..computing import (
    Computation,
    ProgressComputation,
    ReactionRateComputation,
    reaction_rate_chunks,
)
from ..resources import memory_bounded_workers
from ._manifest import component_file


@dataclass(frozen=True, slots=True)
class ComputedDataset:
    """Summary of persisted derived fields."""

    dataset_dir: Path
    fields: tuple[str, ...]
    snapshots: int
    n_jobs: int
    elapsed_seconds: float


def _error(message: str, error_type: type[Exception] = ValueError):
    raise error_type(f"[StackNordic] {message}")


class _Snapshot:
    def __init__(self, root: Path, info: dict[str, object], record: dict[str, object]):
        global_info = info["global"]
        self.root = root
        self.record = record
        self.dtype = np.dtype(global_info["dtype"])
        self.shape = tuple(reversed(tuple(int(value) for value in global_info["Nxyz"])))
        self.fields = {
            name: tuple(components)
            for name, components in global_info["fields"].items()
        }

    def field(self, name: str) -> np.ndarray:
        if name not in self.fields:
            _error(f"computation requested unavailable field {name!r}", KeyError)
        components = self.fields[name]
        if len(components) != 1:
            _error(f"computation input {name!r} must be scalar")
        path = self.root / component_file(self.record, components[0])
        return np.memmap(path, dtype=self.dtype, mode="r", shape=self.shape)


def _compute_snapshot(
    root: Path,
    stage: Path,
    info: dict[str, object],
    record: dict[str, object],
    computations: tuple[Computation, ...],
    dtype: np.dtype,
    progress_callback=None,
) -> tuple[int, dict[str, Path]]:
    snapshot = _Snapshot(root, info, record)
    outputs: dict[str, Path] = {}
    snapshot_id = int(record["id"])
    cell_count = int(np.prod(snapshot.shape))
    chunk_cells = 262_144
    reaction_groups: dict[tuple[object, ...], list[ReactionRateComputation]] = {}
    for definition in computations:
        if isinstance(definition, ReactionRateComputation):
            reaction_groups.setdefault(definition.state_key(), []).append(definition)
            continue
        path = stage / f"{definition.output}_id{snapshot_id:03d}.dat"
        output = np.memmap(path, dtype=dtype, mode="w+", shape=(cell_count,))
        fields = {
            name: np.asarray(snapshot.field(name)).reshape(-1)
            for name in definition.inputs
        }
        for start in range(0, cell_count, chunk_cells):
            stop = min(start + chunk_cells, cell_count)
            values = np.full(stop - start, definition.offset)
            for name, coefficient in definition.coefficients.items():
                values += coefficient * np.asarray(fields[name][start:stop], dtype=np.float64)
            if not np.all(np.isfinite(values)):
                _error(
                    f"computed field {definition.output!r} contains non-finite values "
                    f"at snapshot {snapshot_id}"
                )
            output[start:stop] = values
            if progress_callback is not None:
                progress_callback(stop - start)
        output.flush()
        del output
        outputs[definition.output] = path
    for definitions in reaction_groups.values():
        paths = {
            definition.output: stage / f"{definition.output}_id{snapshot_id:03d}.dat"
            for definition in definitions
        }
        mapped = {
            name: np.memmap(path, dtype=dtype, mode="w+", shape=(cell_count,))
            for name, path in paths.items()
        }
        try:
            for start, stop, values in reaction_rate_chunks(tuple(definitions), snapshot.field):
                for name, chunk in values.items():
                    if not np.all(np.isfinite(chunk)):
                        _error(
                            f"computed field {name!r} contains non-finite values "
                            f"at snapshot {snapshot_id}"
                        )
                    mapped[name][start:stop] = chunk
                if progress_callback is not None:
                    progress_callback(stop - start)
            for output in mapped.values():
                output.flush()
        finally:
            mapped.clear()
        outputs.update(paths)
    return snapshot_id, outputs


def compute(
    *,
    dataset_dir: str | Path,
    computations: Sequence[Computation],
    dtype: str | None = None,
    n_jobs: int | None = None,
    progress: bool = False,
    overwrite: bool = False,
) -> ComputedDataset:
    """Compute and persist declared fields in an exported dataset."""

    started = perf_counter()
    root = Path(dataset_dir).expanduser().resolve()
    try:
        info = json.loads((root / "info.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        _error(f"dataset manifest is unavailable or invalid in {root}")
    global_info = info.get("global", {})
    if global_info.get("layout") != "cartesian" or "Nxyz" not in global_info:
        _error("stored field computation currently requires a verified Cartesian dataset")
    if not isinstance(global_info.get("fields"), dict):
        _error("dataset manifest does not contain logical field mappings")
    records = tuple(info.get("local", ()))
    if not records:
        _error("dataset contains no snapshots")
    definitions = tuple(computations)
    if not definitions:
        _error("computations must contain at least one definition")
    if any(
        not isinstance(item, (ProgressComputation, ReactionRateComputation))
        for item in definitions
    ):
        raise TypeError("computations must contain StackNordic computation contracts")
    names = tuple(item.output for item in definitions)
    if len(set(names)) != len(names):
        _error("computation outputs must be unique")
    existing = set(names).intersection(global_info["fields"])
    if existing and not overwrite:
        _error(f"computed fields already exist: {sorted(existing)}", FileExistsError)
    required = set().union(*(set(item.inputs) for item in definitions))
    unavailable = required.difference(global_info["fields"])
    if unavailable:
        _error(f"computations requested unavailable fields: {sorted(unavailable)}", KeyError)
    output_dtype = np.dtype(global_info["dtype"] if dtype is None else dtype)
    if output_dtype.name not in {"float32", "float64"}:
        _error("dtype must be 'float32', 'float64', or None")
    selected_workers = memory_bounded_workers(
        n_jobs,
        len(records),
        512 * 1024**2,
    )
    if n_jobs == -1 and selected_workers < min(len(records), os.cpu_count() or 1):
        print(
            f"[StackNordic] Computing workers: n_jobs=-1 resolved to {selected_workers} "
            "from the available memory and bounded Cantera workspace."
        )
    stage = Path(tempfile.mkdtemp(prefix=".stacknordic-compute-", dir=root))
    algebraic_count = sum(
        isinstance(definition, ProgressComputation) for definition in definitions
    )
    reaction_count = len(
        {
            definition.state_key()
            for definition in definitions
            if isinstance(definition, ReactionRateComputation)
        }
    )
    work_per_cell = algebraic_count + reaction_count
    total_cells = sum(
        int(record.get("cells", np.prod(tuple(reversed(global_info["Nxyz"])))))
        for record in records
    )
    reporter = (
        start_progress(total_cells * work_per_cell, "StackNordic computing")
        if progress
        else None
    )
    results: dict[int, dict[str, Path]] = {}
    try:
        updates: queue.SimpleQueue[int] = queue.SimpleQueue()

        def task(index: int, callback=None):
            return _compute_snapshot(
                root,
                stage,
                info,
                records[index],
                definitions,
                output_dtype,
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
            for snapshot_id, outputs in completed:
                results[snapshot_id] = outputs
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
                        snapshot_id, outputs = future.result()
                        results[snapshot_id] = outputs
                report_updates()
        data_dir = root / "data"
        data_dir.mkdir(exist_ok=True)
        for record in records:
            snapshot_id = int(record["id"])
            for name, staged_path in results[snapshot_id].items():
                filename = staged_path.name
                staged_path.replace(data_dir / filename)
                record[f"{name} filename"] = f"./data/{filename}"
        for definition in definitions:
            global_info["fields"][definition.output] = [definition.output]
            if definition.output not in global_info["variables"]:
                global_info["variables"].append(definition.output)
        metadata = global_info.setdefault("metadata", {})
        computation_metadata = metadata.setdefault("computing", {})
        for definition in definitions:
            computation_metadata[definition.output] = definition.manifest()
        manifest = stage / "info.json"
        manifest.write_text(json.dumps(info, indent=4) + "\n")
        manifest.replace(root / "info.json")
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        if reporter is not None:
            finish_progress(reporter)
    shutil.rmtree(stage, ignore_errors=True)
    return ComputedDataset(
        dataset_dir=root,
        fields=names,
        snapshots=len(records),
        n_jobs=selected_workers,
        elapsed_seconds=perf_counter() - started,
    )
