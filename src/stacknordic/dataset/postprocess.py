"""Post-process Cartesian StackNordic datasets."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .._progress import finish_progress, start_progress
from ..postprocess import PostprocessOperation
from ..resources import memory_bounded_workers
from ._manifest import component_file


@dataclass(frozen=True, slots=True)
class PostprocessedDataset:
    """Summary of fields added to one Cartesian dataset."""

    dataset_dir: Path
    fields: tuple[str, ...]
    snapshots: int
    elapsed_seconds: float


def _error(message: str, error_type: type[Exception] = ValueError):
    raise error_type(f"[StackNordic] {message}")


class _Dataset:
    def __init__(self, root: Path, info: dict[str, object]):
        global_info = info["global"]
        if global_info.get("layout") != "cartesian" or "Nxyz" not in global_info:
            _error("dataset post-processing requires a Cartesian dataset")
        self.root = root
        self.info = info
        self.records = tuple(info["local"])
        self.dtype = np.dtype(global_info["dtype"])
        self.nxyz = tuple(int(value) for value in global_info["Nxyz"])
        self.shape = tuple(reversed(self.nxyz))
        self.fields = {
            name: tuple(components)
            for name, components in global_info["fields"].items()
        }
        self.coordinates = tuple(
            np.fromfile(root / global_info["grid"][axis], dtype=self.dtype)
            for axis in "xyz"
        )
        if tuple(axis.size for axis in self.coordinates) != self.nxyz:
            _error("Cartesian coordinate lengths do not match Nxyz")
        if any(axis.size > 1 and np.any(np.diff(axis) <= 0.0) for axis in self.coordinates):
            _error("Cartesian coordinates must be strictly increasing")
        times = np.asarray([float(record["time [s]"]) for record in self.records])
        if times.size > 1 and np.any(np.diff(times) <= 0.0):
            _error("dataset times must be strictly increasing")

    def field(self, index: int, name: str) -> np.ndarray:
        if name not in self.fields:
            _error(f"post-processing requested unavailable field {name!r}", KeyError)
        record = self.records[index]
        arrays = [
            np.memmap(
                self.root / component_file(record, component),
                dtype=self.dtype,
                mode="r",
                shape=self.shape,
            )
            for component in self.fields[name]
        ]
        if len(arrays) == 1:
            return np.asarray(arrays[0], dtype=np.float64)
        return np.stack(arrays, axis=-1).astype(np.float64, copy=False)


def _derivative(values: np.ndarray, coordinates: np.ndarray, axis: int) -> np.ndarray:
    if values.shape[axis] == 1:
        return np.zeros_like(values, dtype=np.float64)
    edge_order = 2 if values.shape[axis] > 2 else 1
    return np.gradient(values, coordinates, axis=axis, edge_order=edge_order)


def _gradient(values: np.ndarray, coordinates: tuple[np.ndarray, ...]) -> np.ndarray:
    derivatives = tuple(
        _derivative(values, coordinate, axis)
        for coordinate, axis in zip(reversed(coordinates), range(3), strict=True)
    )
    ordered = tuple(reversed(derivatives))
    if values.ndim == 3:
        return np.stack(ordered, axis=-1)
    return np.stack(ordered, axis=-1).reshape((*values.shape[:3], -1))


def _velocity_gradient(dataset: _Dataset, index: int, field: str) -> np.ndarray:
    values = dataset.field(index, field)
    if values.ndim != 4 or values.shape[-1] != 3:
        _error(f"{field!r} must have three components")
    return _gradient(values, dataset.coordinates).reshape((*dataset.shape, 3, 3))


def _evaluate(
    dataset: _Dataset,
    operation: PostprocessOperation,
    index: int,
) -> dict[str, np.ndarray]:
    kind = operation.kind
    field = operation.field
    if kind == "write_cell_centres":
        x_values, y_values, z_values = dataset.coordinates
        z, y, x = np.meshgrid(z_values, y_values, x_values, indexing="ij")
        return {operation.outputs[0]: np.stack((x, y, z), axis=-1)}
    if kind == "write_cell_volumes":
        if any(axis.size < 2 for axis in dataset.coordinates):
            _error("cell volumes require at least two centres along every Cartesian axis")
        widths = []
        for coordinates in dataset.coordinates:
            faces = np.empty(coordinates.size + 1, dtype=np.float64)
            faces[1:-1] = 0.5 * (coordinates[:-1] + coordinates[1:])
            faces[0] = coordinates[0] - 0.5 * (coordinates[1] - coordinates[0])
            faces[-1] = coordinates[-1] + 0.5 * (coordinates[-1] - coordinates[-2])
            widths.append(np.diff(faces))
        dx, dy, dz = widths
        return {operation.outputs[0]: dz[:, None, None] * dy[None, :, None] * dx[None, None, :]}
    assert field is not None
    values = dataset.field(index, field)
    if kind == "components":
        if values.ndim != 4 or len(operation.outputs) != values.shape[-1]:
            _error(f"components({field}) requires one output per component")
        return {
            name: values[..., component]
            for component, name in enumerate(operation.outputs)
        }
    if kind == "mag":
        return {operation.outputs[0]: np.abs(values) if values.ndim == 3 else np.linalg.norm(values, axis=-1)}
    if kind == "mag_sqr":
        return {operation.outputs[0]: values * values if values.ndim == 3 else np.sum(values * values, axis=-1)}
    if kind == "grad":
        return {operation.outputs[0]: _gradient(values, dataset.coordinates)}
    if kind == "ddt":
        times = np.asarray([float(record["time [s]"]) for record in dataset.records])
        if times.size < 2:
            _error("ddt requires at least two snapshots")
        if index == 0:
            result = (dataset.field(1, field) - values) / (times[1] - times[0])
        elif index == times.size - 1:
            result = (values - dataset.field(index - 1, field)) / (times[index] - times[index - 1])
        else:
            result = (
                dataset.field(index + 1, field) - dataset.field(index - 1, field)
            ) / (times[index + 1] - times[index - 1])
        return {operation.outputs[0]: result}
    gradient = _velocity_gradient(dataset, index, field)
    if kind == "div":
        return {operation.outputs[0]: np.trace(gradient, axis1=-2, axis2=-1)}
    if kind == "vorticity":
        return {
            operation.outputs[0]: np.stack(
                (
                    gradient[..., 2, 1] - gradient[..., 1, 2],
                    gradient[..., 0, 2] - gradient[..., 2, 0],
                    gradient[..., 1, 0] - gradient[..., 0, 1],
                ),
                axis=-1,
            )
        }
    symmetric = 0.5 * (gradient + np.swapaxes(gradient, -1, -2))
    rotation = 0.5 * (gradient - np.swapaxes(gradient, -1, -2))
    if kind == "Q":
        value = 0.5 * (
            np.sum(rotation * rotation, axis=(-2, -1))
            - np.sum(symmetric * symmetric, axis=(-2, -1))
        )
        return {operation.outputs[0]: value}
    if kind == "Lambda2":
        matrix = symmetric @ symmetric + rotation @ rotation
        return {operation.outputs[0]: np.linalg.eigvalsh(matrix)[..., 1]}
    _error(f"unsupported post-processing operation {kind!r}")


def _components(name: str, values: np.ndarray) -> tuple[str, ...]:
    count = 1 if values.ndim == 3 else values.shape[-1]
    suffixes = {
        1: ("",),
        3: ("_x", "_y", "_z"),
        9: tuple(f"_{first}{second}" for first in "xyz" for second in "xyz"),
    }.get(count)
    if suffixes is None:
        _error(f"field {name!r} has unsupported component count {count}")
    return tuple(f"{name}{suffix}" for suffix in suffixes)


def postprocess(
    *,
    dataset_dir: str | Path,
    operations: Sequence[PostprocessOperation],
    n_jobs: int | None = None,
    progress: bool = False,
    overwrite: bool = False,
) -> PostprocessedDataset:
    """Materialise supported operations in a Cartesian dataset."""

    started = perf_counter()
    root = Path(dataset_dir).expanduser().resolve()
    info = json.loads((root / "info.json").read_text(encoding="utf-8"))
    dataset = _Dataset(root, info)
    selected = tuple(operations)
    if not selected or any(not isinstance(item, PostprocessOperation) for item in selected):
        raise TypeError("operations must contain StackNordic post-processing contracts")
    output_names = tuple(name for item in selected for name in item.outputs)
    if len(set(output_names)) != len(output_names):
        _error("post-processing outputs must be unique")
    existing = set(output_names).intersection(dataset.fields)
    if existing and not overwrite:
        _error(f"post-processing fields already exist: {sorted(existing)}", FileExistsError)
    planned: dict[str, tuple[str, ...]] = {}
    for operation in selected:
        for name, values in _evaluate(dataset, operation, 0).items():
            planned[name] = _components(name, values)
    component_owners = {
        component: name
        for name, components in dataset.fields.items()
        for component in components
    }
    new_components: dict[str, str] = {}
    for name, components in planned.items():
        for component in components:
            owner = component_owners.get(component)
            if owner is not None and owner != name:
                _error(
                    f"output component {component!r} is already owned by field {owner!r}"
                )
            previous = new_components.setdefault(component, name)
            if previous != name:
                _error(
                    f"post-processing fields {previous!r} and {name!r} share component "
                    f"{component!r}"
                )
    stage = Path(tempfile.mkdtemp(prefix=".stacknordic-postprocess-", dir=root))
    work = tuple(
        (index, operation)
        for index in range(len(dataset.records))
        for operation in selected
    )
    estimated_worker_bytes = max(1, int(np.prod(dataset.shape))) * 256
    selected_workers = memory_bounded_workers(n_jobs, len(work), estimated_worker_bytes)
    reporter = (
        start_progress(len(work), "StackNordic post-processing")
        if progress
        else None
    )
    results: dict[tuple[int, str], tuple[Path, ...]] = {}
    mappings: dict[str, tuple[str, ...]] = dict(planned)
    published: list[Path] = []
    superseded: list[Path] = []
    committed = False
    try:
        def task(index: int, operation: PostprocessOperation):
            evaluated = _evaluate(dataset, operation, index)
            written: dict[str, tuple[tuple[str, ...], tuple[Path, ...]]] = {}
            for name, values in evaluated.items():
                if not np.all(np.isfinite(values)):
                    _error(f"post-processed field {name!r} contains non-finite values")
                components = _components(name, values)
                if components != planned[name]:
                    _error(f"post-processed field {name!r} changed component shape")
                columns = (values,) if values.ndim == 3 else tuple(
                    values[..., component] for component in range(values.shape[-1])
                )
                paths = []
                for component, column in zip(components, columns, strict=True):
                    path = stage / f"{component}_id{index:03d}.dat"
                    np.asarray(column, dtype=dataset.dtype).tofile(path)
                    paths.append(path)
                written[name] = components, tuple(paths)
            return index, written

        if selected_workers == 1:
            completed = (task(index, operation) for index, operation in work)
            for index, evaluated in completed:
                for name, values in evaluated.items():
                    _, paths = values
                    results[index, name] = paths
                if reporter is not None:
                    reporter.update()
        else:
            with ThreadPoolExecutor(max_workers=selected_workers) as executor:
                futures = [
                    executor.submit(task, index, operation)
                    for index, operation in work
                ]
                for future in as_completed(futures):
                    index, evaluated = future.result()
                    for name, values in evaluated.items():
                        _, paths = values
                        results[index, name] = paths
                    if reporter is not None:
                        reporter.update()
        data_dir = root / "data"
        for index, record in enumerate(dataset.records):
            for name, components in mappings.items():
                for component, path in zip(components, results[index, name], strict=True):
                    destination = data_dir / f"{path.stem}_{stage.name[1:]}{path.suffix}"
                    path.replace(destination)
                    published.append(destination)
                    previous = record.get(f"{component} filename")
                    if previous is not None:
                        superseded.append(root / previous)
                    record[f"{component} filename"] = f"./data/{destination.name}"
        global_info = info["global"]
        for name, components in mappings.items():
            old = global_info["fields"].get(name, ())
            for component in old:
                if component not in components:
                    for record in dataset.records:
                        filename = record.pop(f"{component} filename", None)
                        if filename is not None:
                            superseded.append(root / filename)
            global_info["fields"][name] = list(components)
        global_info["variables"] = [
            component
            for components in global_info["fields"].values()
            for component in components
        ]
        manifest = stage / "info.json"
        manifest.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")
        manifest.replace(root / "info.json")
        committed = True
        for path in superseded:
            if path not in published:
                path.unlink(missing_ok=True)
    except BaseException:
        if not committed:
            for path in published:
                path.unlink(missing_ok=True)
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        if reporter is not None:
            finish_progress(reporter)
    shutil.rmtree(stage, ignore_errors=True)
    return PostprocessedDataset(
        dataset_dir=root,
        fields=tuple(mappings),
        snapshots=len(dataset.records),
        elapsed_seconds=perf_counter() - started,
    )


__all__ = ["PostprocessedDataset", "postprocess"]
