"""Materialise derived scalar fields in stored OpenFOAM layouts."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .._progress import finish_progress, start_progress
from ..computing import ProgressComputation
from ..dataset.export import (
    _available_layouts,
    _mesh_file,
    _processor_payloads,
    _Source,
)
from ..reconstruction import _numeric_directories
from ..resources import workers
from ..selection import TimeSelection, select_times
from .io import _block, collated_blocks, read_field_payload, read_label_payload


@dataclass(frozen=True, slots=True)
class ComputedFields:
    """Summary of derived fields written into an OpenFOAM case."""

    case_dir: Path
    fields: tuple[str, ...]
    times: tuple[float, ...]
    n_jobs: int
    elapsed_seconds: float


def _error(message: str, error_type: type[Exception] = ValueError):
    raise error_type(f"[StackNordic] {message}")


def _cell_counts(source: _Source, time_value: float) -> dict[int, int]:
    path = _mesh_file(source, time_value, "cellProcAddressing")
    return {
        rank: read_label_payload(payload).size
        for rank, payload in _processor_payloads(source, path)
    }


def _transform_boundary(payload: bytes, scale: float, offset: float) -> str:
    block = _block(payload, b"boundaryField")
    if block is None:
        return "boundaryField\n{}\n"
    text = block.decode("utf-8", errors="replace")
    uniform = re.compile(
        r"(?m)^(\s*(?:value|inletValue|refValue)\s+uniform\s+)"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(\s*;)"
    )

    def replace(match: re.Match[str]) -> str:
        value = scale * float(match.group(2)) + offset
        return f"{match.group(1)}{value:.16g}{match.group(3)}"

    text = uniform.sub(replace, text)
    nonuniform = re.compile(
        r"((?:value|inletValue|refValue)\s+nonuniform\s+List<scalar>\s+"
        r"(\d+)\s*\()(.*?)(\)\s*;)",
        re.DOTALL,
    )

    def replace_list(match: re.Match[str]) -> str:
        count = int(match.group(2))
        values = np.fromstring(match.group(3), sep=" ")
        if values.size != count:
            _error("boundary scalar list cannot be decoded safely")
        transformed = scale * values + offset
        body = "\n".join(f"{value:.16g}" for value in transformed)
        return f"{match.group(1)}\n{body}\n{match.group(4)}"

    return f"boundaryField\n{{{nonuniform.sub(replace_list, text)}}}\n"


def _scalar_payload(
    template: bytes,
    values: np.ndarray,
    name: str,
    scale: float,
    offset: float,
) -> bytes:
    dimensions = re.search(rb"(?m)^\s*dimensions\s+([^;]+);", template)
    dimension_set = (
        dimensions.group(1).decode("ascii", errors="replace").strip()
        if dimensions is not None
        else "[0 0 0 0 0 0 0]"
    )
    body = "\n".join(f"{value:.16g}" for value in np.asarray(values).reshape(-1))
    return (
        "FoamFile\n"
        "{\n"
        "    format      ascii;\n"
        "    class       volScalarField;\n"
        f"    object      {name};\n"
        "}\n\n"
        f"dimensions      {dimension_set};\n\n"
        f"internalField   nonuniform List<scalar>\n{values.size}\n(\n{body}\n);\n\n"
        f"{_transform_boundary(template, scale, offset)}"
    ).encode()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _evaluate(
    definition: ProgressComputation,
    payload: bytes,
    cell_count: int | None,
) -> tuple[np.ndarray, float, float]:
    source = definition.inputs[0]
    data = read_field_payload(payload, cell_count=cell_count)
    if data.field_class != "volScalarField":
        _error(f"progress input {source!r} must be a volScalarField")
    scale = float(definition.coefficients[source])
    values = scale * np.asarray(data.values, dtype=np.float64) + definition.offset
    if not np.all(np.isfinite(values)):
        _error(f"computed field {definition.output!r} contains non-finite values")
    return values, scale, definition.offset


def _write_reconstructed(
    case_dir: Path,
    time_name: str,
    definition: ProgressComputation,
    overwrite: bool,
) -> None:
    source = case_dir / time_name / definition.inputs[0]
    destination = case_dir / time_name / definition.output
    if destination.exists() and not overwrite:
        _error(f"field {definition.output!r} already exists at t={time_name}", FileExistsError)
    template = source.read_bytes()
    values, scale, offset = _evaluate(definition, template, None)
    _write_atomic(
        destination,
        _scalar_payload(template, values, definition.output, scale, offset),
    )


def _write_uncollated(
    source: _Source,
    time_value: float,
    time_name: str,
    definition: ProgressComputation,
    overwrite: bool,
) -> None:
    input_path = source.path / time_name / definition.inputs[0]
    output_path = source.path / time_name / definition.output
    if output_path.exists() and not overwrite:
        _error(
            f"field {definition.output!r} already exists at t={time_name} in {source.path.name}",
            FileExistsError,
        )
    template = input_path.read_bytes()
    count = _cell_counts(source, time_value)[source.first_rank]
    values, scale, offset = _evaluate(definition, template, count)
    _write_atomic(
        output_path,
        _scalar_payload(template, values, definition.output, scale, offset),
    )


def _write_collated(
    source: _Source,
    time_value: float,
    time_name: str,
    definition: ProgressComputation,
    overwrite: bool,
) -> None:
    input_path = source.path / time_name / definition.inputs[0]
    output_path = source.path / time_name / definition.output
    if output_path.exists() and not overwrite:
        _error(
            f"field {definition.output!r} already exists at t={time_name} in {source.path.name}",
            FileExistsError,
        )
    counts = _cell_counts(source, time_value)
    blocks: list[tuple[int, bytes]] = []
    for local_rank, template in collated_blocks(input_path):
        rank = local_rank if local_rank >= source.first_rank else local_rank + source.first_rank
        count = counts[rank]
        values, scale, offset = _evaluate(definition, template, count)
        blocks.append(
            (local_rank, _scalar_payload(template, values, definition.output, scale, offset))
        )
    preamble = (
        "FoamFile\n{\n"
        "    format      binary;\n"
        "    class       decomposedBlockData;\n"
        f"    object      {definition.output};\n"
        "}\n"
    ).encode()
    payload = bytearray(preamble)
    for rank, block in blocks:
        payload.extend(f"\n// Processor{rank}\n\n{len(block)}\n(".encode())
        payload.extend(block)
        payload.extend(b")\n")
    _write_atomic(output_path, bytes(payload))


def materialise(
    *,
    case_dir: Path,
    computations: Sequence[ProgressComputation],
    times: TimeSelection,
    n_jobs: int | None,
    progress: bool,
    overwrite: bool,
) -> ComputedFields:
    """Write derived scalar fields into existing OpenFOAM time layouts."""

    started = perf_counter()
    definitions = tuple(computations)
    if not definitions or any(not isinstance(item, ProgressComputation) for item in definitions):
        raise TypeError("computations must contain StackNordic computation contracts")
    if any(len(item.inputs) != 1 for item in definitions):
        _error(
            "case.compute currently accepts single-species progress definitions; "
            "use Dataset.compute for multi-species progress and reaction rates"
        )
    outputs = tuple(item.output for item in definitions)
    if len(set(outputs)) != len(outputs):
        _error("computation outputs must be unique")
    root_times = {
        value: name
        for value, name in _numeric_directories(case_dir).items()
        if all((case_dir / name / item.inputs[0]).is_file() for item in definitions)
    }
    try:
        layouts = _available_layouts(case_dir)
    except FileNotFoundError:
        layouts = {}
    available = dict(layouts)
    available.update(root_times)
    selected = select_times(tuple(available), times)
    work = tuple((value, definition) for value in selected for definition in definitions)
    selected_workers = workers(n_jobs, len(work))
    reporter = start_progress(len(work), "StackNordic computing") if progress else None

    def task(value: float, definition: ProgressComputation) -> None:
        if value in root_times:
            _write_reconstructed(case_dir, root_times[value], definition, overwrite)
            return
        layout = layouts[value]
        for source in layout.sources:
            writer = _write_collated if source.collated else _write_uncollated
            writer(source, value, layout.name, definition, overwrite)

    try:
        if selected_workers == 1:
            for item in work:
                task(*item)
                if reporter is not None:
                    reporter.update()
        else:
            with ThreadPoolExecutor(max_workers=selected_workers) as executor:
                futures = [executor.submit(task, *item) for item in work]
                for future in as_completed(futures):
                    future.result()
                    if reporter is not None:
                        reporter.update()
    finally:
        if reporter is not None:
            finish_progress(reporter)
    return ComputedFields(
        case_dir=case_dir,
        fields=outputs,
        times=selected,
        n_jobs=selected_workers,
        elapsed_seconds=perf_counter() - started,
    )


__all__ = ["ComputedFields", "materialise"]
