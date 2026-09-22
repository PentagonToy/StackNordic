"""OpenFOAM-backed case post-processing."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .._progress import finish_progress, start_progress
from ..postprocess import PostprocessOperation
from ..resources import workers
from ..selection import TimeSelection, select_times
from .environment import _Toolchain


@dataclass(frozen=True, slots=True)
class PostprocessedCase:
    """Summary of OpenFOAM function objects applied to one case."""

    case_dir: Path
    fields: tuple[str, ...]
    times: tuple[float, ...]
    n_jobs: int
    elapsed_seconds: float


def _error(message: str, error_type: type[Exception] = ValueError):
    raise error_type(f"[StackNordic] {message}")


def _function(operation: PostprocessOperation) -> str:
    field = operation.field
    if operation.kind in {"vorticity", "Q", "Lambda2"} and field != "U":
        _error(f"OpenFOAM {operation.kind} currently requires field='U'")
    return {
        "grad": f"grad({field})",
        "div": f"div({field})",
        "ddt": f"ddt({field})",
        "components": f"components({field})",
        "mag": f"mag({field})",
        "mag_sqr": f"magSqr({field})",
        "vorticity": "vorticity",
        "Q": "Q",
        "Lambda2": "Lambda2",
        "write_cell_centres": "writeCellCentres",
        "write_cell_volumes": "writeCellVolumes",
    }[operation.kind]


def _generated(operation: PostprocessOperation) -> tuple[str, ...]:
    field = operation.field
    return {
        "grad": (f"grad({field})",),
        "div": (f"div({field})",),
        "ddt": (f"ddt({field})",),
        "components": tuple(f"{field}{axis}" for axis in "xyz"),
        "mag": (f"mag({field})",),
        "mag_sqr": (f"magSqr({field})",),
        "vorticity": ("vorticity",),
        "Q": ("Q",),
        "Lambda2": ("Lambda2",),
        "write_cell_centres": ("C",),
        "write_cell_volumes": ("V",),
    }[operation.kind]


def _validate_operation(operation: PostprocessOperation) -> None:
    try:
        generated = _generated(operation)
    except KeyError:
        _error(f"unsupported post-processing operation {operation.kind!r}")
    if len(generated) != len(operation.outputs):
        _error(f"{operation.kind} output count does not match its OpenFOAM result")
    _function(operation)


def _run_one(
    case_dir: Path,
    time_name: str,
    operation: PostprocessOperation,
    toolchain: _Toolchain,
    overwrite: bool,
) -> None:
    _validate_operation(operation)
    generated = _generated(operation)
    targets = tuple(case_dir / time_name / name for name in operation.outputs)
    generated_paths = tuple(case_dir / time_name / name for name in generated)
    existing = tuple(path for path in {*targets, *generated_paths} if path.exists())
    if existing and not overwrite:
        _error(
            f"post-processing output already exists at t={time_name}: {existing[0].name}",
            FileExistsError,
        )
    command = toolchain.wrap(
        (
            "postProcess",
            "-case",
            str(case_dir),
            "-func",
            _function(operation),
            "-time",
            time_name,
        )
    )
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        lines = result.stdout.strip().splitlines()
        detail = "\n".join(lines[-20:]) if lines else "postProcess failed without output"
        raise RuntimeError(
            f"[StackNordic] OpenFOAM {_function(operation)} failed at t={time_name}:\n{detail}"
        )
    for source_name, target in zip(generated, targets, strict=True):
        source = case_dir / time_name / source_name
        if not source.is_file():
            _error(
                f"OpenFOAM {_function(operation)} completed without {source_name!r}",
                RuntimeError,
            )
        if source != target:
            if target.exists():
                target.unlink()
            source.replace(target)


def postprocess(
    *,
    case_dir: Path,
    time_names: dict[float, str],
    toolchain: _Toolchain,
    operations: Sequence[PostprocessOperation],
    times: TimeSelection,
    n_jobs: int | None,
    progress: bool,
    overwrite: bool,
) -> PostprocessedCase:
    """Run supported OpenFOAM function objects on reconstructed times."""

    started = perf_counter()
    selected_operations = tuple(operations)
    if not selected_operations or any(
        not isinstance(item, PostprocessOperation) for item in selected_operations
    ):
        raise TypeError("operations must contain StackNordic post-processing contracts")
    for operation in selected_operations:
        _validate_operation(operation)
    selected_times = select_times(tuple(time_names), times)
    work = tuple(
        (value, operation)
        for value in selected_times
        for operation in selected_operations
    )
    selected_workers = workers(n_jobs, len(work))
    reporter = (
        start_progress(len(work), "StackNordic OpenFOAM post-processing")
        if progress
        else None
    )
    try:
        if selected_workers == 1:
            for value, operation in work:
                _run_one(
                    case_dir,
                    time_names[value],
                    operation,
                    toolchain,
                    overwrite,
                )
                if reporter is not None:
                    reporter.update()
        else:
            with ThreadPoolExecutor(max_workers=selected_workers) as executor:
                futures = [
                    executor.submit(
                        _run_one,
                        case_dir,
                        time_names[value],
                        operation,
                        toolchain,
                        overwrite,
                    )
                    for value, operation in work
                ]
                for future in as_completed(futures):
                    future.result()
                    if reporter is not None:
                        reporter.update()
    finally:
        if reporter is not None:
            finish_progress(reporter)
    return PostprocessedCase(
        case_dir=case_dir,
        fields=tuple(name for operation in selected_operations for name in operation.outputs),
        times=selected_times,
        n_jobs=selected_workers,
        elapsed_seconds=perf_counter() - started,
    )


__all__ = ["PostprocessedCase", "postprocess"]
