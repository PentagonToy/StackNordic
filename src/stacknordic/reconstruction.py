"""Concurrent OpenFOAM time reconstruction."""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from ._progress import finish_progress, start_progress
from .openfoam.environment import _Toolchain, toolchain
from .resources import workers as _workers
from .selection import TimeSelection
from .selection import select_times as _select_times

_PROCESSOR = re.compile(r"processor(\d+)$")
_PROCESSORS = re.compile(r"processors(\d+)(?:_(\d+)-(\d+))?$")
_VERSION = re.compile(rb"\bVersion\s*:\s*([A-Za-z0-9_.+-]+)")
_BOUNDARY_COUNT = re.compile(r"\b(\d+)\s*\(")


@dataclass(frozen=True, slots=True)
class _Layout:
    case_dir: Path
    processors: tuple[Path, ...]
    processor_count: int
    collated: bool
    time_names: dict[float, str]
    partial_times: tuple[float, ...]
    source_family: str
    source_versions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Runtime:
    family: str
    version: str


@dataclass(frozen=True, slots=True)
class Reconstruction:
    """Summary of one reconstruction request."""

    case_dir: Path
    output_dir: Path
    processor_count: int
    reconstructed_times: tuple[float, ...]
    skipped_times: tuple[float, ...]
    fields: tuple[str, ...] | None
    source_family: str
    source_versions: tuple[str, ...]
    runtime_family: str
    runtime_version: str
    n_jobs: int
    elapsed_seconds: float


def _numeric_directories(directory: Path) -> dict[float, str]:
    result: dict[float, str] = {}
    for path in directory.iterdir():
        if not path.is_dir():
            continue
        try:
            value = float(path.name)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        if value in result and result[value] != path.name:
            raise ValueError(
                f"OpenFOAM directory {directory} contains duplicate physical time {value:g}"
            )
        result[value] = path.name
    return result


def _processor_directories(case_dir: Path) -> tuple[tuple[Path, ...], int, bool]:
    indexed = []
    collated: dict[int, list[tuple[int, int, Path]]] = {}
    for path in case_dir.iterdir():
        match = _PROCESSOR.fullmatch(path.name)
        if path.is_dir() and match is not None:
            indexed.append((int(match.group(1)), path))
            continue
        grouped = _PROCESSORS.fullmatch(path.name)
        if path.is_dir() and grouped is not None:
            total = int(grouped.group(1))
            first = int(grouped.group(2) or 0)
            last = int(grouped.group(3) or total - 1)
            collated.setdefault(total, []).append((first, last, path))
    if indexed and collated:
        raise ValueError("case contains both processorN and collated processorsN layouts")
    if collated:
        if len(collated) != 1:
            raise ValueError("case contains more than one collated processor count")
        total, ranges = next(iter(collated.items()))
        ranges.sort()
        expected = 0
        for first, last, _ in ranges:
            if first != expected or last < first:
                raise ValueError("collated processor ranges are incomplete or overlap")
            expected = last + 1
        if expected != total:
            raise ValueError("collated processor ranges do not cover their declared count")
        return tuple(path for _, _, path in ranges), total, True
    indexed.sort()
    indices = [index for index, _ in indexed]
    if not indices:
        raise FileNotFoundError(f"No processor directories were found in {case_dir}")
    if indices != list(range(len(indices))):
        raise ValueError("processor directories must be consecutively numbered from processor0")
    return tuple(path for _, path in indexed), len(indexed), False


def _source_identity(case_dir: Path, processors: tuple[Path, ...]) -> tuple[str, tuple[str, ...]]:
    candidates = [case_dir / "system/controlDict"]
    for field in ("U", "p", "T"):
        candidates.append(processors[0] / "0" / field)
    payloads = [path.read_bytes()[:8192] for path in candidates if path.is_file()]
    joined = b"\n".join(payloads).lower()
    if b"openfoam.org" in joined or b"openfoam foundation" in joined:
        family = "foundation"
    elif b"openfoam.com" in joined:
        family = "esi"
    else:
        family = "unknown"
    versions = {
        match.group(1).decode("ascii", errors="replace").strip()
        for payload in payloads
        for match in _VERSION.finditer(payload)
    }
    return family, tuple(sorted(versions))


def _inspect(case_dir: str | Path) -> _Layout:
    selected = Path(case_dir).expanduser().resolve()
    if not selected.is_dir():
        raise FileNotFoundError(f"OpenFOAM case directory does not exist: {selected}")
    for required in (selected / "system/controlDict", selected / "constant"):
        if not required.exists():
            raise FileNotFoundError(f"OpenFOAM case entry is missing: {required}")
    processors, processor_count, collated = _processor_directories(selected)
    rank_times = [_numeric_directories(processor) for processor in processors]
    common = set(rank_times[0])
    union = set(rank_times[0])
    for values in rank_times[1:]:
        common.intersection_update(values)
        union.update(values)
    if not common:
        raise ValueError("processor directories have no common physical times")
    names = {}
    for physical_time in sorted(common):
        rank_names = {values[physical_time] for values in rank_times}
        if len(rank_names) != 1:
            raise ValueError(
                f"processor directories use inconsistent names for time {physical_time:g}"
            )
        names[physical_time] = rank_names.pop()
    family, versions = _source_identity(selected, processors)
    return _Layout(
        case_dir=selected,
        processors=processors,
        processor_count=processor_count,
        collated=collated,
        time_names=names,
        partial_times=tuple(sorted(union.difference(common))),
        source_family=family,
        source_versions=versions,
    )


def _normalise_fields(fields: Sequence[str] | None) -> tuple[str, ...] | None:
    if fields is None:
        return None
    selected = tuple(fields)
    if not selected:
        raise ValueError("fields must contain at least one field name")
    for field in selected:
        if not field or Path(field).name != field:
            raise ValueError("each field must be one non-empty path component")
    if len(set(selected)) != len(selected):
        raise ValueError("fields must not contain duplicates")
    return selected


def _command(
    case_dir: Path,
    time_name: str,
    fields: tuple[str, ...] | None,
    runtime: _Toolchain,
    collated: bool,
) -> tuple[str, ...]:
    command = ["reconstructPar", "-case", str(case_dir), "-time", time_name]
    if collated:
        command.extend(("-fileHandler", "collated"))
    if fields is not None:
        command.extend(("-fields", f"({' '.join(fields)})"))
    return runtime.wrap(tuple(command))


def _control_value(payload: str, name: str, default: str) -> str:
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s+([^;]+);", payload)
    return match.group(1).strip() if match is not None else default


def _minimal_control_dict(source: Path) -> str:
    payload = source.read_text(encoding="utf-8", errors="replace")
    values = {
        "writeFormat": _control_value(payload, "writeFormat", "ascii"),
        "writePrecision": _control_value(payload, "writePrecision", "12"),
        "writeCompression": _control_value(payload, "writeCompression", "off"),
        "timeFormat": _control_value(payload, "timeFormat", "general"),
        "timePrecision": _control_value(payload, "timePrecision", "12"),
    }
    return (
        "FoamFile\n"
        "{\n"
        "    format      ascii;\n"
        "    class       dictionary;\n"
        "    location    \"system\";\n"
        "    object      controlDict;\n"
        "}\n\n"
        "application         reconstructPar;\n"
        "startFrom           latestTime;\n"
        "startTime           0;\n"
        "stopAt              endTime;\n"
        "endTime             1;\n"
        "deltaT              1;\n"
        "writeControl        timeStep;\n"
        "writeInterval       1;\n"
        f"writeFormat         {values['writeFormat']};\n"
        f"writePrecision      {values['writePrecision']};\n"
        f"writeCompression    {values['writeCompression']};\n"
        f"timeFormat          {values['timeFormat']};\n"
        f"timePrecision       {values['timePrecision']};\n"
        "runTimeModifiable   false;\n"
        "functions           {};\n"
    )


def _matching_brace(payload: str, start: int) -> int:
    depth = 0
    for index in range(start, len(payload)):
        if payload[index] == "{":
            depth += 1
        elif payload[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("unterminated OpenFOAM dictionary block")


def _boundary_patches(path: Path) -> tuple[tuple[str, str], ...]:
    payload = path.read_text(encoding="utf-8", errors="strict")
    payload = re.sub(r"/\*.*?\*/", "", payload, flags=re.DOTALL)
    payload = re.sub(r"//[^\r\n]*", "", payload)
    count_match = _BOUNDARY_COUNT.search(payload)
    if count_match is None:
        raise ValueError(f"cannot read boundary patch count from {path}")
    count = int(count_match.group(1))
    cursor = count_match.end()
    patches: list[tuple[str, str]] = []
    name_pattern = re.compile(r"\s*([^\s{}();]+)\s*\{")
    while len(patches) < count:
        name_match = name_pattern.match(payload, cursor)
        if name_match is None:
            raise ValueError(f"cannot read boundary patch {len(patches)} from {path}")
        start = payload.find("{", name_match.start())
        end = _matching_brace(payload, start)
        body = payload[start + 1 : end]
        type_match = re.search(r"(?m)^\s*type\s+([^;]+);", body)
        if type_match is None:
            raise ValueError(f"boundary patch {name_match.group(1)} has no type in {path}")
        patches.append((name_match.group(1), type_match.group(1).strip()))
        cursor = end + 1
    return tuple(patches)


def _write_boundary_addressing(source_case: Path, processor: Path, destination: Path) -> None:
    root_patches = _boundary_patches(source_case / "constant/polyMesh/boundary")
    root_indices = {name: index for index, (name, _) in enumerate(root_patches)}
    processor_patches = _boundary_patches(processor / "constant/polyMesh/boundary")
    addressing = []
    for name, patch_type in processor_patches:
        if patch_type.startswith("processor"):
            addressing.append(-1)
        elif name in root_indices:
            addressing.append(root_indices[name])
        else:
            raise ValueError(f"processor boundary patch {name!r} has no root-mesh match")
    values = "\n".join(str(value) for value in addressing)
    destination.write_text(
        "FoamFile\n"
        "{\n"
        "    format      ascii;\n"
        "    class       labelList;\n"
        "    location    \"constant/polyMesh\";\n"
        "    object      boundaryProcAddressing;\n"
        "}\n\n"
        f"{len(addressing)}\n(\n{values}\n)\n",
        encoding="utf-8",
    )


def _link_processor_view(
    source_case: Path,
    processor: Path,
    destination: Path,
    runtime_family: str,
) -> None:
    destination.mkdir()
    for entry in processor.iterdir():
        if entry.name != "constant":
            (destination / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    source_constant = processor / "constant"
    destination_constant = destination / "constant"
    destination_constant.mkdir()
    for entry in source_constant.iterdir():
        if entry.name != "polyMesh":
            (destination_constant / entry.name).symlink_to(
                entry,
                target_is_directory=entry.is_dir(),
            )
    source_mesh = source_constant / "polyMesh"
    destination_mesh = destination_constant / "polyMesh"
    destination_mesh.mkdir()
    for entry in source_mesh.iterdir():
        if entry.name != "boundaryProcAddressing":
            (destination_mesh / entry.name).symlink_to(
                entry,
                target_is_directory=entry.is_dir(),
            )
    source_addressing = source_mesh / "boundaryProcAddressing"
    destination_addressing = destination_mesh / "boundaryProcAddressing"
    if source_addressing.is_file():
        destination_addressing.symlink_to(source_addressing)
    elif runtime_family == "esi":
        _write_boundary_addressing(source_case, processor, destination_addressing)


@contextmanager
def _staging_case(case_dir: Path, staging_parent: Path, runtime_family: str):
    stage = Path(tempfile.mkdtemp(prefix=".stacknordic-reconstruct-", dir=staging_parent))
    try:
        (stage / "system").mkdir()
        for entry in (case_dir / "system").iterdir():
            if entry.name != "controlDict":
                (stage / "system" / entry.name).symlink_to(
                    entry,
                    target_is_directory=entry.is_dir(),
                )
        (stage / "system" / "controlDict").write_text(
            _minimal_control_dict(case_dir / "system" / "controlDict"),
            encoding="utf-8",
        )
        (stage / "constant").symlink_to(case_dir / "constant", target_is_directory=True)
        processors, _, collated = _processor_directories(case_dir)
        for processor in processors:
            if collated:
                (stage / processor.name).symlink_to(processor, target_is_directory=True)
            else:
                _link_processor_view(
                    case_dir,
                    processor,
                    stage / processor.name,
                    runtime_family,
                )
        yield stage
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _execute(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _runtime_identity(toolchain: _Toolchain) -> _Runtime:
    command = toolchain.wrap(("reconstructPar", "-help"))
    try:
        result = _execute(command)
    except OSError as error:
        raise RuntimeError(f"OpenFOAM runtime preflight failed: {error}") from error
    if result.returncode != 0:
        detail = result.stdout.strip().splitlines()
        tail = "\n".join(detail[-20:]) if detail else "reconstructPar is unavailable"
        raise RuntimeError(f"OpenFOAM runtime preflight failed:\n{tail}")
    lowered = result.stdout.lower()
    if "openfoam.com" in lowered:
        family = "esi"
    elif "openfoam.org" in lowered or "openfoam foundation" in lowered:
        family = "foundation"
    else:
        family = "unknown"
    versions = _VERSION.findall(result.stdout.encode())
    if versions:
        version = versions[-1].decode("ascii", errors="replace")
    else:
        match = re.search(
            r"\bOpenFOAM(?: Foundation version|-)[ ]?v?(\d+(?:\.\d+)*)",
            result.stdout,
            re.IGNORECASE,
        )
        version = match.group(1) if match is not None else "unknown"
    return _Runtime(family=family, version=version)


def _reconstruct_one(
    case_dir: Path,
    output_dir: Path,
    runtime_family: str,
    physical_time: float,
    time_name: str,
    fields: tuple[str, ...] | None,
    toolchain: _Toolchain,
    collated: bool,
) -> float:
    target = output_dir / time_name
    with _staging_case(case_dir, output_dir.parent, runtime_family) as stage:
        result = _execute(_command(stage, time_name, fields, toolchain, collated))
        staged_target = stage / time_name
        if result.returncode != 0:
            detail = result.stdout.strip().splitlines()
            tail = "\n".join(detail[-20:]) if detail else "reconstructPar failed without output"
            raise RuntimeError(f"reconstructPar failed at time {time_name}:\n{tail}")
        if not staged_target.is_dir():
            raise RuntimeError(f"reconstructPar completed without producing time {time_name}")
        if target.exists():
            raise FileExistsError(f"reconstructed time already exists: {target}")
        staged_target.replace(target)
    return physical_time


def reconstruct(
    *,
    case_dir: str | Path,
    output_dir: str | Path | None = None,
    times: TimeSelection = None,
    fields: Sequence[str] | None = None,
    n_jobs: int | None = None,
    progress: bool = False,
    _toolchain_override: _Toolchain | None = None,
) -> Reconstruction:
    """Reconstruct complete processor times with the configured OpenFOAM runtime."""

    if not isinstance(progress, bool):
        raise TypeError("progress must be a boolean")
    layout = _inspect(case_dir)
    destination = (
        layout.case_dir
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"reconstruction output is not a directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    selected_fields = _normalise_fields(fields)
    selected_times = _select_times(tuple(layout.time_names), times)
    root_times = _numeric_directories(destination)
    pending = tuple(value for value in selected_times if value not in root_times)
    skipped = tuple(value for value in selected_times if value in root_times)
    workers = _workers(n_jobs, len(pending)) if pending else 0
    selected_toolchain = _toolchain_override or toolchain()
    runtime = (
        _runtime_identity(selected_toolchain)
        if pending
        else _Runtime("unknown", "unknown")
    )
    if pending and runtime.family == "unknown":
        print(
            "[StackNordic] Warning: reconstructPar is available, but its OpenFOAM "
            "distribution could not be identified."
        )
    if (
        pending
        and layout.source_family != "unknown"
        and runtime.family != "unknown"
        and layout.source_family != runtime.family
    ):
        source_versions = ", ".join(layout.source_versions) or "unknown"
        print(
            f"[StackNordic] Warning: source case uses {layout.source_family} OpenFOAM "
            f"{source_versions}; reconstruction uses {runtime.family} OpenFOAM "
            f"{runtime.version}."
        )
    if layout.partial_times:
        values = ", ".join(f"{value:g}" for value in layout.partial_times)
        print(f"[StackNordic] Skipping incomplete processor times: {values}.")
    started = perf_counter()
    completed: list[float] = []
    reporter = (
        start_progress(len(pending), "StackNordic reconstruction")
        if progress and pending
        else None
    )
    try:
        if workers == 1:
            for physical_time in pending:
                completed.append(
                    _reconstruct_one(
                        layout.case_dir,
                        destination,
                        runtime.family,
                        physical_time,
                        layout.time_names[physical_time],
                        selected_fields,
                        selected_toolchain,
                        layout.collated,
                    )
                )
                if reporter is not None:
                    reporter.update()
        elif workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _reconstruct_one,
                        layout.case_dir,
                        destination,
                        runtime.family,
                        physical_time,
                        layout.time_names[physical_time],
                        selected_fields,
                        selected_toolchain,
                        layout.collated,
                    ): physical_time
                    for physical_time in pending
                }
                for future in as_completed(futures):
                    completed.append(future.result())
                    if reporter is not None:
                        reporter.update()
    finally:
        if reporter is not None:
            finish_progress(reporter)
    return Reconstruction(
        case_dir=layout.case_dir,
        output_dir=destination,
        processor_count=layout.processor_count,
        reconstructed_times=tuple(sorted(completed)),
        skipped_times=skipped,
        fields=selected_fields,
        source_family=layout.source_family,
        source_versions=layout.source_versions,
        runtime_family=runtime.family,
        runtime_version=runtime.version,
        n_jobs=workers,
        elapsed_seconds=perf_counter() - started,
    )
