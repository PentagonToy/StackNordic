"""OpenFOAM-backed derivation of recognised stored fields."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

from ..openfoam.environment import _Toolchain


def gradient_source(name: str) -> str | None:
    """Return the source field for the conventional ``grad<field>`` name."""

    if not name.startswith("grad") or len(name) == 4:
        return None
    return name[4:]


def reconstruction_fields(fields: Iterable[str]) -> tuple[str, ...]:
    """Replace recognised derived fields with the fields needed to compute them."""

    result: list[str] = []
    for field in fields:
        source = gradient_source(field) or field
        if source not in result:
            result.append(source)
    return tuple(result)


def _run(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def materialise_gradients(
    case_dir: Path,
    source_case: Path,
    times: tuple[tuple[float, str], ...],
    fields: Iterable[str],
    toolchain: _Toolchain,
) -> None:
    """Compute missing recognised gradients in an isolated reconstructed case."""

    system = case_dir / "system"
    constant = case_dir / "constant"
    if not system.exists():
        system.symlink_to(source_case / "system", target_is_directory=True)
    if not constant.exists():
        constant.symlink_to(source_case / "constant", target_is_directory=True)
    for field in fields:
        source = gradient_source(field)
        if source is None:
            continue
        expression = f"grad({source})"
        missing = tuple((value, name) for value, name in times if not (case_dir / name / field).is_file())
        if not missing:
            continue
        print(
            f"[StackNordic] Field {field!r} was not detected. "
            f"Using OpenFOAM postProcess for {expression}."
        )
        for _, time_name in missing:
            command = toolchain.wrap(
                ("postProcess", "-case", str(case_dir), "-func", expression, "-time", time_name)
            )
            result = _run(command)
            if result.returncode != 0:
                detail = result.stdout.strip().splitlines()
                tail = "\n".join(detail[-20:]) if detail else "postProcess failed without output"
                raise RuntimeError(
                    f"[StackNordic] OpenFOAM postProcess could not generate {field!r} "
                    f"at t={time_name}:\n{tail}"
                )
            generated = case_dir / time_name / expression
            target = case_dir / time_name / field
            if generated.is_file() and generated != target:
                shutil.move(generated, target)
            if not target.is_file():
                raise RuntimeError(
                    f"[StackNordic] OpenFOAM postProcess completed without generating "
                    f"{field!r} at t={time_name}"
                )
