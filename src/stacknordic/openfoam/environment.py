"""OpenFOAM runtime selection."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class _Toolchain:
    command: str | None
    shell: str

    def wrap(self, command: tuple[str, ...]) -> tuple[str, ...]:
        if self.command is None:
            return command
        payload = shlex.join(command)
        words = tuple(shlex.split(self.command))
        if words and Path(words[0]).name == "openfoam":
            return (*words, "-c", payload)
        return (self.shell, "-lc", f"{self.command} && {payload}")


def _normalise_command(of_cmd: str | None) -> str | None:
    if of_cmd is None:
        return None
    value = of_cmd.strip()
    if not value:
        raise ValueError("of_cmd must not be empty")
    if "/" in value and not any(character.isspace() for character in value):
        return f"module load {shlex.quote(value)}"
    return value


def toolchain(*, of_cmd: str | None = None, shell: str = "bash") -> _Toolchain:
    """Create one case-local OpenFOAM toolchain."""
    if not shell.strip():
        raise ValueError("shell must not be empty")
    return _Toolchain(_normalise_command(of_cmd), shell)


def _toolchain() -> _Toolchain:
    return _Toolchain(None, "bash")


def __dir__() -> list[str]:
    return []
