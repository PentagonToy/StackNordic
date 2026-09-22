"""OpenFOAM runtime and stored-data support."""

from .case import case

__all__ = ["case"]


def __dir__() -> list[str]:
    return sorted(__all__)
