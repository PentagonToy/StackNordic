"""Portable post-processing operation declarations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PostprocessOperation:
    """One post-processing operation shared by supported backends."""

    kind: str
    field: str | None
    outputs: tuple[str, ...]


def _field_operation(kind: str, field: str, output: str | None) -> PostprocessOperation:
    if not field:
        raise ValueError("field must be a non-empty name")
    selected = output or f"{kind}{field[0].upper()}{field[1:]}"
    if not selected:
        raise ValueError("output must be a non-empty name")
    return PostprocessOperation(kind, field, (selected,))


class Postprocess:
    """Factories for the supported OpenFOAM and Cartesian operations."""

    @staticmethod
    def grad(*, field: str, output: str | None = None) -> PostprocessOperation:
        """Return the spatial gradient of a scalar or vector field."""

        return _field_operation("grad", field, output or f"grad{field}")

    @staticmethod
    def div(*, field: str, output: str | None = None) -> PostprocessOperation:
        """Return the divergence of a vector field."""

        return _field_operation("div", field, output or f"div{field}")

    @staticmethod
    def ddt(*, field: str, output: str | None = None) -> PostprocessOperation:
        """Return the temporal derivative of a stored field."""

        return _field_operation("ddt", field, output or f"ddt{field}")

    @staticmethod
    def components(
        *,
        field: str,
        outputs: tuple[str, ...] | None = None,
    ) -> PostprocessOperation:
        """Split a vector field into scalar components."""

        if not field:
            raise ValueError("field must be a non-empty name")
        selected = outputs or tuple(f"{field}{axis}" for axis in "xyz")
        if not selected or any(not name for name in selected):
            raise ValueError("outputs must contain non-empty names")
        return PostprocessOperation("components", field, tuple(selected))

    @staticmethod
    def mag(*, field: str, output: str | None = None) -> PostprocessOperation:
        """Return a field magnitude."""

        return _field_operation("mag", field, output or f"mag{field}")

    @staticmethod
    def mag_sqr(*, field: str, output: str | None = None) -> PostprocessOperation:
        """Return a squared field magnitude."""

        return _field_operation("mag_sqr", field, output or f"magSqr{field}")

    @staticmethod
    def vorticity(*, field: str = "U", output: str = "vorticity") -> PostprocessOperation:
        """Return the curl of a velocity field."""

        return _field_operation("vorticity", field, output)

    @staticmethod
    def Q(*, field: str = "U", output: str = "Q") -> PostprocessOperation:
        """Return the second invariant of the velocity-gradient tensor."""

        return _field_operation("Q", field, output)

    @staticmethod
    def Lambda2(*, field: str = "U", output: str = "Lambda2") -> PostprocessOperation:
        """Return the Lambda2 vortex criterion."""

        return _field_operation("Lambda2", field, output)

    @staticmethod
    def write_cell_centres(*, output: str = "C") -> PostprocessOperation:
        """Return cell-centre coordinates."""

        return PostprocessOperation("write_cell_centres", None, (output,))

    @staticmethod
    def write_cell_volumes(*, output: str = "V") -> PostprocessOperation:
        """Return cell volumes."""

        return PostprocessOperation("write_cell_volumes", None, (output,))


__all__ = ["Postprocess", "PostprocessOperation"]


def __dir__() -> list[str]:
    return sorted(__all__)
