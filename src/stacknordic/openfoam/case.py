"""Stored OpenFOAM case orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..computing import ProgressComputation
from ..dataset.export import ExportedDataset, export
from ..postprocess import PostprocessOperation
from ..reconstruction import (
    Reconstruction,
    _numeric_directories,
    _processor_directories,
    _source_identity,
    reconstruct,
)
from ..selection import TimeSelection
from .environment import _Toolchain, toolchain
from .postprocess import PostprocessedCase, postprocess
from .write import ComputedFields, materialise


def _runtime_family(command: str | None) -> str:
    value = (command or "").lower()
    if "openfoam-org" in value or "openfoam.org" in value:
        return "foundation"
    if "openfoam" in value:
        return "esi"
    return "unknown"


def _family_name(family: str) -> str:
    return {
        "esi": "OpenFOAM.com",
        "foundation": "OpenFOAM Foundation",
        "unknown": "unknown OpenFOAM",
    }[family]


def _warn_runtime_mismatch(source_family: str, selected_toolchain: _Toolchain) -> None:
    runtime_family = _runtime_family(selected_toolchain.command)
    if source_family == "unknown" or runtime_family in {"unknown", source_family}:
        return
    print(
        f"[StackNordic] Active runtime: {_family_name(runtime_family)}; "
        f"case: {_family_name(source_family)}."
    )
    print("[StackNordic] Direct reads remain available; OpenFOAM commands may need a matching runtime.")


@dataclass(frozen=True, slots=True)
class Case:
    """Inspected OpenFOAM case with a case-local runtime."""

    case_dir: Path
    family: str
    source_versions: tuple[str, ...]
    reconstructed_times: tuple[float, ...]
    decomposed_times: tuple[float, ...]
    processor_count: int
    collated: bool
    _toolchain: _Toolchain = field(repr=False)

    def compute(
        self,
        *,
        computations: Sequence[ProgressComputation],
        times: TimeSelection = None,
        n_jobs: int | None = None,
        progress: bool = False,
        overwrite: bool = False,
    ) -> ComputedFields:
        """Materialise derived fields in the existing OpenFOAM time layout."""

        return materialise(
            case_dir=self.case_dir,
            computations=computations,
            times=times,
            n_jobs=n_jobs,
            progress=progress,
            overwrite=overwrite,
        )

    def export(
        self,
        *,
        output_dir: str | Path,
        fields: Mapping[str, str | Sequence[str]],
        times: TimeSelection = None,
        dtype: str = "float32",
        mode: str = "direct",
        metadata: Mapping[str, object] | None = None,
        n_jobs: int | None = None,
        progress: bool = False,
        overwrite: bool = False,
    ) -> ExportedDataset:
        """Export stored fields as one analysis-ready dataset."""

        return export(
            case_dir=self.case_dir,
            output_dir=output_dir,
            fields=fields,
            times=times,
            dtype=dtype,
            mode=mode,
            metadata=metadata,
            n_jobs=n_jobs,
            progress=progress,
            overwrite=overwrite,
            _toolchain_override=self._toolchain,
        )

    def postprocess(
        self,
        *,
        operations: Sequence[PostprocessOperation],
        times: TimeSelection = None,
        n_jobs: int | None = None,
        progress: bool = False,
        overwrite: bool = False,
    ) -> PostprocessedCase:
        """Run supported OpenFOAM function objects on reconstructed times."""

        if not self.reconstructed_times:
            raise ValueError(
                "[StackNordic] case post-processing requires reconstructed times; "
                "export decomposed data directly for Cartesian dataset post-processing"
            )
        names = _numeric_directories(self.case_dir)
        return postprocess(
            case_dir=self.case_dir,
            time_names={value: names[value] for value in self.reconstructed_times},
            toolchain=self._toolchain,
            operations=operations,
            times=times,
            n_jobs=n_jobs,
            progress=progress,
            overwrite=overwrite,
        )

    def reconstruct(self, **kwargs) -> Reconstruction:
        """Reconstruct selected stored fields with the configured OpenFOAM runtime."""

        return reconstruct(
            case_dir=self.case_dir,
            _toolchain_override=self._toolchain,
            **kwargs,
        )


def case(
    *,
    case_dir: str | Path,
    of_cmd: str | None = None,
    shell: str = "bash",
) -> Case:
    """Inspect an OpenFOAM case without changing its stored results."""

    root = Path(case_dir).expanduser().resolve()
    if not (root / "system/controlDict").is_file() or not (root / "constant").is_dir():
        raise FileNotFoundError(f"[StackNordic] OpenFOAM case is incomplete: {root}")
    reconstructed = tuple(sorted(_numeric_directories(root)))
    try:
        processors, processor_count, collated = _processor_directories(root)
        rank_times = [_numeric_directories(path) for path in processors]
        common = set(rank_times[0])
        for values in rank_times[1:]:
            common.intersection_update(values)
        decomposed = tuple(sorted(common))
    except FileNotFoundError:
        processors, processor_count, collated, decomposed = (root,), 0, False, ()
    family, versions = _source_identity(root, processors)
    selected_toolchain = toolchain(of_cmd=of_cmd, shell=shell)
    _warn_runtime_mismatch(family, selected_toolchain)
    return Case(
        case_dir=root,
        family=family,
        source_versions=versions,
        reconstructed_times=reconstructed,
        decomposed_times=decomposed,
        processor_count=processor_count,
        collated=collated,
        _toolchain=selected_toolchain,
    )


__all__ = ["Case", "case"]
