"""Declarative physical filtering rules for dataset reduction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType


def _names(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{label} must contain non-empty field names")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicate field names")
    return result


def _mapping(values: Mapping[str, str], label: str) -> Mapping[str, str]:
    result = dict(values)
    if any(not source or not target for source, target in result.items()):
        raise ValueError(f"{label} must map non-empty field names")
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class Filtering:
    """Complete physical reduction contract for one downsampling operation."""

    density: str | None = None
    mean: tuple[str, ...] = ()
    favre: tuple[str, ...] = ()
    variance: Mapping[str, str] = field(default_factory=dict)
    stress: Mapping[str, str] = field(default_factory=dict)
    energy: Mapping[str, str] = field(default_factory=dict)
    specific: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.density is not None and not self.density:
            raise ValueError("density must be a non-empty field name or None")
        object.__setattr__(self, "mean", _names(self.mean, "mean"))
        object.__setattr__(self, "favre", _names(self.favre, "favre"))
        object.__setattr__(self, "variance", _mapping(self.variance, "variance"))
        object.__setattr__(self, "stress", _mapping(self.stress, "stress"))
        object.__setattr__(self, "energy", _mapping(self.energy, "energy"))
        object.__setattr__(self, "specific", _mapping(self.specific, "specific"))
        overlap = set(self.mean).intersection(self.favre)
        if overlap:
            raise ValueError(f"fields cannot use both mean and Favre filtering: {sorted(overlap)}")
        if self.favre and self.density is None:
            raise ValueError("Favre filtering requires density")
        if self.specific and self.density is None:
            raise ValueError("specific source conversion requires density")
        missing_specific = set(self.specific).difference(self.mean)
        if missing_specific:
            raise ValueError("specific keys must name fields declared by mean")
        missing_stress = set(self.energy).difference(self.stress.values())
        if missing_stress:
            raise ValueError("energy keys must name outputs declared by stress")

    @classmethod
    def flow(
        cls,
        *,
        regime: str,
        velocity: str = "U",
        pressure: str = "p",
        density: str | None = None,
        sgs_stress: str | None = None,
        sgs_energy: str | None = None,
    ) -> Filtering:
        """Build an incompressible or compressible flow filtering contract."""

        if regime not in {"incompressible", "compressible"}:
            raise ValueError("flow regime must be 'incompressible' or 'compressible'")
        if regime == "compressible" and density is None:
            raise ValueError("compressible flow filtering requires density")
        mean = [pressure]
        favre: list[str] = []
        if density is not None:
            mean.append(density)
        if regime == "compressible":
            favre.append(velocity)
        else:
            mean.append(velocity)
        stress = {velocity: sgs_stress} if sgs_stress is not None else {}
        energy = {sgs_stress: sgs_energy} if sgs_stress and sgs_energy else {}
        return cls(
            density=density,
            mean=tuple(mean),
            favre=tuple(favre),
            stress=stress,
            energy=energy,
        )

    @classmethod
    def combustion(
        cls,
        *,
        regime: str,
        flow: str,
        density: str | None = None,
        velocity: str = "U",
        pressure: str = "p",
        temperature: str = "T",
        progress: str | None = None,
        progress_variance: str | None = None,
        mixture_fraction: str | None = None,
        mixture_fraction_variance: str | None = None,
        species: Sequence[str] = (),
        reaction_rate: str | None = None,
        specific_reaction_rate: str | None = None,
        sgs_stress: str | None = None,
        sgs_energy: str | None = None,
    ) -> Filtering:
        """Build a premixed or non-premixed combustion filtering contract."""

        if regime not in {"premixed", "non-premixed"}:
            raise ValueError("combustion regime must be 'premixed' or 'non-premixed'")
        if flow not in {"incompressible", "compressible"}:
            raise ValueError("combustion flow must be 'incompressible' or 'compressible'")
        if flow == "compressible" and density is None:
            raise ValueError("compressible combustion filtering requires density")
        if regime == "premixed":
            if progress is None:
                raise ValueError("premixed combustion filtering requires progress")
            coordinate = progress
            variance_name = progress_variance
        else:
            if mixture_fraction is None:
                raise ValueError("non-premixed combustion filtering requires mixture_fraction")
            coordinate = mixture_fraction
            variance_name = mixture_fraction_variance
        mean = [pressure]
        if density is not None:
            mean.insert(0, density)
        if reaction_rate is not None:
            mean.append(reaction_rate)
        if specific_reaction_rate is not None and reaction_rate is None:
            raise ValueError("specific_reaction_rate requires reaction_rate")
        transported = [velocity, temperature, coordinate, *_names(species, "species")]
        favre = transported if flow == "compressible" else []
        if flow == "incompressible":
            mean.extend(transported)
        variance = {coordinate: variance_name} if variance_name is not None else {}
        stress = {velocity: sgs_stress} if sgs_stress is not None else {}
        energy = {sgs_stress: sgs_energy} if sgs_stress and sgs_energy else {}
        return cls(
            density=density,
            mean=tuple(mean),
            favre=tuple(favre),
            variance=variance,
            stress=stress,
            energy=energy,
            specific=(
                {reaction_rate: specific_reaction_rate}
                if reaction_rate is not None and specific_reaction_rate is not None
                else {}
            ),
        )
