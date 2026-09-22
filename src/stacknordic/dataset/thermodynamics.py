"""Derive stored thermodynamic fields from explicit OpenFOAM contracts."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_UNIVERSAL_GAS_CONSTANT = 8314.46261815324
_COMMENTS = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)


def _block(payload: str, name: str) -> str | None:
    match = re.search(rf"\b{re.escape(name)}\s*\{{", payload)
    if match is None:
        return None
    depth = 1
    for index in range(match.end(), len(payload)):
        if payload[index] == "{":
            depth += 1
        elif payload[index] == "}":
            depth -= 1
            if depth == 0:
                return payload[match.end() : index]
    raise ValueError(f"[StackNordic] unterminated OpenFOAM dictionary for {name!r}")


def _expanded_thermophysical_properties(case_dir: Path) -> str:
    root = case_dir / "constant/thermophysicalProperties"
    seen: set[Path] = set()

    def expand(path: Path) -> str:
        selected = path.resolve()
        if selected in seen:
            return ""
        seen.add(selected)
        payload = path.read_text(errors="replace")

        def include(match: re.Match[str]) -> str:
            value = match.group(1).replace("$FOAM_CASE", str(case_dir))
            target = Path(value)
            if not target.is_absolute():
                target = path.parent / target
            return expand(target) if target.is_file() else match.group(0)

        return re.sub(r'^\s*#include\s+["<]([^">]+)[">]\s*$', include, payload, flags=re.MULTILINE)

    return _COMMENTS.sub(" ", expand(root))


@dataclass(frozen=True, slots=True)
class PerfectGasMixture:
    """Mass-fraction perfect-gas density contract."""

    molecular_weights: dict[str, float]

    @property
    def species(self) -> tuple[str, ...]:
        return tuple(self.molecular_weights)

    def density(self, field: Callable[[str], np.ndarray]) -> np.ndarray:
        pressure = np.asarray(field("p"), dtype=np.float64)
        temperature = np.asarray(field("T"), dtype=np.float64)
        reciprocal_weight = np.zeros_like(pressure)
        mass_fraction_sum = np.zeros_like(pressure)
        for name, molecular_weight in self.molecular_weights.items():
            mass_fraction = np.asarray(field(name), dtype=np.float64)
            reciprocal_weight += mass_fraction / molecular_weight
            mass_fraction_sum += mass_fraction
        if pressure.shape != temperature.shape or reciprocal_weight.shape != pressure.shape:
            raise ValueError("[StackNordic] thermodynamic fields must share one shape")
        if np.any(~np.isfinite(pressure)) or np.any(pressure <= 0.0):
            raise ValueError("[StackNordic] pressure must be finite and positive to derive rho")
        if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
            raise ValueError("[StackNordic] temperature must be finite and positive to derive rho")
        if not np.allclose(mass_fraction_sum, 1.0, rtol=1.0e-6, atol=1.0e-8):
            raise ValueError("[StackNordic] species mass fractions must sum to one to derive rho")
        return pressure / (_UNIVERSAL_GAS_CONSTANT * temperature * reciprocal_weight)


def perfect_gas_mixture(case_dir: Path) -> PerfectGasMixture | None:
    """Read a supported multicomponent perfect-gas contract from one case."""

    path = case_dir / "constant/thermophysicalProperties"
    if not path.is_file():
        return None
    payload = _expanded_thermophysical_properties(case_dir)
    thermo_type = _block(payload, "thermoType")
    if thermo_type is None or re.search(r"\bequationOfState\s+perfectGas\s*;", thermo_type) is None:
        return None
    species_match = re.search(r"\bspecies\s+\d+\s*\((.*?)\)\s*;", payload, re.DOTALL)
    if species_match is None:
        return None
    species = tuple(species_match.group(1).split())
    molecular_weights: dict[str, float] = {}
    for name in species:
        definition = _block(payload, name)
        specie = _block(definition or "", "specie")
        match = re.search(
            r"\bmolWeight\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*;",
            specie or "",
        )
        if match is None or float(match.group(1)) <= 0.0:
            return None
        molecular_weights[name] = float(match.group(1))
    return PerfectGasMixture(molecular_weights)


__all__ = ["PerfectGasMixture", "perfect_gas_mixture"]
