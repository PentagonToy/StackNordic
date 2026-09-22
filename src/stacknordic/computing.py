"""Explicit field computations applied before spatial reduction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np


def _cantera():
    try:
        import cantera as ct
    except ImportError as error:
        raise ImportError("[StackNordic] This computation requires Cantera.") from error
    return ct


def _composition(
    values: Mapping[str, float],
    name: str,
    *,
    allow_negative: bool = False,
) -> Mapping[str, float]:
    result = MappingProxyType({str(key): float(value) for key, value in values.items()})
    if not result:
        raise ValueError(f"{name} must not be empty")
    if not allow_negative and any(value < 0.0 for value in result.values()):
        raise ValueError(f"{name} values must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class ProgressComputation:
    """Normalised linear progress variable with fixed reference states."""

    output: str
    coefficients: Mapping[str, float]
    offset: float
    weights: Mapping[str, float]
    mechanism: str
    equivalence_ratio: float
    fuel: Mapping[str, float]
    oxidizer: Mapping[str, float]
    temperature: float
    pressure: float

    @property
    def inputs(self) -> tuple[str, ...]:
        return tuple(self.coefficients)

    def evaluate(self, field: Callable[[str], np.ndarray]) -> np.ndarray:
        values: np.ndarray | None = None
        for name, coefficient in self.coefficients.items():
            contribution = coefficient * np.asarray(field(name), dtype=np.float64)
            values = contribution if values is None else values + contribution
        if values is None:
            raise RuntimeError("progress computation has no active species")
        return values + self.offset

    def manifest(self) -> dict[str, object]:
        return {
            "kind": "progress",
            "output": self.output,
            "inputs": list(self.inputs),
            "weights": dict(self.weights),
            "coefficients": dict(self.coefficients),
            "offset": self.offset,
            "units": "1",
            "reference": {
                "provider": "cantera",
                "mechanism": self.mechanism,
                "equivalence_ratio": self.equivalence_ratio,
                "fuel": dict(self.fuel),
                "oxidizer": dict(self.oxidizer),
                "temperature_K": self.temperature,
                "pressure_Pa": self.pressure,
                "composition_basis": "mole",
                "burnt_state": "HP",
            },
        }


@dataclass(frozen=True, slots=True)
class ReactionRateComputation:
    """Cantera species or progress-variable mass source term."""

    output: str
    mechanism: str
    species_fields: Mapping[str, str]
    temperature_field: str
    pressure_field: str | None
    density_field: str | None
    target_coefficients: Mapping[str, float]
    target: str
    units: str

    @property
    def inputs(self) -> tuple[str, ...]:
        names = [*self.species_fields.values(), self.temperature_field]
        names.append(self.pressure_field or self.density_field or "")
        return tuple(dict.fromkeys(name for name in names if name))

    def evaluate(self, field: Callable[[str], np.ndarray]) -> np.ndarray:
        source = np.asarray(field(self.temperature_field))
        result = np.empty(source.size)
        for start, stop, outputs in reaction_rate_chunks((self,), field):
            result[start:stop] = outputs[self.output]
        return result.reshape(source.shape)

    def manifest(self) -> dict[str, object]:
        return {
            "kind": "reaction_rate",
            "output": self.output,
            "inputs": list(self.inputs),
            "provider": "cantera",
            "mechanism": self.mechanism,
            "species_fields": dict(self.species_fields),
            "temperature_field": self.temperature_field,
            "pressure_field": self.pressure_field,
            "density_field": self.density_field,
            "target": self.target,
            "target_coefficients": dict(self.target_coefficients),
            "units": self.units,
        }

    def state_key(self) -> tuple[object, ...]:
        """Return the immutable state contract used for execution fusion."""

        return (
            self.mechanism,
            tuple(self.species_fields.items()),
            self.temperature_field,
            self.pressure_field,
            self.density_field,
        )


def reaction_rate_chunks(
    definitions: tuple[ReactionRateComputation, ...],
    field: Callable[[str], np.ndarray],
    *,
    chunk_cells: int | None = None,
):
    """Yield fused reaction-rate outputs for one compatible state contract."""

    if not definitions:
        return
    first = definitions[0]
    if any(item.state_key() != first.state_key() for item in definitions[1:]):
        raise ValueError("fused reaction-rate definitions must share one state contract")
    ct = _cantera()
    gas = ct.Solution(first.mechanism)
    if chunk_cells is None:
        bytes_per_cell = max((2 * gas.n_species + 8) * np.dtype(np.float64).itemsize, 1)
        chunk_cells = max(4_096, min(262_144, (64 * 1024**2) // bytes_per_cell))
    temperature = np.asarray(field(first.temperature_field)).reshape(-1)
    species_values = {
        species: np.asarray(field(name)).reshape(-1)
        for species, name in first.species_fields.items()
    }
    if first.pressure_field is not None:
        thermodynamic = np.asarray(field(first.pressure_field)).reshape(-1)
        thermodynamic_name = "pressure"
    else:
        thermodynamic = np.asarray(field(first.density_field or "")).reshape(-1)
        thermodynamic_name = "density"
    size = temperature.size
    if thermodynamic.size != size or any(values.size != size for values in species_values.values()):
        raise ValueError("reaction-rate fields must share one shape")
    indices = {name: gas.species_index(name) for name in first.species_fields}
    molecular_weights = gas.molecular_weights
    for start in range(0, size, chunk_cells):
        stop = min(start + chunk_cells, size)
        chunk_temperature = np.asarray(temperature[start:stop], dtype=np.float64)
        chunk_thermodynamic = np.asarray(thermodynamic[start:stop], dtype=np.float64)
        if np.any(~np.isfinite(chunk_temperature)) or np.any(chunk_temperature <= 0.0):
            raise ValueError("reaction-rate temperature must be finite and positive")
        if np.any(~np.isfinite(chunk_thermodynamic)) or np.any(chunk_thermodynamic <= 0.0):
            raise ValueError(f"reaction-rate {thermodynamic_name} must be finite and positive")
        mass_fractions = np.zeros((stop - start, gas.n_species))
        for species, values in species_values.items():
            mass_fractions[:, indices[species]] = np.asarray(
                values[start:stop], dtype=np.float64
            )
        totals = mass_fractions.sum(axis=1)
        if np.any(~np.isfinite(mass_fractions)) or np.any(mass_fractions < 0.0):
            raise ValueError("reaction-rate mass fractions must be finite and non-negative")
        if not np.allclose(totals, 1.0, rtol=1.0e-6, atol=1.0e-8):
            raise ValueError(
                "reaction-rate mass fractions must include a complete state summing to one"
            )
        states = ct.SolutionArray(gas, stop - start)
        state = (chunk_temperature, chunk_thermodynamic, mass_fractions)
        if first.pressure_field is not None:
            states.TPY = state
        else:
            states.TDY = state
        rates = states.net_production_rates
        outputs: dict[str, np.ndarray] = {}
        for definition in definitions:
            values = np.zeros(stop - start)
            for species, coefficient in definition.target_coefficients.items():
                index = gas.species_index(species)
                values += coefficient * molecular_weights[index] * rates[:, index]
            if definition.units == "1/s":
                values /= states.density
            outputs[definition.output] = values
        yield start, stop, outputs


Computation = ProgressComputation | ReactionRateComputation


class Computing:
    """Factories for explicit pre-filter field computations."""

    @staticmethod
    def progress(
        *,
        mechanism: str | Path,
        equivalence_ratio: float,
        fuel: Mapping[str, float],
        oxidizer: Mapping[str, float],
        species: Mapping[str, float],
        temperature: float,
        pressure: float,
        output: str = "c",
    ) -> ProgressComputation:
        """Define a normalised linear progress variable."""

        ct = _cantera()
        mechanism_path = str(Path(mechanism).expanduser())
        weights = _composition(species, "species", allow_negative=True)
        if not output:
            raise ValueError("output must be a non-empty field name")
        if equivalence_ratio <= 0.0:
            raise ValueError("equivalence_ratio must be positive")
        if temperature <= 0.0 or pressure <= 0.0:
            raise ValueError("temperature and pressure must be positive")
        gas = ct.Solution(mechanism_path)
        fuel_composition = _composition(fuel, "fuel")
        oxidizer_composition = _composition(oxidizer, "oxidizer")
        missing = set(weights).difference(gas.species_names)
        if missing:
            raise ValueError(f"mechanism does not contain progress species: {sorted(missing)}")
        gas.TP = float(temperature), float(pressure)
        gas.set_equivalence_ratio(
            phi=float(equivalence_ratio),
            fuel=dict(fuel_composition),
            oxidizer=dict(oxidizer_composition),
            basis="mole",
        )
        indices = np.asarray([gas.species_index(name) for name in weights])
        alpha = np.asarray(tuple(weights.values()))
        unburnt = float(alpha @ gas.Y[indices])
        gas.equilibrate("HP")
        burnt = float(alpha @ gas.Y[indices])
        denominator = burnt - unburnt
        tolerance = np.finfo(np.float64).eps * 64.0
        if abs(denominator) <= tolerance:
            raise ValueError("progress-variable reference denominator is negligible")
        coefficients = MappingProxyType(
            {name: float(weight / denominator) for name, weight in weights.items()}
        )
        return ProgressComputation(
            output=output,
            coefficients=coefficients,
            offset=-unburnt / denominator,
            weights=weights,
            mechanism=mechanism_path,
            equivalence_ratio=float(equivalence_ratio),
            fuel=fuel_composition,
            oxidizer=oxidizer_composition,
            temperature=float(temperature),
            pressure=float(pressure),
        )

    @staticmethod
    def reaction_rate(
        *,
        mechanism: str | Path,
        species: Mapping[str, str],
        temperature: str,
        target: str | ProgressComputation,
        output: str,
        pressure: str | None = None,
        density: str | None = None,
        units: str = "kg/m^3/s",
    ) -> ReactionRateComputation:
        """Define a species or progress-variable reaction rate."""

        ct = _cantera()
        mechanism_path = str(Path(mechanism).expanduser())
        gas = ct.Solution(mechanism_path)
        fields = MappingProxyType({str(name): str(field) for name, field in species.items()})
        if not fields or any(not name for name in fields.values()):
            raise ValueError("species must map mechanism names to non-empty field names")
        missing = set(fields).difference(gas.species_names)
        if missing:
            raise ValueError(f"mechanism does not contain state species: {sorted(missing)}")
        if (pressure is None) == (density is None):
            raise ValueError("exactly one of pressure or density must be provided")
        if not temperature or not output or (pressure == "") or (density == ""):
            raise ValueError("temperature, thermodynamic input, and output names must be non-empty")
        if units not in {"kg/m^3/s", "1/s"}:
            raise ValueError("units must be 'kg/m^3/s' or '1/s'")
        if isinstance(target, ProgressComputation):
            target_coefficients = MappingProxyType(dict(target.coefficients))
            target_name = target.output
        elif isinstance(target, str) and target:
            if target not in gas.species_names:
                raise ValueError(f"mechanism does not contain target species: {target!r}")
            target_coefficients = MappingProxyType({target: 1.0})
            target_name = target
        else:
            raise TypeError("target must be a species name or ProgressComputation")
        unavailable = set(target_coefficients).difference(fields)
        missing_targets = set(target_coefficients).difference(gas.species_names)
        if missing_targets:
            raise ValueError(
                f"mechanism does not contain target species: {sorted(missing_targets)}"
            )
        if unavailable:
            raise ValueError(
                "reaction-rate target species require state fields: "
                f"{sorted(unavailable)}"
            )
        return ReactionRateComputation(
            output=output,
            mechanism=mechanism_path,
            species_fields=fields,
            temperature_field=temperature,
            pressure_field=pressure,
            density_field=density,
            target_coefficients=target_coefficients,
            target=target_name,
            units=units,
        )


__all__ = ["Computation", "Computing", "ProgressComputation", "ReactionRateComputation"]


def __dir__() -> list[str]:
    return sorted(__all__)
