from __future__ import annotations

import pytest

import stacknordic as sno


def test_compressible_flow_contract() -> None:
    filtering = sno.Filtering.flow(
        regime="compressible",
        density="rho",
        velocity="U",
        pressure="p",
        sgs_stress="tau_sgs",
        sgs_energy="k_sgs",
    )

    assert filtering.mean == ("p", "rho")
    assert filtering.favre == ("U",)
    assert dict(filtering.stress) == {"U": "tau_sgs"}
    assert dict(filtering.energy) == {"tau_sgs": "k_sgs"}


def test_flow_contract_does_not_add_sgs_outputs_by_default() -> None:
    filtering = sno.Filtering.flow(
        regime="incompressible",
        velocity="U",
        pressure="p",
    )

    assert filtering.mean == ("p", "U")
    assert dict(filtering.stress) == {}
    assert dict(filtering.energy) == {}


def test_premixed_combustion_contract() -> None:
    filtering = sno.Filtering.combustion(
        regime="premixed",
        flow="compressible",
        density="rho",
        velocity="U",
        pressure="p",
        temperature="T",
        progress="c",
        progress_variance="c_var",
        species=("H2", "O2", "H2O"),
        reaction_rate="omega_c_mass",
        specific_reaction_rate="omega_c",
    )

    assert filtering.mean == ("rho", "p", "omega_c_mass")
    assert filtering.favre == ("U", "T", "c", "H2", "O2", "H2O")
    assert dict(filtering.variance) == {"c": "c_var"}
    assert dict(filtering.specific) == {"omega_c_mass": "omega_c"}


def test_invalid_regime_is_rejected() -> None:
    with pytest.raises(ValueError, match="flow regime"):
        sno.Filtering.flow(regime="variable", density="rho")


def test_incompressible_combustion_uses_ordinary_means() -> None:
    filtering = sno.Filtering.combustion(
        regime="premixed",
        flow="incompressible",
        density=None,
        velocity="U",
        pressure="p",
        temperature="T",
        progress="c",
        species=("H2", "O2"),
    )

    assert filtering.density is None
    assert filtering.favre == ()
    assert filtering.mean == ("p", "U", "T", "c", "H2", "O2")
