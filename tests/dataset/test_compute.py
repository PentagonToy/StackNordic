from __future__ import annotations

import importlib
import json
from pathlib import Path

import cantera as ct
import numpy as np
import pytest

import stacknordic as sno


def _dataset(tmp_path: Path, fields: dict[str, np.ndarray]) -> Path:
    root = tmp_path / "dataset"
    data = root / "data"
    grid = root / "grid"
    data.mkdir(parents=True)
    grid.mkdir()
    record: dict[str, object] = {"id": 0, "time [s]": 0.1, "cells": 2}
    mappings: dict[str, list[str]] = {}
    for name, values in fields.items():
        filename = f"{name}_id000.dat"
        np.asarray(values, dtype="float32").tofile(data / filename)
        record[f"{name} filename"] = f"./data/{filename}"
        mappings[name] = [name]
    for axis in "XYZ":
        np.arange(2 if axis == "X" else 1, dtype="float32").tofile(grid / f"{axis}_m.dat")
    info = {
        "global": {
            "snapshots": 1,
            "variables": list(fields),
            "dtype": "float32",
            "layout": "cartesian",
            "grid": {axis.lower(): f"./grid/{axis}_m.dat" for axis in "XYZ"},
            "fields": mappings,
            "Nxyz": [2, 1, 1],
            "metadata": {},
        },
        "local": [record],
    }
    (root / "info.json").write_text(json.dumps(info))
    return root


def test_compute_reads_legacy_component_paths(tmp_path: Path) -> None:
    source = _dataset(tmp_path, {"H2": np.array([0.1, 0.0])})
    manifest_path = source / "info.json"
    manifest = json.loads(manifest_path.read_text())
    record = manifest["local"][0]
    record["H2"] = record.pop("H2 filename")
    manifest_path.write_text(json.dumps(manifest))
    definition = sno.Computing.progress(
        mechanism="h2o2.yaml",
        equivalence_ratio=0.45,
        fuel={"H2": 1.0},
        oxidizer={"O2": 0.21, "N2": 0.79},
        species={"H2": -1.0},
        temperature=300.0,
        pressure=ct.one_atm,
        output="c",
    )

    sno.Dataset.compute(
        dataset_dir=source,
        computations=(definition,),
        dtype="float32",
        n_jobs=1,
        progress=False,
        overwrite=False,
    )

    assert (source / "data/c_id000.dat").is_file()


def test_compute_reports_chunk_progress_from_zero(tmp_path: Path, monkeypatch) -> None:
    source = _dataset(tmp_path, {"H2": np.array([0.1, 0.0])})
    definition = sno.Computing.progress(
        mechanism="h2o2.yaml",
        equivalence_ratio=0.45,
        fuel={"H2": 1.0},
        oxidizer={"O2": 0.21, "N2": 0.79},
        species={"H2": -1.0},
        temperature=300.0,
        pressure=ct.one_atm,
        output="c",
    )
    updates: list[int] = []

    class Reporter:
        def update(self, value=1):
            updates.append(value)

        def finish(self):
            return None

    module = importlib.import_module("stacknordic.dataset.compute")

    def start_progress(total, description):
        reporter = Reporter()
        reporter.update(0)
        return reporter

    monkeypatch.setattr(module, "start_progress", start_progress)

    sno.Dataset.compute(
        dataset_dir=source,
        computations=(definition,),
        dtype="float32",
        n_jobs=1,
        progress=True,
        overwrite=False,
    )

    assert updates == [0, 2]


def test_progress_computation_is_persisted_with_mole_reference(tmp_path: Path) -> None:
    gas = ct.Solution("h2o2.yaml")
    gas.TP = 300.0, ct.one_atm
    gas.set_equivalence_ratio(
        phi=0.45,
        fuel={"H2": 1.0},
        oxidizer={"O2": 0.21, "N2": 0.79},
        basis="mole",
    )
    unburnt_h2 = gas["H2"].Y[0]
    gas.equilibrate("HP")
    burnt_h2 = gas["H2"].Y[0]
    source = _dataset(tmp_path, {"H2": np.array([unburnt_h2, burnt_h2])})
    definition = sno.Computing.progress(
        mechanism="h2o2.yaml",
        equivalence_ratio=0.45,
        fuel={"H2": 1.0},
        oxidizer={"O2": 0.21, "N2": 0.79},
        species={"H2": 1.0},
        temperature=300.0,
        pressure=ct.one_atm,
        output="c",
    )

    result = sno.Dataset.compute(
        dataset_dir=source,
        computations=(definition,),
        dtype="float32",
        n_jobs=-1,
        progress=False,
        overwrite=False,
    )

    values = np.fromfile(source / "data/c_id000.dat", dtype="float32")
    np.testing.assert_allclose(values, [0.0, 1.0], atol=1.0e-6)
    assert result.fields == ("c",)
    manifest = json.loads((source / "info.json").read_text())
    assert manifest["global"]["fields"]["c"] == ["c"]
    assert manifest["global"]["metadata"]["computing"]["c"]["reference"][
        "composition_basis"
    ] == "mole"
    assert manifest["global"]["metadata"]["computing"]["c"]["reference"]["oxidizer"] == {
        "N2": 0.79,
        "O2": 0.21,
    }
    assert not tuple(source.glob(".stacknordic-compute-*"))


def test_progress_requires_a_nonempty_species_definition() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        sno.Computing.progress(
            mechanism="h2o2.yaml",
            equivalence_ratio=0.45,
            fuel={"H2": 1.0},
            oxidizer={"O2": 0.21, "N2": 0.79},
            species={},
            temperature=300.0,
            pressure=ct.one_atm,
            output="c",
        )


def test_progress_accepts_a_weighted_species_sum(tmp_path: Path) -> None:
    gas = ct.Solution("h2o2.yaml")
    gas.TP = 300.0, ct.one_atm
    gas.set_equivalence_ratio(0.45, {"H2": 1.0}, {"O2": 0.21, "N2": 0.79})
    unburnt = {name: gas[name].Y[0] for name in ("H2", "H2O")}
    gas.equilibrate("HP")
    burnt = {name: gas[name].Y[0] for name in ("H2", "H2O")}
    source = _dataset(
        tmp_path,
        {
            name: np.array([unburnt[name], burnt[name]])
            for name in ("H2", "H2O")
        },
    )
    definition = sno.Computing.progress(
        mechanism="h2o2.yaml",
        equivalence_ratio=0.45,
        fuel={"H2": 1.0},
        oxidizer={"O2": 0.21, "N2": 0.79},
        species={"H2": -1.0, "H2O": 1.0},
        temperature=300.0,
        pressure=ct.one_atm,
        output="c",
    )

    sno.Dataset.compute(
        dataset_dir=source,
        computations=(definition,),
        dtype="float64",
        n_jobs=1,
        progress=False,
        overwrite=False,
    )

    np.testing.assert_allclose(
        np.fromfile(source / "data/c_id000.dat", dtype="float64"),
        [0.0, 1.0],
        atol=1.0e-8,
    )


def test_species_and_progress_reaction_rates_have_explicit_units(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gas = ct.Solution("h2o2.yaml")
    gas.TPX = 1100.0, ct.one_atm, {"H2": 2.0, "O2": 1.0, "N2": 3.76}
    fields = {
        name: np.full(2, gas[name].Y[0])
        for name in gas.species_names
    }
    fields["T"] = np.full(2, gas.T)
    fields["p"] = np.full(2, gas.P)
    fields["rho"] = np.full(2, gas.density)
    source = _dataset(tmp_path, fields)
    progress = sno.Computing.progress(
        mechanism="h2o2.yaml",
        equivalence_ratio=1.0,
        fuel={"H2": 1.0},
        oxidizer={"O2": 1.0, "N2": 3.76},
        species={"H2": -1.0, "H2O": 1.0},
        temperature=300.0,
        pressure=ct.one_atm,
        output="c",
    )
    state_fields = {name: name for name in gas.species_names}
    species_rate = sno.Computing.reaction_rate(
        mechanism="h2o2.yaml",
        species=state_fields,
        temperature="T",
        pressure="p",
        target="H2",
        output="omega_H2_mass",
        units="kg/m^3/s",
    )
    progress_rate = sno.Computing.reaction_rate(
        mechanism="h2o2.yaml",
        species=state_fields,
        temperature="T",
        pressure="p",
        target=progress,
        output="omega_c",
        units="1/s",
    )
    module = importlib.import_module("stacknordic.dataset.compute")
    original = module.reaction_rate_chunks
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        yield from original(*args, **kwargs)

    monkeypatch.setattr(module, "reaction_rate_chunks", counted)

    sno.Dataset.compute(
        dataset_dir=source,
        computations=(species_rate, progress_rate),
        dtype="float64",
        n_jobs=1,
        progress=False,
        overwrite=False,
    )

    mass_rates = gas.net_production_rates * gas.molecular_weights
    expected_species = mass_rates[gas.species_index("H2")]
    expected_progress = sum(
        coefficient * mass_rates[gas.species_index(name)]
        for name, coefficient in progress.coefficients.items()
    ) / gas.density
    np.testing.assert_allclose(
        np.fromfile(source / "data/omega_H2_mass_id000.dat", dtype="float64"),
        expected_species,
    )
    np.testing.assert_allclose(
        np.fromfile(source / "data/omega_c_id000.dat", dtype="float64"),
        expected_progress,
    )
    manifest = json.loads((source / "info.json").read_text())
    metadata = manifest["global"]["metadata"]["computing"]
    assert metadata["omega_H2_mass"]["units"] == "kg/m^3/s"
    assert metadata["omega_c"]["units"] == "1/s"
    assert calls == 1
