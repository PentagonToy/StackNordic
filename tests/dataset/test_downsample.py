from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

import stacknordic as sno


def _dataset(tmp_path: Path, *, constant: bool = False) -> Path:
    root = tmp_path / "source"
    data = root / "data"
    grid = root / "grid"
    data.mkdir(parents=True)
    grid.mkdir()
    shape = (2, 2, 4)
    x = np.ones(shape) if constant else np.broadcast_to([0.0, 2.0, 4.0, 6.0], shape)
    rho = np.ones(shape) if constant else np.broadcast_to([1.0, 1.0, 3.0, 3.0], shape)
    fields = {
        "UX": x,
        "UY": np.zeros(shape),
        "UZ": np.zeros(shape),
        "P": np.full(shape, 2.0),
        "RHO": rho,
        "OMEGA": 2.0 * rho,
    }
    record: dict[str, object] = {"id": 0, "time [s]": 0.1, "cells": 16}
    for name, values in fields.items():
        filename = f"{name}_id000.dat"
        np.asarray(values, dtype="float32").tofile(data / filename)
        record[f"{name} filename"] = f"./data/{filename}"
    for axis, values in {
        "X": np.arange(4, dtype="float32") + 0.5,
        "Y": np.arange(2, dtype="float32") + 0.5,
        "Z": np.arange(2, dtype="float32") + 0.5,
    }.items():
        values.tofile(grid / f"{axis}_m.dat")
    info = {
        "global": {
            "snapshots": 1,
            "variables": list(fields),
            "dtype": "float32",
            "layout": "cartesian",
            "grid": {axis.lower(): f"./grid/{axis}_m.dat" for axis in "XYZ"},
            "fields": {
                "U": ["UX", "UY", "UZ"],
                "p": ["P"],
                "rho": ["RHO"],
                "omega_c_mass": ["OMEGA"],
            },
            "Nxyz": [4, 2, 2],
            "metadata": {},
        },
        "local": [record],
    }
    (root / "info.json").write_text(json.dumps(info))
    return root


def test_box_downsampling_applies_favre_moments(tmp_path: Path) -> None:
    source = _dataset(tmp_path)
    result = sno.Dataset.downsample(
        dataset_dir=source,
        output_dir=tmp_path / "les",
        factor=(2, 2, 2),
        filtering=sno.Filtering.flow(
            regime="compressible",
            density="rho",
            velocity="U",
            pressure="p",
            sgs_stress="tau_sgs",
            sgs_energy="k_sgs",
        ),
        kernel="box",
        dtype="float32",
        n_jobs=-1,
        progress=False,
        overwrite=False,
    )

    assert result.factor == (2, 2, 2)
    np.testing.assert_allclose(np.fromfile(tmp_path / "les/data/UX_id000.dat", "float32"), [1, 5])
    np.testing.assert_allclose(
        np.fromfile(tmp_path / "les/data/tau_sgs_xx_id000.dat", "float32"), [1, 1]
    )
    np.testing.assert_allclose(
        np.fromfile(tmp_path / "les/data/k_sgs_id000.dat", "float32"), [0.5, 0.5]
    )


def test_gaussian_downsampling_preserves_constant_fields(tmp_path: Path) -> None:
    source = _dataset(tmp_path, constant=True)
    sno.Dataset.downsample(
        dataset_dir=source,
        output_dir=tmp_path / "les",
        factor=(2, 2, 2),
        filtering=sno.Filtering.flow(
            regime="incompressible",
            velocity="U",
            pressure="p",
            sgs_stress=None,
            sgs_energy=None,
        ),
        kernel="gaussian",
        sigma=(1.0, 1.0, 1.0),
        boundary="reflect",
        dtype="float32",
        n_jobs=1,
        progress=False,
        overwrite=False,
    )

    np.testing.assert_allclose(np.fromfile(tmp_path / "les/data/UX_id000.dat", "float32"), 1.0)
    np.testing.assert_allclose(np.fromfile(tmp_path / "les/data/P_id000.dat", "float32"), 2.0)


def test_combustion_filtering_derives_specific_reaction_rate(tmp_path: Path) -> None:
    source = _dataset(tmp_path)
    sno.Dataset.downsample(
        dataset_dir=source,
        output_dir=tmp_path / "les",
        factor=(2, 2, 2),
        filtering=sno.Filtering(
            density="rho",
            mean=("rho", "omega_c_mass"),
            specific={"omega_c_mass": "omega_c"},
        ),
        kernel="box",
        dtype="float32",
        n_jobs=1,
        progress=False,
        overwrite=False,
    )

    np.testing.assert_allclose(
        np.fromfile(tmp_path / "les/data/omega_c_id000.dat", "float32"),
        2.0,
    )


def _package(source: Path, output_dir: Path) -> sno.PackagedDataset:
    return sno.Dataset.package(
        dataset_dir=source,
        output_dir=output_dir,
        name="flow",
        fields={
            "U": "U",
            "p": "p",
            "rho": "rho",
            "omega_c_mass": "omega_c_mass",
        },
        n_jobs=1,
    )


def _downsample(source: Path, output_dir: Path) -> None:
    sno.Dataset.downsample(
        dataset_dir=source,
        output_dir=output_dir,
        factor=(2, 2, 2),
        filtering=sno.Filtering.flow(
            regime="compressible",
            density="rho",
            velocity="U",
            pressure="p",
            sgs_stress="tau_sgs",
            sgs_energy="k_sgs",
        ),
        n_jobs=1,
    )


def test_downsample_accepts_snapshot_hdf5_packages(tmp_path: Path) -> None:
    source = _dataset(tmp_path)
    packaged = _package(source, tmp_path / "packaged")
    _downsample(source, tmp_path / "native-les")
    _downsample(packaged.output_dir, tmp_path / "hdf5-les")

    for name in ("UX", "UY", "UZ", "P", "RHO", "tau_sgs_xx", "k_sgs"):
        np.testing.assert_array_equal(
            np.fromfile(tmp_path / f"native-les/data/{name}_id000.dat", "float32"),
            np.fromfile(tmp_path / f"hdf5-les/data/{name}_id000.dat", "float32"),
        )


def test_downsample_accepts_compiled_hdf5_series(tmp_path: Path) -> None:
    source = _dataset(tmp_path)
    packaged = _package(source, tmp_path / "packaged")
    compiled = sno.Dataset.compile(
        dataset_dir=packaged.output_dir,
        output_file=tmp_path / "flow.h5",
        pattern="*.h5",
    )
    _downsample(source, tmp_path / "native-les")
    _downsample(compiled.output_file, tmp_path / "series-les")

    for name in ("UX", "UY", "UZ", "P", "RHO", "tau_sgs_xx", "k_sgs"):
        np.testing.assert_array_equal(
            np.fromfile(tmp_path / f"native-les/data/{name}_id000.dat", "float32"),
            np.fromfile(tmp_path / f"series-les/data/{name}_id000.dat", "float32"),
        )


def test_downsample_writes_compiled_hdf5_without_persistent_intermediates(
    tmp_path: Path,
) -> None:
    source = _dataset(tmp_path)
    output = tmp_path / "les/output/flow.h5"
    result = sno.Dataset.downsample(
        dataset_dir=source,
        output_file=output,
        factor=(2, 2, 2),
        filtering=sno.Filtering.flow(
            regime="compressible",
            density="rho",
            velocity="U",
            pressure="p",
            sgs_stress="tau_sgs",
            sgs_energy="k_sgs",
        ),
        fields={
            "U": "fields / U",
            "p": "fields / p",
            "rho": "fields / rho",
            "tau_sgs": "fields / tau_sgs",
            "k_sgs": "fields / k_sgs",
        },
        compression="lzf",
        n_jobs=-1,
    )

    assert result.output_file == output
    assert not tuple(output.parent.glob(".stacknordic-downsample-*"))
    with h5py.File(output) as file:
        definition = json.loads(file.attrs["stacknordic_contract"])
        snapshots = json.loads(file.attrs["stacknordic_snapshots"])
        assert definition["schema"] == "stacknordic-hdf5-series-v1"
        assert definition["shape_zyx"] == [1, 1, 2]
        assert snapshots == [{"id": 0, "time_s": 0.1, "group": "id_000"}]
        np.testing.assert_allclose(file["id_000/fields/U"][..., 0], [[[1, 5]]])
        np.testing.assert_allclose(file["id_000/fields/k_sgs"], [[[0.5, 0.5]]])
