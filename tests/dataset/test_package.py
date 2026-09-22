from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import stacknordic as sno


def _dataset(tmp_path: Path, *, snapshots: int = 1) -> Path:
    root = tmp_path / "dataset"
    data = root / "data"
    grid = root / "grid"
    data.mkdir(parents=True)
    grid.mkdir()
    fields = {
        "c": ("c",),
        "T": ("T_K",),
        "U": ("Ux", "Uy", "Uz"),
        "H2O2": ("YH2O2",),
    }
    values = {
        "c": np.arange(4),
        "T_K": np.arange(4) + 300.0,
        "Ux": np.arange(4) + 1.0,
        "Uy": np.arange(4) + 2.0,
        "Uz": np.arange(4) + 3.0,
        "YH2O2": np.arange(4) / 10.0,
    }
    records = []
    for snapshot_id in range(snapshots):
        record: dict[str, object] = {
            "id": snapshot_id,
            "time [s]": 0.001 * (snapshot_id + 1),
            "cells": 4,
        }
        for index, (name, value) in enumerate(values.items()):
            filename = f"{name}_id{snapshot_id:03d}.dat"
            np.asarray(value + snapshot_id, dtype="float32").tofile(data / filename)
            key = name if index % 2 else f"{name} filename"
            record[key] = f"./data/{filename}"
        records.append(record)
    for axis, value in zip("XYZ", (np.arange(2), np.arange(2), np.arange(1)), strict=True):
        np.asarray(value, dtype="float32").tofile(grid / f"{axis}_m.dat")
    info = {
        "global": {
            "snapshots": snapshots,
            "variables": list(values),
            "dtype": "float32",
            "layout": "cartesian",
            "grid": {axis.lower(): f"./grid/{axis}_m.dat" for axis in "XYZ"},
            "fields": {name: list(components) for name, components in fields.items()},
            "Nxyz": [2, 2, 1],
            "metadata": {"equivalence_ratio": 0.45, "source": "test"},
        },
        "local": records,
    }
    (root / "info.json").write_text(json.dumps(info))
    return root


def test_package_writes_blastnet_compatible_hdf5(tmp_path: Path) -> None:
    source = _dataset(tmp_path)
    result = sno.Dataset.package(
        dataset_dir=source,
        output_dir=tmp_path / "dns",
        name="phi045",
        format="hdf5",
        fields={
            "c": "c",
            "T": "T_K",
            "U": "U",
            "H2O2": " species_fields / H2O2 ",
        },
        compression="lzf",
        n_jobs=-1,
        progress=False,
        overwrite=False,
    )

    assert result.snapshots == 1
    assert result.files == (tmp_path / "dns/phi045_id000.h5",)
    with h5py.File(result.files[0]) as file:
        assert file.attrs["phi"] == pytest.approx(0.45)
        assert file.attrs["time_s"] == pytest.approx(0.001)
        np.testing.assert_array_equal(file["c"], np.arange(4).reshape(1, 2, 2))
        np.testing.assert_array_equal(file["T_K"], (np.arange(4) + 300).reshape(1, 2, 2))
        assert file["U"].shape == (1, 2, 2, 3)
        np.testing.assert_array_equal(
            file["species_fields/H2O2"],
            (np.arange(4) / 10).astype("float32").reshape(1, 2, 2),
        )
        assert tuple(file["grid"]) == ("x_axis", "y_axis", "z_axis")


def test_package_rejects_non_hdf5_formats(tmp_path: Path) -> None:
    source = _dataset(tmp_path)
    with pytest.raises(ValueError, match="format must be 'hdf5'"):
        sno.Dataset.package(
            dataset_dir=source,
            output_dir=tmp_path / "dns",
            name="phi045",
            format="parquet",
            fields={"c": "c"},
        )


def test_package_rejects_conflicting_hdf5_paths(tmp_path: Path) -> None:
    source = _dataset(tmp_path)
    with pytest.raises(ValueError, match="conflicts"):
        sno.Dataset.package(
            dataset_dir=source,
            output_dir=tmp_path / "dns",
            name="phi045",
            fields={"c": "species", "T": "species / T"},
        )


def test_compile_writes_self_describing_time_series(tmp_path: Path) -> None:
    source = _dataset(tmp_path, snapshots=2)
    packaged = sno.Dataset.package(
        dataset_dir=source,
        output_dir=tmp_path / "dns",
        name="phi045",
        fields={"c": "c", "U": "flow/U", "H2O2": "species/H2O2"},
        n_jobs=1,
    )

    result = sno.Dataset.compile(
        dataset_dir=packaged.output_dir,
        output_file=tmp_path / "phi045.h5",
        pattern="*phi045_id*.h5",
    )

    assert result.snapshots == 2
    assert result.source_files == packaged.files
    assert not tuple(tmp_path.glob(".phi045-*.h5"))
    with h5py.File(result.output_file) as file:
        contract = json.loads(file.attrs["stacknordic_contract"])
        snapshots = json.loads(file.attrs["stacknordic_snapshots"])
        assert contract["schema"] == "stacknordic-hdf5-series-v1"
        assert contract["fields"] == {
            "c": "c",
            "U": "flow/U",
            "H2O2": "species/H2O2",
        }
        assert [item["time_s"] for item in snapshots] == pytest.approx([0.001, 0.002])
        np.testing.assert_array_equal(file["id_000/c"], np.arange(4).reshape(1, 2, 2))
        np.testing.assert_array_equal(
            file["id_001/species/H2O2"],
            (np.arange(4) / 10 + 1).astype("float32").reshape(1, 2, 2),
        )
        assert file["id_001/flow/U"].shape == (1, 2, 2, 3)


def test_compile_rejects_contract_mismatch(tmp_path: Path) -> None:
    source = _dataset(tmp_path, snapshots=2)
    packaged = sno.Dataset.package(
        dataset_dir=source,
        output_dir=tmp_path / "dns",
        name="phi045",
        fields={"c": "c"},
        n_jobs=1,
    )
    with h5py.File(packaged.files[1], "r+") as file:
        definition = json.loads(file.attrs["stacknordic_contract"])
        definition["dtype"] = "float64"
        file.attrs["stacknordic_contract"] = json.dumps(definition)

    with pytest.raises(ValueError, match="contract differs"):
        sno.Dataset.compile(
            dataset_dir=packaged.output_dir,
            output_file=tmp_path / "phi045.h5",
            pattern="*.h5",
        )
    assert not (tmp_path / "phi045.h5").exists()
