from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import stacknordic as sno


def _dataset(root: Path, *, layout: str = "cartesian") -> Path:
    data = root / "data"
    grid = root / "grid"
    data.mkdir(parents=True)
    grid.mkdir()
    coordinates = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    for axis in "XYZ":
        coordinates.tofile(grid / f"{axis}_m.dat")
    z, y, x = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    base = (x, y, z)
    records = []
    for index, time in enumerate((0.0, 1.0)):
        record: dict[str, object] = {"time [s]": time}
        for component, values in zip(("Ux", "Uy", "Uz"), base, strict=True):
            path = data / f"{component}_id{index:03d}.dat"
            (values + time).tofile(path)
            record[f"{component} filename"] = f"./data/{path.name}"
        pressure = data / f"p_id{index:03d}.dat"
        (x + 2.0 * y + 3.0 * z + time).tofile(pressure)
        record["p filename"] = f"./data/{pressure.name}"
        records.append(record)
    manifest = {
        "global": {
            "snapshots": 2,
            "variables": ["Ux", "Uy", "Uz", "p"],
            "dtype": "float64",
            "layout": layout,
            "grid": {axis.lower(): f"./grid/{axis}_m.dat" for axis in "XYZ"},
            "Nxyz": [3, 3, 3],
            "fields": {"U": ["Ux", "Uy", "Uz"], "p": ["p"]},
            "metadata": {},
        },
        "local": records,
    }
    (root / "info.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _field(root: Path, name: str, index: int, components: int = 1) -> np.ndarray:
    info = json.loads((root / "info.json").read_text())
    names = info["global"]["fields"][name]
    arrays = [
        np.fromfile(root / info["local"][index][f"{component} filename"], "float64")
        for component in names
    ]
    values = np.stack(arrays, axis=-1) if len(arrays) > 1 else arrays[0]
    return values.reshape((3, 3, 3, components)) if components > 1 else values.reshape(3, 3, 3)


def test_cartesian_dataset_postprocesses_supported_operations(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset")
    result = sno.Dataset.postprocess(
        dataset_dir=dataset,
        operations=(
            sno.Postprocess.grad(field="p"),
            sno.Postprocess.div(field="U"),
            sno.Postprocess.ddt(field="U"),
            sno.Postprocess.components(
                field="U",
                outputs=("U_component_x", "U_component_y", "U_component_z"),
            ),
            sno.Postprocess.mag(field="U"),
            sno.Postprocess.mag_sqr(field="U"),
            sno.Postprocess.vorticity(),
            sno.Postprocess.Q(),
            sno.Postprocess.Lambda2(),
            sno.Postprocess.write_cell_centres(),
            sno.Postprocess.write_cell_volumes(),
        ),
        n_jobs=2,
        progress=False,
        overwrite=False,
    )

    assert result.snapshots == 2
    np.testing.assert_allclose(
        _field(dataset, "gradp", 0, 3),
        np.broadcast_to([1.0, 2.0, 3.0], (3, 3, 3, 3)),
    )
    np.testing.assert_allclose(_field(dataset, "divU", 0), 3.0)
    np.testing.assert_allclose(_field(dataset, "ddtU", 0, 3), 1.0)
    np.testing.assert_allclose(_field(dataset, "vorticity", 0, 3), 0.0)
    np.testing.assert_allclose(_field(dataset, "Q", 0), -1.5)
    np.testing.assert_allclose(_field(dataset, "Lambda2", 0), 1.0)
    np.testing.assert_allclose(_field(dataset, "V", 0), 1.0)
    centres = _field(dataset, "C", 0, 3)
    np.testing.assert_allclose(centres[0, 0, 0], [-1.0, -1.0, -1.0])
    assert not tuple(dataset.glob(".stacknordic-postprocess-*"))


def test_dataset_postprocess_rejects_non_cartesian_layout(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset", layout="cells")

    with pytest.raises(ValueError, match="requires a Cartesian dataset"):
        sno.Dataset.postprocess(
            dataset_dir=dataset,
            operations=(sno.Postprocess.mag(field="p"),),
        )


def test_postprocess_namespace_is_discoverable() -> None:
    expected = {
        "grad",
        "div",
        "ddt",
        "components",
        "mag",
        "mag_sqr",
        "vorticity",
        "Q",
        "Lambda2",
        "write_cell_centres",
        "write_cell_volumes",
    }

    assert expected.issubset(dir(sno.Postprocess))
