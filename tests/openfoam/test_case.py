from __future__ import annotations

import json
from pathlib import Path

import cantera as ct
import numpy as np

import stacknordic as sno


def _field(name: str, values: np.ndarray) -> str:
    body = "\n".join(f"{value:.16g}" for value in values)
    return (
        "FoamFile\n{\n"
        "    format ascii;\n"
        "    class volScalarField;\n"
        f"    object {name};\n"
        "}\n"
        f"internalField nonuniform List<scalar> {values.size}\n(\n{body}\n);\n"
        "boundaryField {}\n"
    )


def _case(tmp_path: Path, values: np.ndarray) -> Path:
    root = tmp_path / "case"
    (root / "constant").mkdir(parents=True)
    (root / "system").mkdir()
    (root / "0.1").mkdir()
    (root / "system/controlDict").write_text(
        "// OpenFOAM Foundation\napplication solver;\n"
    )
    (root / "system/blockMeshDict").write_text(
        "convertToMeters 1;\n"
        "xMin 0;\nxMax 2;\nyMin 0;\nyMax 1;\nzMin 0;\nzMax 1;\n"
        "nx 2;\nny 1;\nnz 1;\n"
        "blocks ( hex (0 1 2 3 4 5 6 7) ($nx $ny $nz) simpleGrading (1 1 1) );\n"
    )
    (root / "0.1/H2").write_text(_field("H2", values))
    return root


def test_case_inspects_layout_and_materialises_virtual_progress(tmp_path: Path) -> None:
    gas = ct.Solution("h2o2.yaml")
    gas.TP = 300.0, ct.one_atm
    gas.set_equivalence_ratio(0.45, {"H2": 1.0}, {"O2": 0.21, "N2": 0.79})
    unburnt = gas["H2"].Y[0]
    gas.equilibrate("HP")
    burnt = gas["H2"].Y[0]
    case = sno.OpenFOAM.case(case_dir=_case(tmp_path, np.array([unburnt, burnt])))
    progress = sno.Computing.progress(
        mechanism="h2o2.yaml",
        equivalence_ratio=0.45,
        fuel={"H2": 1.0},
        oxidizer={"O2": 0.21, "N2": 0.79},
        species={"H2": 1.0},
        temperature=300.0,
        pressure=ct.one_atm,
        output="c",
    )

    computed = case.compute(
        computations=(progress,),
        times=None,
        n_jobs=None,
        progress=False,
        overwrite=False,
    )
    dataset = case.export(
        output_dir=tmp_path / "dataset",
        fields={"c": "progress"},
        times=None,
        dtype="float32",
        mode="direct",
        metadata=None,
        n_jobs=None,
        progress=False,
        overwrite=False,
    )

    assert case.family == "foundation"
    assert case.reconstructed_times == (0.1,)
    assert computed.fields == ("c",)
    assert (case.case_dir / "0.1/c").is_file()
    np.testing.assert_allclose(
        np.fromfile(dataset.output_dir / "data/progress_id000.dat", "float32"),
        [0.0, 1.0],
        atol=1.0e-6,
    )
    manifest = json.loads((dataset.output_dir / "info.json").read_text())
    assert manifest["global"]["fields"] == {"c": ["progress"]}
    assert not (dataset.output_dir / "data/H2_id000.dat").exists()


def test_runtime_mismatch_uses_two_short_lines(tmp_path: Path, capsys) -> None:
    root = _case(tmp_path, np.array([0.0, 1.0]))
    sno.OpenFOAM.case(
        case_dir=root,
        of_cmd="openfoam/2512",
        shell="bash",
    )

    assert capsys.readouterr().out.splitlines() == [
        "[StackNordic] Active runtime: OpenFOAM.com; case: OpenFOAM Foundation.",
        "[StackNordic] Direct reads remain available; OpenFOAM commands may need a matching runtime.",
    ]
