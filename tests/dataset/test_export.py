from __future__ import annotations

import importlib
import json
from pathlib import Path

import cantera as ct
import numpy as np
import pytest

import stacknordic as sno
from stacknordic.openfoam.io import collated_blocks, read_field_payload, read_label_payload

dataset_module = importlib.import_module("stacknordic.dataset.export")


def _header(field_class: str, object_name: str) -> str:
    return (
        "FoamFile\n{\n"
        "    format ascii;\n"
        f"    class {field_class};\n"
        f"    object {object_name};\n"
        "}\n"
    )


def _labels(values: list[int]) -> bytes:
    body = "\n".join(str(value) for value in values)
    return (_header("labelList", "cellProcAddressing") + f"{len(values)}\n(\n{body}\n)\n").encode()


def _field(field_class: str, kind: str, name: str, values: list[object]) -> bytes:
    if kind == "scalar":
        body = "\n".join(str(value) for value in values)
    else:
        body = "\n".join("(" + " ".join(str(item) for item in value) + ")" for value in values)
    return (
        _header(field_class, name)
        + f"internalField nonuniform List<{kind}>\n{len(values)}\n(\n{body}\n);\n"
        + "boundaryField {}\n"
    ).encode()


def _collated(
    path: Path,
    name: str,
    blocks: dict[int, bytes],
    *,
    shared_header: bool = False,
) -> None:
    payload = (
        "FoamFile\n{\n"
        "    format binary;\n"
        "    class decomposedBlockData;\n"
        f"    object {name};\n"
        "}\n"
    ).encode()
    for index, (rank, block) in enumerate(blocks.items()):
        if shared_header and index:
            block = block[block.index(b"}") + 1 :]
        payload += f"\n// Processor{rank}\n\n{len(block)}\n(".encode() + block + b")\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _case(tmp_path: Path) -> Path:
    case = tmp_path / "case"
    (case / "system").mkdir(parents=True)
    (case / "constant").mkdir()
    (case / "system/controlDict").write_text("application solver;\n")
    (case / "system/blockMeshDict").write_text(
        "convertToMeters 1;\n"
        "xMin 0;\n"
        "xMax 2;\n"
        "yMin 0;\n"
        "yMax 1;\n"
        "zMin 0;\n"
        "zMax 1;\n"
        "nx 2;\n"
        "ny 1;\n"
        "nz 1;\n"
        "blocks ( hex (0 1 2 3 4 5 6 7) ($nx $ny $nz) simpleGrading (1 1 1) );\n"
    )
    group = case / "processors2"
    _collated(
        group / "constant/polyMesh/cellProcAddressing",
        "cellProcAddressing",
        {0: _labels([1]), 1: _labels([0])},
    )
    _collated(
        group / "0.1/U",
        "U",
        {
            0: _field("volVectorField", "vector", "U", [(10, 11, 12)]),
            1: _field("volVectorField", "vector", "U", [(20, 21, 22)]),
        },
    )
    _collated(
        group / "0.1/p",
        "p",
        {
            0: _field("volScalarField", "scalar", "p", [100]),
            1: _field("volScalarField", "scalar", "p", [200]),
        },
    )
    return case


def test_literal_single_block_is_recognised_as_cartesian(tmp_path: Path) -> None:
    case = tmp_path / "literal"
    (case / "system").mkdir(parents=True)
    (case / "system/blockMeshDict").write_text(
        "scale 2;\n"
        "vertices\n(\n"
        "    (0 0 0)\n    (2 0 0)\n    (2 1 0)\n    (0 1 0)\n"
        "    (0 0 1)\n    (2 0 1)\n    (2 1 1)\n    (0 1 1)\n"
        ");\n"
        "blocks\n(\n"
        "    hex (0 1 2 3 4 5 6 7) (2 1 1) simpleGrading (1 1 1)\n"
        ");\n",
        encoding="utf-8",
    )

    grid = dataset_module._cartesian_grid(case, 2, np.dtype("float64"))

    assert grid is not None
    np.testing.assert_allclose(grid["X"], [1.0, 3.0])
    np.testing.assert_allclose(grid["Y"], [1.0])
    np.testing.assert_allclose(grid["Z"], [1.0])


def test_decodes_collated_blocks(tmp_path: Path) -> None:
    path = tmp_path / "U"
    first = _field("volScalarField", "scalar", "p", [1, 2])
    second = _field("volScalarField", "scalar", "p", [3])
    _collated(path, "p", {4: first, 5: second}, shared_header=True)

    blocks = dict(collated_blocks(path))

    np.testing.assert_array_equal(read_field_payload(blocks[4]).values, [1, 2])
    np.testing.assert_array_equal(read_field_payload(blocks[5]).values, [3])


def test_grouped_collated_local_ranks_map_to_declared_global_range(tmp_path: Path) -> None:
    path = tmp_path / "processors4_2-3" / "U"
    _collated(
        path,
        "U",
        {
            0: _field("volScalarField", "scalar", "U", [1]),
            1: _field("volScalarField", "scalar", "U", [2]),
        },
    )
    source = dataset_module._Source(path.parent, 2, 3, True)

    assert [rank for rank, _ in dataset_module._processor_payloads(source, path)] == [2, 3]


def test_decodes_label_payload() -> None:
    np.testing.assert_array_equal(read_label_payload(_labels([2, 0, 1])), [2, 0, 1])


def test_decodes_binary_volume_field() -> None:
    values = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="<f8")
    payload = (
        b"FoamFile\n{\n"
        b"    format binary;\n"
        b'    arch "LSB;label=32;scalar=64";\n'
        b"    class volVectorField;\n"
        b"    object U;\n"
        b"}\n"
        b"internalField nonuniform List<vector> 2\n(\n"
    ) + values.tobytes() + b"\n);\nboundaryField {}\n"

    decoded = read_field_payload(payload)

    assert decoded.field_class == "volVectorField"
    np.testing.assert_array_equal(decoded.values, values)


def test_direct_export_restores_global_cell_order(tmp_path: Path) -> None:
    case = _case(tmp_path)
    output = tmp_path / "dataset"

    result = sno.Dataset.export(
        case_dir=case,
        output_dir=output,
        fields={"U": ("UX", "UY", "UZ"), "p": "P"},
        times=["0.1"],
        dtype="float32",
        mode="direct",
        metadata={"case": "fixture"},
        n_jobs=-1,
        progress=False,
        overwrite=False,
    )

    assert result.times == (0.1,)
    np.testing.assert_array_equal(np.fromfile(output / "data/UX_id000.dat", "float32"), [20, 10])
    np.testing.assert_array_equal(np.fromfile(output / "data/P_id000.dat", "float32"), [200, 100])
    np.testing.assert_allclose(np.fromfile(output / "grid/X_m.dat", "float32"), [0.5, 1.5])
    info = json.loads((output / "info.json").read_text())
    assert info["global"]["Nxyz"] == [2, 1, 1]
    assert info["global"]["metadata"] == {"case": "fixture"}


def test_export_accepts_precreated_empty_output_directory(tmp_path: Path) -> None:
    case = _case(tmp_path)
    output = tmp_path / "dataset"
    output.mkdir()

    sno.Dataset.export(
        case_dir=case,
        output_dir=output,
        fields={"p": "P"},
        times=[0.1],
        mode="direct",
        overwrite=False,
    )

    assert (output / "info.json").is_file()


def test_export_rejects_nonempty_output_directory_without_overwrite(tmp_path: Path) -> None:
    case = _case(tmp_path)
    output = tmp_path / "dataset"
    output.mkdir()
    (output / "existing").write_text("preserve")

    with pytest.raises(FileExistsError, match="output directory is not empty"):
        sno.Dataset.export(
            case_dir=case,
            output_dir=output,
            fields={"p": "P"},
            times=[0.1],
            mode="direct",
            overwrite=False,
        )

    assert (output / "existing").read_text() == "preserve"


def test_export_removes_staging_after_keyboard_interrupt(tmp_path: Path, monkeypatch) -> None:
    case = _case(tmp_path)

    def interrupt(_task):
        raise KeyboardInterrupt

    monkeypatch.setattr(dataset_module, "_run_direct_field_task", interrupt)

    with pytest.raises(KeyboardInterrupt):
        sno.Dataset.export(
            case_dir=case,
            output_dir=tmp_path / "dataset",
            fields={"p": "P"},
            times=[0.1],
            mode="direct",
            n_jobs=None,
        )

    assert not tuple(tmp_path.glob(".stacknordic-export-*"))


def test_direct_export_derives_missing_perfect_gas_density(tmp_path: Path) -> None:
    case = _case(tmp_path)
    group = case / "processors2"
    (case / "constant/foam").mkdir()
    (case / "constant/thermophysicalProperties").write_text(
        "thermoType\n{\n    equationOfState perfectGas;\n}\n"
        '#include "$FOAM_CASE/constant/foam/species.foam"\n'
        '#include "$FOAM_CASE/constant/foam/thermo.foam"\n'
    )
    (case / "constant/foam/species.foam").write_text("species 2 ( H2 N2 );\n")
    (case / "constant/foam/thermo.foam").write_text(
        "H2 { specie { molWeight 2; } }\n"
        "N2 { specie { molWeight 4; } }\n"
    )
    for name, values in {
        "T": (300.0, 400.0),
        "H2": (0.25, 0.50),
        "N2": (0.75, 0.50),
    }.items():
        _collated(
            group / f"0.1/{name}",
            name,
            {
                0: _field("volScalarField", "scalar", name, [values[0]]),
                1: _field("volScalarField", "scalar", name, [values[1]]),
            },
        )

    sno.Dataset.export(
        case_dir=case,
        output_dir=tmp_path / "density",
        fields={"rho": "RHO"},
        times=[0.1],
        dtype="float64",
        mode="direct",
    )

    pressure = np.asarray([200.0, 100.0])
    temperature = np.asarray([400.0, 300.0])
    h2 = np.asarray([0.50, 0.25])
    n2 = 1.0 - h2
    expected = pressure / (8314.46261815324 * temperature * (h2 / 2.0 + n2 / 4.0))
    np.testing.assert_allclose(
        np.fromfile(tmp_path / "density/data/RHO_id000.dat", "float64"),
        expected,
    )


def test_case_computes_progress_in_collated_layout(tmp_path: Path) -> None:
    case_dir = _case(tmp_path)
    group = case_dir / "processors2"
    gas = ct.Solution("h2o2.yaml")
    gas.TP = 300.0, ct.one_atm
    gas.set_equivalence_ratio(0.45, {"H2": 1.0}, {"O2": 0.21, "N2": 0.79})
    unburnt = gas["H2"].Y[0]
    gas.equilibrate("HP")
    burnt = gas["H2"].Y[0]
    _collated(
        group / "0.1/H2",
        "H2",
        {
            0: _field("volScalarField", "scalar", "H2", [burnt]),
            1: _field("volScalarField", "scalar", "H2", [unburnt]),
        },
    )
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

    sno.OpenFOAM.case(case_dir=case_dir).compute(
        computations=(definition,),
        times=[0.1],
        n_jobs=None,
        progress=False,
        overwrite=False,
    )
    sno.Dataset.export(
        case_dir=case_dir,
        output_dir=tmp_path / "progress",
        fields={"c": "c"},
        times=[0.1],
        mode="direct",
    )

    assert (group / "0.1/c").is_file()
    np.testing.assert_allclose(
        np.fromfile(tmp_path / "progress/data/c_id000.dat", "float32"),
        [0.0, 1.0],
        atol=1.0e-6,
    )


def test_direct_export_prefers_complete_reconstructed_time(tmp_path: Path) -> None:
    case = _case(tmp_path)
    (case / "0.1").mkdir()
    (case / "0.1/p").write_bytes(
        _field("volScalarField", "scalar", "p", [7, 8])
    )

    sno.Dataset.export(
        case_dir=case,
        output_dir=tmp_path / "dataset",
        fields={"p": "P"},
        times=[0.1],
        mode="direct",
    )

    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "dataset/data/P_id000.dat", "float32"), [7, 8]
    )


def test_reconstructed_time_local_mesh_disables_static_cartesian_grid(tmp_path: Path) -> None:
    case = _case(tmp_path)
    (case / "0.1/polyMesh").mkdir(parents=True)
    (case / "0.1/p").write_bytes(
        _field("volScalarField", "scalar", "p", [7, 8])
    )

    sno.Dataset.export(
        case_dir=case,
        output_dir=tmp_path / "dataset",
        fields={"p": "P"},
        times=[0.1],
        mode="direct",
    )

    info = json.loads((tmp_path / "dataset/info.json").read_text())
    assert info["global"]["layout"] == "cells"
    assert info["global"]["grid"] is None


def test_vector_requires_three_output_names(tmp_path: Path) -> None:
    case = _case(tmp_path)

    with pytest.raises(ValueError, match=r"\[StackNordic\].*requires 3 output component names"):
        sno.Dataset.export(
            case_dir=case,
            output_dir=tmp_path / "dataset",
            fields={"U": "velocity"},
            times=[0.1],
            mode="direct",
        )

    assert not tuple(tmp_path.glob(".stacknordic-export-*"))


def test_direct_export_explains_how_to_generate_missing_gradient(tmp_path: Path) -> None:
    case = _case(tmp_path)

    with pytest.raises(FileNotFoundError, match="mode='reconstruct'.*grad\\(U\\)"):
        sno.Dataset.export(
            case_dir=case,
            output_dir=tmp_path / "dataset",
            fields={"gradU": tuple(f"gradU{index}" for index in range(9))},
            times=[0.1],
            mode="direct",
        )


def test_time_local_addressing_replaces_constant_mapping(tmp_path: Path) -> None:
    case = _case(tmp_path)
    group = case / "processors2"
    _collated(
        group / "0.2/polyMesh/cellProcAddressing",
        "cellProcAddressing",
        {0: _labels([0]), 1: _labels([1])},
    )
    _collated(
        group / "0.2/p",
        "p",
        {
            0: _field("volScalarField", "scalar", "p", [300]),
            1: _field("volScalarField", "scalar", "p", [400]),
        },
    )

    sno.Dataset.export(
        case_dir=case,
        output_dir=tmp_path / "dataset",
        fields={"p": "P"},
        times=[0.2],
        mode="direct",
    )

    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "dataset/data/P_id000.dat", "float32"), [300, 400]
    )
    info = json.loads((tmp_path / "dataset/info.json").read_text())
    assert info["global"]["layout"] == "cells"


def test_all_cpu_workers_are_limited_by_available_memory(monkeypatch) -> None:
    resources = importlib.import_module("stacknordic.resources")
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(dataset_module, "_available_memory", lambda: 10_000)

    assert dataset_module._export_workers(-1, 20, 3_000) == 2
    assert dataset_module._export_workers(8, 20, 3_000) == 8


def test_all_cpu_workers_respect_process_and_slurm_limits(monkeypatch) -> None:
    resources = importlib.import_module("stacknordic.resources")
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(
        resources.os,
        "sched_getaffinity",
        lambda _: set(range(12)),
        raising=False,
    )
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")

    assert resources.workers(-1, 32) == 8


def test_each_time_selects_its_own_complete_decomposition(tmp_path: Path) -> None:
    case = _case(tmp_path)
    group = case / "processors1"
    _collated(
        group / "constant/polyMesh/cellProcAddressing",
        "cellProcAddressing",
        {0: _labels([0, 1])},
    )
    _collated(
        group / "0.2/p",
        "p",
        {0: _field("volScalarField", "scalar", "p", [300, 400])},
    )

    sno.Dataset.export(
        case_dir=case,
        output_dir=tmp_path / "dataset",
        fields={"p": "P"},
        times=[0.1, 0.2],
        mode="direct",
    )

    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "dataset/data/P_id000.dat", "float32"), [200, 100]
    )
    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "dataset/data/P_id001.dat", "float32"), [300, 400]
    )
