from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import stacknordic as sno
import stacknordic.openfoam.postprocess as postprocess_module


def _case(tmp_path: Path) -> Path:
    root = tmp_path / "case"
    (root / "constant").mkdir(parents=True)
    (root / "system").mkdir()
    (root / "0.1").mkdir()
    (root / "system/controlDict").write_text("application solver;\n")
    return root


def test_case_postprocess_runs_declared_openfoam_function(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _case(tmp_path)

    def run(command, **kwargs):
        assert command == (
            "postProcess",
            "-case",
            str(root),
            "-func",
            "mag(U)",
            "-time",
            "0.1",
        )
        (root / "0.1/mag(U)").write_text("field")
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setattr(postprocess_module.subprocess, "run", run)
    case = sno.OpenFOAM.case(case_dir=root)
    result = case.postprocess(
        operations=(sno.Postprocess.mag(field="U"),),
        times=[0.1],
        n_jobs=None,
        progress=False,
        overwrite=False,
    )

    assert result.fields == ("magU",)
    assert (root / "0.1/magU").read_text() == "field"
    assert not (root / "0.1/mag(U)").exists()


def test_case_postprocess_rejects_nonstandard_q_field(tmp_path: Path) -> None:
    case = sno.OpenFOAM.case(case_dir=_case(tmp_path))

    with pytest.raises(ValueError, match="currently requires field='U'"):
        case.postprocess(
            operations=(sno.Postprocess.Q(field="velocity"),),
            times=[0.1],
        )
