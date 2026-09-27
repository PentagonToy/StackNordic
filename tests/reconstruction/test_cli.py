from pathlib import Path

import pytest

from stacknordic import cli
from stacknordic.reconstruction import Reconstruction


def _result(case_dir: Path, output_dir: Path) -> Reconstruction:
    return Reconstruction(
        case_dir=case_dir,
        output_dir=output_dir,
        processor_count=4,
        reconstructed_times=(0.1, 0.2),
        skipped_times=(0.0,),
        fields=("U", "p"),
        source_family="foundation",
        source_versions=("14",),
        runtime_family="foundation",
        runtime_version="14",
        n_jobs=2,
        elapsed_seconds=1.25,
    )


def test_version(capsys):
    with pytest.raises(SystemExit, match="0"):
        cli.main(["--version"])

    assert capsys.readouterr().out == "stacknordic 0.1.0\n"


@pytest.mark.parametrize("command", ["reconstruct", "-reconstructPar"])
def test_reconstruct_commands_dispatch_to_python_api(monkeypatch, tmp_path, command, capsys):
    captured = {}

    def fake_reconstruct(**kwargs):
        captured.update(kwargs)
        return _result(kwargs["case_dir"], kwargs["output_dir"])

    monkeypatch.setattr(cli, "reconstruct", fake_reconstruct)
    output = tmp_path / "output"

    status = cli.main(
        [
            command,
            "--case-dir",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--time",
            "0.1",
            "--time",
            "0.2",
            "--fields",
            "U",
            "p",
            "--n-jobs",
            "2",
            "--progress",
        ]
    )

    assert status == 0
    assert captured == {
        "case_dir": tmp_path,
        "output_dir": output,
        "times": [0.1, 0.2],
        "fields": ["U", "p"],
        "n_jobs": 2,
        "progress": True,
    }
    assert "Reconstructed times: 0.1, 0.2" in capsys.readouterr().out


def test_reconstruct_time_range(monkeypatch, tmp_path):
    captured = {}

    def fake_reconstruct(**kwargs):
        captured.update(kwargs)
        return _result(kwargs["case_dir"], tmp_path)

    monkeypatch.setattr(cli, "reconstruct", fake_reconstruct)

    assert cli.main(["reconstruct", "--case-dir", str(tmp_path), "--time-range", "0", "1"]) == 0
    assert captured["times"] == (0.0, 1.0)


def test_reconstruct_defaults_to_current_directory(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.chdir(tmp_path)

    def fake_reconstruct(**kwargs):
        captured.update(kwargs)
        return _result(kwargs["case_dir"], tmp_path)

    monkeypatch.setattr(cli, "reconstruct", fake_reconstruct)

    assert cli.main(["reconstruct"]) == 0
    assert captured["case_dir"] == tmp_path
    assert captured["output_dir"] is None
    assert captured["times"] is None


def test_reconstruct_reports_expected_errors(monkeypatch, capsys):
    def fake_reconstruct(**kwargs):
        raise FileNotFoundError("No processor directories were found")

    monkeypatch.setattr(cli, "reconstruct", fake_reconstruct)

    assert cli.main(["reconstruct"]) == 1
    assert "No processor directories were found" in capsys.readouterr().err
