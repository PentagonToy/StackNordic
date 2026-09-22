from __future__ import annotations

import subprocess
from pathlib import Path

from stacknordic.dataset import derived
from stacknordic.openfoam.environment import toolchain


def test_reconstruction_fields_replace_gradients_with_their_sources() -> None:
    assert derived.reconstruction_fields(("T", "gradU", "gradp", "U")) == ("T", "U", "p")


def test_materialise_gradient_uses_isolated_postprocess_case(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    case = tmp_path / "stage"
    (source / "system").mkdir(parents=True)
    (source / "constant").mkdir()
    (case / "0.1").mkdir(parents=True)

    def run(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        (case / "0.1" / "grad(U)").write_text("field")
        return subprocess.CompletedProcess(command, 0, "ok")

    monkeypatch.setattr(derived, "_run", run)

    derived.materialise_gradients(
        case,
        source,
        ((0.1, "0.1"),),
        ("gradU",),
        toolchain(),
    )

    assert (case / "system").is_symlink()
    assert (case / "constant").is_symlink()
    assert (case / "0.1" / "gradU").read_text() == "field"
