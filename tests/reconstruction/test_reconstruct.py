import subprocess
from pathlib import Path

import pytest

import stacknordic as sno
from stacknordic import reconstruction
from stacknordic.openfoam import environment as openfoam
from stacknordic.resources import workers
from stacknordic.selection import select_times


def _write_header(path: Path, *, family: str = "org", version: str = "12") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"/* OpenFOAM Foundation: {family} Version: {version} */\n"
        "FoamFile\n"
        "{\n"
        "    format ascii;\n"
        "}\n",
        encoding="utf-8",
    )


def _write_boundary(path: Path, patches: tuple[tuple[str, str], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = "\n".join(
        f"{name}\n{{\n    type {patch_type};\n    nFaces 1;\n    startFace 0;\n}}"
        for name, patch_type in patches
    )
    path.write_text(f"{len(patches)}\n(\n{entries}\n)\n", encoding="utf-8")


def _case(tmp_path: Path) -> Path:
    case_dir = tmp_path / "case"
    _write_header(case_dir / "system" / "controlDict")
    (case_dir / "constant" / "polyMesh").mkdir(parents=True)
    _write_boundary(
        case_dir / "constant/polyMesh/boundary",
        (("walls", "wall"),),
    )
    for rank in range(2):
        _write_boundary(
            case_dir / f"processor{rank}/constant/polyMesh/boundary",
            (("walls", "wall"), (f"procBoundary{rank}", "processor")),
        )
        for name in ("points", "faces", "owner", "neighbour"):
            (case_dir / f"processor{rank}/constant/polyMesh" / name).write_text(
                "fixture",
                encoding="utf-8",
            )
        for time_name in ("0", "0.1", "0.2"):
            _write_header(case_dir / f"processor{rank}" / time_name / "U")
    (case_dir / "0").mkdir()
    return case_dir


@pytest.fixture(autouse=True)
def _reset_toolchain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        reconstruction,
        "_runtime_identity",
        lambda selected: reconstruction._Runtime("esi", "2512"),
    )


def test_reconstructs_missing_times_and_preserves_existing_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_dir = _case(tmp_path)
    commands: list[tuple[str, ...]] = []

    def execute(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        staged_case = Path(command[command.index("-case") + 1])
        time_name = command[command.index("-time") + 1]
        (staged_case / time_name).mkdir()
        return subprocess.CompletedProcess(command, 0, "ok")

    monkeypatch.setattr(reconstruction, "_execute", execute)

    result = sno.reconstruct(
        case_dir=case_dir,
        times=None,
        fields=["U", "p"],
        n_jobs=2,
        progress=False,
    )

    assert result.reconstructed_times == (0.1, 0.2)
    assert result.skipped_times == (0.0,)
    assert result.output_dir == case_dir
    assert result.processor_count == 2
    assert result.source_family == "foundation"
    assert result.source_versions == ("12",)
    assert result.fields == ("U", "p")
    assert result.n_jobs == 2
    assert len(commands) == 2
    assert all(command[-2:] == ("-fields", "(U p)") for command in commands)


def test_failed_reconstruction_removes_new_partial_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_dir = _case(tmp_path)

    def execute(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        staged_case = Path(command[command.index("-case") + 1])
        time_name = command[command.index("-time") + 1]
        (staged_case / time_name).mkdir()
        return subprocess.CompletedProcess(command, 1, "fatal reconstruction error")

    monkeypatch.setattr(reconstruction, "_execute", execute)

    with pytest.raises(RuntimeError, match="fatal reconstruction error"):
        sno.reconstruct(case_dir=case_dir, times=[0.1])

    assert not (case_dir / "0.1").exists()
    assert not tuple(tmp_path.glob(".stacknordic-reconstruct-*"))


def test_reconstructs_into_separate_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_dir = _case(tmp_path)
    output_dir = tmp_path / "reconstructed"

    def execute(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        staged_case = Path(command[command.index("-case") + 1])
        time_name = command[command.index("-time") + 1]
        (staged_case / time_name).mkdir()
        return subprocess.CompletedProcess(command, 0, "ok")

    monkeypatch.setattr(reconstruction, "_execute", execute)

    result = sno.reconstruct(
        case_dir=case_dir,
        output_dir=output_dir,
        times=[0.1],
    )

    assert result.output_dir == output_dir.resolve()
    assert (output_dir / "0.1").is_dir()
    assert not (case_dir / "0.1").exists()


def test_rejects_time_missing_from_one_processor(tmp_path: Path) -> None:
    case_dir = _case(tmp_path)
    (case_dir / "processor1" / "0.2" / "U").unlink()
    (case_dir / "processor1" / "0.2").rmdir()

    with pytest.raises(KeyError, match="incomplete or unavailable"):
        sno.reconstruct(case_dir=case_dir, times=[0.2])


def test_case_wraps_reconstruction_in_selected_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_dir = _case(tmp_path)
    commands: list[tuple[str, ...]] = []

    def execute(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        staged_case = Path(command[2].split(" -case ", maxsplit=1)[1].split()[0])
        (staged_case / "0.1").mkdir()
        return subprocess.CompletedProcess(command, 0, "ok")

    monkeypatch.setattr(reconstruction, "_execute", execute)
    case = sno.OpenFOAM.case(
        case_dir=case_dir,
        of_cmd="openfoam/2512",
        shell="bash",
    )
    case.reconstruct(times=[0.1])

    assert commands[0][:2] == ("bash", "-lc")
    assert "module load openfoam/2512" in commands[0][2]
    assert "reconstructPar" in commands[0][2]


def test_runtime_preflight_accepts_foundation_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()
    monkeypatch.setattr(
        reconstruction,
        "_execute",
        lambda command: subprocess.CompletedProcess(
            command,
            0,
            "OpenFOAM Foundation version 12 - openfoam.org",
        ),
    )

    runtime = reconstruction._runtime_identity(openfoam.toolchain())

    assert runtime == reconstruction._Runtime("foundation", "12")


def test_foundation_module_shorthand() -> None:
    assert openfoam._normalise_command("openfoam-org/14") == "module load openfoam-org/14"


def test_runtime_preflight_reports_missing_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()
    monkeypatch.setattr(
        reconstruction,
        "_execute",
        lambda command: (_ for _ in ()).throw(FileNotFoundError("reconstructPar")),
    )

    with pytest.raises(RuntimeError, match="runtime preflight failed"):
        reconstruction._runtime_identity(openfoam.toolchain())


def test_minus_one_uses_available_cpu_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("stacknordic.resources.available_cpus", lambda: 32)

    assert workers(-1, 20) == 20
    assert workers(-1, 2) == 2


def test_time_tuple_selects_an_inclusive_range() -> None:
    available = (0.0, 0.1, 0.2, 0.3)

    assert select_times(available, (0.1, 0.2)) == (0.1, 0.2)


def test_time_list_selects_exact_values_even_with_two_entries() -> None:
    available = (0.0, 0.1, 0.2, 0.3)

    assert select_times(available, [0.1, 0.3]) == (0.1, 0.3)


def test_generates_foundation_boundary_addressing(tmp_path: Path) -> None:
    case_dir = _case(tmp_path)
    destination = tmp_path / "boundaryProcAddressing"

    reconstruction._write_boundary_addressing(
        case_dir,
        case_dir / "processor0",
        destination,
    )

    payload = destination.read_text(encoding="utf-8")
    assert "2\n(\n0\n-1\n)" in payload


@pytest.mark.parametrize("n_jobs", [0, -2, True])
def test_rejects_invalid_worker_count(n_jobs: int) -> None:
    with pytest.raises(ValueError, match="n_jobs"):
        workers(n_jobs, 2)
