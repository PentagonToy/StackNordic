"""Command-line interface for StackNordic."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .reconstruction import Reconstruction, reconstruct


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stacknordic",
        description="Prepare stored OpenFOAM results.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    reconstruction = commands.add_parser(
        "reconstruct",
        help="reconstruct decomposed OpenFOAM times",
    )
    reconstruction.add_argument(
        "--case-dir",
        type=Path,
        default=Path.cwd(),
        help="OpenFOAM case directory (default: current directory)",
    )
    reconstruction.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="destination directory (default: case directory)",
    )
    times = reconstruction.add_mutually_exclusive_group()
    times.add_argument(
        "--time",
        type=float,
        action="append",
        dest="selected_times",
        metavar="VALUE",
        help="exact physical time; repeat for multiple times",
    )
    times.add_argument(
        "--time-range",
        type=float,
        nargs=2,
        metavar=("START", "END"),
        help="inclusive physical-time range",
    )
    reconstruction.add_argument(
        "--fields",
        nargs="+",
        default=None,
        metavar="FIELD",
        help="fields to reconstruct (default: all fields)",
    )
    reconstruction.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="concurrent physical times; -1 uses all available CPUs",
    )
    reconstruction.add_argument(
        "--progress",
        action="store_true",
        help="show reconstruction progress",
    )
    return parser


def _times(arguments: argparse.Namespace) -> list[float] | tuple[float, float] | None:
    if arguments.selected_times is not None:
        return arguments.selected_times
    if arguments.time_range is not None:
        return tuple(arguments.time_range)
    return None


def _print_reconstruction(result: Reconstruction) -> None:
    reconstructed = ", ".join(f"{value:g}" for value in result.reconstructed_times) or "none"
    skipped = ", ".join(f"{value:g}" for value in result.skipped_times) or "none"
    print(f"[StackNordic] Output: {result.output_dir}")
    print(f"[StackNordic] Reconstructed times: {reconstructed}")
    print(f"[StackNordic] Existing times: {skipped}")
    print(f"[StackNordic] Workers: {result.n_jobs}")
    print(f"[StackNordic] Elapsed time: {result.elapsed_seconds:.3f} s")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the StackNordic command-line interface."""

    selected = list(sys.argv[1:] if argv is None else argv)
    if selected and selected[0] == "-reconstructPar":
        selected[0] = "reconstruct"
    arguments = _parser().parse_args(selected)
    try:
        if arguments.command == "reconstruct":
            result = reconstruct(
                case_dir=arguments.case_dir,
                output_dir=arguments.output_dir,
                times=_times(arguments),
                fields=arguments.fields,
                n_jobs=arguments.n_jobs,
                progress=arguments.progress,
            )
            _print_reconstruction(result)
            return 0
    except (FileNotFoundError, NotADirectoryError, ValueError, RuntimeError) as error:
        print(f"[StackNordic] Error: {error}", file=sys.stderr)
        return 1
    raise RuntimeError(f"unsupported command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
