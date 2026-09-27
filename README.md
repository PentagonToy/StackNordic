# StackNordic

StackNordic reconstructs and prepares stored OpenFOAM results for DNS, LES and machine-learning workflows. It reads OpenFOAM.com and OpenFOAM Foundation cases, including reconstructed, uncollated, collated and grouped-collated layouts.

## Installation

Install the released package:

```console
uv pip install stacknordic
```

Install the current repository version when working ahead of a release:

```console
uv pip install "git+https://github.com/PentagonToy/StackNordic.git"
```

Install a local checkout for development:

```console
uv pip install --editable .
```

## Reconstruct a case

Activate the Python environment and the OpenFOAM runtime, enter a decomposed case, and run:

```console
stacknordic reconstruct --n-jobs -1 --progress
```

The OpenFOAM-style alias is equivalent:

```console
stacknordic -reconstructPar --n-jobs -1 --progress
```

Select times, fields and another destination explicitly when required:

```console
stacknordic reconstruct \
    --case-dir cases/TaylorGreenVortex3D \
    --output-dir output/TaylorGreenVortex3D \
    --time-range 0 0.1 \
    --fields U p nut \
    --n-jobs 8 \
    --progress
```

StackNordic isolates concurrent physical-time reconstructions, preserves the source case and removes temporary case views after success or failure. See the [reconstruction API and CLI contract](docs/api/reconstruction.md).

## Core workflows

- [Export reconstructed or decomposed fields directly](docs/api/dataset.md), including time-local decomposition and mesh addressing.
- [Compute progress variables and reaction rates](docs/api/computing.md) from declared Cantera contracts.
- [Package and compile Cartesian datasets](docs/api/packaging.md) as self-describing HDF5 snapshots or time series.
- [Filter and downsample DNS data](docs/api/filtering.md) with declared Reynolds, Favre, variance and SGS reductions.
- [Run a closed set of post-processing operations](docs/api/postprocess.md) on OpenFOAM cases or exported Cartesian datasets.

StackNordic is independent of [FoamNordic](https://github.com/PentagonToy/FoamNordic), but the projects are designed to be used together. FoamNordic owns solver execution and resident model coupling; StackNordic owns reconstruction, stored-data preparation, packaging and filtering.

See the [documentation index](docs/README.md) for the complete API, architecture and validation records.

## Development

```console
uv sync --dev
uv run pytest
uv run ruff check .
uv build
```
