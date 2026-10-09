<picture>
  <source media="(prefers-color-scheme: dark)" srcset="others/assets/banner_dark.svg">
  <img src="others/assets/banner_light.svg" alt="StackNordic" width="960">
</picture>

# StackNordic

StackNordic reconstructs and prepares stored OpenFOAM results for DNS, LES and machine-learning workflows. It reads OpenFOAM.com and OpenFOAM Foundation cases, including reconstructed, uncollated, collated and grouped-collated layouts.

[Documentation](https://pentagontoy.github.io/FoamNordic/stacknordic/) · [Installation](https://pentagontoy.github.io/FoamNordic/stacknordic/installation/) · [API](https://pentagontoy.github.io/FoamNordic/stacknordic/api/)

## Installation

Use Python 3.11 or 3.12. Install the PyPI release with pip:

```console
pip install stacknordic
```

Or use uv in an existing virtual environment:

```console
uv pip install stacknordic
```

Follow the public [installation guide](https://pentagontoy.github.io/FoamNordic/stacknordic/installation/) for optional environment setup, OpenFOAM requirements, installation checks and main or editable installation. The [repository copy](docs/installation.md) contains the same guide.

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

StackNordic isolates concurrent physical-time reconstructions, preserves the source case and removes temporary case views after success or failure. See the [reconstruction API and CLI contract](docs/api/openfoam.md#reconstruction).

Read the [case-to-dataset example](https://pentagontoy.github.io/FoamNordic/stacknordic/tutorials/case-to-dataset/) for copyable Python steps and the expected output structure.

## Core workflows

- [Export reconstructed or decomposed fields directly](docs/api/dataset.md), including time-local decomposition and mesh addressing.
- [Compute progress variables and reaction rates](docs/api/fields.md#field-computing) from declared Cantera contracts.
- [Package and compile Cartesian datasets](docs/api/dataset.md#snapshot-packaging) as self-describing HDF5 snapshots or time series.
- [Filter and downsample DNS data](docs/api/filtering.md) with declared Reynolds, Favre, variance and SGS reductions.
- [Run a closed set of post-processing operations](docs/api/fields.md#post-processing) on OpenFOAM cases or exported Cartesian datasets.

StackNordic is independent of [FoamNordic](https://github.com/PentagonToy/FoamNordic), but the projects are designed to be used together. FoamNordic owns solver execution and resident model coupling; StackNordic owns reconstruction, stored-data preparation, packaging and filtering.

Read the [shared documentation site](https://pentagontoy.github.io/FoamNordic/stacknordic/) or the [repository index](docs/README.md) for API, architecture and validation references.

## Development

```console
uv sync --dev
uv run pytest
uv run ruff check .
uv build
```
