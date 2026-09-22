# StackNordic

StackNordic prepares stored OpenFOAM results for DNS, LES, and machine-learning workflows. It supports reconstructed, uncollated, collated, and grouped-collated results from OpenFOAM.com and OpenFOAM Foundation.

## Installation

```console
uv pip install "git+https://github.com/PentagonToy/StackNordic.git"
```

From a local checkout:

```console
uv pip install .
```

## Workflow

```python
from pathlib import Path
import stacknordic as sno

case = sno.OpenFOAM.case(
    case_dir=Path("cases/compressibleChannel"),
    of_cmd="openfoam/2512",
    shell="bash",
)

dataset = case.export(
    output_dir=Path("dataset/rawdata"),
    fields={
        "U": ("Ux", "Uy", "Uz"),
        "p": "p",
        "T": "T",
        "rho": "rho",
    },
    times=None,
    dtype="float32",
    mode="direct",
    metadata=None,
    n_jobs=-1,
    progress=True,
    overwrite=False,
)
```

```python
filtering = sno.Filtering.flow(
    regime="compressible",
    density="rho",
    velocity="U",
    pressure="p",
    sgs_stress="tau_sgs",
    sgs_energy="k_sgs",
)

les = dataset.downsample(
    output_dir=Path("dataset/les"),
    factor=(8, 8, 8),
    filtering=filtering,
    kernel="box",
    sigma=None,
    boundary="reflect",
    dtype="float32",
    n_jobs=-1,
    progress=True,
    overwrite=False,
)
```

See the [documentation index](docs/README.md) for reconstruction, field computation, post-processing, export, filtering, and benchmarks.

## Development

```console
uv pip install --editable .
uv run pytest
uvx ruff check .
```
