# Reconstruction

StackNordic reconstructs complete physical times from decomposed OpenFOAM cases produced by OpenFOAM Foundation or OpenFOAM.com.

## `case.reconstruct`

```python
case.reconstruct(
    *,
    output_dir=None,
    times=None,
    fields=None,
    n_jobs=None,
    progress=False,
)
```

Reconstructs selected fields and physical times from consecutive `processor0` to `processorN` directories with the runtime configured by `sno.OpenFOAM.case()`.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `output_dir` | `str \| Path \| None` | `None` | Reconstructed-time destination. `None` writes completed times into `case_dir`. |
| `times` | `list[float] \| tuple[float, float] \| None` | `None` | Exact physical times, an inclusive `(start, end)` range, or every time present on all processor ranks. |
| `fields` | `Sequence[str] \| None` | `None` | Field names passed to `reconstructPar`; `None` reconstructs all available fields. |
| `n_jobs` | `int \| None` | `None` | Concurrent physical-time reconstructions. `None` uses one worker; `-1` uses up to one worker per available CPU and selected time. |
| `progress` | `bool` | `False` | Show an Onsaemiro progress display. |

```python
from pathlib import Path
import stacknordic as sno

case = sno.OpenFOAM.case(
    case_dir=Path("cases/TaylorGreenVortex3D"),
    of_cmd="openfoam/2512",
    shell="bash",
)

result = case.reconstruct(
    output_dir=Path("output/TaylorGreenVortex3D"),
    times=(0.0, 0.1),
    fields=["U", "p", "nut"],
    n_jobs=4,
    progress=True,
)
```

The return value is an immutable `Reconstruction` summary containing the source case, output directory, processor count, reconstructed and skipped times, fields, detected source identity, worker count, and elapsed time.

Each worker runs one `reconstructPar -time ...` process across every processor directory. `n_jobs` controls concurrent times, not the number of OpenFOAM ranks.

Existing root time directories are preserved. Incomplete processor times are excluded, and an explicitly requested incomplete time raises an error. Each worker uses an isolated runtime-compatible case view and moves only a completed time into the output directory. When a Foundation decomposition is read by an OpenFOAM.com runtime, StackNordic supplies the missing ESI boundary addressing inside that temporary view. Temporary views are removed after success or failure.

`times=None` selects every complete time. A tuple selects an inclusive range, while a list selects exactly its members. This distinction applies consistently to reconstruction and dataset export.

## Command line

Activate the Python environment and OpenFOAM runtime before running the command. The current directory is used as the case by default.

```console
stacknordic reconstruct --n-jobs -1 --progress
```

Select exact times and fields:

```console
stacknordic reconstruct \
    --case-dir cases/TaylorGreenVortex3D \
    --output-dir output/TaylorGreenVortex3D \
    --time 0.05 \
    --time 0.1 \
    --fields U p nut \
    --n-jobs 4 \
    --progress
```

Use `--time-range START END` for an inclusive range. The OpenFOAM-style alias `stacknordic -reconstructPar` accepts the same options.
