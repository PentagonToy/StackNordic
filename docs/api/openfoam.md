# OpenFOAM cases and reconstruction

Inspect stored cases, configure the runtime and reconstruct selected physical times.

- [Case inspection and runtime](#case-inspection-and-runtime)
- [Reconstruction](#reconstruction)

## Case inspection and runtime

`sno.OpenFOAM.case()` inspects stored case data without changing the OpenFOAM case. It recognises reconstructed, `processorN`, and collated `processorsN` layouts.

```python
case = sno.OpenFOAM.case(
    case_dir=CASE_DIR,
    of_cmd="openfoam/2512",
    shell="bash",
)
```

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `case_dir` | `str \| Path` | Required | OpenFOAM case directory containing `system/controlDict` and `constant/`. |
| `of_cmd` | `str \| None` | `None` | Environment command evaluated before this case invokes OpenFOAM. A shorthand such as `openfoam/2512` expands to `module load openfoam/2512`. |
| `shell` | `str` | `"bash"` | Shell used to evaluate `of_cmd`. |

The returned case exposes `family`, `source_versions`, `reconstructed_times`, `decomposed_times`, `processor_count`, and `collated`. Direct exports prefer a complete reconstructed time and otherwise assemble the corresponding processor or collated data without running `reconstructPar`.

Use `case.compute()` to write derived fields into the existing reconstructed, `processorN`, or collated layout. The generated files are valid OpenFOAM volume fields and are immediately available to `case.export()`.

```python
sno.OpenFOAM.case(...).compute(
    *,
    computations,
    times=None,
    n_jobs=None,
    progress=False,
    overwrite=False,
)
```

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `computations` | `Sequence[ProgressComputation]` | Required | Derived scalar fields to materialise. |
| `times` | `list[float] \| tuple[float, float] \| None` | `None` | Exact physical times, an inclusive range, or every complete time. |
| `n_jobs` | `int \| None` | `None` | Concurrent time/field computations. `-1` uses the available CPU and work limits. |
| `progress` | `bool` | `False` | Show a Hedgehogs progress display. |
| `overwrite` | `bool` | `False` | Replace an existing output field after the replacement is complete. |

```python
computed = case.compute(
    computations=(progress,),
    times=None,
    n_jobs=-1,
    progress=True,
    overwrite=False,
)

dataset = case.export(
    output_dir=DATASET_DIR,
    fields={"H2": "YH2", "c": "c"},
    times=None,
    dtype="float32",
    mode="direct",
    metadata=None,
    n_jobs=-1,
    progress=True,
    overwrite=False,
)
```

Each output is committed atomically beside its source fields. Existing source fields are never rewritten. Explicit scalar boundary values are transformed with the same definition as the internal field, while the source patch structure and boundary types are retained. Computing a field does not rewrite a site-specific decomposition into a different layout.

See [Dataset export](dataset.md) for the complete `case.export()` parameter contract.

Use `case.postprocess()` for the closed set of supported OpenFOAM function objects. The case-local `of_cmd` and `shell` configuration is reused automatically. See [Post-processing](fields.md#post-processing) for the operation and parameter contracts.

Module names are site-specific. StackNordic accepts OpenFOAM Foundation and OpenFOAM.com runtimes. When the configured runtime family differs from an identifiable case family, inspection reports two short lines and keeps direct stored-data operations available.

## Reconstruction

Selected time zero is included when reconstructing into a new destination.

StackNordic reconstructs complete physical times from decomposed OpenFOAM cases produced by OpenFOAM Foundation or OpenFOAM.com.

### `case.reconstruct`

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

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `output_dir` | `str \| Path \| None` | `None` | Reconstructed-time destination. `None` writes completed times into `case_dir`. |
| `times` | `list[float] \| tuple[float, float] \| None` | `None` | Exact physical times, an inclusive `(start, end)` range, or every time present on all processor ranks. |
| `fields` | `Sequence[str] \| None` | `None` | Field names passed to `reconstructPar`; `None` reconstructs all available fields. |
| `n_jobs` | `int \| None` | `None` | Concurrent physical-time reconstructions. `None` uses one worker; `-1` uses up to one worker per available CPU and selected time. |
| `progress` | `bool` | `False` | Show a Hedgehogs progress display. |

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

Existing root times are preserved. Incomplete processor times are excluded; explicitly requesting one raises an error. Each worker publishes completed times from an isolated case view, then removes the view after success or failure.

For Foundation decompositions read by OpenFOAM.com, the temporary view supplies the missing ESI boundary addressing.

Time selection follows the [API conventions](README.md#reference-conventions), shared with dataset export.

### Command line

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
