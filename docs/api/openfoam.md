# OpenFOAM cases and runtime

`sno.OpenFOAM.case()` inspects stored case data without changing the OpenFOAM case. It recognises reconstructed, `processorN`, and collated `processorsN` layouts.

```python
case = sno.OpenFOAM.case(
    case_dir=CASE_DIR,
    of_cmd="openfoam/2512",
    shell="bash",
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `case_dir` | `str \| Path` | required | OpenFOAM case directory containing `system/controlDict` and `constant/`. |
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

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `computations` | `Sequence[ProgressComputation]` | required | Derived scalar fields to materialise. |
| `times` | `list[float] \| tuple[float, float] \| None` | `None` | Exact physical times, an inclusive range, or every complete time. |
| `n_jobs` | `int \| None` | `None` | Concurrent time/field computations. `-1` uses the available CPU and work limits. |
| `progress` | `bool` | `False` | Show an Onsaemiro progress display. |
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

```python
case.export(
    *,
    output_dir,
    fields,
    times=None,
    dtype="float32",
    mode="direct",
    metadata=None,
    n_jobs=None,
    progress=False,
    overwrite=False,
)
```

See [Dataset export](dataset.md) for the complete `case.export()` parameter contract.

Use `case.postprocess()` for the closed set of supported OpenFOAM function objects. The case-local `of_cmd` and `shell` configuration is reused automatically. See [Post-processing](postprocess.md) for the operation and parameter contracts.

Module names are site-specific. StackNordic accepts OpenFOAM Foundation and OpenFOAM.com runtimes. When the configured runtime family differs from an identifiable case family, inspection reports two short lines and keeps direct stored-data operations available.
