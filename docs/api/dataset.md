# Dataset export

`case.export()` converts selected OpenFOAM volume fields into NumPy-compatible scalar arrays and an `info.json` manifest. Create the case with `sno.OpenFOAM.case()` so OpenFOAM-backed operations share one explicit runtime.

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

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `output_dir` | `str \| Path` | required | Dataset destination. A missing or empty directory is accepted. |
| `fields` | `Mapping[str, str \| Sequence[str]]` | required | OpenFOAM field names mapped to exported scalar component names. |
| `times` | `list[float] \| tuple[float, float] \| None` | `None` | Exact times, an inclusive range, or every complete time. |
| `dtype` | `"float32" \| "float64"` | `"float32"` | Stored floating-point type. |
| `mode` | `"direct" \| "reconstruct"` | `"direct"` | Read existing reconstructed or decomposed data directly, or reconstruct selected results in an isolated workspace. |
| `metadata` | `Mapping[str, object] \| None` | `None` | JSON-compatible user metadata. |
| `n_jobs` | `int \| None` | `None` | Concurrent export workers. `-1` uses the time-field work count, allocated CPUs and available-memory limit. |
| `progress` | `bool` | `False` | Show completed time-field tasks with an Onsaemiro progress display. |
| `overwrite` | `bool` | `False` | Replace a non-empty existing dataset only after the new export completes. |

`times=None` selects every complete time. A tuple such as `(0.00114, 0.00120)` selects the inclusive range. A list such as `[0.00114, 0.00120]` selects exactly those two values; a two-element list is not interpreted as a range.

`mode="direct"` prefers a complete reconstructed time and otherwise reads `processorN`, `processorsN` or `processorsN_a-b` data without creating reconstructed time directories. Decomposition and `cellProcAddressing` are resolved for every time, so a changing rank layout or time-local mesh is not treated as static.

For decomposed data, mesh addressing is indexed once per mesh generation and shared read-only through a memory map. Independent time-field tasks run in worker processes and write directly to staged arrays. Automatic worker selection respects the active Slurm or container memory limit before using the allocated CPUs.

If `rho` is requested but not stored, direct mode derives it from `p`, `T`, complete species mass fractions and molecular weights when the case declares a multicomponent `perfectGas` equation of state. Stored density remains authoritative, and unsupported thermodynamic contracts fail explicitly.

Each scalar requires one output name, a vector three names, a symmetric tensor six names and a tensor nine names. `gradU` and `gradp` may be generated automatically with OpenFOAM `postProcess` in `mode="reconstruct"`. Direct mode never starts an implicit full reconstruction; it reports the required derived-field action instead.

```python
from pathlib import Path
import stacknordic as sno

case = sno.OpenFOAM.case(
    case_dir=Path("cases/Premixed_H2"),
    of_cmd="openfoam/2512",
    shell="bash",
)

dataset = case.export(
    output_dir=Path("dataset/rawdata/output/phi045"),
    fields={
        "U": ("UX_ms-1", "UY_ms-1", "UZ_ms-1"),
        "p": "P_Pa",
        "T": "T_K",
    },
    times=[0.00114, 0.00120],
    dtype="float32",
    mode="direct",
    metadata=None,
    n_jobs=-1,
    progress=True,
    overwrite=False,
)
```

For a verified single-block Cartesian mesh, StackNordic records `Nxyz` and one-dimensional cell-centre coordinates. General, moving and topology-changing meshes remain cell-indexed; no fixed Cartesian geometry is invented. Export is staged beside `output_dir`, committed atomically and removed completely after failure.
