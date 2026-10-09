# Dataset export and storage

Export case fields into arrays, package snapshots and compile HDF5 time series.

- [Dataset export](#dataset-export)
- [Snapshot packaging](#snapshot-packaging)
- [Time-series compilation](#snodatasetcompile)

## Dataset export

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

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `output_dir` | `str \| Path` | Required | Dataset destination. A missing or empty directory is accepted. |
| `fields` | `Mapping[str, str \| Sequence[str]]` | Required | OpenFOAM field names mapped to exported scalar component names. |
| `times` | `list[float] \| tuple[float, float] \| None` | `None` | Exact times, an inclusive range, or every complete time. |
| `dtype` | `"float32" \| "float64"` | `"float32"` | Stored floating-point type. |
| `mode` | `"direct" \| "reconstruct"` | `"direct"` | Read existing reconstructed or decomposed data directly, or reconstruct selected results in an isolated workspace. |
| `metadata` | `Mapping[str, object] \| None` | `None` | JSON-compatible user metadata. |
| `n_jobs` | `int \| None` | `None` | Concurrent export workers. `-1` uses the time-field work count, allocated CPUs and available-memory limit. |
| `progress` | `bool` | `False` | Show completed time-field tasks with a Hedgehogs progress display. |
| `overwrite` | `bool` | `False` | Replace a non-empty existing dataset only after the new export completes. |

`times=None` selects every complete time. A tuple such as `(0.00114, 0.00120)` selects the inclusive range. A list such as `[0.00114, 0.00120]` selects exactly those two values; a two-element list is not interpreted as a range.

Uniform reconstructed fields obtain their cell count from the selected-time mesh.

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

## Snapshot packaging

`sno.Dataset.package()` writes each Cartesian snapshot as one portable HDF5 file. Source arrays remain memory-mapped and are copied in bounded slabs.

```python
sno.Dataset.package(
    *,
    dataset_dir,
    output_dir,
    name,
    format="hdf5",
    fields,
    compression="lzf",
    n_jobs=None,
    progress=False,
    overwrite=False,
)
```

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `dataset_dir` | `str \| Path` | Required | Cartesian StackNordic dataset. |
| `output_dir` | `str \| Path` | Required | Destination for packaged snapshots. |
| `name` | `str` | Required | Filename stem; snapshot identifiers are appended automatically. |
| `format` | `str` | `"hdf5"` | Package format. Only `"hdf5"` is currently supported. |
| `fields` | `Mapping[str, str]` | Required | Logical source fields mapped to relative HDF5 paths. |
| `compression` | `None \| "lzf" \| "gzip"` | `"lzf"` | HDF5 dataset compression. |
| `n_jobs` | `int \| None` | `None` | Concurrent snapshots. `-1` is bounded by snapshots, CPUs and estimated memory. |
| `progress` | `bool` | `False` | Show a Hedgehogs progress display. |
| `overwrite` | `bool` | `False` | Replace an existing package only after the new package completes. |

Each `/`-separated output segment is stripped of surrounding whitespace. `"species_fields/H2O2"` and `"species_fields / H2O2"` therefore select the same dataset path. Empty segments, parent traversal and dataset/group collisions are rejected.

```python
packaged = sno.Dataset.package(
    dataset_dir=RAWDATA_DIR / "output",
    output_dir=DNS_DIR / "output",
    name="phi045",
    format="hdf5",
    fields={
        "c": "c",
        "T": "T_K",
        "p": "P_Pa",
        "rho": "RHO",
        "U": "U",
        "omega_c_mass": "omega_c_mass",
        "omega_c_chem": "omega_c_chem",
        "H2": "species_fields/H2",
        "O2": "species_fields/O2",
        "H2O": "species_fields/H2O",
    },
    compression="lzf",
    n_jobs=-1,
    progress=True,
    overwrite=False,
)
```

Scalars use `(z, y, x)` order; multi-component fields add a final component axis, giving `(z, y, x, 3)` for velocity. Grid axes use `grid/x_axis`, `grid/y_axis` and `grid/z_axis`.

HDF5 attributes retain physical time, source precision and dataset metadata. The embedded contract records logical field paths, component order, grid paths, shape and dtype.

Packaging is staged beside `output_dir` and published atomically. Incomplete files are removed after failure or interruption.

### `sno.Dataset.compile`

`sno.Dataset.compile()` combines compatible snapshot packages into one self-describing HDF5 time series. It validates the embedded contracts before writing and does not infer fields from arbitrary HDF5 layouts.

```python
sno.Dataset.compile(
    *,
    dataset_dir,
    output_file,
    pattern,
    progress=False,
    overwrite=False,
)
```

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `dataset_dir` | `str \| Path` | Required | Directory containing StackNordic snapshot packages. |
| `output_file` | `str \| Path` | Required | Compiled HDF5 destination. |
| `pattern` | `str` | Required | Glob selecting snapshot packages, for example `"*phi045_id*.h5"`. |
| `progress` | `bool` | `False` | Show a Hedgehogs progress display. |
| `overwrite` | `bool` | `False` | Replace an existing file only after the new series completes. |

```python
compiled = sno.Dataset.compile(
    dataset_dir=DNS_DIR / "output",
    output_file=DNS_DIR / "phi045.h5",
    pattern="*phi045_id*.h5",
    progress=True,
    overwrite=False,
)
```

The compiled file stores the Cartesian grid once and places each snapshot under an `id_NNN` group. Snapshot identifiers and physical times remain available in the series metadata. Publication is atomic and incomplete temporary files are removed.
