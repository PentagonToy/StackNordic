# Dataset packaging

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

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset_dir` | `str \| Path` | required | Cartesian StackNordic dataset. |
| `output_dir` | `str \| Path` | required | Destination for packaged snapshots. |
| `name` | `str` | required | Filename stem; snapshot identifiers are appended automatically. |
| `format` | `str` | `"hdf5"` | Package format. Only `"hdf5"` is currently supported. |
| `fields` | `Mapping[str, str]` | required | Logical source fields mapped to relative HDF5 paths. |
| `compression` | `None \| "lzf" \| "gzip"` | `"lzf"` | HDF5 dataset compression. |
| `n_jobs` | `int \| None` | `None` | Concurrent snapshots. `-1` is bounded by snapshots, CPUs and estimated memory. |
| `progress` | `bool` | `False` | Show an Onsaemiro progress display. |
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

Scalar fields retain `(z, y, x)` order. Multi-component fields add a final component axis, so velocity is stored as `(z, y, x, 3)`. Grid axes are written as `grid/x_axis`, `grid/y_axis` and `grid/z_axis`. Physical time, source precision and dataset metadata are stored as HDF5 attributes. Each file also carries a StackNordic contract that maps logical field names to HDF5 paths and records component order, grid paths, shape and dtype.

Packaging is staged beside `output_dir` and published atomically. Incomplete files are removed after failure or interruption.

## `sno.Dataset.compile`

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

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset_dir` | `str \| Path` | required | Directory containing StackNordic snapshot packages. |
| `output_file` | `str \| Path` | required | Compiled HDF5 destination. |
| `pattern` | `str` | required | Glob selecting snapshot packages, for example `"*phi045_id*.h5"`. |
| `progress` | `bool` | `False` | Show an Onsaemiro progress display. |
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
