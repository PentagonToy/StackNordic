# Filtering and downsampling

`sno.Filtering` declares the physical reduction applied to each logical field. `Dataset.downsample()` applies that contract with a box or Gaussian spatial kernel.

## `sno.Filtering`

```python
sno.Filtering(
    density=None,
    mean=(),
    favre=(),
    variance={},
    stress={},
    energy={},
    specific={},
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `density` | `str \| None` | `None` | Density field used by Favre reductions. |
| `mean` | `Sequence[str]` | `()` | Fields reduced by an unweighted spatial mean. |
| `favre` | `Sequence[str]` | `()` | Fields reduced as $\widetilde{\phi}=\overline{\rho\phi}/\overline{\rho}$. |
| `variance` | `Mapping[str, str]` | `{}` | Scalar source fields mapped to filtered variance outputs. |
| `stress` | `Mapping[str, str]` | `{}` | Three-component velocity fields mapped to six-component SGS stress outputs. |
| `energy` | `Mapping[str, str]` | `{}` | SGS stress outputs mapped to SGS kinetic-energy outputs. |
| `specific` | `Mapping[str, str]` | `{}` | Filtered volumetric sources mapped to specific-source outputs obtained by division by filtered density. |

`Filtering.flow()` provides incompressible and compressible presets. `Filtering.combustion()` provides premixed and non-premixed presets.

Derived moments are optional. Omitting `sgs_stress`, `sgs_energy`, `progress_variance` or `reaction_rate` performs only the requested field reductions.

```python
filtering = sno.Filtering.flow(
    regime="compressible",
    density="rho",
    velocity="U",
    pressure="p",
    sgs_stress="tau_sgs",
    sgs_energy="k_sgs",
)
```

```python
filtering = sno.Filtering.combustion(
    regime="premixed",
    flow="compressible",
    density="rho",
    velocity="U",
    pressure="p",
    temperature="T",
    progress="c",
    progress_variance="c_var",
    species=("H2", "H", "O2", "OH", "O", "H2O", "HO2", "H2O2", "N2"),
    reaction_rate="omega_c_mass",
    specific_reaction_rate="omega_c",
)
```

`flow="compressible"` requires `density` and applies Favre filtering to transported fields. `flow="incompressible"` applies ordinary spatial means; `density` may be omitted.

`reaction_rate` is filtered as a volumetric source. When `specific_reaction_rate` is provided, StackNordic derives the specific source after filtering:

$$
\dot{\omega}_c = \frac{\overline{\dot{\omega}_{c,m}}}{\overline{\rho}}.
$$

## `Dataset.downsample`

```python
dataset.downsample(
    *,
    factor,
    filtering,
    output_dir=None,
    output_file=None,
    fields=None,
    kernel="box",
    sigma=None,
    boundary="reflect",
    dtype="float32",
    compression="lzf",
    n_jobs=None,
    progress=False,
    overwrite=False,
)
```

The same operation is available as `sno.Dataset.downsample(dataset_dir=..., ...)` when the input dataset already exists on disk. `dataset_dir` may identify a native StackNordic dataset, a directory of StackNordic HDF5 snapshot packages, or one compiled StackNordic HDF5 series.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset_dir` | `str \| Path` | required for class call | Native dataset directory, snapshot-package directory or compiled HDF5 series. |
| `factor` | `tuple[int, int, int]` | required | Output stride in public `(x, y, z)` order. |
| `filtering` | `sno.Filtering` | required | Physical field-reduction contract. |
| `output_dir` | `str \| Path \| None` | `None` | Native StackNordic dataset destination. Mutually exclusive with `output_file`. |
| `output_file` | `str \| Path \| None` | `None` | Self-describing compiled HDF5 destination. Mutually exclusive with `output_dir`. |
| `fields` | `Mapping[str, str] \| None` | `None` | Logical output fields mapped to HDF5 paths. Required with `output_file`. |
| `kernel` | `"box" \| "gaussian"` | `"box"` | Spatial filtering kernel. |
| `sigma` | `tuple[float, float, float] \| None` | `None` | Gaussian standard deviation in source-cell units and public `(x, y, z)` order. Required for `kernel="gaussian"`. |
| `boundary` | `"reflect" \| "nearest" \| "wrap"` | `"reflect"` | Gaussian boundary extension. |
| `dtype` | `"float32" \| "float64"` | `"float32"` | Stored output precision. |
| `compression` | `None \| "lzf" \| "gzip"` | `"lzf"` | HDF5 compression used with `output_file`. |
| `n_jobs` | `int \| None` | `None` | Concurrent snapshots. `-1` is bounded by work, CPUs and estimated memory. |
| `progress` | `bool` | `False` | Show an Onsaemiro progress display. |
| `overwrite` | `bool` | `False` | Replace an existing output only after the new dataset completes. |

```python
les = dataset.downsample(
    output_dir=LES_DIR,
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

Box filtering uses each non-overlapping block and trims only incomplete cells at the upper end of an axis. Gaussian filtering smooths the source field and samples at output-cell centres. The output manifest records the kernel, factor, Gaussian parameters and source dataset.

Passing `output_file` filters snapshots in parallel and commits each completed snapshot to one HDF5 series through a single writer. Temporary snapshots are bounded by the worker count, deleted after commit and removed after failure or interruption.

```python
les = sno.Dataset.downsample(
    dataset_dir=DNS_DIR / "phi045.h5",
    output_file=LES_DIR / "output/phi045.h5",
    factor=(8, 8, 8),
    filtering=filtering,
    fields={
        "c": "fields/c",
        "c_var": "fields/c_var",
        "T": "fields/T",
        "rho": "fields/rho",
        "U": "fields/U",
        "H2": "species/H2",
    },
    kernel="box",
    dtype="float32",
    compression="lzf",
    n_jobs=-1,
    progress=True,
    overwrite=False,
)
```
