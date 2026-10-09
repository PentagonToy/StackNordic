# Field computation and post-processing

Declare combustion fields and apply supported spatial or temporal operations to cases and Cartesian datasets.

- [Field computing](#field-computing)
- [Post-processing](#post-processing)

## Field computing

`sno.Computing` declares physical fields evaluated before filtering. `Dataset.compute()` supports every contract below. `case.compute()` currently supports a single-species progress variable because writing a multi-field boundary transformation requires additional OpenFOAM boundary semantics.

### `sno.Computing.progress`

```python
sno.Computing.progress(
    *,
    mechanism,
    equivalence_ratio,
    fuel,
    oxidizer,
    species,
    temperature,
    pressure,
    output="c",
)
```

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `mechanism` | `str \| Path` | Required | Cantera mechanism. |
| `equivalence_ratio` | `float` | Required | Positive equivalence ratio used for the reference states. |
| `fuel` | `Mapping[str, float]` | Required | Fuel mole composition. |
| `oxidizer` | `Mapping[str, float]` | Required | Oxidiser mole composition. |
| `species` | `Mapping[str, float]` | Required | Species mass-fraction fields and their linear weights $\alpha_i$. |
| `temperature` | `float` | Required | Unburnt reference temperature in kelvin. |
| `pressure` | `float` | Required | Reference pressure in pascals. |
| `output` | `str` | `"c"` | Stored logical field name. |

The unnormalised progress variable and its normalised form are

$$
Y_c = \sum_i \alpha_i Y_i,
$$

$$
c = \frac{Y_c-Y_{c,u}}{Y_{c,b}-Y_{c,u}}.
$$

The unburnt state is the declared mixture. The burnt state is its constant-enthalpy, constant-pressure Cantera equilibrium state. Negative weights may represent consumed reactants; positive weights may represent products.

```python
progress = sno.Computing.progress(
    mechanism=MECHANISM,
    equivalence_ratio=0.45,
    fuel={"H2": 1.0},
    oxidizer={"O2": 0.21, "N2": 0.79},
    species={"H2": -1.0, "H2O": 1.0},
    temperature=300.0,
    pressure=101325.0,
    output="c",
)
```

### `sno.Computing.reaction_rate`

```python
sno.Computing.reaction_rate(
    *,
    mechanism,
    species,
    temperature,
    target,
    output,
    pressure=None,
    density=None,
    units="kg/m^3/s",
)
```

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `mechanism` | `str \| Path` | Required | Cantera mechanism. |
| `species` | `Mapping[str, str]` | Required | Cantera species names mapped to stored mass-fraction fields. The fields must describe a complete state summing to one. |
| `temperature` | `str` | Required | Stored temperature field in kelvin. |
| `target` | `str \| ProgressComputation` | Required | One Cantera species or a progress-variable definition. |
| `output` | `str` | Required | Stored reaction-rate field name. |
| `pressure` | `str \| None` | `None` | Stored pressure field in pascals. Exactly one of `pressure` and `density` is required. |
| `density` | `str \| None` | `None` | Stored density field in kilograms per cubic metre. Exactly one of `pressure` and `density` is required. |
| `units` | `str` | `"kg/m^3/s"` | Output basis: `"kg/m^3/s"` or `"1/s"`. |

For species $i$, Cantera supplies the molar production rate $\dot{\omega}_i$ and StackNordic computes

$$
\dot{\omega}_{i,m} = W_i\dot{\omega}_i \quad [\mathrm{kg\,m^{-3}\,s^{-1}}].
$$

For the normalised progress variable, the coefficients are inherited from `ProgressComputation`:

$$
\dot{\omega}_{c,m} = \sum_i \frac{\alpha_i}{Y_{c,b}-Y_{c,u}}\dot{\omega}_{i,m}.
$$

Selecting `units="1/s"` returns the corresponding specific source $\dot{\omega}/\rho$. StackNordic records the selected basis in dataset metadata and does not infer it from the output name.

Compatible reaction-rate declarations are fused automatically. Definitions sharing the mechanism, species fields, temperature field and pressure or density field use one Cantera state evaluation per bounded cell chunk; requesting species and progress-variable rates therefore does not repeat kinetics evaluation.

With `progress=True`, the display starts before computation and advances by completed cell chunks across all snapshots. Long Cantera evaluations therefore remain visible before the first snapshot completes.

```python
state = {name: name for name in ("H2", "H", "O2", "OH", "O", "H2O", "HO2", "H2O2", "N2")}

omega_h2 = sno.Computing.reaction_rate(
    mechanism=MECHANISM,
    species=state,
    temperature="T",
    pressure="p",
    target="H2",
    output="omega_H2_mass",
    units="kg/m^3/s",
)

omega_c = sno.Computing.reaction_rate(
    mechanism=MECHANISM,
    species=state,
    temperature="T",
    density="rho",
    target=progress,
    output="omega_c",
    units="1/s",
)

computed = dataset.compute(
    computations=(progress, omega_h2, omega_c),
    dtype="float32",
    n_jobs=-1,
    progress=True,
    overwrite=False,
)
```

## Post-processing

`sno.Postprocess` declares the supported field operations. A declaration can be executed by `case.postprocess()` through OpenFOAM or by `dataset.postprocess()` on an exported Cartesian dataset.

### Operations

| Factory | Default output | OpenFOAM case | Cartesian dataset | Definition |
| --- | --- | --- | --- | --- |
| `grad(field=..., output=None)` | `grad<field>` | yes | yes | Spatial gradient of a scalar or vector field. |
| `div(field=..., output=None)` | `div<field>` | yes | yes | Divergence of a vector field. |
| `ddt(field=..., output=None)` | `ddt<field>` | yes | yes | OpenFOAM temporal derivative or finite difference between stored dataset snapshots. |
| `components(field=..., outputs=None)` | `<field>x`, `<field>y`, `<field>z` | yes | yes | Scalar components of a three-component vector. |
| `mag(field=..., output=None)` | `mag<field>` | yes | yes | Scalar absolute value or vector magnitude. |
| `mag_sqr(field=..., output=None)` | `magSqr<field>` | yes | yes | Squared scalar or vector magnitude. |
| `vorticity(field="U", output="vorticity")` | `vorticity` | yes | yes | Curl of a three-component velocity field. |
| `Q(field="U", output="Q")` | `Q` | yes | yes | Second invariant of the velocity-gradient tensor. |
| `Lambda2(field="U", output="Lambda2")` | `Lambda2` | yes | yes | Intermediate eigenvalue of $S^2 + \Omega^2$. |
| `write_cell_centres(output="C")` | `C` | yes | yes | Cell-centre coordinates. |
| `write_cell_volumes(output="V")` | `V` | yes | yes | Cell volumes. |

The OpenFOAM implementations of `vorticity`, `Q`, and `Lambda2` currently require the conventional velocity field `U`. Dataset-native implementations accept another three-component field explicitly. `components()` accepts custom output names and requires one name per component.

### OpenFOAM case

```python
case.postprocess(
    *,
    operations,
    times=None,
    n_jobs=None,
    progress=False,
    overwrite=False,
)
```

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `operations` | `Sequence[PostprocessOperation]` | Required | Operations created by `sno.Postprocess`. |
| `times` | `list[float] \| tuple[float, float] \| None` | `None` | Exact physical times, an inclusive range, or every reconstructed time. |
| `n_jobs` | `int \| None` | `None` | Concurrent time/operation jobs. `-1` uses the available CPU and work limits. |
| `progress` | `bool` | `False` | Show a Hedgehogs progress display. |
| `overwrite` | `bool` | `False` | Replace existing generated fields. |

```python
result = case.postprocess(
    operations=(
        sno.Postprocess.grad(field="U"),
        sno.Postprocess.vorticity(),
        sno.Postprocess.Q(),
    ),
    times=(0.0, 0.1),
    n_jobs=4,
    progress=True,
    overwrite=False,
)
```

This path requires reconstructed time directories and invokes only the listed OpenFOAM function objects through the case-local runtime. Direct decomposed export remains available independently.

### Cartesian dataset

```python
dataset.postprocess(
    *,
    operations,
    n_jobs=None,
    progress=False,
    overwrite=False,
)
```

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `operations` | `Sequence[PostprocessOperation]` | Required | Operations created by `sno.Postprocess`. |
| `n_jobs` | `int \| None` | `None` | Concurrent snapshot/operation jobs. `-1` uses the available CPU and work limits. |
| `progress` | `bool` | `False` | Show a Hedgehogs progress display. |
| `overwrite` | `bool` | `False` | Replace existing logical fields without overwriting components owned by another field. |

```python
result = dataset.postprocess(
    operations=(
        sno.Postprocess.grad(field="U"),
        sno.Postprocess.mag(field="U"),
        sno.Postprocess.write_cell_volumes(),
    ),
    n_jobs=-1,
    progress=True,
    overwrite=False,
)
```

Dataset-native post-processing is intentionally limited to single-block Cartesian DNS or LES data with `Nxyz` and one-dimensional cell-centre coordinates. It rejects cell-indexed, moving, curvilinear, unstructured, and topology-changing datasets rather than inferring invalid geometry.
