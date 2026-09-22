# Field computing

`sno.Computing` declares physical fields evaluated before filtering. `Dataset.compute()` supports every contract below. `case.compute()` currently supports a single-species progress variable because writing a multi-field boundary transformation requires additional OpenFOAM boundary semantics.

## `sno.Computing.progress`

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

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `mechanism` | `str \| Path` | required | Cantera mechanism. |
| `equivalence_ratio` | `float` | required | Positive equivalence ratio used for the reference states. |
| `fuel` | `Mapping[str, float]` | required | Fuel mole composition. |
| `oxidizer` | `Mapping[str, float]` | required | Oxidiser mole composition. |
| `species` | `Mapping[str, float]` | required | Species mass-fraction fields and their linear weights $\alpha_i$. |
| `temperature` | `float` | required | Unburnt reference temperature in kelvin. |
| `pressure` | `float` | required | Reference pressure in pascals. |
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

## `sno.Computing.reaction_rate`

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

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `mechanism` | `str \| Path` | required | Cantera mechanism. |
| `species` | `Mapping[str, str]` | required | Cantera species names mapped to stored mass-fraction fields. The fields must describe a complete state summing to one. |
| `temperature` | `str` | required | Stored temperature field in kelvin. |
| `target` | `str \| ProgressComputation` | required | One Cantera species or a progress-variable definition. |
| `output` | `str` | required | Stored reaction-rate field name. |
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
