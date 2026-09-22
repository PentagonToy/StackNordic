# Dataset workflow

StackNordic provides one route from stored OpenFOAM results to analysis-ready DNS and LES data:

```text
OpenFOAM case
    │
    ├── inspect distribution, decomposition, times and topology
    ├── decode directly or reconstruct in an isolated workspace
    ├── derive requested fields with OpenFOAM or a declared numerical provider
    ├── validate and export canonical snapshots
    ├── filter or downsample with explicit physical rules
    └── publish atomically with provenance and validation results
```

The dataset manifest records physical time, layout, component order, dtype, coordinates when verified, and provenance metadata.

## Reduction rules

| Quantity | Default declared reduction |
| --- | --- |
| Density and pressure | Arithmetic block mean; equivalent to a volume mean on a uniform Cartesian grid |
| Velocity, temperature, progress variable and species mass fraction | Favre mean when density is available |
| Volumetric source term such as `omega_c_mass` | Volume mean |
| Specific source term such as `omega_c` | Derive from the filtered volumetric source and filtered density |
| Progress-variable variance | Favre second moment minus the square of the Favre mean |
| SGS stress and kinetic energy | Favre velocity second moments |

Reduction rules are declared through `Filtering`, `Filtering.flow()`, or `Filtering.combustion()`; field names do not select a rule implicitly.

Box reduction trims incomplete blocks at the upper end of each axis. A filtered DNS grid can therefore differ from an independently generated LES grid even when both use the same nominal filter width. Compare physical times and grid axes before computing cell-wise errors. Interpolation aligns coordinates but does not reproduce a conservative volume filter.

## Execution

Parallel work is bounded by selected work, available CPUs, and estimated memory. Temporary data remains beside its destination and is removed after success or failure.
