# Post-processing

`sno.Postprocess` declares the supported field operations. A declaration can be executed by `case.postprocess()` through OpenFOAM or by `dataset.postprocess()` on an exported Cartesian dataset.

## Operations

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

## OpenFOAM case

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

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `operations` | `Sequence[PostprocessOperation]` | required | Operations created by `sno.Postprocess`. |
| `times` | `list[float] \| tuple[float, float] \| None` | `None` | Exact physical times, an inclusive range, or every reconstructed time. |
| `n_jobs` | `int \| None` | `None` | Concurrent time/operation jobs. `-1` uses the available CPU and work limits. |
| `progress` | `bool` | `False` | Show an Onsaemiro progress display. |
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

## Cartesian dataset

```python
dataset.postprocess(
    *,
    operations,
    n_jobs=None,
    progress=False,
    overwrite=False,
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `operations` | `Sequence[PostprocessOperation]` | required | Operations created by `sno.Postprocess`. |
| `n_jobs` | `int \| None` | `None` | Concurrent snapshot/operation jobs. `-1` uses the available CPU and work limits. |
| `progress` | `bool` | `False` | Show an Onsaemiro progress display. |
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
