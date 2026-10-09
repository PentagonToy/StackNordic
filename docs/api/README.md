# Python API reference

StackNordic inspects stored OpenFOAM cases, reconstructs selected times and prepares datasets. Examples use `import stacknordic as sno`. OpenFOAM retains mesh, field and boundary semantics.

## Public namespace

| Group | Main API | Use it for |
| --- | --- | --- |
| Stored cases | `OpenFOAM.case`, `case.reconstruct`, `case.export` | Inspect, reconstruct and export existing results |
| Dataset storage | `Dataset.package`, `Dataset.compile` | Package Cartesian snapshots and compile time series |
| Derived fields | `Computing`, `case.compute`, `dataset.compute` | Declare and evaluate thermochemical fields |
| Field operations | `Postprocess`, `case.postprocess`, `dataset.postprocess` | Apply supported spatial and temporal operations |
| Reduction | `Filtering`, `dataset.downsample`, `Dataset.downsample` | Declare physical reductions and produce coarser datasets |

## Reference map

| Task | Reference |
| --- | --- |
| Inspect cases, configure a runtime or reconstruct times | [OpenFOAM cases and reconstruction](openfoam.md) |
| Export arrays, package snapshots or compile HDF5 | [Dataset export and storage](dataset.md) |
| Compute progress variables, reaction rates or field operations | [Field computation and post-processing](fields.md) |
| Filter and downsample with declared physical rules | [Filtering and downsampling](filtering.md) |

## Reference conventions

Signatures mark keyword-only arguments with `*`. Paths accept strings or `Path` objects. `times=None` selects all complete times, a tuple selects an inclusive range and a list selects exact values. Worker limits and overwrite behaviour are documented per operation.

Export, storage and reduction operations return result summaries. A reconstruction also returns its source identity, selected times and execution details. Data-changing operations require explicit fields or declarations; names alone do not choose physical reduction rules. See [Stored-data layout](../architecture/data-layout.md) for geometry constraints and [Dataset workflow](../architecture/workflow.md) for provenance and publication.
