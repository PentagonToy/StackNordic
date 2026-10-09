<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../others/assets/banner_dark.svg">
  <img src="../others/assets/banner_light.svg" alt="StackNordic" width="960">
</picture>

# StackNordic documentation

StackNordic reconstructs stored OpenFOAM results and prepares datasets for DNS, LES and machine learning. It operates independently of FoamNordic, which owns solver execution and resident model coupling.

| Area | Use it for |
| --- | --- |
| [Installation](installation.md) | Package setup, environment checks and OpenFOAM requirements |
| [Worked example](tutorials/case-to-dataset.md) | Inspect a case, export U/p and package Cartesian data |
| [Python API](api/README.md) | Inspect cases, reconstruct times, export arrays and reduce datasets |
| [Architecture](architecture/principles.md) | Ownership, stored-data layout and physical reduction rules |
| [Developer guide](developer/verification.md) | Select checks for source changes |
| [Validation](benchmarks/README.md) | Numerical parity and measured performance |

## Reading paths

| Goal | Start here | Continue with |
| --- | --- | --- |
| Inspect or reconstruct a case | [OpenFOAM cases](api/openfoam.md) | [Stored-data layout](architecture/data-layout.md) |
| Export fields and build a time series | [Dataset export and storage](api/dataset.md) | [Dataset workflow](architecture/workflow.md) |
| Compute combustion fields | [Field computation](api/fields.md#field-computing) | [Filtering](api/filtering.md) |
| Filter DNS results for LES analysis | [Filtering](api/filtering.md) | [Reduction rules](architecture/workflow.md#reduction-rules) |
| Change or validate an implementation | [Verification](developer/verification.md) | [Validation records](benchmarks/README.md) |

The [shared documentation site](https://pentagontoy.github.io/FoamNordic/stacknordic/) provides navigation and search. Return to the [project README](../README.md) for a reconstruction example.
