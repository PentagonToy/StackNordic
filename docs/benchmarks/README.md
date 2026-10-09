# Benchmarks and validation

These records describe numerical parity and performance under the stated case, runtime and resource conditions. Repeated measurements report the median and maximum absolute deviation; single runs are labelled separately. Results are engineering evidence for those conditions rather than general performance guarantees.

| Record | Scope |
| --- | --- |
| [Reconstruction](reconstruction.md) | Native OpenFOAM comparison, command-line overhead and cross-distribution compatibility |
| [Dataset export and reduction](dataset.md) | Direct collated decoding, memory use and filtering kernels |

Keep case size, field selection, runtime, worker count and timing boundaries attached to each comparison. Preserve source cases and raw measurements outside the source tree. See [Verification](../developer/verification.md) for checks after implementation changes.
