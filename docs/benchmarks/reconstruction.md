# Reconstruction benchmark

This record compares direct OpenFOAM reconstruction with StackNordic orchestration. It measures reconstruction overhead, numerical parity and cross-distribution compatibility; it does not measure solver performance.

## Reporting

Repeated measurements report the median and maximum absolute deviation from the median. End-to-end wall time includes process start-up and shutdown. Reconstruction wall time covers the operation reported by StackNordic after package import.

## Validation matrix

| Case | Source | Runtime | Processor directories | Selection | Runs |
| --- | --- | --- | ---: | --- | ---: |
| Taylor–Green vortex LES | OpenFOAM.com v2512 | OpenFOAM.com v2512 | 32 | `U` at `0.011574954` | 1 warm-up + 7 measured |
| Premixed hydrogen LES | OpenFOAM Foundation 12/14 | OpenFOAM.com v2512 | 384 | `U` at `0.00114` | 1 |
| Premixed hydrogen LES | OpenFOAM Foundation 12/14 | OpenFOAM Foundation 14 | 384 | `U` at `0.00114` | 1 |

## Direct reconstruction

The Taylor–Green vortex comparison alternated direct `reconstructPar` and `case.reconstruct()` after one warm-up per path.

| Path | End-to-end wall | Reconstruction wall | Peak process-tree RSS |
| --- | ---: | ---: | ---: |
| Direct `reconstructPar` | 3.344 ± 0.067 s | 3.344 ± 0.067 s | 1901.0 ± 40.6 MiB |
| `case.reconstruct()` | 5.932 ± 0.091 s | 4.235 ± 0.037 s | 1929.6 ± 52.2 MiB |

The StackNordic reconstruction ratio was `1.267`; the end-to-end ratio was `1.774` when Python start-up, package import, case inspection, runtime preflight, compatibility staging and cleanup were included. All fourteen measured outputs had the same SHA-256 digest and were therefore byte-identical.

## Command-line validation

The installed `stacknordic reconstruct` command was validated on Roihu with OpenFOAM.com v2512. The 32-directory Taylor–Green vortex case reconstructed `U` at `0.011574954`. Direct and StackNordic runs were alternated after one warm-up per path.

| Path | End-to-end wall | Runs | Output parity |
| --- | ---: | ---: | --- |
| Direct `reconstructPar` | 2.178 ± 0.020 s | 7 | SHA-256 identical |
| `stacknordic reconstruct` | 4.062 ± 0.035 s | 7 | SHA-256 identical |

The CLI ratio was `1.864`. Its end-to-end time includes Python start-up, package import, runtime preflight, isolated case staging and cleanup. Roihu job `1569534` performed the measurement on one interactive CPU node; the raw CSV and summary are retained with the project validation artefacts.

## Cross-distribution reconstruction

| Runtime | Reconstruction wall | Slurm wall | Peak RSS | Compatibility action |
| --- | ---: | ---: | ---: | --- |
| OpenFOAM.com v2512 | 18.42 s | 23 s | 243 MiB | Generated ESI `boundaryProcAddressing` in the isolated case view |
| OpenFOAM Foundation 14 | 16.01 s | 19 s | 278 MiB | Used the native Foundation decomposition |

Both 384-rank runs completed successfully, preserved the source case and removed their isolated case views.
