# Dataset export benchmark

This record measures direct decoding of one OpenFOAM Foundation collated decomposition containing 768 ranks and 25,750,872 cells. Each result is a single measured run and includes Python start-up, stored-data inspection, global-order restoration, type conversion and output writes.

| Field | Components | Reader | Wall time | Peak RSS |
| --- | ---: | --- | ---: | ---: |
| `T` | 1 | Buffered collated file | 26.53 s | 2.54 GiB |
| `T` | 1 | Streaming blocks and memory-mapped output | 27.18 s | 1.92 GiB |
| `U` | 3 | Buffered collated file | 45.72 s | 4.24 GiB |
| `U` | 3 | Streaming blocks and memory-mapped output | 46.25 s | 1.93 GiB |

Streaming preserved wall time within approximately 2% while reducing peak RSS by approximately 24% for `T` and 55% for `U`. The detected Cartesian shape was `811 × 252 × 126`. A corresponding OpenFOAM reconstruction with 16 GiB was terminated by the scheduler for exceeding memory after 2 min 32 s. This comparison establishes the practical value of direct export; it is not a controlled throughput comparison between equivalent implementations.

## Filtering and downsampling

One `25,750,872`-cell snapshot from the same dataset was reduced by a factor of `8 × 8 × 8` on a Roihu interactive CPU node using one worker. Both runs produced `101 × 31 × 15` cells with finite Favre velocity, density, SGS stress and SGS kinetic energy.

| Kernel | Wall time | Peak RSS |
| --- | ---: | ---: |
| Box | 1.41 s | 1.85 GiB |
| Gaussian, $\sigma=(4,4,4)$ source cells | 8.50 s | 2.04 GiB |

Peak RSS is the process high-water mark after direct export and the reported reduction, not the isolated kernel allocation.
