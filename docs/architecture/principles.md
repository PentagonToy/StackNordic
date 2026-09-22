# Principles

StackNordic owns stored-result inspection, reconstruction orchestration, derived fields, dataset export and reproducible data reduction. OpenFOAM retains mesh and field semantics.

- Accept OpenFOAM Foundation and OpenFOAM.com stored results without rewriting the source case.
- Treat decomposition, topology and mesh geometry as time-dependent unless verified otherwise.
- Prefer direct decoding when OpenFOAM reconstruction is unnecessary.
- Isolate OpenFOAM operations and remove their temporary state after success or failure.
- Commit exported datasets atomically.
- Stream fields through bounded workspaces instead of materialising complete multi-snapshot datasets in memory.
- Require an explicit physical rule for every reduction; field names alone do not decide whether a quantity uses Reynolds, Favre, conservative or moment filtering.
- Validate finite values, density, species bounds, species sums and requested conservation properties before publishing an output.
- Keep public operations explicit and usable without FoamNordic.
- Bound automatic parallelism by work, CPU and memory rather than CPU count alone.
