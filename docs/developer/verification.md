# Verification

Choose checks according to the changed behaviour. Keep generated cases, datasets and scheduler logs outside the source tree.

| Change | Check |
| --- | --- |
| Documentation | Read the diff; check links, signatures, units and rendering |
| Stored-case inspection or decoding | Test layouts, time-local addressing, component order and source preservation |
| Reconstruction | Compare selected outputs with the native OpenFOAM tool; check failure cleanup |
| Packaging or compilation | Check field contracts, grid metadata, snapshot identity and atomic publication |
| Computing or reduction | Check declared physical definitions, finite values, units and requested conservation properties |
| Worker selection | Check work, CPU and memory bounds and failure propagation |

## Portable checks

Run from the repository root in the development environment:

```console
uv run pytest
uvx ruff check .
```

Documentation-only changes require link and rendering checks, without solver execution or runtime tests. Run relevant regression cases during functional work and the full portable suite after a broad change.

## OpenFOAM and HPC

Use saved batch scripts for representative decomposed cases on the target system. Record the source commit, OpenFOAM family/version, case, decomposition, field/time selection and allocation. Preserve the original case and retain numerical parity and resource evidence with the [validation records](../benchmarks/README.md).
