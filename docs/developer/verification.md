# Verification

Run the portable checks from the repository root:

```console
uv run pytest
uvx ruff check .
```

OpenFOAM integration and representative decomposed cases are validated through saved batch scripts on the target HPC system. Temporary cases, scheduler logs and generated datasets remain outside the repository unless they are intentional fixtures or benchmark evidence.
