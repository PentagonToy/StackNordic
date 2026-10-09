# Installation

Install StackNordic from PyPI, check the Python environment, then choose a stored-data or OpenFOAM workflow. StackNordic works independently of FoamNordic and requires no FoamNordic integration build.

## Before starting

Use Python 3.11 or 3.12 on Linux or macOS. The package installs NumPy, SciPy, h5py, Cantera and Hedgehogs as dependencies. The commands below use a bash or zsh terminal.

StackNordic is under active development. The site records its [source revision](https://pentagontoy.github.io/FoamNordic/versions/); a PyPI release can differ from the current repository API. Use a matching release or pinned source revision for reproducible work.

## 1. Install the Python package

Choose pip or uv in the Python environment used for the project. uv requires a virtual environment; an existing one is sufficient.

<!-- tabs -->
### pip

```console
pip install stacknordic
```

### uv

```console
uv pip install stacknordic
```

<!-- /tabs -->

<details>
<summary>Create a virtual environment if needed</summary>

Skip this step if a Python 3.11 or 3.12 environment is already active. Otherwise, create and activate one before installing StackNordic.

<!-- tabs -->
### pip

```console
python3.12 -m venv .venv
source .venv/bin/activate
```

### uv

```console
uv venv --python 3.12
source .venv/bin/activate
```

<!-- /tabs -->

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) if uv is unavailable.

</details>

## 2. Check the installation

```console
stacknordic --version
stacknordic --help
python -c "import sys, stacknordic; print(sys.executable); print(stacknordic.__file__)"
```

The commands should report the installed version, available commands and package location. For notebooks, select a kernel from the same Python environment.

## 3. Choose a workflow

### Read stored data

Case inspection, direct reads of stored fields and operations on exported datasets do not require an active OpenFOAM runtime. Inspect an existing case without launching a solver:

```python
import stacknordic as sno

case = sno.OpenFOAM.case(case_dir="cases/cavity")
print(case.reconstructed_times)
print(case.decomposed_times)
```

Replace the path with a case containing `system/controlDict` and `constant/`. Continue with [dataset export and storage](api/dataset.md) or [filtering](api/filtering.md). Export requests that invoke OpenFOAM, such as reconstruction or runtime-derived gradients, need the runtime below.

### Run OpenFOAM utilities

Reconstruction and OpenFOAM-backed post-processing require an installed OpenFOAM runtime. Installing StackNordic does not install OpenFOAM. Select a runtime compatible with the case.

<!-- tabs -->
### Linux / HPC

Load the site's OpenFOAM module; the module name below is an example:

```console
module load openfoam/2512
reconstructPar -help
```

For a standalone installation, source its `etc/bashrc` instead, following its compiler and MPI setup instructions.

### Apple Silicon

Use OpenFOAM.app and its `openfoam` launcher:

```console
openfoam reconstructPar -help
```

Run StackNordic through the same launcher to preserve the selected Python executable:

```console
openfoam "$(command -v stacknordic)" reconstruct --case-dir cases/cavity --n-jobs 1 --progress
```

<!-- /tabs -->

Continue with the [reconstruction example](../README.md#reconstruct-a-case). In Python, declare the environment through `sno.OpenFOAM.case(of_cmd=..., shell=...)`: use the local module command with `shell="bash"`, or `of_cmd="openfoam"` with `shell="zsh"` on macOS.

## If installation stops

| Symptom | Next step |
| --- | --- |
| Python version rejected | Select Python 3.11 or 3.12. |
| `stacknordic: command not found` | Activate the installation environment and check its executable path. |
| Missing case entries | Check the case path and its `system/controlDict` and `constant/` entries. |
| `reconstructPar` unavailable | Activate a compatible OpenFOAM runtime before reconstruction. |
| Import fails only in a notebook | Select the matching Python kernel and restart it. |
| Documented API absent | Compare the installed release with the site's source revision. |

For an unresolved failure, retain the command, error output, installed version and Python path. Include the OpenFOAM version when a utility was involved.

## Install from main

To use the current repository version ahead of a release:

<!-- tabs -->
### pip

```console
pip install "git+https://github.com/PentagonToy/StackNordic.git@main"
```

### uv

```console
uv pip install "git+https://github.com/PentagonToy/StackNordic.git@main"
```

<!-- /tabs -->

Replace `main` with a tag or commit for reproducible work.

## Editable installation

Use an editable installation when modifying the source:

```console
git clone https://github.com/PentagonToy/StackNordic.git
cd StackNordic
```

<!-- tabs -->
### pip

```console
pip install -e .
```

### uv

```console
uv pip install -e .
```

<!-- /tabs -->

Follow the [development instructions](../README.md#development) and [verification guide](developer/verification.md) for contributor checks.
