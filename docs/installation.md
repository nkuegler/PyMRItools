# Installation

PyMRItools requires Python ≥ 3.11 (see `pyproject.toml`).
Clone the repository first:

```shell
git clone https://github.com/schmidt-jo/PyMRItools.git
cd PyMRItools
```

## uv (recommended)

[uv](https://docs.astral.sh/uv/) sets up a virtual environment from `pyproject.toml` and `uv.lock`.

1. Install uv, see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).
   Optionally, you can choose a custom installation directory:

    ```shell
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/custom/path" sh
    ```

2. Create the environment and install PyMRItools in editable mode:

    ```shell
    uv sync
    ```

3. Run tools inside the environment with `uv run <command>`, or activate it with `source .venv/bin/activate`.

!!! note "PyTorch and CUDA"
    On Linux and Windows, `pyproject.toml` pulls PyTorch from the CUDA 13.0 wheel index (`pytorch-cu130`).
    On macOS the default (CPU / MPS) wheels are used.
    If your driver or CUDA version differs, adapt the `[tool.uv.sources]` and `[[tool.uv.index]]` sections in `pyproject.toml`.
    The package is mainly tested on Linux with NVIDIA GPUs; CPU-only use is possible for most tools but not systematically tested.

`twixtools` and `pypulseq` are installed directly from their GitHub repositories.

## Conda

For development, a conda environment can be used instead.
We prefer [conda-forge](https://conda-forge.org/download/) and `mamba`, but `conda` works as well.
From within the repository directory, create the `mri_tools_env` environment from the provided environment file:

```shell
mamba env create -f envrionment.yml
```

!!! note
    The environment file in the repository root is currently named `envrionment.yml` (sic).
    It pins an older set of dependencies than `pyproject.toml`; if an import fails, install the missing package
    (e.g. `torch-kmeans`, `rich`, `scikit-image`) with `pip`.

Activate the environment and install PyMRItools in editable mode.
Since the dependencies are already provided by conda, use `--no-deps`:

```shell
mamba activate mri_tools_env
pip install --no-deps -e .
```

Changes to the code are directly reflected in the environment.

## Apptainer container for HPC

The `apptainer/` directory contains a definition file to build an [Apptainer](https://apptainer.org/) container
with ROCm (AMD GPU) support, written for the MPCDF Viper-GPU cluster:

- `viper_apptainer.def`: container definition (ROCm Ubuntu 24.04 base image, Miniforge, PyTorch for ROCm).
- `viper_environment.yml`: conda environment used inside the container.

Build and use the container:

```shell
cd apptainer
sudo apptainer build pymritools.sif viper_apptainer.def

# run a single command
apptainer exec pymritools.sif <command>
# or open a shell
apptainer shell pymritools.sif
```

The container activates the `mri_tools_env` environment automatically.
See `apptainer/README.md` for details.

## Verifying the installation

The command line tools listed in [Command Line Tools](cli.md) are available once the environment is active:

```shell
uv run recon_loraks --help
```
