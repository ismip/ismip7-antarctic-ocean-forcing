# Installation

## Requirements

- Python >= 3.10
- A conda installation
  ([Miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#requirements-and-installers)
  is recommended). A conda environment from conda-forge is required. Some
  required tools (e.g., `nco`, `moab`, `mpas_tools`) are available only on
  conda-forge.

If you already have conda but aren’t sure the `conda-forge` channel is set:

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
```

## Conda environment

Create and activate an environment using the `dev-spec.txt` file:

```bash
git clone https://github.com/ismip/ismip7-antarctic-ocean-forcing.git
cd ismip7-antarctic-ocean-forcing
conda create -y -n ismip7_dev --file dev-spec.txt
conda activate ismip7_dev
python -m pip install -e . --no-deps --no-build-isolation
```
