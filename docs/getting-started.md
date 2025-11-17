# Getting Started

This project requires a consistent setup for everyone, including compiled Fortran executables used by the extrapolation workflow. Follow this guide to install, build, and verify your environment.

## Requirements

- Python >= 3.10
- A conda installation (we recommend Miniforge)
- The conda-forge channel with strict priority

If you aren’t sure the `conda-forge` channel is set:

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
```

## 1) Install rattler-build in your base environment

Install once into your base conda environment to enable local package builds:

```bash
conda install -n base -c conda-forge rattler-build
```

## 2) Clone the repository

```bash
git clone https://github.com/ismip/ismip7-antarctic-ocean-forcing.git
cd ismip7-antarctic-ocean-forcing
```

## 3) Build a local conda package (with Fortran executables)

This produces a package that installs the Fortran executables used by `i7aof.extrap`:

```bash
# Build inside your current (base) environment; this sets a local channel path
LOCAL_CHANNEL=${CONDA_PREFIX}/conda-bld

# Pick the variant matching the Python you plan to use, e.g.:
VARIANT=conda/variants/linux_64_python3.13.____cp313.yaml

# Build the package (reads conda/recipe.yaml)
rattler-build build -m "$VARIANT" --output-dir "$LOCAL_CHANNEL" -r conda/recipe.yaml
```

This build provides two executables:

- `i7aof_extrap_horizontal`
- `i7aof_extrap_vertical`

## 4) Create the environment using the local package

Create your development environment pulling everything from conda-forge, with this package coming from your local channel (put it first):

```bash
conda create -y -n ismip7_dev -c "$LOCAL_CHANNEL" -c conda-forge --file dev-spec.txt ismip7-antarctic-ocean-forcing
conda activate ismip7_dev
```

Then install the repo in editable mode (keeps Python code live while preserving the compiled executables from the package):

```bash
pip install --no-deps --no-build-isolation -e .
```

Optional but recommended for consistent formatting and linting:

```bash
pre-commit install
```

## 5) Verify the setup

```bash
which i7aof_extrap_horizontal
which i7aof_extrap_vertical
python -c "import i7aof.extrap as ex; print(ex.load_template_text()[:120])"
```

If the `which` commands return paths inside the active environment and the Python import succeeds, your setup is correct.

## 6) Rebuilding after Fortran source changes

If you modify files under `fortran/`, rebuild and then recreate the environment with the newly built local package:

```bash
# ensure you're in the base environment
conda deactivate || true
conda deactivate || true
conda activate base

LOCAL_CHANNEL=${CONDA_PREFIX}/conda-bld

# delete any previously built packages for this project
rm -rf ${LOCAL_CHANNEL}/*/ismip7-antarctic-ocean-forcing*.conda

# pick the appropriate Python variant again, e.g.:
VARIANT=conda/variants/linux_64_python3.13.____cp313.yaml

# rebuild the package
rattler-build build -m "$VARIANT" --output-dir "$LOCAL_CHANNEL" -r conda/recipe.yaml

# recreate the environment
conda create -y -n ismip7_dev -c "$LOCAL_CHANNEL" -c conda-forge --file dev-spec.txt ismip7-antarctic-ocean-forcing
conda activate ismip7_dev

# reinstall the Python code in editable mode
pip install --no-deps --no-build-isolation -e .
```

## Troubleshooting

- If builds seem stale, remove `build/` subdirectories under rattler-build’s cache (see its terminal output for cache paths) and rebuild.
- Ensure `conda-forge` channel priority is strict, and your local channel is listed before conda-forge when creating the environment.
- Use a variant file under `conda/variants/` matching the Python version you want in your environment.
