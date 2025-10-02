# Contributing

We welcome contributions! Please open issues and pull requests.

- Follow PEP8 and the configured Ruff rules.
- Add or update tests when changing behavior.
- Keep public APIs documented.
- Small PRs are easier to review.

## Dev setup

Use conda-forge for the environment and create it directly from `dev-spec.txt`:

```bash
conda create -n ismip7_dev -c conda-forge --file dev-spec.txt
conda activate ismip7_dev
pip install --no-deps --no-build-isolation -e .
pre-commit install
```

If you don't already have a conda base installation set up, we recommend
starting with [Miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#requirements-and-installers).

Build docs locally when editing documentation:

```bash
sphinx-build -b html docs docs/_build/html
```

Add test scripts to `scripts/` that can be used to smoke test features.

## Building and testing the Fortran extrapolation tools locally

The project now includes two small Fortran executables that perform
horizontal and vertical extrapolation used by the `i7aof.extrap`
package. There is currently no Python fallback; without these
executables the extrapolation workflow cannot be exercised. Until a
conda-forge package including them is published, you need to build and
install them locally for any extrapolation development or testing.

### 1. Ensure `rattler-build` base dependencies

Install (once) into your *base* conda environment so you can perform
local package builds:

```bash
conda install -n base -c conda-forge rattler-build
```

### 2. Build a local package

From the repository root:

```bash
# you could also build in any temp directory outside of the source repo
LOCAL_CHANNEL=${CONDA_PREFIX}/conda-bld

# pick the right variant file for the python version you want to use, e.g.:
VARIANT=conda/variants/linux_64_python3.13.____cp313.yaml

# Build the conda package (this reads conda/recipe.yaml)
rattler-build build -m "$VARIANT" --output-dir "$LOCAL_CHANNEL" -r conda/recipe.yaml
```

This creates a local channel containing a build of
`ismip7-antarctic-ocean-forcing` with the compiled Fortran executables:

- `i7aof_extrap_horizontal`
- `i7aof_extrap_vertical`

### 3. Create (or recreate) the dev environment including local build

You can now create your development environment pulling everything from
conda-forge except this package which will come from your local channel.
Include the channel *before* conda-forge so it has priority.

```bash
conda create -y -n ismip7_dev -c "$LOCAL_CHANNEL" -c conda-forge --file dev-spec.txt ismip7-antarctic-ocean-forcing
conda activate ismip7_dev
```

Then install the repo in editable mode (this keeps Python code live
while preserving the compiled executables placed by the package):

```bash
pip install --no-deps --no-build-isolation -e .
```

To activate the code linting and formatting tools (required), run:

```bash
pre-commit install
```

### 4. Verify executable availability

```bash
which i7aof_extrap_horizontal
which i7aof_extrap_vertical
python -c "import i7aof.extrap as ex; print(ex.load_template_text()[:120])"
```

If the `which` commands return paths inside the active environment and
the Python import succeeds, the setup is correct.

### 5. Rebuilding after changes

If you modify the Fortran sources under `fortran/`, rebuild and then
recreate the environment with the new local build:

```bash
# make sure you're back in the base environment
conda deactivate
conda deactivate
conda activate base
LOCAL_CHANNEL=${CONDA_PREFIX}/conda-bld
# delete any packages you already built
rm -rf ${LOCAL_CHANNEL}/*/ismip7-antarctic-ocean-forcing*.conda
# pick the right variant file for the python version you want to use, e.g.:
VARIANT=conda/variants/linux_64_python3.13.____cp313.yaml
# build the package
rattler-build build -m "$VARIANT" --output-dir "$LOCAL_CHANNEL" -r conda/recipe.yaml
# recreate the environment
conda create -y -n ismip7_dev -c "$LOCAL_CHANNEL" -c conda-forge --file dev-spec.txt ismip7-antarctic-ocean-forcing
# activate the environment
conda activate ismip7_dev
# install the python package in editable mode
pip install --no-deps --no-build-isolation -e .
```

### Notes / troubleshooting

- Clean previous build artifacts by removing `build/` subdirectories that
  `rattler-build` creates under its cache (see its terminal output for paths)
  if you suspect a stale build.
