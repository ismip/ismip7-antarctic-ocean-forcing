# ismip7-antarctic-ocean-forcing

A package for generating Antarctic ocean forcing for the
[ISMIP7]() activity

## Documentation

Full documentation, including user and developer guides, is available at the
project website:
[ismip7-antarctic-ocean-forcing documentation](https://ismip.github.io/ismip7-antarctic-ocean-forcing/).

## Required conda environment

To generate the ocean forcing data, you need the `conda` package manager. If
you don't have a conda base environment, please download and install
[Miniforge3](https://conda-forge.org/download/).

If you already have `conda` installed but aren't sure if you have the
`conda-forge` channel, please run:
```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
```

Then, this repository can be cloned and a conda environment can be set up with
the required packages as follows:

```bash
mkdir ismip7
cd ismip7
git clone git@github.com:ismip/ismip7-antarctic-ocean-forcing.git main
cd main
conda create -y -n ismip7_dev --file dev-spec.txt
conda activate ismip7_dev
python -m pip install -e . --no-deps --no-build-isolation
pre-commit install
```

### Fortran extrapolation executables

The package includes Fortran executables that are required for the
horizontal and vertical extrapolation steps. There is currently no
Python-only implementation of these routines; without the executables
the extrapolation workflow cannot be run. A conda-forge build that
ships the executables automatically is planned, but until that
feedstock is published you must build them locally to enable
extrapolation.

Quick start (after creating/activating the `ismip7_dev` environment above; and
with `rattler-build` installed in your base environment):

```bash
# from repo root; use your base env's conda-bld as a local channel
conda activate base
LOCAL_CHANNEL=${CONDA_PREFIX}/conda-bld

# remove previous local builds to avoid stale artifacts
rm -rf ${LOCAL_CHANNEL}/*/ismip7-antarctic-ocean-forcing*.conda

# pick the right variant file for the python version you want to use, e.g.:
VARIANT=conda/variants/linux_64_python3.13.____cp313.yaml

# build the package (writes into $LOCAL_CHANNEL)
rattler-build build -m "$VARIANT" --output-dir "$LOCAL_CHANNEL" -r conda/recipe.yaml

# install the compiled executables into your dev environment
conda install -n ismip7_dev -c "$LOCAL_CHANNEL" ismip7-antarctic-ocean-forcing
```

Re-run the cleanup, build, and install commands whenever you change the Fortran
sources.

If you need to modify or experiment with the Fortran sources locally,
an additional (lightweight) local package build step using
`rattler-build` is required. A concise workflow and troubleshooting
notes are documented in the developer guide: see
`docs/dev/contributing.md` (section: "Building and testing the Fortran
extrapolation tools locally"). The short version is: build the recipe
into a local channel with `rattler-build`, then create your development
environment using that channel ahead of `conda-forge`.

If you skip the local Fortran build you will not be able to run the
extrapolation step. Once the conda-forge package becomes available this
manual build step will no longer be necessary for typical users.

To use this environment, you simply run:
```bash
conda activate ismip7_dev
```
Note: if the `conda` command is not found, `conda` was not added to your
`.bashrc` or equivalent.  You will need to run something like:
```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate
```
to get the base environment with the `conda` command.

### Developing in a new directory

If you move to a new development directory (e.g. using `git worktree` or
cloning the repository again), you need to make sure to install `i7aof` from
the new directory:
```bash
conda activate ismip7_dev
python -m pip install -e . --no-deps --no-build-isolation
```

### Updating the conda environment

If you need to update the conda environment (e.g. because the dependencies
have changed or you just want the latest versions), we recommend that you
just start fresh:
```bash
conda create -y -n ismip7_dev --file dev-spec.txt
conda activate ismip7_dev
python -m pip install -e . --no-deps --no-build-isolation
```
This will delete the old environment and create a new one with the required
dependencies.  Then, you need to install `i7aof` again.

It is also possible to update the environment you have:
```bash
conda activate ismip7_dev
conda install -y --file dev-spec.txt
```
This is okay to do but typically takes just about as long as starting fresh
and can occasionally lead to messier dependencies.

## Building the Documentation

Built with Sphinx and MyST. Theme: Furo.

Build locally with conda-forge packages:

```bash
conda activate ismip7_dev
sphinx-build -b html docs docs/_build/html
```
Open `docs/_build/html/index.html` in your favorite browser.
