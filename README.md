# ismip7-antarctic-ocean-forcing

A package for generating Antarctic ocean forcing for the
[ISMIP7]() activity

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

Then, this repository can be cloned and a conda  environment can be set up with
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
to ge the base environment with the `conda` command.

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

## Documentation

Built with Sphinx and MyST. Theme: Furo.

Build locally with conda-forge packages:

```bash
conda activate ismip7_dev
sphinx-build -b html docs docs/_build/html
```
Open `docs/_build/html/index.html` in your favorite browser.
