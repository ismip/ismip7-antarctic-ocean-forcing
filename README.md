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
conda create -n ismip7_dev --file dev-spec.txt
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
