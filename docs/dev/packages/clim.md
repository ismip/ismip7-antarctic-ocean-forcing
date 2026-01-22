# i7aof.clim

Purpose: Bundled climatology configurations used across the workflows (remap → extrapolate → bias correction → TF, etc.). This package ships versioned config files and serves as a stable reference for climatology metadata.

## Public Python API (by module)

- Module: {py:mod}`i7aof.clim` (package)
  - No public functions or classes; the package provides packaged configuration files (`*.cfg`). These are loaded via {py:func}`i7aof.config.load_config`.

## How configuration is loaded

- {py:func}`i7aof.config.load_config` merges, in order:
  1. Base defaults from `i7aof/default.cfg`.
  2. Optional model config from {py:mod}`i7aof.cmip` (if a model is specified).
  3. Climatology config from {py:mod}`i7aof.clim` when `clim_name` is specified: `config.add_from_package('i7aof.clim', f'{clim_name}.cfg')`.
  4. Optional user config file.

- The argument `clim_name` is the stem of the file, e.g., `zhou_annual_06_nov` (without the `.cfg` extension).

## What’s included

These packaged climatologies reflect curated variants. Current files include (examples):

- `zhou_annual_06_nov.cfg` (preferred v2)
- `zhou_2000_annual_06_nov.cfg` (v2, 2000-only variant)
- `zhou_summer_06_nov.cfg` (v2, summer-only)
- Legacy `30_sep` variants retained for completeness and reproducibility.

See the user docs for recommendations and workflows that prefer the `06_nov` (v2) set.

## Config schema (section: `[climatology]`)

Common options across the packaged files:

- Variable and dimension names:
  - `lat_var`, `lon_var`, `lev_var`
  - `lat_dim`, `lon_dim`, `lev_dim`
- Data arrays:
  - `ct_var` (Conservative Temperature)
  - `sa_var` (Absolute Salinity)
  - Optional MSE arrays: `ct_mse_var`, `sa_mse_var`
- Quality control:
  - `mse_threshold` (omit values where MSE exceeds this threshold)
- Input path:
  - `filename`: relative path to the source NetCDF under the configured input directory
- Bias-correction window:
  - `start_year`, `end_year` used when forming the CMIP model climatology for bias correction

Example (`zhou_annual_06_nov.cfg`):

```ini
[climatology]
lat_var = latitude
lon_var = longitude
lat_dim = ny
lon_dim = nx
lev_var = pressure
lev_dim = nz
filename = Updated_TS_Climatology/OI_Climatology_v2/OI_Climatology.nc
ct_var = ct
sa_var = sa
ct_mse_var = ct_mse
sa_mse_var = sa_mse
mse_threshold = 1e9
start_year = 1995
end_year = 2024
```

## Required config options

- None in code. These files are consumed by higher-level steps that expect the keys above.

## Outputs

- No direct outputs from this package; consuming steps write outputs (remap/extrap/biascorr/convert/time).

## Data model

- The schema above is consumed by {py:mod}`i7aof.remap.clim` for preprocessing/remapping, by {py:mod}`i7aof.extrap.clim` for vertical extrapolation, and by {py:mod}`i7aof.biascorr.classic` to define the climatology time window and input paths.

## Runtime and external requirements

- None beyond the project’s standard environment; these are static configuration assets.

## Usage

Programmatic:

```python
from i7aof.config import load_config

config = load_config(clim_name="zhou_annual_06_nov", workdir="/work", inputdir="/inputs")
rel_path = config.get('climatology', 'filename')
```

CLI (via consuming steps):

- Remap climatology: `ismip7-antarctic-remap-clim -c zhou_annual_06_nov ...`
- Extrapolate climatology: `ismip7-antarctic-extrap-clim -c zhou_annual_06_nov ...`
- Convert climatology CT/SA → TF: `ismip7-antarctic-clim-ct-sa-to-tf -c zhou_annual_06_nov ...`

## Internals (for maintainers)

- Packaged under `i7aof/clim/*.cfg` and included via setuptools package data; see `pyproject.toml` `[tool.setuptools.package-data]`.
- Consuming code references the `[climatology]` section keys to determine variable names, dimensions, thresholds, and source filename.

## Edge cases / validations

- Dim/var casing differences exist across upstream datasets (e.g., `NY`/`NX` vs `ny`/`nx`); the configs reflect each file’s reality.
- If a required key is missing, consuming code will raise an error when it attempts to access it.

## Extension points

- Adding new climatologies: contribute new `*.cfg` files following the schema above.
- Consider adding validation for the config schema during load, and docstring links in consuming modules to the config keys they require.
