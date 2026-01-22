# i7aof.cmip

Purpose: Model-specific CMIP configuration and light utilities used to drive the CMIP branch of the pipeline (split → convert → remap → extrapolate → bias-correct → TF → annual, etc.).

## Public Python API (by module)

- Module: {py:mod}`i7aof.cmip`
  - {py:func}`get_model_prefix() <i7aof.cmip.get_model_prefix>`: Normalize model names to a lowercase, dash-to-underscore prefix used to look up packaged model configs (e.g., `CESM2-WACCM` → `cesm2_waccm`).

## How configuration is loaded

- {py:func}`i7aof.config.load_config` merges, in order:
  1. Base defaults from `i7aof/default.cfg` (includes general CMIP defaults: dataset dims/vars, split, convert/remap/extrap chunk sizes, etc.).
  2. Model config from {py:mod}`i7aof.cmip` when a model is specified: `config.add_from_package('i7aof.cmip', f'{get_model_prefix(model)}.cfg')`.
  3. Optional climatology config (if `clim_name` provided) from {py:mod}`i7aof.clim`.
  4. Optional user config file.

- The model config augments/overrides defaults with model-specific variable/dimension names and input file lists for each scenario.

## Packaged model configs

- Example: `i7aof/cmip/cesm2_waccm.cfg` with sections:
  - `[cmip_dataset]`: dataset variable and dimension names (`lat_var`, `lon_var`, `lat_dim`, `lon_dim`).
  - `[convert_cmip]`: conversion settings (e.g., `depth_var = lev`, `time_chunk`).
  - `[remap_cmip]`: vertical/horizontal chunk sizes (`vert_time_chunk`, `horiz_time_chunk`).
  - `[extrap_cmip]`: extrapolation chunking and parallelism (`time_chunk`, `num_workers`).
  - `[historical_files]`: list of relative monthly input files for `thetao` and `so`.
  - `[ssp585_files]`: list of relative monthly input files for `thetao` and `so` for SSP5-8.5.

These override or extend defaults defined in `i7aof/default.cfg`:

- `[cmip_dataset]` defaults use `lat/lon` dims and vars named `lat`/`lon`.
- `[split_cmip]` provides `months_per_file` for splitting large inputs.
- `[convert_cmip]`, `[remap_cmip]`, `[extrap_cmip]` set reasonable chunk sizes.

## Required config options

- None required by this package directly. Downstream steps will expect sections above to be present (and will error if missing keys are accessed).

## Outputs

- No direct outputs from this package; consuming steps create outputs (convert/remap/extrap/biascorr/time).

## Data model

- The CMIP config defines:
  - Dataset layout: coordinate variable names and dimension names.
  - Processing parameters: chunk sizes and extrapolation concurrency.
  - Input discovery: lists of relative paths to monthly `thetao`/`so` files for each scenario.
- Consuming tools use `model` to select the packaged config file via `get_model_prefix(model)`.

## Runtime and external requirements

- None beyond the project environment; these are static assets plus a simple utility function.

## Usage

Programmatic config load:

```python
from i7aof.config import load_config

config = load_config(model="CESM2-WACCM", workdir="/work", inputdir="/inputs")
lat_dim = config.get('cmip_dataset', 'lat_dim')
```

Downstream CLIs (examples using the loaded model context):

- Split: `ismip7-antarctic-split-cmip -m CESM2-WACCM ...`
- Convert: `ismip7-antarctic-convert-cmip-to-ct-sa -m CESM2-WACCM ...`
- Remap: `ismip7-antarctic-remap-cmip -m CESM2-WACCM ...`
- Extrapolate: `ismip7-antarctic-extrap-cmip -m CESM2-WACCM ...`
- Bias-correct (classic): `ismip7-antarctic-bias-corr-classic -m CESM2-WACCM -c zhou_annual_06_nov ...`
- TF (CMIP path): `ismip7-antarctic-cmip-ct-sa-to-tf -m CESM2-WACCM -c zhou_annual_06_nov ...`
- Annual: `ismip7-antarctic-cmip-annual-averages -m CESM2-WACCM -c zhou_annual_06_nov ...`

## Internals (for maintainers)

- `get_model_prefix` maps model names to a filesystem-friendly, lowercase, underscore-separated prefix used to look up `[prefix].cfg` under `i7aof/cmip/`.
- Packaged configs (`*.cfg`) are included via setuptools package data; see `pyproject.toml` `[tool.setuptools.package-data]` for inclusion rules.

## Edge cases / validations

- Model names with dashes (`-`) are normalized to underscores (`_`). Unknown models will simply not find a packaged config; `load_config` will still return defaults unless the downstream code requires specific keys.
- If required keys are missing for a step (e.g., depth var name for conversion), the consuming code will raise an error when accessing them.

## Extension points

- Add new model configs in `i7aof/cmip/*.cfg` following the `cesm2_waccm.cfg` structure; consider covering additional scenarios (e.g., `ssp126`, `ssp245`).
- Provide a schema validator for model configs to catch typos early.
