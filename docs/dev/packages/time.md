# i7aof.time

Purpose: Time utilities for annual averaging, CMIP-focused annual drivers, and time-bounds helpers used across workflows.

## Public Python API (by module)

- Module: {py:mod}`i7aof.time.average`
  - {py:func}`annual_average() <i7aof.time.average.annual_average>`: Compute weighted annual means from monthly inputs; outputs one annual file per input. CLI: `ismip7-antarctic-annual-average`.

- Module: {py:mod}`i7aof.time.cmip`
  - {py:func}`compute_cmip_annual_averages() <i7aof.time.cmip.compute_cmip_annual_averages>`: Driver that discovers bias-corrected monthly CT/SA and TF files per model/scenario/climatology and writes annual means into `Oyr/ct_sa_tf`. CLI: `ismip7-antarctic-cmip-annual-averages`.

- Module: {py:mod}`i7aof.time.bounds`
  - {py:func}`capture_time_bounds() <i7aof.time.bounds.capture_time_bounds>`: Capture existing `time_bnds` if present, returning the variable name and DataArray (optional tuple).
  - {py:func}`inject_time_bounds() <i7aof.time.bounds.inject_time_bounds>`: Inject previously captured `time_bnds` back into a dataset and set `time.attrs['bounds']` accordingly.

## Required config options

- {py:mod}`i7aof.time.average`: None.
- {py:mod}`i7aof.time.bounds`: None.
- {py:mod}`i7aof.time.cmip`: Uses {py:func}`i7aof.config.load_config` to resolve `workdir` and grid metadata; no required keys beyond the standard configuration used elsewhere.

## Outputs

- Annual-mean NetCDF files written alongside inputs or into an explicit `out_dir`.
- Naming:
  - If an input basename contains `Omon_`, it is replaced with `Oyr_`.
  - Otherwise, `_ann` is inserted before the extension.
- The CMIP driver combines CT/SA and TF into a shared `Oyr/ct_sa_tf` directory under `workdir/biascorr/<model>/<scenario>/<clim_name>/`.

## Data model

- Weighted monthly-to-annual means using month length via `time.dt.days_in_month`.
- Requires complete years (exactly 12 months per year); errors otherwise.
- Annual time stamps are at start-of-year with CF `time_bnds` covering `[start_of_year, start_of_next_year]`.
- Handles common CF calendars (gregorian, noleap, 360_day, proleptic_gregorian) using cftime.
- Passes through non-time data vars and coordinates; only data variables with a `time` dimension are averaged.
- Fill values and compression: annual outputs are written via {py:func}`i7aof.io.write_netcdf` with an explicit policy that only `'ct'`, `'sa'`, and `'tf'` carry `_FillValue` and compression; coords and bounds have fills suppressed.

## Runtime and external requirements

- Core: `xarray`, `cftime`, `numpy`.
- Interop: {py:mod}`i7aof.io` for robust NetCDF writing and CF time normalization; {py:mod}`i7aof.coords` for grid coordinate/bounds attachment and fill suppression on non-data.
- CLI: None beyond the project’s standard dependencies; see `dev-spec.txt` for the full conda-forge environment.

## Usage

Python API (monthly → annual):

```python
from i7aof.time.average import annual_average

outs = annual_average(["/path/to/monthly_*.nc"], out_dir="/out/annual", overwrite=True)
for p in outs:
    print(p)
```

CLI (monthly → annual):

```bash
ismip7-antarctic-annual-average \
  "/path/to/*.nc" \
  --outdir "/out/annual" \
  --overwrite
```

CMIP driver (bias-corrected CT/SA + TF):

```python
from i7aof.time.cmip import compute_cmip_annual_averages

outs = compute_cmip_annual_averages(
    model="NorESM2-MM", scenario="ssp585", clim_name="06_nov",
    workdir="/scratch/work", overwrite=False, progress=True)
```

```bash
ismip7-antarctic-cmip-annual-averages \
  -m NorESM2-MM -s ssp585 -c 06_nov \
  -w /scratch/work --overwrite
```

Time-bounds helpers:

```python
from i7aof.time.bounds import capture_time_bounds, inject_time_bounds
import xarray as xr

ds = xr.open_dataset("in.nc")
time_bounds = capture_time_bounds(ds)
# ... operations that drop/modify bounds ...
inject_time_bounds(ds, time_bounds)
```

## Internals (for maintainers)

- annual_average:
  - Expands globs, deduplicates inputs, and validates presence of a `time` dimension.
  - Heuristic chunking along `time` (12) to avoid full loads; leverages `read_dataset` for cftime decoding.
  - Verifies exactly 12 months per year; raises a helpful error listing offending years when possible.
  - Weights by days-in-month; casts float64 to float32 for data vars before averaging; preserves attributes.
  - Constructs `time` and `time_bnds` with cftime classes based on the detected calendar.
  - Keeps other coords and non-time data variables; sets `time.attrs['bounds'] = 'time_bnds'`.
  - Suppresses fill on non-data vars via `strip_fill_on_non_data` and writes using `write_netcdf` with `has_fill_values=['ct','sa','tf']` and `compression=['ct','sa','tf']`.
  - Writes to a temp file in the output directory and atomically renames to the final filename.
- cmip:
  - Loads config to resolve `workdir`; discovers monthly inputs under `biascorr/.../Omon/{ct_sa,tf}`; writes annual outputs to `Oyr/ct_sa_tf`.
  - After averaging, re-opens each output to attach grid coordinates/bounds and re-applies the fill/compression policy before atomically replacing the file.
- bounds:
  - `capture_time_bounds` looks up `time.attrs['bounds']` and returns `(name, dataarray)` if present.
  - `inject_time_bounds` puts the bounds variable back into the target dataset and resets the `time` coordinate `bounds` attribute.

## Edge cases / validations

- Missing `time` dimension: error with a clear message.
- Non-12-month years: error with a list of offending years when available.
- No data variables with a `time` dimension: error.
- Calendar handling: defaults to proleptic gregorian when not specified; cftime classes used for supported calendars.
- Safe writes: temp-file then atomic rename avoids partial outputs and supports resuming after interruptions.
- Performance: chunking along `time` only; all scans/writes are lazy-friendly.

## Extension points

- Seasonal or custom-period averages; mid-year time labeling options.
- Configurable list of variables to carry `_FillValue`/compression.
- Multi-file annual averaging with alignment; inclusion of additional metadata.
