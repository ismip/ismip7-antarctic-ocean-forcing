# i7aof.convert

Purpose: Conversion utilities for TEOS-10 and workflows to generate CT/SA
(Conservative Temperature/Absolute Salinity) from CMIP thetao/so on the native
source grid.

## Public Python API (by module)

- Module: {py:mod}`i7aof.convert.teos10`
  - {py:func}`compute_sa() <i7aof.convert.teos10.compute_sa>`:
      SA from SP + depth/pressure + lon/lat.
  - {py:func}`compute_ct() <i7aof.convert.teos10.compute_ct>`:
      CT from pt + SA.
  - {py:func}`compute_ct_sa() <i7aof.convert.teos10.compute_ct_sa>`:
      Convenience to compute SA, then CT.
  - {py:func}`convert_dataset_to_ct_sa() <i7aof.convert.teos10.convert_dataset_to_ct_sa>`:
      High-level helper to convert paired thetao/so datasets to a ct/sa
      dataset with coordinates preserved.

- Module: {py:mod}`i7aof.convert.paths`
  - {py:func}`get_ct_sa_output_paths() <i7aof.convert.paths.get_ct_sa_output_paths>`:
      Build output filenames for ct/sa derived from thetao/so lists in config;
      used by both conversion and remapping to ensure consistent naming.

- Module: {py:mod}`i7aof.convert.cmip`
  - {py:func}`convert_cmip() <i7aof.convert.cmip.convert_cmip>`:
      Convert all thetao/so monthly pairs for a model/scenario to native-grid
      ct/sa files under ``convert/{model}/{scenario}/Omon/ct_sa``.

## Required config options

- `[workdir] base_dir` — required unless ``workdir`` arg is provided.
- `[inputdir] base_dir` — required unless ``inputdir`` arg is provided.
- `[cmip_dataset]`
  - `lon_var`, `lat_var`: variable/dimension names on input
- `[convert_cmip]` (optional overrides)
  - `depth_var`: name of source depth coordinate (default 'lev').
  - `time_chunk`: integer or None; conversion time chunk size.
- Scenario sections, e.g. `[historical_files]`, `[ssp585_files]` must define:
  - `thetao`, `so`: expressions expanding to lists of relative input paths.

## Outputs

- Native-grid ct/sa monthly files under:
  ``convert/{model}/{scenario}/Omon/ct_sa``
- Filenames mirror thetao basenames with the variable token replaced by
  ``ct_sa``. Existing outputs are skipped for resumability.

## Usage

Python
```python
from i7aof.convert.cmip import convert_cmip

convert_cmip(
  model='CESM2-WACCM',
  scenario='historical',
  user_config_filename='my-config.cfg',
)
```

CLI
```text
ismip7-antarctic-convert-cmip \
  --model CESM2-WACCM \
  --scenario historical \
  --config my-config.cfg
```

## Internals (for maintainers)

- TEOS-10 now uses direct NumPy calls to GSW (gsw) for speed and simplicity,
  with explicit broadcasting:
  - Pressure: `p = gsw.p_from_z(z, lat)` with shapes (Z,1,1) and (Y,X)
  - Salinity: `SA = gsw.SA_from_SP(SP, p, lon, lat)`
  - Temperature: `CT = gsw.CT_from_pt(SA, PT)`
  Inputs are eagerly `.load()`ed per time chunk to avoid large dask graphs.
- Longitudes can be normalized to [0, 360).
- Depth to TEOS-10 z conversion handled by `_depth_to_z` with CF-compliant
  attribute detection; meters and centimeters are supported.
- Conversion runs in a manual time-chunk loop; each chunk is written to a
  unique temporary NetCDF file (adjacent to the final output), then combined
  via `xarray.open_mfdataset` and the temp directory is removed. A tqdm
  progress bar reports chunk progress.
- Conversion keeps ct and sa together; types are cast to float32 for size.
- Output path derivation centralized in `i7aof.convert.paths`.
- Optional debugging: set environment variable `I7AOF_DEBUG_TEOS10=1` to
  print timings and shapes around the heavy TEOS-10 computations.

## Edge cases / validations

- thetao and so list lengths must match; otherwise a ValueError is raised.
- Inputs must align exactly in time and depth; strict alignment is enforced.
- Depth units in meters or centimeters are supported (assumed meters if
  unit-less); other units raise a ValueError.
