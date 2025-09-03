# i7aof.io

Purpose: Shared I/O helpers for NetCDF writing and robust file downloads.

```{note}
This page documents both modules: {py:mod}`i7aof.io` and {py:mod}`i7aof.download`.
```

## Public Python API (by module)

Import paths and brief descriptions by module:

- Module: {py:mod}`i7aof.io`
  - {py:func}`write_netcdf() <i7aof.io.write_netcdf>`: Write an xarray.Dataset to NetCDF with appropriate fill values and optional progress bar; supports a conversion path for 64-bit NetCDF3.

```{note}
Why use {py:func}`i7aof.io.write_netcdf` instead of `xarray.Dataset.to_netcdf`?

- Typed fill values, not NaNs: xarray’s default can leave NaNs for missing
  values; many tools (e.g., NCO) expect a typed `_FillValue`. This wrapper maps
  NumPy dtypes to NetCDF fill values (via `netCDF4.default_fillvals`), skips
  string fills, and only writes `_FillValue` when a variable actually contains
  NaNs.
- Unlimited time by default: automatically marks `time` as an unlimited
  dimension when present (and clears unlimited dims when not), improving
  compatibility with tools that append or concatenate over time.
- Reliable CDF5 output: `format='NETCDF3_64BIT_DATA'` is inconsistently
  supported by backends. Here, data are written as NETCDF4 and converted to
  CDF5 using NCO (`ncks -5`), then the temp file is removed. If `engine='scipy'`
  is requested for this path, it is coerced to `netcdf4` to avoid failures.
- Better user experience for large writes: optional Dask progress bar without
    changing user code.

**Note within a note**: CDF5 conversion uses `ncks` (NCO). In the conda-forge
environment defined by `dev-spec.txt`, NCO is included and available on PATH.
```

- Module: {py:mod}`i7aof.download`
  - {py:func}`download_file() <i7aof.download.download_file>`: Download a
      single file to a directory or explicit path, with progress bar.
  - {py:func}`download_files() <i7aof.download.download_files>`: Download
      many files from a base URL to a directory, preserving subpaths.

## Required config options

None. Functions accept explicit arguments and use no global config.

## Outputs

- NetCDF files written via {py:func}`i7aof.io.write_netcdf` to the provided
  filename (may use a temporary file during format conversion).
- Downloaded files saved to requested paths; parent directories are created.

## Data model

- Encodes `_FillValue` per variable based on NumPy dtype → NetCDF mapping,
  skipping string types; time is set as an unlimited dimension when present.
- For `format='NETCDF3_64BIT_DATA'`, the dataset is written as NETCDF4 then
  converted using `ncks -5`, and the temporary file is removed.

## Runtime and external requirements

- Core: `xarray`, `netCDF4`, `numpy`, `dask` (for the optional progress bar).
- Tools: `ncks` from NCO if converting to `NETCDF3_64BIT_DATA`.
- Downloads: `requests`, `tqdm` for progress.
- For the authoritative conda-forge environment, see `dev-spec.txt` (note
  `pyproject.toml` lists a PyPI-only subset).

## Usage

```python
import xarray as xr
from i7aof.io import write_netcdf

ds = xr.Dataset({'a': ('x', [1, 2, 3])})
write_netcdf(ds, 'out.nc')
```

```python
from i7aof.download import download_file

download_file('https://example.com/file.txt', 'data/', quiet=True)
```

## Internals (for maintainers)

- `write_netcdf` builds an encoding dict over all data variables and coords,
  sets per-dtype `_FillValue`, and manages unlimited dims.
- When converting to `NETCDF3_64BIT_DATA`, a temporary NETCDF4 file is created
  and NCO is invoked with `ncks -O -5` to produce the final output.
- Downloads stream content in 1KB chunks and update a `tqdm` progress bar.

## Edge cases / validations

- Fill values are only applied when variables contain NaNs; string dtypes
  intentionally have no `_FillValue`.
- If `engine='scipy'` is requested for a conversion path, it is forced to
  `netcdf4` to avoid incompatibilities.
- Download functions do nothing if destination files exist unless
  `overwrite=True` is passed.

## Extension points

- Add retry/backoff to downloads; allow custom chunk sizes.
