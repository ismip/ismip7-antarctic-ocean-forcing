# i7aof.io, i7aof.io_zarr and i7aof.download

Purpose: Shared I/O helpers for NetCDF writing, Zarr finalization, and robust file downloads.

## Public Python API (by module)

Import paths and brief descriptions by module:

- Module: {py:mod}`i7aof.io`
  - {py:func}`read_dataset() <i7aof.io.read_dataset>`: Open a dataset with package defaults and normalized CF time metadata (cftime decoding, matching units/calendar on `time`/`time_bnds`).
  - {py:func}`write_netcdf() <i7aof.io.write_netcdf>`: Write an xarray.Dataset to NetCDF with per-variable fill values, optional compression, time encoding normalization, and optional progress bar; supports a conversion path for 64-bit NetCDF3 (CDF5).

```{note}
Why use {py:func}`i7aof.io.write_netcdf` instead of `xarray.Dataset.to_netcdf`?

- Typed fill values, not NaNs: xarray’s default can leave NaNs for missing
  values; many tools (e.g., NCO) expect a typed `_FillValue`. This wrapper maps
  NumPy dtypes to NetCDF fill values (via `netCDF4.default_fillvals`), skips
  string fills, and only writes `_FillValue` when a variable actually contains
  NaNs (unless explicitly directed).
- Per-variable control: enable/disable `_FillValue` and compression for all
  variables or a selected list via simple flags.
- Unlimited time by default: automatically marks `time` as an unlimited
  dimension when present (and clears unlimited dims when not), improving
  compatibility with tools that append or concatenate over time.
- Deterministic CF-time encoding: normalizes `time`/`time_bnds` encodings to
  use `units='days since 1850-01-01'`, sets `calendar`, and ensures numeric
  dtype for consistent serialization across backends.
- Reliable CDF5 output: `format='NETCDF3_64BIT_DATA'` is inconsistently
  supported by backends. Here, data are written as NETCDF4 and converted to
  CDF5 using NCO (`ncks -5`), then the temp file is removed. If `engine='scipy'`
  is requested for this path, it is coerced to `netcdf4` to avoid failures.
- Better user experience for large writes: optional Dask progress bar without
    changing user code.

**Note within a note**: CDF5 conversion uses `ncks` (NCO). In the conda-forge
environment defined by `dev-spec.txt`, NCO is included and available on PATH.
```

- Module: {py:mod}`i7aof.io_zarr`
  - {py:func}`append_to_zarr() <i7aof.io_zarr.append_to_zarr>`: Append one or more chunked Datasets to a temporary Zarr store, creating it on the first write; idempotent when a previous run already completed the same segment(s).
  - {py:func}`finalize_zarr_to_netcdf() <i7aof.io_zarr.finalize_zarr_to_netcdf>`: Open the Zarr store (non-consolidated), optionally postprocess the Dataset, then write the final NetCDF atomically and clean up the Zarr store.

- Module: {py:mod}`i7aof.download`
  - {py:func}`download_file() <i7aof.download.download_file>`: Download a
      single file to a directory or explicit path, with progress bar.
  - {py:func}`download_files() <i7aof.download.download_files>`: Download
      many files from a base URL to a directory, preserving subpaths.

## Required config options

None. Functions accept explicit arguments and use no global config.

## Outputs

- NetCDF files written via {py:func}`i7aof.io.write_netcdf` to the provided
  filename (may use a temporary file during CDF5 conversion).
- When using {py:mod}`i7aof.io_zarr`: a temporary Zarr store during chunked
  appends, and a single final NetCDF produced at finalize time. The Zarr
  store is removed after a successful finalize.
- Downloaded files saved to requested paths; parent directories are created.

## Data model

- Fill values: `_FillValue` decisions are made per variable, using
  NumPy-dtype → NetCDF mappings (string types are skipped). Behavior is
  controlled by the `has_fill_values` argument:
  - `True`: apply `_FillValue` to all variables using type-appropriate defaults
  - `False`: explicitly suppress `_FillValue` (sets `None` to disable backend defaults)
  - `list[str]`: apply only to the named variables
  - `None` (default): lazily scan each variable for missing values and set
    `_FillValue` only when needed
- Compression: per-variable compression is controlled by the `compression`
  argument with the same forms as above (`True`/`False`/`list[str]`/`None`).
  Default compression options (when enabled as a boolean) are:
  `{'zlib': True, 'complevel': 4, 'shuffle': True}`. If compression is
  requested and no engine is specified, `h5netcdf` is preferred. The `scipy`
  engine does not support compression and will be ignored with a warning.
- Time encoding: `write_netcdf` normalizes CF-time metadata for `time` and
  `time_bnds` to use numeric days since `1850-01-01` with a declared
  `calendar` (defaulting to `proleptic_gregorian`), clears conflicting attrs,
  and sets a numeric dtype to ensure consistent serialization.
- Unlimited dimension: when a `time` dimension exists, it is marked as
  unlimited; otherwise, any unlimited dims are cleared.
- CDF5 conversion: for `format='NETCDF3_64BIT_DATA'`, the dataset is written as
  NETCDF4 then converted using `ncks -5`, and the temporary file is removed.
- Zarr workflow: {py:mod}`i7aof.io_zarr` appends chunked results to a Zarr
  store and later finalizes to a single NetCDF. An internal ready marker file
  (`.i7aof_zarr_ready`) inside the store enables idempotent reruns (skip
  appends if Zarr is already ready). Finalization writes to `out.nc.tmp` first
  and atomically moves it to `out.nc` on success, then removes the Zarr store
  and any legacy `.complete` marker.

## Runtime and external requirements

- Core: `xarray`, `netCDF4`, `numpy`, `dask` (for the optional progress bar).
- Tools: `ncks` from NCO if converting to `NETCDF3_64BIT_DATA`.
- Zarr: `zarr` (via xarray’s Zarr backend) for append/finalize workflows.
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

### Per-variable fill values and compression

```python
import xarray as xr
from i7aof.io import write_netcdf

ds = xr.tutorial.load_dataset('air_temperature').isel(time=slice(0, 12))

# Apply fill values to only selected variables and compress those variables
write_netcdf(
  ds,
  'air.nc',
  has_fill_values=['air'],           # apply _FillValue only to 'air'
  compression=['air'],               # compress only 'air'
  compression_opts={'zlib': True, 'complevel': 4, 'shuffle': True},
)

# Enable defaults for all variables
write_netcdf(
  ds,
  'air_all.nc',
  has_fill_values=True,
  compression=True,  # uses default options
)
```

### Zarr append and atomic finalize

```python
import xarray as xr
from i7aof.io_zarr import append_to_zarr, finalize_zarr_to_netcdf

zstore = 'tmp.zarr'
first = True

for chunk in range(3):
  ds = xr.Dataset({'x': ('t', [chunk])}, coords={'t': [chunk]})
  first = append_to_zarr(ds=ds, zarr_store=zstore, first=first, append_dim='t')

def post(ds: xr.Dataset) -> xr.Dataset:
  ds.attrs['history'] = 'finalized'
  return ds

finalize_zarr_to_netcdf(
  zarr_store=zstore,
  out_nc='final.nc',
  postprocess=post,
  # pass-through to write_netcdf:
  compression=True,
)
```

## Internals (for maintainers)

- `read_dataset` opens datasets with `decode_times=CFDatetimeCoder(use_cftime=True)`
  by default and normalizes `time`/`time_bnds` metadata.
- `write_netcdf` builds an encoding dict over all data variables and coords,
  applies `_FillValue` and compression decisions, normalizes time encodings,
  and manages unlimited dims.
- When converting to `NETCDF3_64BIT_DATA`, a temporary NETCDF4 file is created
  and NCO is invoked with `ncks -O -5` to produce the final output.
- {py:mod}`i7aof.io_zarr` writes a hidden `.i7aof_zarr_ready` marker inside
  the Zarr store upon successful open to enable idempotent reruns, then writes
  NetCDF to `out.nc.tmp` and atomically moves it to `out.nc` on success.
- Downloads stream content in 1KB chunks and update a `tqdm` progress bar.

## Edge cases / validations

- Fill values: when `has_fill_values is None`, variables are lazily scanned
  for missing values and `_FillValue` is only written when necessary; string
  dtypes intentionally have no `_FillValue`.
- Compression: if `engine='scipy'` is selected, compression directives are
  ignored with a warning (backend limitation). When compression is requested
  and engine is unspecified, `h5netcdf` is preferred.
- Time dtype: writing with `numpy.datetime64` `time`/`time_bnds` is rejected;
  use `read_dataset` (cftime) or numeric CF time with supported units
  (`'days since'` or `'seconds since'`).
- If `engine='scipy'` is requested for a CDF5 conversion path, it is forced to
  `netcdf4` to avoid incompatibilities.
- Zarr: `append_to_zarr` is idempotent—skips when a segment or the entire
  store is already complete/ready. Finalization always removes the Zarr store
  and cleans any legacy external `.complete` marker.
- Download functions do nothing if destination files exist unless
  `overwrite=True` is passed.

## Extension points

- Add retry/backoff to downloads; allow custom chunk sizes.
- Make compression options configurable globally; add per-variable chunk/codec
  control if needed.
