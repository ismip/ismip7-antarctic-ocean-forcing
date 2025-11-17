# i7aof.extrap

Purpose: Orchestrate horizontal and vertical extrapolation of CMIP-derived
ct/sa to the ISMIP grid using external Fortran executables.

## Public Python API (by module)

- Module: {py:mod}`i7aof.extrap`
  - {py:func}`load_template_text() <i7aof.extrap.load_template_text>`:
      Load the combined horizontal/vertical namelist template text.

- Module: {py:mod}`i7aof.extrap.cmip`
  - {py:func}`extrap_cmip() <i7aof.extrap.cmip.extrap_cmip>`:
      Orchestrate per-file, per-variable extrapolation in time chunks with
      optional parallel workers and per-chunk logs.
  - Post-extrap conservative resampling (z_extrap → z) is performed using a
    Zarr-first, append-by-time workflow implemented in
    {py:mod}`i7aof.extrap.shared`.

## Required config options

- `[workdir] base_dir` — required unless `workdir` arg is provided.
- `[extrap_cmip]`
  - `time_chunk`: int; time chunk size for extrapolation.
  - `num_workers`: int or 'auto'/'0'; controls process parallelism.
  - `time_chunk_resample`: int; time chunk size for post-extrap vertical
    resampling to z-levels (Zarr append chunk length).
- `[extrap]`
  - `mask_under_ice`: bool; if true, mask values where `ice_frac` exceeds the threshold before extrapolation.
  - `under_ice_threshold`: float; threshold for masking under ice.

## Outputs

- Per-variable vertically extrapolated monthly files under:
  `extrap/{model}/{scenario}/Omon/ct_sa/*ismip<res>_extrap.nc`
- Per-chunk intermediates and logs under `*_tmp/` next to the final output:
  - `input_<i0>_<i1>.nc` (prepared inputs)
  - `horizontal_<i0>_<i1>.nc`, `vertical_<i0>_<i1>.nc`
  - `logs/<var>_t<i0>-<i1>.log` containing Fortran output and Python tracebacks

- Post-extrap conservative resampled outputs (on `z` levels) are written next
  to the extrapolated file with the ISMIP resolution component updated from
  `ismip<hres>_<dz_extrap>` to `ismip<hres>_<dz>`. If the resolution string is
  unchanged, a `_z` suffix is used, e.g., `..._extrap_z.nc`. A temporary
  `<basename>.zarr/` store is created during resampling and removed after the
  final NetCDF is written.

## Data model

- Ensures `x`, `y` coordinates and retains only required variables for the
  Fortran tools (target variable, time, x, y, and z/z_extrap).
- Final concatenation injects ISMIP grid coordinates and related variables.

## Runtime and external requirements

- Core: `xarray`, `numpy`, `dask` (scheduler control), `mpas-tools` (config/logging).
- Tools: Fortran executables `i7aof_extrap_horizontal` and `i7aof_extrap_vertical`.
- Environment: `HDF5_USE_FILE_LOCKING=FALSE` set by default; OMP/BLAS/MKL threads
  set to 1 per worker; `stdbuf` used for unbuffered Fortran output when available.

## Usage

```python
from i7aof.extrap.cmip import extrap_cmip

extrap_cmip(
  model='CESM2-WACCM',
  scenario='historical',
  user_config_filename='my-config.cfg',
  num_workers='auto',  # or an int
)
```

CLI
```text
ismip7-antarctic-extrap-cmip \
  --model CESM2-WACCM \
  --scenario historical \
  --config my-config.cfg \
  --num_workers auto
```

## Internals (for maintainers)

- Time chunking computed from source metadata; chunks run serially or in a
  process pool (`spawn` start method). Each chunk writes a per-chunk input with
  Dask `scheduler='synchronous'` for safer HDF5 writes, then runs horizontal and
  vertical Fortran steps with unbuffered stdout/stderr captured to the same log.
- Worker failures raise a `ChunkFailed(i0, i1, log_path, message)` that the parent
  logs verbosely before cancelling outstanding futures. Pool crashes log
  completed vs pending chunk indices and point to the logs directory.
- Finalization concatenates vertical outputs and injects grid coordinates/vars.

### Post-extrap vertical resampling (z_extrap → z)

- Strategy: open the extrapolated NetCDF lazily (optionally chunked by time),
  iterate over time chunks of length `[extrap_cmip] time_chunk_resample`,
  apply {py:class}`i7aof.vert.resamp.VerticalResampler` to conservatively map
  intensive fields onto `z`, and append each chunk to a Zarr store using
  `append_dim='time'` when applicable. After all chunks, open the Zarr store
  once, preserve per-variable chunk encodings, write a single NetCDF, and
  delete the Zarr store.
- Output variable dimension order is enforced to `(time?, z, y, x)`.

## Edge cases / validations

- Missing required dims (`x`, `y`) or target variable raise clear errors.
- Grid/field dimension mismatches raise `ValueError` in preparation phase.
- If final output exists, the file is skipped entirely. If a chunk’s vertical
  output exists, it won’t be recomputed (idempotent per-chunk work).
- If the final resampled NetCDF already exists, resampling is skipped.

## Extension points

- Add support for additional variables or alternate Fortran executables.
- Provide alternative scheduling strategies (e.g., per-node pools) or adaptive
  chunk sizing.
- Make faulthandler and logging verbosity configurable.
