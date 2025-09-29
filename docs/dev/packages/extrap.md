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

## Required config options

- `[workdir] base_dir` — required unless `workdir` arg is provided.
- `[cmip_dataset]`
  - `lon_var`, `lat_var`: variable/dimension names on input
- `[extrap_cmip]`
  - `time_chunk`: int; time chunk size for extrapolation.
  - `num_workers`: int or 'auto'/'0'; controls process parallelism.

## Outputs

- Per-variable vertically extrapolated monthly files under:
  `extrap/{model}/{scenario}/Omon/ct_sa/*ismip<res>_extrap.nc`
- Per-chunk intermediates and logs under `*_tmp/` next to the final output:
  - `input_<i0>_<i1>.nc` (prepared inputs)
  - `horizontal_<i0>_<i1>.nc`, `vertical_<i0>_<i1>.nc`
  - `logs/<var>_t<i0>-<i1>.log` containing Fortran output and Python tracebacks

## Data model

- Ensures `x`, `y` coordinates and retains only required variables for the
  Fortran tools (target variable, time, x, y, and z/z_extrap).
- Final concatenation injects ISMIP grid coordinates and related variables.

## Runtime and external requirements

- Core: `xarray`, `numpy`, `dask` (scheduler control), `mpas-tools` (config/logging).
- Tools: Fortran executables `i7aof_extrap_horizontal` and `i7aof_extrap_vertical`.
- Environment: `HDF5_USE_FILE_LOCKING=FALSE` set by default; OMP/BLAS/MKL threads
  set to 1 per worker. `stdbuf` used for unbuffered Fortran output when available.

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

## Edge cases / validations

- Missing required dims (`x`, `y`) or target variable raise clear errors.
- Grid/field dimension mismatches raise `ValueError` in preparation phase.
- If final output exists, the file is skipped entirely. If a chunk’s vertical
  output exists, it won’t be recomputed (idempotent per-chunk work).

## Extension points

- Add support for additional variables or alternate Fortran executables.
- Provide alternative scheduling strategies (e.g., per-node pools) or adaptive
  chunk sizing.
- Make faulthandler and logging verbosity configurable.
