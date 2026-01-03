# Architecture Overview

This overview reflects the current project layout and workflow graph.

## Packages

- Core workflows
    - `i7aof.convert` — CMIP conversions (thetao/so → ct/sa; ct/sa → TF; back-conversion of annual ct/sa/tf → thetao/so).
    - `i7aof.remap` — vertical interpolation to `z_extrap` and horizontal remapping to the ISMIP grid.
    - `i7aof.extrap` — horizontal and vertical extrapolation using external Fortran executables; includes post-extrap conservative resampling (`z_extrap` → `z`).
    - `i7aof.biascorr` — classic bias-correction of CMIP ct/sa toward observational climatology.
    - `i7aof.time` — monthly → annual averaging utilities and a CMIP driver for annual means.

- Shared data and helpers
    - `i7aof.grid` — ISMIP grid specification and helpers.
    - `i7aof.vert` — vertical interpolation and conservative resampling utilities.
    - `i7aof.io` — NetCDF writing and CF time normalization; `i7aof.io_zarr` — Zarr append/finalize helpers.
    - `i7aof.coords` — coordinate helpers: attach grid coords, strip `_FillValue` on non-data, time-bounds handling.
    - `i7aof.topo` — topography datasets (download/preprocess/remap to ISMIP).
    - `i7aof.imbie` — IMBIE basin masks (download/generate on demand).
    - `i7aof.config` — configuration loader/merger (`default.cfg` + model/climatology/user configs).
    - `i7aof.cmip` and `i7aof.clim` — packaged model and climatology configs consumed by `i7aof.config`.
    - `i7aof.download` — simple HTTP download helpers with progress.

## Data flow (high level)

End-to-end CMIP path (per model; most steps run per scenario):

1) Split (optional) — cut very large CMIP monthly files into manageable chunks (`split`).
2) Convert — thetao/so → ct/sa on native grid (`convert`).
3) Remap — vertical to `z_extrap`, then horizontal to ISMIP grid (`remap`).
4) Extrapolate — fill shelves/cavities/ice/bathymetry using Fortran executables; then conservatively resample to `z` (`extrap`).
5) Bias-correct (classic) — run once per model + future scenario with a reference climatology; applies to historical and that future scenario (`biascorr`).
6) Thermal forcing — compute TF from bias-corrected CT/SA (`convert.ct_sa_to_tf`).
7) Annual means — compute weighted annual means for CT/SA/TF as needed (`time`).
8) Optional back-conversion — convert annual ct/sa/tf back to thetao/so (`convert.ct_sa_to_thetao_so`).

Climatology-only path (no time dimension):

1) Remap → 2) Extrapolate → 3) TF → 4) Optional back-conversion. The TF and back-convert CLIs have dedicated “clim” entry points.

I/O patterns and reliability:

- Zarr-first for chunked operations (append by time), then finalize to a single NetCDF with atomic rename; internal ready markers enable idempotent reruns.
- NetCDF writes use consistent CF time encoding (numeric days since 1850, declared calendar), per-variable `_FillValue` and optional compression, and unlimited `time` when present.

See package pages for developer details and the User Guide for end-to-end examples.
