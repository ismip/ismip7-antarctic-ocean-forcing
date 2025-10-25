# CMIP Inputs

Guidance on obtaining and preparing CMIP6/CMIP7 data for `i7aof`.

- Variables: `thetao`, `so`, and `zos`.
- Convert `thetao`/`so` to TEOS-10 `ct`/`sa` on the native grid first using
  `i7aof.convert.cmip_to_ct_sa` or the
  `ismip7-antarctic-convert-cmip-to-ct-sa` CLI, then
  remap to ISMIP grids with `i7aof.remap.cmip` or the
  `ismip7-antarctic-remap-cmip` CLI.
  After bias correction, compute TF with `ismip7-antarctic-cmip-ct-sa-to-tf`.
- Configure paths and patterns in your `.cfg` files.

## Tips

- Time chunking can significantly impact performance. For CESM2-WACCM,
  12-month chunks have shown ~25% speedup vs 1-month chunks. Configure via
  `[convert_cmip] time_chunk = 12` in your model config.
- Set `I7AOF_DEBUG_TEOS10=1` to print profiling info for the TEOS-10 step if
  you need to troubleshoot performance.

## Post-extrap vertical resampling

After running `ismip7-antarctic-extrap-cmip`, a conservative resampling step
maps `ct`/`sa` from `z_extrap` to `z` (20 m → 60 m by default). This now uses a
Zarr-first workflow that appends results by time chunk and converts once to
NetCDF, significantly improving performance and reducing memory usage.

- Configure the resampling time chunk length with
  `[extrap_cmip] time_chunk_resample` in your config.
- Final outputs use the `ismip<hres>_<dz>` resolution tag (or `_z` suffix if
  the tag would be unchanged). A temporary `<basename>.zarr/` folder appears
  during processing and is cleaned up automatically.
