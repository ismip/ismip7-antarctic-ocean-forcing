# CMIP Inputs

Guidance on obtaining and preparing CMIP6/CMIP7 data for `i7aof`.

- Variables: `thetao`, `so`, and `zos`.
- Convert `thetao`/`so` to TEOS-10 `ct`/`sa` on the native grid first using
  `i7aof.convert.cmip` or the `ismip7-antarctic-convert-cmip` CLI, then
  remap to ISMIP grids with `i7aof.remap.cmip` or the
  `ismip7-antarctic-remap-cmip` CLI.
- Configure paths and patterns in your `.cfg` files.

## Tips

- Time chunking can significantly impact performance. For CESM2-WACCM,
  12-month chunks have shown ~25% speedup vs 1-month chunks. Configure via
  `[convert_cmip] time_chunk = 12` in your model config.
- Set `I7AOF_DEBUG_TEOS10=1` to print profiling info for the TEOS-10 step if
  you need to troubleshoot performance.
