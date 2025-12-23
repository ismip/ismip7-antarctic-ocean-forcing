# Extrapolation

Fill spatial gaps and extend fields vertically on the ISMIP grid. This page
covers the extrapolation steps for both CMIP and climatology and how to tune
them without duplicating the full {doc}`workflows` page.

## What gets extrapolated

- CMIP path (per scenario): `ismip7-antarctic-extrap-cmip`
- Climatology path (once): `ismip7-antarctic-extrap-clim`

Inputs are remapped CT/SA on the ISMIP grid with a dense `z_extrap` vertical
axis.

## Algorithm overview

1. Horizontal “fill” to extend fields into masked/coastal regions while honoring
   available neighbors and masks.
2. Vertical extrapolation to populate missing values in the water column.
3. Conservative resampling from dense `z_extrap` to coarser `z` (e.g., 20 m →
   60 m) for downstream steps.

Climatology processing temporarily adds a singleton `time` dimension for
compatibility with the Fortran executables; it is removed afterward.

## Configuration keys

```
[extrap_cmip]
time_chunk = 12
# Resample z_extrap → z after extrapolation
 time_chunk_resample = 12
```

Tips:

- Increase `time_chunk` cautiously—extrapolation is memory intensive.
- If resampling is slow, try adjusting `time_chunk_resample`.

## Outputs

```
# CMIP
<workdir>/extrap/<model>/<scenario>/Omon/ct_sa/*_{ct,sa}_extrap_*.nc

# Climatology
<workdir>/extrap/climatology/<clim>/*_{ct,sa}_extrap.nc
<workdir>/extrap/climatology/<clim>/*_z.nc
```

CMIP filenames retain monthly time ranges; climatology files are static.

## Validation checklist

- No large voids along ice shelf fronts or cavities after extrapolation.
- Vertical profiles remain physically reasonable (no spurious inversions).
- z→z_extrap resampling preserves large‑scale structure; check a few columns.

## Common pitfalls

- Running extrapolation before remap (order matters).
- Forgetting the z resample step (done automatically in the CMIP workflow, and
  included in the climatology step here).

## Minimal examples

```python
from i7aof.extrap.cmip import extrap_cmip
from i7aof.extrap.clim import extrap_climatology

extrap_cmip('CESM2-WACCM', 'historical', user_config_filename='my.cfg')
extrap_climatology('zhou_annual_06_nov', user_config_filename='my.cfg')
```
