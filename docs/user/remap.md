# Remapping

This page focuses on how remapping works in `i7aof` and what you can configure.
It complements the end‑to‑end {doc}`workflows` page without repeating it.

## What is remapped and when

- CMIP CT/SA after TEOS‑10 conversion (per scenario): `ismip7-antarctic-remap-cmip`
- Climatology CT/SA (once): `ismip7-antarctic-remap-clim`

Both paths prepare the vertical coordinate to a dense `z_extrap` before the
horizontal remap to the ISMIP grid.

## ISMIP grid definitions

Grid descriptions, coordinates, and bounds are provided by `i7aof.grid.ismip`.
The target grid is specified in your config, and filenames include an
`ismip<res>` tag (e.g., `ismip8km`).

## Methods and when to use them

The remap method is selected via `[remap] method` in your config and may differ
by variable:

- Bilinear: good default for smoothly varying `ct` and `sa`.
- Conservative: use when integral preservation across cell areas matters.
- Nearest‑stod: fallback for categorical masks or extremely sparse regions.

Mask handling: invalid cells are masked prior to interpolation; a valid‑fraction
normalization is applied during the vertical stage to minimize edge artifacts.

## Vertical preparation (to `z_extrap`)

- Convert the model/climatology vertical coordinate to a monotonic height axis.
- Interpolate to dense `z_extrap` levels to support robust extrapolation later.
- Enforce dimension order `(z_extrap, y, x)` for consistency.

## Configuration keys

```
[remap_cmip]
vert_time_chunk = 1
horiz_time_chunk = 120
method = bilinear        # or conservative, nearest

[remap_clim]
method = bilinear
```

Tune `horiz_time_chunk` to balance memory and throughput; keep
`vert_time_chunk = 1` unless vertical processing becomes dominant.

**Note**: For now, it is not recommended that you change the remapping method,
since we have only tested bilinear and there is reason to think that the
other methods would lead to unphysically blocky results.

## Outputs

```
<workdir>/remap/<source>/<tag>/.../*_{ct,sa}_remap.nc           # CMIP
<workdir>/remap/climatology/<clim>/*_ismip<res>.nc              # Climatology
```

## Validation checklist

- Coordinates carry ISMIP grid bounds; CF attributes preserved.
- No large holes introduced near complex coastlines (inspect masks).
- Values are within physical ranges; spot‑check against native grid statistics.

## Tips

- Use bilinear for speed unless conservation is essential.
- If performance is I/O‑bound, increase `horiz_time_chunk`; if memory spikes,
  reduce it.

See also: {doc}`clim` for recommended climatology choices and {doc}`cmip` for
CMIP input preparation.
