# IMBIE Polygon Export

This page documents how to export combined + extended IMBIE basin polygons
from the rasterized basin masks.

## Recommended command

The recommended settings for a robust, simplified output are:

```bash
ismip7-antarctic-imbie-polygons \
  --simplify-tolerance-m 16e3 \
  --min-hole-area-m2 0 \
  --validate
```

These options:

- simplify basin boundaries with a 16 km tolerance,
- remove all extension holes before final topology repair,
- validate that final exported basins are geometrically valid and do not
  contain unacceptable overlaps.

## Inputs and outputs

Input prerequisites:

- A valid working directory (`[workdir] base_dir`) with topography and ISMIP
  grid configuration in your config.
- IMBIE2 shapefiles are downloaded automatically if missing.
- Rasterized extended basin masks are generated automatically if missing.

Primary output:

- `imbie2/extended_basin_polygons.shp` (plus `.shx`, `.dbf`, `.prj`)

## Debug outputs

The command also supports debug exports for troubleshooting polygon topology:

- `--debug-raster-only-shapefile <PATH>`
  - Writes polygons built directly from rasterized extended basins without
    unioning with original IMBIE polygons.
- `--debug-raster-only-simplified-shapefile <PATH>`
  - Same as above, but includes simplification and hole-removal settings.

Example:

```bash
ismip7-antarctic-imbie-polygons \
  --simplify-tolerance-m 16e3 \
  --min-hole-area-m2 0 \
  --debug-raster-only-shapefile imbie2/raster_only_debug.shp \
  --debug-raster-only-simplified-shapefile imbie2/raster_only_simplified_debug.shp \
  --validate
```

## Notes on topology behavior

The export pipeline enforces a shared-boundary partition for raster-derived
extensions, then combines with exact original IMBIE polygons basin-by-basin:

1. Build repaired extension polygons from raster masks.
2. Subtract the union of all original IMBIE polygons from each extension basin.
3. Union each cleaned extension basin with the exact original basin polygon.

This preserves original IMBIE geometry while producing clean extended basins.
