# Architecture Overview

This section sketches the intended layout and workflow graph. The project is
evolving; details may change.

## Packages

- `i7aof.biascorr` — bias-correction time-slice and projection methods.
- `i7aof.grid` — grid definitions and helpers (e.g., ISMIP grid spec).
- `i7aof.imbie` — IMBIE masks and downloads.
- `i7aof.io` — shared IO utilities and config parsing.
- `i7aof.remap` — routines to remap CMIP variables to ISMIP grids.
- `i7aof.topo` — topography datasets and utilities.
- `i7aof.vert` — vertical interpolation utilities.

## Data flow (high level)

1. Download inputs (CMIP monthly ocean data, topography, IMBIE basins,
   observationally derived climatologies).
2. Remap monthly model data to ISMIP grid.
3. Compute annual means.
4. Extrapolate to shelves, cavities, ice, and bathymetry.
5. Apply bias correction using observational climatology.
6. Package outputs for downstream ice-sheet models.

See module pages for details and examples.
