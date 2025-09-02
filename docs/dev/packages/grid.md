# i7aof.grid

Purpose: define the ISMIP grid and helpers for writing grid NetCDF files and
for computing resolution strings used elsewhere.

```{note}
This page follows a structure you can reuse in documenting other packages:
Purpose → Public Python API → Required config options → Outputs → Data model →
Runtime and external requirements → Usage → Internals → Edge cases →
Extension points.
```

## Public Python API (by module)

Import paths and brief descriptions by module:

- Module: {py:mod}`i7aof.grid.ismip`
  - {py:func}`write_ismip_grid() <i7aof.grid.ismip.write_ismip_grid>`:
    Create the ISMIP grid file if it doesn’t already exist.
  - {py:func}`get_ismip_grid_filename() <i7aof.grid.ismip.get_ismip_grid_filename>`:
    Return the output grid filename based on resolution.
  - {py:func}`get_horiz_res_string() <i7aof.grid.ismip.get_horiz_res_string>`: e.g., "10km"
  - {py:func}`get_ver_res_string() <i7aof.grid.ismip.get_ver_res_string>`: e.g., "50m"
  - {py:func}`get_res_string() <i7aof.grid.ismip.get_res_string>`: e.g., "10km_50m"

Note: implementation for this package currently lives in the module {py:mod}`i7aof.grid.ismip`.

## Required config options

Section: `[ismip_grid]` in your config. Required keys:

- `dx` (float, meters): target horizontal spacing in x.
- `dy` (float, meters): target horizontal spacing in y.
- `dz` (float, meters): vertical layer thickness used for the `z` coordinate.
- `dz_extrap` (float, meters): vertical layer thickness used for the
  `z_extrap` coordinate.

Constraints and behavior:

- The grid spans 6080 km × 6080 km (EPSG:3031) with fixed extents. The number
  of points along x/y is derived from dx/dy, then the exact dx/dy are adjusted
  to fit the domain endpoints.
- If `dx`, `dy`, `dz`, or `dz_extrap` are missing, a KeyError/Config error will
  be raised by the caller or during access.


```{note}
The `z` coordinate is used for the final version of the datasets provided to
ice-sheet modelers.  It is typically coarser (60 m spacing by default) than
the extrapolation coordinate (`z_extrap` has 20 m spacing by default).  The
higher resolution for `z_extrap` helps ensure that bed topography is
represented more accurately during extrapolation. At higher resolution,
troughs can allow water masses to enter ice-shelf cavities or sills can block
them in ways that would be missed at coarser vertical resolution.
```

## Outputs

- Path: `ismip/ismip_{hres}_{vres}_grid.nc` (created if missing)
  - Example: `ismip/ismip_10km_50m_grid.nc`
- Conventions: CF-1.10; `crs` variable with EPSG:3031 metadata.

## Data model (coordinates, dims, attributes)

Coordinates and dimensions written by `write_ismip_grid`:

- Horizontal:
  - `x`: 1D, length `nx`; units `m`; attrs include `axis='X'`, `bounds='x_bnds'`.
  - `y`: 1D, length `ny`; units `m`; attrs include `axis='Y'`, `bounds='y_bnds'`.
  - `x_bnds`: shape `(nx, 2)`; `y_bnds`: shape `(ny, 2)`.
  - `lon`, `lat`: 2D, shape `(y, x)`; degrees east/north; bounds
    `lon_bnds`, `lat_bnds` with shape `(y, x, 4)` (cell corners).
  - `crs`: scalar DataArray with projection metadata (EPSG:3031).

- Vertical (two coordinates):
  - `z` and `z_extrap`: 1D, length `nz` and `nz_extrap` respectively; units `m`;
    `positive='up'`; bounds `z_bnds` and `z_extrap_bnds` with shape `(n, 2)`.

Global attrs include `Conventions='CF-1.10'`, projection strings and metadata.

## Runtime and external requirements

- Core: `numpy`, `xarray`, `pyproj`; writes via `i7aof.io.write_netcdf`.
- CRS: EPSG:3031 (Antarctic Polar Stereographic); requires PROJ data at runtime.
- For the authoritative conda-forge environment, see `dev-spec.txt` (note `pyproject.toml` lists a PyPI-only subset).

## Usage

Minimal example using an MPAS-style config parser (keys listed above):

```python
from i7aof.grid.ismip import write_ismip_grid, get_ismip_grid_filename

# config is an mpas_tools.config.MpasConfigParser with [ismip_grid] section
write_ismip_grid(config)
print(get_ismip_grid_filename(config))
```

## Internals (for maintainers)

Implementation details (private helpers; not part of the public API):

- `_add_horiz_grid(ds, config)` — compute `x/y`, `lon/lat`, bounds, CRS.
- `_add_vert_levels(ds, coord_name, dz)` — construct vertical midpoints and bounds.

The package `__init__.py` is currently empty (no public re-exports). Consider
re-exporting the primary functions there if a flatter import path is desired.

## Edge cases / validations

- Existing file: `write_ismip_grid` is a no-op if the output file already exists.
- Horizontal resolution rounding: `nx`, `ny` are derived from reference size;
  actual `dx`, `dy` may differ slightly from requested `ismip_grid.dx/dy` to
  fit exact domain endpoints.
- Longitude normalization: longitudes are wrapped to `[-180, 180)`.

## Extension points

- Expose `write_ismip_grid(..., overwrite=True)` behavior (optional) if
  regeneration is needed.


