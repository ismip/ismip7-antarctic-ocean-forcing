# i7aof.imbie

Purpose: IMBIE (Ice Sheet Mass Balance Inter-comparison Exercise) basin
downloads and masks on the ISMIP grid.

## Public Python API (by module)

Import paths and brief descriptions by module:

- Module: {py:mod}`i7aof.imbie.download`
  - {py:func}`download_imbie() <i7aof.imbie.download.download_imbie>`:
      Download the IMBIE2 basin shapefiles (no-op if already present).

- Module: {py:mod}`i7aof.imbie.masks`
  - {py:func}`make_imbie_masks() <i7aof.imbie.masks.make_imbie_masks>`:
      Generate merged basin masks on the ISMIP grid and write a NetCDF file.

## Required config options

Section: `[ismip_grid]` in your config (shared with grid package). Required keys:

- `dx` (float, meters): target horizontal spacing in x.
- `dy` (float, meters): target horizontal spacing in y.

Behavior:

- IMBIE masks are generated on the ISMIP grid derived from `dx`/`dy`.
- The ISMIP grid NetCDF must exist; if not, generate it using
  {py:func}`i7aof.grid.ismip.write_ismip_grid` before calling
  {py:func}`i7aof.imbie.masks.make_imbie_masks`.

## Outputs

- Path: `imbie/basinNumbers_{hres}.nc` (created if missing)
  - Example: `imbie/basinNumbers_10km.nc`
- Also downloads/ships:
  - `imbie/ANT_Basins_IMBIE2_v1.6/` folder (unzipped shapefiles)
  - Source ZIP cached at `imbie/ANT_Basins_IMBIE2_v1.6.zip`

## Data model

Variables written by {py:func}`i7aof.imbie.masks.make_imbie_masks`:

- `basinNumber(y, x)` — int; merged IMBIE basin index per cell.
- Coordinates: `x(x)`, `y(y)` — meters (projected, EPSG:3031).

Indexing convention: the function merges IMBIE sub-basins into named groups
(`A-Ap`, `Ap-B`, …, `K-A`) and assigns 0..N-1 to those groups in
iteration order.

## Runtime and external requirements

- Core: `numpy`, `xarray`, `shapely` (GEOS), `pyshp` (`shapefile`), `inpoly`, `scikit-fmm`, `tqdm`.
- Network/data: downloads IMBIE2 shapefiles from imbie.org and caches under `imbie/`.
- For the authoritative conda-forge environment, see `dev-spec.txt` (note `pyproject.toml` lists a PyPI-only subset).

## Usage

Minimal example using an MPAS-style config parser:

```python
from i7aof.grid.ismip import write_ismip_grid
from i7aof.imbie.masks import make_imbie_masks

# config is an mpas_tools.config.MpasConfigParser with [ismip_grid] section
write_ismip_grid(config)      # ensure grid exists
make_imbie_masks(config)      # writes imbie/basinNumbers_{hres}.nc
```

## Internals (for maintainers)

Implementation details in `i7aof/imbie/masks.py` (private helpers):

- `_get_basin_definitions()` — mapping of merged basin groups to sub-basins.
- `_load_ismip_grid(filename)` — reads `x/y` and constructs point cloud.
- `_load_basin_shapes(shapefile_path)` — reads shapefile records to Shapely.
- `_rasterize_basins(points, nx, ny, basins, in_basin_data)` — point-in-polygon
  rasterization using `inpoly2`.
- `_extend_basins_to_ocean(basin_number, num_basins)` — distance transform to
  assign nearest basin in ocean cells (via `skfmm.distance`).
- `_write_basin_mask(x, y, basin_number, filename)` — writes NetCDF via
  `i7aof.io.write_netcdf`.

## Edge cases / validations

- Existing files: `download_imbie()` and `make_imbie_masks()` are no-ops if the
  expected outputs already exist.
- Shapefile geometry: if a merged basin union is neither Polygon nor
  MultiPolygon, that basin is skipped with a warning.
- Grid alignment: relies on the ISMIP grid extents; ensure `dx`/`dy` match the
  grid.

## Extension points

- Expose overwrite flags to force regeneration of outputs.
- Parameterize the distance transform behavior (e.g., max ocean extension).
