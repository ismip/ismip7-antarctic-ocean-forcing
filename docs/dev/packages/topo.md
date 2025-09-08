# i7aof.topo

Purpose: Topography datasets and utilities for producing standardized surface,
draft, thickness, and bed layers on the ISMIP grid with consistent
area-fraction fields and ocean-masked variants.

## Public Python API (by module)

Import paths and brief descriptions by module:

- Module: {py:mod}`i7aof.topo`
  - {py:func}`get_topo() <i7aof.topo.get_topo>`: Factory returning a concrete
    topography class based on config option `[topo] dataset`.

- Module: {py:mod}`i7aof.topo.topo_base`
  - {py:class}`TopoBase <i7aof.topo.topo_base.TopoBase>`: Abstract base class
    with common helpers and validations for topography datasets.
    - {py:meth}`download_and_preprocess_topo() <i7aof.topo.topo_base.TopoBase.download_and_preprocess_topo>`:
      Subclass should download and preprocess, then call `super().download_and_preprocess_topo()`
      to validate fields.
    - {py:meth}`get_preprocessed_topo_path() <i7aof.topo.topo_base.TopoBase.get_preprocessed_topo_path>`:
      Return the pre-remap (source grid) file path.
    - {py:meth}`get_topo_on_ismip_path() <i7aof.topo.topo_base.TopoBase.get_topo_on_ismip_path>`:
      Return the final ISMIP-grid output path.
    - {py:meth}`remap_topo_to_ismip() <i7aof.topo.topo_base.TopoBase.remap_topo_to_ismip>`:
      Perform horizontal remap to ISMIP grid, then renormalize.
    - {py:meth}`renormalize_topo_fields(in_filename, out_filename) <i7aof.topo.topo_base.TopoBase.renormalize_topo_fields>`:
      Divide non-fraction fields by area-fractions with a small-fraction
      threshold.
    - {py:meth}`check() <i7aof.topo.topo_base.TopoBase.check>`: Ensure all
      expected variables exist (see Data model below).

- Module: {py:mod}`i7aof.topo.bedmachine`
  - {py:class}`BedMachineAntarcticaV3 <i7aof.topo.bedmachine.BedMachineAntarcticaV3>`:
    BedMachine Antarctica v3 ingestion and remap.
    - Implements the TopoBase abstract methods; requires a local copy of
      `BedMachineAntarctica-v3.nc`.

- Module: {py:mod}`i7aof.topo.bedmap`
  - {py:class}`Bedmap3 <i7aof.topo.bedmap.Bedmap3>`: Bedmap3 ingestion and
    remap, including automatic download from BAS.

## Required config options

Sections and keys used by this package (some are shared with other packages):

- `[topo]`
  - `dataset`: one of `bedmachine_antarctica_v3`, `bedmap3`.
  - `remap_method`: one of {'bilinear', 'neareststod', 'conserve'} — passed to
    the remap backend.
  - `renorm_threshold`: float in [0, 1]; minimum area-fraction for safe
    renormalization of non-fraction fields (values below become NaN).

- `[download]`
  - `quiet` (bool): suppress download progress for Bedmap3 source file.

- `[ismip_grid]` (shared with {py:mod}`i7aof.grid.ismip`)
  - `dx`, `dy` (meters): define horizontal resolution; used to compute the
    resolution string (e.g., `10km`). The ISMIP grid file must exist (create
    with {py:func}`i7aof.grid.ismip.write_ismip_grid`).

- `[remap]` (shared with {py:mod}`i7aof.remap`)
  - Common remap backend settings (e.g., `tool`, paths to ESMF/MOAB, `cores`,
    `parallel_exec`), as used by
    {py:func}`i7aof.remap.remap_projection_to_ismip`.

Behavior and constraints:

- BedMachine v3 cannot be auto-downloaded; place the file at
  `topo/BedMachineAntarctica-v3.nc` before running.
- Bedmap3 is auto-downloaded to `topo/bedmap3.nc`.

## Outputs

- Final ISMIP-grid topography:
  - Path: `topo/{name}_ismip_{hres}.nc`
    - Examples: `topo/bedmap3_ismip_10km.nc`,
      `topo/BedMachineAntarctica-v3_ismip_10km.nc`

- Intermediates (created or overwritten as needed):
  - Preprocessed source-grid file: `topo/intermediate/{name}_processed.nc`
  - Remapped (pre-renormalization): `topo/intermediate/{name}_remapped.nc`
  - Remap weights mapfile(s) under `topo/`, named like
    `map_{in_grid_name}_to_{out_mesh_name}_{method}.nc`

## Data model

Variables produced on the ISMIP grid (after remap and renormalization):

- Topography fields (meters):
  - `bed(y, x)` — bed elevation (below sea level negative).
  - `surface(y, x)` — ice surface elevation.
  - `thickness(y, x)` — ice thickness.
  - `draft(y, x)` — ice draft = `surface - thickness`.

- Ocean-masked variants (meters):
  - `ocean_masked_bed(y, x)` — bed where ocean is present, else 0.0.
  - `ocean_masked_surface(y, x)` — surface where ocean is present, else 0.0.
  - `ocean_masked_thickness(y, x)` — thickness where ocean is present, else 0.0.
  - `ocean_masked_draft(y, x)` — draft where ocean is present, else 0.0.

- Area fractions (unitless, in [0, 1]):
  - `ice_frac(y, x)` — any ice present (grounded or floating).
  - `ocean_frac(y, x)` — ocean present (open ocean or beneath shelf).
  - `grounded_frac(y, x)` — grounded ice share.
  - `floating_frac(y, x)` — floating ice share.
  - `rock_frac(y, x)` — bare rock share.

Coordinates and attributes:

- `x(x)`, `y(y)` — meters in EPSG:3031; on the ISMIP grid extents.
- CF conventions per {py:mod}`i7aof.grid.ismip` output; global attrs may include
  `Conventions`, projection metadata, etc.

## Runtime and external requirements

- Core: `numpy`, `xarray`.
- This package relies on other project components:
  - IO: {py:mod}`i7aof.io` for NetCDF writing.
  - Remapping: {py:mod}`i7aof.remap` (ESMF/MOAB via `pyremap`, `mpas-tools`).
  - Downloads (Bedmap3 only): {py:mod}`i7aof.download` plus `requests`, `tqdm`.
- CRS: EPSG:3031 for both source datasets and ISMIP target.
- For the authoritative conda-forge environment, see `dev-spec.txt` (note
  `pyproject.toml` lists a PyPI-only subset).

## Usage

Minimal end-to-end example using an MPAS-style config parser and a logger:

```python
from i7aof.grid.ismip import write_ismip_grid
from i7aof.topo import get_topo

# config is an mpas_tools.config.MpasConfigParser with required sections
topo = get_topo(config, logger)         # selects Bedmap3 or BedMachine v3
topo.download_and_preprocess_topo()     # fetch/preprocess + validate fields

# ensure ISMIP grid exists for the chosen dx/dy
write_ismip_grid(config)

# remap to ISMIP and renormalize; final path is:
topo.remap_topo_to_ismip()
print(topo.get_topo_on_ismip_path())
```

Notes:

- For BedMachine v3, place `BedMachineAntarctica-v3.nc` under `topo/` before
  calling `download_and_preprocess_topo()` (auto-download is not permitted).
- For Bedmap3, the source file is downloaded automatically.

## Internals (for maintainers)

High-level flow shared by both dataset classes:

1) Preprocess on source grid (`_preprocess_topo` in each class):
   - Derive `draft = surface - thickness` (meters).
   - Build area-fraction masks from dataset-specific `mask` coding.
   - Create `ocean_masked_*` variables with 0.0 outside ocean.
   - Preserve/propagate variable attributes where applicable.

2) Remap to ISMIP grid:
   - {py:func}`i7aof.remap.remap_projection_to_ismip` with `in_proj4='epsg:3031'`,
     `map_dir='topo'`, and `method=[topo].remap_method`.

3) Renormalize (`TopoBase.renormalize_topo_fields`):
   - Divide `draft`, `surface`, `thickness` by `ice_frac` and
     `ocean_masked_*` by `floating_frac`/`ocean_frac` as appropriate when the
     area fraction exceeds `[topo].renorm_threshold`; otherwise set to NaN.
   - Writes the final ISMIP file.

Validation:

- {py:meth}`TopoBase.check` enforces the presence of all expected variables:
  `bed`, `draft`, `surface`, `thickness`, `ocean_masked_*`, and all fraction
  fields noted above.

Dataset-specific notes:

- BedMachine v3 (`i7aof.topo.bedmachine`): uses integer `mask` (0: ocean,
  1: ice-free land, 2: grounded, 3: floating, 4: Lake Vostok) to construct
  fractions; requires manual placement of the source NetCDF.
- Bedmap3 (`i7aof.topo.bedmap`): renames `bed_topography`, `surface_topography`,
  `ice_thickness`; uses `mask` with values (FillValue: ocean, 1: grounded,
  2: transiently grounded shelf, 3: floating shelf, 4: rock). For `surface`
  and `thickness`, non-ice cells are set to 0.0 before remap.

## Edge cases / validations

- Missing source file (BedMachine): raises `FileNotFoundError` with guidance to
  manually download from the NSIDC data page.
- Existing outputs: current implementation overwrites intermediate and final
  files; consider adding overwrite flags if needed.
- Renormalization safety: values where area fractions are ≤ `renorm_threshold`
  become NaN to avoid division by small numbers.
- Grid consistency: ensure the ISMIP grid was generated with the same `dx/dy`
  you intend to use; otherwise remap target dims will not match expectations.
- Variable availability (Bedmap3): missing inputs are warned and skipped during
  preprocessing.

## Extension points

- Add new datasets by subclassing {py:class}`TopoBase` and implementing:
  `download_and_preprocess_topo`, `get_preprocessed_topo_path`,
  `get_topo_on_ismip_path`, and `remap_topo_to_ismip`.
- Extend `i7aof.topo.get_topo` to recognize the new dataset key.
- Add optional overwrite flags and expose EPSG/projection strings if future
  datasets differ from EPSG:3031.
