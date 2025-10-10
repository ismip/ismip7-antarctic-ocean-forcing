# i7aof.vert

Purpose: Vertical interpolation utilities to map source profiles onto the
ISMIP vertical coordinates (`z` or `z_extrap`), with mask-aware
renormalization.

## Public Python API (by module)

Import paths and brief descriptions by module:

- Module: {py:mod}`i7aof.vert.interp`
  - {py:class}`VerticalInterpolator <i7aof.vert.interp.VerticalInterpolator>`:
    End-to-end helper for masking, interpolating to ISMIP vertical coords, and
    renormalizing by valid-source fraction.
    - Attributes (selected): `src_valid`, `src_coord`, `dst_coord`,
      `src_frac_interp`, `z_src`, `threshold`.
    - {py:meth}`mask_and_sort(da) <i7aof.vert.interp.VerticalInterpolator.mask_and_sort>`:
      Apply source-valid mask and ensure the source coordinate is aligned.
    - {py:meth}`interp(da_masked) <i7aof.vert.interp.VerticalInterpolator.interp>`:
      Interpolate to `z` or `z_extrap` using linear interpolation with
      extrapolation at edges.
    - {py:meth}`normalize(da_interp) <i7aof.vert.interp.VerticalInterpolator.normalize>`:
      Divide by `src_frac_interp` where the fraction exceeds a threshold,
      otherwise set to NaN.
  - {py:func}`fix_src_z_coord() <i7aof.vert.interp.fix_src_z_coord>`:
    Ensure the source vertical coordinate is positive up and in meters,
    applying the same to bounds; returns corrected `z` and `z_bnds`.

- Module: {py:mod}`i7aof.vert.resamp`
  - {py:class}`VerticalResampler <i7aof.vert.resamp.VerticalResampler>`:
    Conservative vertical resampling using layer-overlap weights, designed
    to map intensive fields (e.g., CT and SA) from `z_extrap` to `z`. The
    resampler computes per-layer overlap thickness between source and
    destination bounds and returns a thickness-weighted mean over valid
    source contributions, normalized by the valid overlap. A coverage
    threshold is applied from `[vert_interp] threshold`.
    - Typical usage: post-extrapolation coarsening (e.g., 20 m → 60 m) on
      the ISMIP grid to deliver standard `z`-level outputs.

Note: package `__init__.py` currently has no public re-exports.

## Required config options

Sections and keys used by this package:

- `[vert_interp]`
  - `threshold` (float): minimum valid-source fraction to allow
    renormalization; values below are treated as invalid (NaN).
  - Used by both the linear interpolator and the conservative resampler.

- `[ismip_grid]` (shared with {py:mod}`i7aof.grid.ismip`)
  - `dz`, `dz_extrap` define the vertical coordinates written to the ISMIP
    grid file, which is then read by the interpolator. Ensure the grid file
    exists via {py:func}`i7aof.grid.ismip.write_ismip_grid`.

```{note}
Typically, we will use `z_extrap` since this interpoation is performed
before extrapolation into invalid regions.  The higher resolution `z_extrap`
coordinate helps ensure that bed topography is represented more accurately
during extrapolation. At higher vertical resolution, troughs can allow water
masses to  enter ice-shelf cavities or sills can block them in ways that would
be missed at coarser vertical resolution.
```


## Outputs

- No files are written by this package directly. The interpolator returns
  in-memory {py:class}`xarray.DataArray` results and stores
  `src_frac_interp` on the instance for reuse.
  The resampler similarly returns in-memory results; workflow drivers write
  the resampled files.

## Data model

- Source inputs:
  - `src_valid(..., z_src, ...)` — boolean mask of valid source layers.
  - `src_coord` (str) — name of the source vertical coordinate in your data.
  - Data arrays to interpolate must share `src_coord` and dims compatible with
    `src_valid` (extra dims like `time` allowed).

- Destination coordinates (from ISMIP grid):
  - `dst_coord` is `'z'` or `'z_extrap'`; 1D, units `m`, `positive='up'`.

- Intermediate/outputs:
  - `src_frac_interp(..., dst_coord, ...)` — fraction of valid source overlap
    after vertical interpolation; reused for normalization.
  - `interp(...)` returns a DataArray on `dst_coord` with original attrs; the
    source vertical coord is dropped.
  - `normalize(...)` divides by `src_frac_interp` where > `threshold`, else
    sets values to NaN; attrs preserved.

## Runtime and external requirements

- Core: `xarray`.
- Internal: {py:mod}`i7aof.grid.ismip` to read the ISMIP grid file for `z` and
  `z_extrap` coordinates.
- For environment details, see `dev-spec.txt`.

## Usage

Minimal example using a dataset `ds` with a vertical coordinate `lev` and
valid mask `valid_mask` (True where values are valid):

```python
import xarray as xr
from i7aof.grid.ismip import write_ismip_grid
from i7aof.vert.interp import VerticalInterpolator, fix_src_z_coord

# ensure ISMIP grid exists with desired vertical coords
write_ismip_grid(config)

# fix source z coordinate to meters and positive-up
z_src, z_bnds_src = fix_src_z_coord(ds, z_coord='lev', z_bnds='lev_bnds')
ds = ds.assign_coords(lev=z_src)

# build a valid-data mask (boolean). Example: temperature not NaN
valid_mask = ds['thetao'].notnull()

vi = VerticalInterpolator(
    src_valid=valid_mask,
    src_coord='lev',
    dst_coord='z_extrap',
    config=config,
)

da_masked = vi.mask_and_sort(ds['thetao'])
da_interp = vi.interp(da_masked)
da_norm = vi.normalize(da_interp)

# src_frac_interp can be reused across time steps for same grid
src_frac = vi.src_frac_interp

Conservative resampling from `z_extrap` to `z` (intensive fields):

```python
from i7aof.vert.resamp import VerticalResampler

z_src = ds_ismip['z_extrap']
src_valid = xr.DataArray(
  np.ones_like(z_src, dtype=np.float32), dims=('z_extrap',), coords={'z_extrap': z_src}
)
vr = VerticalResampler(
  src_valid=src_valid, src_coord='z_extrap', dst_coord='z', config=config
)
ct_z = vr.resample(ds_extrap['ct'])
sa_z = vr.resample(ds_extrap['sa'])
```
```

## Internals (for maintainers)

- The interpolator reads the destination vertical coordinate from the ISMIP
  grid written by {py:mod}`i7aof.grid.ismip` and computes an interpolated
  `src_frac_interp` by remapping a binary validity mask (1/0) with linear
  interpolation and edge extrapolation.
- `mask_and_sort` aligns `src_coord` values and applies `src_valid`. Attributes
  are preserved on the data variable.
- `interp` uses {py:meth}`xarray.DataArray.interp` with
  `kwargs={'fill_value': 'extrapolate'}` and drops the source coord from the
  result, preserving attrs.
- `normalize` applies `src_frac_interp > threshold` and divides by the masked
  fraction so state variables (e.g., temperature) aren’t biased low by partial
  overlap; values below threshold become NaN.
- {py:func}`fix_src_z_coord` flips sign if `positive='down'` and converts units
  from centimeters to meters for both coord and bounds; sets `units='m'` and
  `positive='up'` on the coord; bounds attrs cleared (not required by CF).

- Resampler computes an overlap matrix `w[i,j]` between destination and source
  layer bounds and evaluates `sum_j w[i,j] * x_j * valid_j / sum_j w[i,j] * valid_j`.
  The coverage fraction `f_i = (sum_j w[i,j] * valid_j) / dz_dst[i]` is compared
  to the threshold; layers below threshold are set to invalid (NaN). The method
  is conservative for integrals and appropriate for intensive variables.

## Edge cases / validations

- Source vertical coordinate may be positive-down or in centimeters; use
  `fix_src_z_coord` to standardize before constructing the interpolator.
- If `src_frac_interp` is 0 or below the `threshold`, outputs become NaN to
  avoid dividing by small numbers.
- Time dimension: `src_valid` can omit `time`; broadcasting across time is
  supported as long as non-time dims match.
- Extrapolation: linear extrapolation is used at the destination edges; ensure
  this is acceptable for your variable.

## Extension points

- Add alternate interpolation schemes or slope-limited/exact-integration
  options by extending `VerticalInterpolator`.
- Expose different thresholds per variable or pass-in a custom mask for
  normalization instead of `src_frac_interp`.
