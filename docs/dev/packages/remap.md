# i7aof.remap

Purpose: Utilities and workflows to remap source data (e.g., CMIP) to the
ISMIP grid with vertical interpolation/normalization and horizontal remapping.
For CMIP workflows, inputs are expected to be the pre-converted ct/sa files
produced by {py:mod}`i7aof.convert.cmip` and written under the workdir.

## Public Python API (by module)

Import paths and brief descriptions by module:

- Module: {py:mod}`i7aof.remap`
  - {py:func}`remap_projection_to_ismip() <i7aof.remap.remap_projection_to_ismip>`:
      Remap a dataset defined by a projection string to the ISMIP grid.
  - {py:func}`remap_lat_lon_to_ismip() <i7aof.remap.remap_lat_lon_to_ismip>`:
      Remap a lat-lon dataset (1D/2D lon/lat) to the ISMIP grid.
  - {py:func}`add_periodic_lon() <i7aof.remap.add_periodic_lon>`: Add a
      periodic longitude column when needed (avoid dateline seam).

- Module: {py:mod}`i7aof.remap.cmip`
  - {py:func}`remap_cmip() <i7aof.remap.cmip.remap_cmip>`: Orchestrate per-file
    vertical interpolation/normalization and horizontal remapping of CMIP ct/sa
    monthly data to the ISMIP grid.

## Required config options

These sections/keys are used; defaults come from package configs, but user
overrides are typical:

- `[workdir] base_dir` — required unless passed as `workdir` arg.
- `[inputdir] base_dir` — required unless passed as `inputdir` arg.
- `[remap]`
  - `method`: one of {'bilinear', 'neareststod', 'conserve'}
  - `cores`: integer task count for remapping tools
  - `tool`: backend ('esmf' or 'moab')
  - `esmf_path`, `moab_path`, `parallel_exec`: paths/exec or 'None'
  - `threshold`: float; horizontal renormalization threshold
- `[cmip_dataset]`
  - `lon_var`, `lat_var`, `lon_dim`: variable/dimension names on input
- `[remap_cmip]`
  - `vert_time_chunk`: int; time chunk for vertical steps
  - `horiz_time_chunk`: int; time chunk for horizontal remap
  (Inputs are discovered from the convert step; you no longer need to specify
  per-scenario file lists here.)

## Outputs

- Remapped monthly files under:
  `remap/{model}/{scenario}/Omon/{variable}/..._ismip{hres_vres}.nc`
- Temporary intermediates per input file (cleaned automatically):
  - `tmp_vert_interp_*` (mask, interp, normalized)
  - `tmp_horiz_remap_*` (mask and time-chunk outputs)

## Data model

- Carries source time/lon/lat bounds through both stages.
- Adds `src_frac_interp` from the vertical stage; reused across time for
  horizontal stage and copied into final output.
- ISMIP vertical coordinate `z_extrap_bnds` attached after vertical interp.

## Runtime and external requirements

- Core: `xarray`, `numpy`, `pyremap`, `mpas-tools` (config/logging).
- Tools: ESMF/ESMPy or MOAB stack via pyremap backends; see `dev-spec.txt`.
- Internal APIs: grid helpers, IO helpers, vertical interpolator.

## Usage

Remap a model and scenario (using default package configs plus a user config
for workdir). Conversion must be run first to generate ct/sa inputs under
`convert/<model>/<scenario>/Omon/ct_sa/`.

```python
from i7aof.remap.cmip import remap_cmip

remap_cmip(
  model='CESM2-WACCM',
  scenario='historical',
  user_config_filename='my-config.cfg',
)
```

## Internals (for maintainers)

- Vertical pipeline (`_vert_mask`, `_vert_interp`, `_vert_normalize`):
  masking → interp to `z_extrap` → renormalization using `src_frac_interp`.
- Horizontal remap (`_remap_horiz`):
  optional periodic lon → remap ancillary mask once → remap data in time
  chunks with renormalization → concat and attach `src_frac_interp` → write.
- Map-file naming: `map_{in_grid_name}_to_{out_mesh_name}_{method}.nc`.

## Edge cases / validations

- Input grid must be native (filenames contain 'gn'); otherwise error.
- For bilinear remapping, a periodic lon is added if needed to avoid seams.
- If output files exist, both vertical and horizontal interpolation are
  skipped.
- Strict scenario variable lists must be provided in config.

## Extension points

- Support additional variables and alternative source grids.
