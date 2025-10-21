## Climatology Workflows

The package supports remapping and extrapolating several observational
temperature/salinity climatologies (provided privately by Shenjie Zhou on
30-Sep-2025). These are currently preliminary and not yet published; the
metadata and provenance will be expanded once a public release occurs.

### Supported Climatologies

Each climatology is selected with a `clim_name` passed to the remap or
extrapolation CLI/ API. Internally, a configuration file under
`i7aof/clim/*.cfg` supplies variable/dimension names and the relative
input path (under the ISMIP base input directory) for that climatology.

| clim_name | Description (brief) | Source file (relative) |
|-----------|---------------------|------------------------|
| `zhou_annual_30_sep` | Annual mean over full observational period (exact years TBD) | `Updated_TS_Climatology/OI_Climatology.nc` |
| `zhou_summer_30_sep` | Summer-only (austral summer months) mean over full period | `Updated_TS_Climatology/OI_summer_Climatology.nc` |
| `zhou_2000_annual_30_sep` | Annual mean restricted to observations from 2000 onward | `Updated_TS_Climatology/OI_2000_Climatology.nc` |

Variables provided: conservative temperature `ct` and absolute salinity
`sa` (with optional *_mse uncertainty fields if present in future
updates). The raw vertical coordinate is pressure (dbar) and is
converted to a monotonically increasing height coordinate (meters,
positive up) during preprocessing.

### Remapping

Remapping performs two stages:
1. Vertical pipeline: mask invalid cells, interpolate to the ISMIP
	 `z_extrap` levels, and vertically renormalize using a valid fraction
	 mask.
2. Horizontal remap to the ISMIP projected grid (bilinear, conservative,
	 or nearest-stod per `[remap] method`).

Command-line example:

```bash
ismip7-antarctic-remap-clim \
	--clim zhou_annual_30_sep \
	--config my-config.cfg
```

Output path pattern:
```
<workdir>/remap/climatology/<clim_name>/<original>_ismip<res>.nc
```

### Extrapolation

Climatology extrapolation adds a dummy singleton `time` dimension (size
1) so the legacy Fortran extrapolation executables (which assume a time
axis) can be reused. This dimension is removed again in the finalized
output.

```bash
ismip7-antarctic-extrap-clim \
	--clim zhou_annual_30_sep \
	--config my-config.cfg
```

Outputs (one per variable):
```
<workdir>/extrap/climatology/<clim_name>/<remapped_stem>_<var>_extrap.nc
```

### Post-extrap vertical resampling

After extrapolation, a conservative resampling step maps `z_extrap` to `z`
levels (from 20 m to 60 m by default). The workflow uses the same Zarr-first
approach as CMIP (single write without time appends) and converts to a final
NetCDF alongside the Extrap output. Dimension order is `(z, y, x)` for
climatologies (no time).

### Dimension Ordering

Climatology remap outputs enforce variable dimension order
`(z_extrap, y, x)` prior to extrapolation. During extrapolation a dummy
`time` is inserted as the most slowly varying dimension resulting in
`(time, z_extrap, y, x)`—mirroring CMIP workflow inputs so the Fortran
code reads the data consistently.

### Configuration Keys

Each `i7aof/clim/zhou_*.cfg` file defines:

```
[climatology]
lat_var, lon_var, lat_dim, lon_dim
lev_var, lev_dim (pressure) -> converted to lev (meters)
filename (relative path under input base dir)
ct_var, sa_var (variable names)
```

User overrides (e.g. input base directory) can be supplied through a
user config passed with `--config` or via the Python API.

### Planned Enhancements

- Publish provenance and observational period ranges.
- Support inclusion of mean-square-error or uncertainty variables when
	provided.
- Add validation utilities for vertical coordinate consistency.

### Python API

Programmatic usage mirrors the CMIP workflows:

```python
from i7aof.remap.clim import remap_climatology
from i7aof.extrap.clim import extrap_climatology

remap_climatology('zhou_annual_30_sep', user_config_filename='my.cfg')
extrap_climatology('zhou_annual_30_sep', user_config_filename='my.cfg')
```

Refer also to {doc}`workflows` for integrating climatology products in a
full CMIP+climatology processing chain.

