# Climatology Workflows

This page covers the climatology-only pipeline. No CMIP inputs are required.
You can run the full end-to-end processing for an observational
climatology in four steps: Remap → Extrapolate → Thermal Forcing → Back‑convert.

## Source

The climatologies were provided by Zhou and collaborators, and are similar
to (but updated from) the publicly available dataset
[Zhou et al. (2024)](https://doi.org/10.17882/103946).



## Supported Climatologies (prefer v2: 06_nov)

Select a climatology with `--clim <clim_name>` on the CLIs. Each
`clim_name` corresponds to a config file under `i7aof/clim/*.cfg` that
defines variable/dimension names and the relative input path under your
input base directory.

Recommended (v2, 06_nov):

- `zhou_annual_06_nov` → `Updated_TS_Climatology/OI_Climatology_v2/OI_Climatology.nc`
- `zhou_summer_06_nov` → `Updated_TS_Climatology/OI_Climatology_v2/OI_summer_Climatology.nc`
- `zhou_2000_annual_06_nov` → `Updated_TS_Climatology/OI_Climatology_v2/OI_2000_Climatology.nc`

Also available (earlier 30_sep set):

- `zhou_annual_30_sep` → `Updated_TS_Climatology/OI_Climatology.nc`
- `zhou_summer_30_sep` → `Updated_TS_Climatology/OI_summer_Climatology.nc`
- `zhou_2000_annual_30_sep` → `Updated_TS_Climatology/OI_2000_Climatology.nc`

Variables provided: conservative temperature `ct` and absolute salinity `sa`,
with mean‑square‑error fields `ct_mse` and `sa_mse` in v2. The raw vertical
coordinate is pressure (dbar) and is converted to height (meters, positive up)
during preprocessing.

## Remap (Step 3a)

Two stages are performed: (1) vertical preparation to the ISMIP `z_extrap`
levels with masking and normalization, then (2) horizontal remapping to the
ISMIP grid (method set by `[remap] method`).

CLI:

```bash
ismip7-antarctic-remap-clim \
  --clim zhou_annual_06_nov \
  --config my.cfg
```

Job script: `example_job_scripts/03_remap/job_script_remap_clim.bash`

Outputs:

```
<workdir>/remap/climatology/<clim_name>/*_ismip<res>.nc
```

## Extrapolate + Resample (Step 4a)

We fill spatial gaps horizontally then vertically (Fortran routines). A dummy
singleton `time` dimension is added for processing and removed in the final
outputs. After extrapolation, we conservatively resample from `z_extrap` to `z`
levels (e.g., 20 m → 60 m).

CLI:

```bash
ismip7-antarctic-extrap-clim \
  --clim zhou_annual_06_nov \
  --config my.cfg
```

Job script: `example_job_scripts/04_extrap/job_script_extrap_clim.bash`

Outputs:

```
<workdir>/extrap/climatology/<clim_name>/*_{ct,sa}_extrap.nc
<workdir>/extrap/climatology/<clim_name>/*_z.nc
```

Dimension order for final climatology products is `(z, y, x)` (no time).

Note: Remap outputs are `(z_extrap, y, x)`. During extrapolation a dummy `time`
is inserted to form `(time, z_extrap, y, x)` for the Fortran step.

## Thermal Forcing (Step 6b)

Compute TF from CT/SA derived from the climatology.

CLI:

```bash
ismip7-antarctic-clim-ct-sa-to-tf \
	--clim zhou_annual_06_nov \
	--config my.cfg
```

Job script: `example_job_scripts/06_ct_sa_to_tf/job_script_tf_clim.bash`

Outputs:

```
<workdir>/extrap/climatology/<clim_name>/*_tf_extrap.nc
```

## Back‑convert to thetao/so (Step 8b)

Provide static `thetao/so` fields derived from climatology CT/SA.

CLI:

```bash
ismip7-antarctic-clim-ct-sa-to-thetao-so \
	--clim zhou_annual_06_nov \
	--config my.cfg
```

Job script: `example_job_scripts/08_ct_sa_to_thetao_so/job_script_thetao_clim.bash`

Outputs:

```
<workdir>/extrap/climatology/<clim_name>/*_{thetao,so}_extrap.nc
```

## Configuration Keys

Each `i7aof/clim/zhou_*.cfg` file defines, for its source:

```
[climatology]
lat_var, lon_var, lat_dim, lon_dim
lev_var, lev_dim (pressure) -> converted to lev (meters)
filename (relative path under input base dir)
ct_var, sa_var (and optionally ct_mse_var, sa_mse_var)
```

You can override paths and settings via a user config passed with `--config`.

## Notes and Tips

- Prefer the 06_nov (v2) climatologies; they include MSE variables and updated
  metadata. The 30_sep set remains available for continuity.
- For heavy remapping, tune chunk sizes in your config to match your system.

## Python API

Minimal climatology-only example:

```python
from i7aof.remap.clim import remap_climatology
from i7aof.extrap.clim import extrap_climatology
from i7aof.convert.ct_sa_to_tf import clim_ct_sa_to_tf
from i7aof.convert.ct_sa_to_thetao_so import clim_ct_sa_to_thetao_so

clim = 'zhou_annual_06_nov'
cfg = 'my.cfg'

remap_climatology(clim, user_config_filename=cfg)
extrap_climatology(clim, user_config_filename=cfg)
clim_ct_sa_to_tf(clim, user_config_filename=cfg)
clim_ct_sa_to_thetao_so(clim, user_config_filename=cfg)
```

See also {doc}`workflows` for how the climatology ties into the full CMIP
pipeline.

