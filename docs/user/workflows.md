# End‑to‑End Workflows

This is the canonical 8‑step processing chain for turning raw CMIP6/7 model
outputs and observational climatologies into ISMIP-ready forcing products.
All steps are required for each CMIP model: the observational climatology
is processed alongside the CMIP data and is used to bias‑correct the CMIP
outputs before subsequent steps.

You may run each step either:

1. Via the provided example job scripts (`example_job_scripts/<STEP>/`),
2. Directly with the listed CLI commands, or
3. Programmatically through the Python API modules (import paths given).

You may wish to prepare one or more configuration files (`scripts/*.cfg`
examples) that set input base directories (`[inputdir] base_dir`), work
directory (`[workdir] base_dir`), grids, chunk lengths, and time windows. Most
step‑specific options live under section names matching the step (e.g.
`[convert_cmip]`, `[biascorr]`).

A few options can be provided on the command-line via flags instead of config
options.  These include the input and working base directories, the CMIP
and scenario, and the name of the climatology to use for bias correction.
See `example_job_scripts/` for examples of this approach.

> Tip: For large models, tune time chunk sizes (e.g.
> `[convert_cmip] time_chunk = 12`) to improve TEOS‑10 and IO throughput.

---

## Workflow overview (compact)

- **Step 1** — Split CMIP files
	- **Purpose:** split monthly `thetao`/`so` into N‑month blocks.
	- **Run:** historical and future.
	- **CLI:** `ismip7-antarctic-split-cmip`
	- **Job:** `example_job_scripts/01_split/job_script_cmip_{hist,ssp}_split.bash`
	- **Inputs:** raw CMIP under input base dir
	- **Outputs:** `split/<model>/<scenario>/Omon/{thetao,so}/*_<YYYY>-<YYYY>.nc`

- **Step 2** — Convert to CT/SA (TEOS‑10)
	- **Purpose:** `thetao/so` → `ct/sa` on native model grid.
	- **Run:** historical and future.
	- **CLI:** `ismip7-antarctic-convert-cmip-to-ct-sa`
	- **Job:** `example_job_scripts/02_cmip_to_ct_sa/job_script_cmip_{hist,ssp}_to_ct_sa.bash`
	- **Inputs:** Step 1 outputs
	- **Outputs:** `convert/<model>/<scenario>/Omon/ct_sa/*_{ct,sa}_native.nc`

- **Step 3a** — Remap climatology
	- **Purpose:** remap climatology CT/SA to ISMIP grid and `z_extrap`.
	- **Run:** once (no scenario).
	- **CLI:** `ismip7-antarctic-remap-clim`
	- **Job:** `example_job_scripts/03_remap/job_script_remap_clim.bash`
	- **Inputs:** climatology under input base dir
	- **Outputs:** `remap/climatology/<clim>/*_ismip<res>.nc`

- **Step 3b** — Remap CMIP
	- **Purpose:** remap CMIP CT/SA to ISMIP grid and `z_extrap`.
	- **Run:** historical and future.
	- **CLI:** `ismip7-antarctic-remap-cmip`
	- **Job:** `example_job_scripts/03_remap/job_script_remap_{hist,ssp}.bash`
	- **Inputs:** Step 2 outputs
	- **Outputs:** `remap/<model>/<scenario>/Omon/ct_sa/*_{ct,sa}_remap.nc`

- **Step 4a** — Extrapolate climatology + resample
	- **Purpose:** fill gaps; resample `z_extrap→z`.
	- **Run:** once.
	- **CLI:** `ismip7-antarctic-extrap-clim`
	- **Job:** `example_job_scripts/04_extrap/job_script_extrap_clim.bash`
	- **Inputs:** Step 3a outputs
	- **Outputs:** `extrap/climatology/<clim>/*_{ct,sa}_extrap.nc` and `*_z.nc`

- **Step 4b** — Extrapolate CMIP + resample
	- **Purpose:** fill gaps; resample `z_extrap→z`.
	- **Run:** historical and future.
	- **CLI:** `ismip7-antarctic-extrap-cmip`
	- **Job:** `example_job_scripts/04_extrap/job_script_extrap_{hist,ssp}.bash`
	- **Inputs:** Step 3b outputs
	- **Outputs:** `extrap/<model>/<scenario>/Omon/ct_sa/*_{ct,sa}_extrap_*.nc`

- **Step 5** — Bias correction (classic)
	- **Purpose:** bias‑correct CMIP CT/SA toward the extrapolated climatology.
	- **Run:** once; uses historical + future internally; writes both scenarios.
	- **CLI:** `ismip7-antarctic-bias-corr-classic`
	- **Job:** `example_job_scripts/05_biascorr/job_script_biascorr.bash`
	- **Inputs:** Step 4a and 4b (historical + future)
	- **Outputs:** `biascorr/<model>/{historical,<future>}/<clim>/Omon/ct_sa/*_biascorr_*.nc`

- **Step 6a** — Thermal Forcing (climatology)
	- **Purpose:** compute TF from extrapolated climatology CT/SA.
	- **Run:** once.
	- **CLI:** `ismip7-antarctic-clim-ct-sa-to-tf`
	- **Job:** `example_job_scripts/06_ct_sa_to_tf/job_script_tf_clim.bash`
	- **Inputs:** Step 4a outputs
	- **Outputs:** `extrap/climatology/<clim>/*_tf_extrap.nc`

- **Step 6b** — Thermal Forcing (CMIP)
	- **Purpose:** compute TF from bias‑corrected CMIP CT/SA.
	- **Run:** historical and future.
	- **CLI:** `ismip7-antarctic-cmip-ct-sa-to-tf`
	- **Job:** `example_job_scripts/06_ct_sa_to_tf/job_script_tf_{hist,ssp}.bash`
	- **Inputs:** Step 5 CMIP outputs
	- **Outputs:** `biascorr/<model>/<scenario>/<clim>/Omon/tf/*_tf_*.nc`

- **Step 7** — Annual averages (CMIP)
	- **Purpose:** annual means of CT, SA, and TF.
	- **Run:** historical and future.
	- **CLI:** `ismip7-antarctic-cmip-annual-averages`
	- **Job:** `example_job_scripts/07_annual/job_script_ann_{hist,ssp}.bash`
	- **Inputs:** Step 5 and 6a outputs
	- **Outputs:** `biascorr/<model>/<scenario>/<clim>/Oyr/ct_sa_tf/*_ann.nc`

- **Step 8a** — Back‑convert climatology CT/SA → `thetao/so`
	- **Purpose:** provide static `thetao/so` derived from climatology CT/SA.
	- **Run:** once.
	- **CLI:** `ismip7-antarctic-clim-ct-sa-to-thetao-so`
	- **Job:** `example_job_scripts/08_ct_sa_to_thetao_so/job_script_thetao_clim.bash`
	- **Inputs:** Step 4a climatology outputs
	- **Outputs:** `extrap/climatology/<clim>/*_{thetao,so}_extrap.nc`

- **Step 8b** — Back‑convert CMIP annual CT/SA → `thetao/so`
	- **Purpose:** provide CMIP annual `thetao/so` (TF carried alongside).
	- **Run:** historical and future.
	- **CLI:** `ismip7-antarctic-cmip-annual-ct-sa-to-thetao-so`
	- **Job:** `example_job_scripts/08_ct_sa_to_thetao_so/job_script_thetao_{hist,ssp}.bash`
	- **Inputs:** Step 7 CMIP annual outputs
	- **Outputs:** `biascorr/<model>/<scenario>/<clim>/Oyr/thetao_so_tf/*_{thetao,so,tf}_ann.nc`

---

## Step Details

### 1. Split CMIP
Chunk long monthly CMIP streams into N‑month files for better workflow performance.
Configure `[split_cmip] months_per_file`. Produces manageable, uniform ranges,
the default is 10 years (120 months).

CLI:
```bash
ismip7-antarctic-split-cmip --inputdir <INPUTDIR> --workdir <WORKDIR> --model <MODEL> --scenario <SCENARIO> --config my.cfg
```
### 2. Convert to CT/SA
Reads `thetao/so`, applies TEOS‑10 (GSW) to compute conservative temperature `ct` and absolute salinity `sa`.
Important config keys: `[convert_cmip] time_chunk`, grid/vertical naming.

CLI:
```bash
ismip7-antarctic-convert-cmip-to-ct-sa --workdir <WORKDIR> --model <MODEL> --scenario <SCENARIO> --config my.cfg
```

Outputs remain on the model’s native horizontal grid.

### 3. Remap (Climatology and CMIP)
Vertically prepare to `z_extrap` levels and horizontally map to ISMIP grid resolution tag (e.g. `ismip2km`). See {doc}`remap` and {doc}`clim`.

CMIP CLI:
```bash
ismip7-antarctic-remap-cmip --workdir <WORKDIR> --model <MODEL> --scenario <SCENARIO> --config my.cfg
```
Climatology CLI:
```bash
ismip7-antarctic-remap-clim --workdir <WORKDIR> --clim <CLIM_NAME> --config my.cfg
```

### 4. Extrapolate + Resample
Fill spatial gaps (coastal cavities etc.) horizontally then vertically (Fortran routines). Resample from dense `z_extrap` (e.g. 20 m) to coarser `z` (e.g. 60 m) using conservative integration. Climatology adds and later removes a dummy `time` dimension.

CMIP CLI:
```bash
ismip7-antarctic-extrap-cmip --workdir <WORKDIR> --model <MODEL> --scenario <SCENARIO> --config my.cfg
```
Climatology CLI:
```bash
ismip7-antarctic-extrap-clim --workdir <WORKDIR> --clim <CLIM_NAME> --config my.cfg
```

Tune `[extrap_cmip] time_chunk_resample` (CMIP only) for throughput.

### 5. Bias Correction
Align model extrapolated climatology to an extrapolated observational reference.
Configure historical window `[biascorr] climatology_start_year` / `_end_year` and chunking.

CLI:
```bash
ismip7-antarctic-bias-corr-classic \
  --workdir <WORKDIR> --model <MODEL> --scenario <SCENARIO> --clim <CLIM_NAME> --config my.cfg
```

### 6. Thermal Forcing (TF)
Compute TF from CT/SA (bias‑corrected for CMIP; extrapolated for climatology). Internally computes in-situ freezing point and subtracts. For CMIP, run per scenario once bias‑corrected CT/SA exist for that scenario (both scenarios are available after Step 5).

CMIP CLI:
```bash
ismip7-antarctic-cmip-ct-sa-to-tf --workdir <WORKDIR> --model <MODEL> --scenario <SCENARIO> --clim <CLIM_NAME> --config my.cfg
```
Climatology CLI:
```bash
ismip7-antarctic-clim-ct-sa-to-tf --workdir <WORKDIR> --clim <CLIM_NAME> --config my.cfg
```

### 7. Annual Averages (CMIP)
Aggregate monthly bias‑corrected CT, SA, and TF into annual means. Run per scenario once TF exists.

CLI:
```bash
ismip7-antarctic-cmip-annual-averages --workdir <WORKDIR> --model <MODEL> --scenario <SCENARIO> --clim <CLIM_NAME> --config my.cfg
```

### 8. Back‑conversion to `thetao/so`
Provide annual or static `thetao/so` (and copy TF for CMIP annuals). Run per CMIP scenario once annual CT/SA/TF are available and once for the climatology.

CMIP CLI:
```bash
ismip7-antarctic-cmip-annual-ct-sa-to-thetao-so --workdir <WORKDIR> --model <MODEL> --scenario <SCENARIO> --clim <CLIM_NAME> --config my.cfg
```
Climatology CLI:
```bash
ismip7-antarctic-clim-ct-sa-to-thetao-so --workdir <WORKDIR> --clim <CLIM_NAME> --config my.cfg
```

---

## Configuration Skeleton
Minimal user config excerpt:
```ini
[inputdir]
base_dir = /path/to/cmip_and_clim_inputs

[workdir]
base_dir = /scratch/work_i7aof

[split_cmip]
months_per_file = 120

[convert_cmip]
time_chunk = 120

[remap_cmip]
vert_time_chunk = 1
horiz_time_chunk = 120

[extrap_cmip]
time_chunk = 12
time_chunk_resample = 12
```

Adjust as needed for a given CMIP model or HPC system.

---

## Troubleshooting Quickies
- Out of memory: try reducing relevant `*time_chunk` config option(s).
- Missing files in a step: re‑check earlier output directory naming
  conventions; each job script echoes the expected patterns.

---

## Programmatic Example
```python
from i7aof.convert.split import split_cmip
from i7aof.convert.cmip_to_ct_sa import convert_cmip_to_ct_sa
from i7aof.remap.cmip import remap_cmip
from i7aof.remap.clim import remap_climatology
from i7aof.extrap.cmip import extrap_cmip
from i7aof.extrap.clim import extrap_climatology
from i7aof.biascorr.classic import biascorr_cmip
from i7aof.convert.ct_sa_to_tf import cmip_ct_sa_to_tf
from i7aof.convert.ct_sa_to_tf import clim_ct_sa_to_tf
from i7aof.time.cmip import compute_cmip_annual_averages
from i7aof.convert.ct_sa_to_thetao_so import (
  cmip_ct_sa_ann_to_thetao_so_tf,
  clim_ct_sa_to_thetao_so,
)

model = 'CESM2-WACCM'
future_scenario = 'ssp585'
clim_name = 'zhou_annual_06_nov'
cfg = 'my.cfg'

scenarios = ['historical', future_scenario]

for scenario in scenarios:
	split_cmip(model, scenario, user_config_filename=cfg)
for scenario in scenarios:
	convert_cmip_to_ct_sa(model, scenario, user_config_filename=cfg)
for scenario in scenarios:
	remap_cmip(model, scenario, user_config_filename=cfg)
remap_climatology(clim_name, user_config_filename=cfg)
for scenario in scenarios:
	extrap_cmip(model, scenario, user_config_filename=cfg)
extrap_climatology(clim_name, user_config_filename=cfg)
biascorr_cmip(
  model, future_scenario, clim_name=clim_name, user_config_filename=cfg
)
for scenario in scenarios:
	cmip_ct_sa_to_tf(
		model, scenario, clim_name=clim_name, user_config_filename=cfg
	)
clim_ct_sa_to_tf(clim_name, user_config_filename=cfg)

for scenario in scenarios:
	compute_cmip_annual_averages(
	model=model,
	scenario=scenario,
	clim_name=clim_name,
	user_config_filename=cfg,
	)

for scenario in scenarios:
	cmip_ct_sa_ann_to_thetao_so_tf(
	model=model,
	scenario=scenario,
	clim_name=clim_name,
	user_config_filename=cfg
	)
clim_ct_sa_to_thetao_so(clim_name, user_config_filename=cfg)
```
