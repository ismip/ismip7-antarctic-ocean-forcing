# Quickstart

This page is a fast on‑ramp. If you want the full 8‑step pipeline, jump to the
[End‑to‑End Workflows](workflows.md) page.

## Prerequisites

- Install the package and dependencies: see [Install](install.md)
- Prepare an input base directory and a working directory
- Optionally copy and adapt one of the example configs in `scripts/*.cfg`

## 60‑second smoke check

Verify the CLIs are available and show their options:

```bash
ismip7-antarctic-split-cmip --help
ismip7-antarctic-remap-cmip --help
ismip7-antarctic-extrap-cmip --help
ismip7-antarctic-bias-corr-classic --help
```

If these commands print usage information, you’re ready to run the workflow.

## Run your first model (simplest path)

Use the provided job scripts as a starting point—they encode sane defaults and
the correct step ordering. Pick a CMIP model and a future scenario (e.g.
`ssp585`) and run the scripts in order for historical and the chosen future:

- `example_job_scripts/01_split/job_script_cmip_{hist,ssp}_split.bash`
- `example_job_scripts/02_cmip_to_ct_sa/job_script_cmip_{hist,ssp}_to_ct_sa.bash`
- `example_job_scripts/03_remap/job_script_remap_{clim,hist,ssp}.bash`
- `example_job_scripts/04_extrap/job_script_extrap_{clim,hist,ssp}.bash`
- `example_job_scripts/05_biascorr/job_script_biascorr.bash` (runs once; writes both scenarios)
- `example_job_scripts/06_ct_sa_to_tf/job_script_tf_{clim,hist,ssp}.bash`
- `example_job_scripts/07_annual/job_script_ann_{hist,ssp}.bash`
- `example_job_scripts/08_ct_sa_to_thetao_so/job_script_thetao_{clim,hist,ssp}.bash`

Each script echoes the expected inputs and outputs and accepts overrides for the
input/working directories, model, scenario, and climatology.

## Where next

- Full details, inputs/outputs, and programmatic example: see
    [End‑to‑End Workflows](workflows.md)
- Background on climatologies, CMIP inputs, and remapping choices: see
    [Climatology](clim.md), [CMIP](cmip.md), and [Remapping](remap.md)
