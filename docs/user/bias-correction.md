# Bias Correction

Aligns CMIP extrapolated CT/SA to an observational climatology on the ISMIP
grid and writes bias‑corrected monthly CT/SA. This page covers the classic
geographic‑space method; a timeslice/projection approach also exists for
basin‑wise T–S corrections.

## Key behavior (classic)

- Single invocation writes both scenarios: run once with the future scenario
  argument; internally it reads historical and future extrapolated files,
  computes the bias using the configured historical window, and writes
  bias‑corrected monthly CT/SA for both scenarios.
- Climatology: prefer the 06_nov (v2) set (e.g., `zhou_annual_06_nov`).
- Inputs must be extrapolated CMIP and extrapolated climatology on the ISMIP grid.

## CLI

Run after remapping and extrapolation (Steps 3–4):

```bash
ismip7-antarctic-bias-corr-classic \
  --model <MODEL> \
  --scenario <FUTURE_SCENARIO> \
  --clim <CLIM_NAME> \
  --workdir <WORKDIR> \
  --config <CONFIG>
```

Inputs under `<WORKDIR>`:

- CMIP extrapolated monthly: `extrap/<MODEL>/{historical,<future>}/Omon/ct_sa/*_{ct,sa}_extrap_*.nc`
- Climatology extrapolated: `extrap/climatology/<CLIM_NAME>/*_{ct,sa}_extrap.nc`

Outputs:

- Bias‑corrected CT/SA: `biascorr/<MODEL>/{historical,<future>}/<CLIM_NAME>/Omon/ct_sa/*_{ct,sa}_biascorr_*.nc`

## Algorithm (classic)

1. Build a model climatology from CMIP extrapolated monthly `ct` and `sa` over
   the configured historical window (`[biascorr] climatology_start_year/_end_year`).
2. Read the extrapolated reference climatology (`ct`, `sa`) on the ISMIP grid.
3. Compute `bias = model_climatology - reference` for each variable.
4. Subtract the bias from every CMIP extrapolated monthly file; preserve
   coordinates and ISMIP bounds; write outputs per scenario.

## Configuration

```
[biascorr]
climatology_start_year = 1995
climatology_end_year   = 2024
time_chunk = 12
```

Set `time_chunk` to balance IO/memory. The climatology window should match the
reference period intended for alignment and be fully covered by the CMIP
historical input.

## Python API

```python
from i7aof.biascorr.classic import biascorr_cmip

biascorr_cmip(
    model='CESM2-WACCM',
    future_scenario='ssp585',
    clim_name='zhou_annual_06_nov',
    user_config_filename='my.cfg',
)
```

See also:

- {py:mod}`i7aof.biascorr.classic` — implementation details
- {doc}`workflows` — where bias correction fits in the pipeline
- Timeslice/projection approach: {py:mod}`i7aof.biascorr.timeslice`, {py:mod}`i7aof.biascorr.projection`
