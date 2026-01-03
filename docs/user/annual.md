# Annual Averages (CMIP)

Compute annual means for CMIP bias‑corrected `ct`, `sa`, and TF. This page
covers inputs, weighting, and outputs without repeating the full
{doc}`workflows` page.

## What is averaged

- Inputs: monthly bias‑corrected `ct`, `sa`, and monthly TF for each scenario
- CLI: `ismip7-antarctic-cmip-annual-averages`
- Python: {py:mod}`i7aof.time.cmip`

Climatology products are static and do not require annual averaging.

## Weighting and calendars

- Weighted by the number of days per month, respecting the CF calendar of the
  input time axis (e.g., `gregorian`, `proleptic_gregorian`, `365_day`).
- Time bounds are propagated or constructed as needed for the annual mean.

## Outputs

```
<workdir>/biascorr/<model>/<scenario>/<clim>/Oyr/ct_sa_tf/*_ann.nc
```

Annual files contain the three variables (ct, sa, tf) under a single `Oyr`
product tree for convenience.

## Validation checklist

- Year coverage matches the monthly inputs; no missing years.
- Annual means align with independent monthly averaging over a spot check.
- Metadata (units, attributes) preserved for all variables.

## Minimal example

```python
from i7aof.time.cmip import compute_cmip_annual_averages

compute_cmip_annual_averages(
    model='CESM2-WACCM',
    scenario='ssp585',
    clim_name='zhou_annual_06_nov',
    user_config_filename='my.cfg',
)
```
