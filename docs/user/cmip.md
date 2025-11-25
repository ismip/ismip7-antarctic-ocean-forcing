# CMIP Model Inputs

This page focuses on CMIP6/CMIP7 model data: what you need, how it is
interpreted, and how to prepare and validate it before and during the
`i7aof` pipeline. It complements (but does not repeat) the full 8‑step
workflow described in the {doc}`workflows` page.

## Scope & Role

CMIP ocean monthly fields (`thetao`, `so`, optionally `zos`) provide the time‑varying
forcing foundation. They are transformed to TEOS‑10 conservative temperature
(`ct`) and absolute salinity (`sa`), remapped, extrapolated, bias‑corrected
against an observational climatology, then used to derive thermal forcing (TF)
and annual products. All CMIP steps (except the bias correction invocation) run
per scenario (`historical`, plus one future scenario like `ssp585`).

## Required Inputs & Minimum Coverage

You should supply:

- Monthly `thetao` and `so` (Omon) for the full intended historical climatology window
  (e.g. 1995–2024 as set by `[biascorr] climatology_start_year/end_year`) and the chosen future scenario span.
- Consistent horizontal grid and vertical coordinate (depth/lev); model native vertical
  must be monotonic increasing in depth or pressure.
- Optional `zos` (sea surface height) if downstream workflows or validation need it.
- Complete months (no gaps, no duplicates). Leap‑year handling should match CF conventions.

Minimum viable dataset: a continuous monthly time series covering the configured
climatology window plus at least the first decade of the future scenario for testing.

## Scenario Handling & Bias Correction Interaction

Steps 1–4 (split, convert, remap, extrapolate) are executed independently for
`historical` and the future scenario. Step 5 (bias correction) is invoked once
with the future scenario argument; internally it reads extrapolated CT/SA from
both scenarios and writes bias‑corrected monthly outputs for each. Subsequent CMIP steps (TF,
annual averages, back‑conversion) run per scenario using those corrected fields.

## Directory & Naming Conventions (CMIP portion)

The pipeline builds a predictable hierarchy under `<workdir>`:

```
split/<model>/<scenario>/Omon/{thetao,so}/...
convert/<model>/<scenario>/Omon/ct_sa/*_{ct,sa}_native.nc
remap/<model>/<scenario>/Omon/ct_sa/*_{ct,sa}_remap.nc
extrap/<model>/<scenario>/Omon/ct_sa/*_{ct,sa}_extrap_*.nc
biascorr/<model>/<scenario>/<clim>/Omon/ct_sa/*_{ct,sa}_biascorr_*.nc
biascorr/<model>/<scenario>/<clim>/Omon/tf/*_tf_*.nc
biascorr/<model>/<scenario>/<clim>/Oyr/ct_sa_tf/*_ann.nc
biascorr/<model>/<scenario>/<clim>/Oyr/thetao_so_tf/*_{thetao,so,tf}_ann.nc
```

Key points:

- Scenario directory names must exactly match the names you pass (`historical`, `ssp585`, etc.).
- Filenames carry variable tags (`_ct_native`, `_sa_remap`, `_ct_biascorr`) for clarity and automated discovery.
- Annual products use `Oyr` while monthly products use `Omon`.

## Configuration Blocks & Tuning

Relevant sections in your config (`*.cfg`):

```
[inputdir] base_dir = /path/to/raw_cmip_and_clim
[workdir] base_dir = /scratch/work_i7aof

[split_cmip]
months_per_file = 120          # 10-year blocks (adjust for I/O patterns)

[convert_cmip]
time_chunk = 12                # TEOS-10 compute chunk length (months)

[remap_cmip]
vert_time_chunk = 1            # Vertical interpolation chunk
horiz_time_chunk = 120         # Horizontal remap chunk

[extrap_cmip]
time_chunk = 12                # Extrapolation chunk for Fortran steps
time_chunk_resample = 12       # Post-extrap vertical resample chunk

[biascorr]
climatology_start_year = 1995
climatology_end_year   = 2024
time_chunk = 12               # Bias application chunk
```

For each CMIP scenario you want to process, you will also define a
`[<scenario>_files]` section (for example, `[historical_files]` or
`[ssp585_files]`). Within each of these sections you provide one or more
expressions for `thetao` and `so` input files, typically glob patterns
relative to `[inputdir] base_dir`.

Optionally, you can restrict the split to a subset of years using
integer `start_year` and/or `end_year` options in the same
`[<scenario>_files]` section. When either or both are provided,
`split_cmip` (and the `ismip7-antarctic-split-cmip` CLI) will first
subset each input dataset to the overlapping year range and will skip
files that do not overlap at all. This is the recommended way to work
with a limited time span of the CMIP input data without modifying the
original files.

Tuning guidance:

- Increase `months_per_file` for fewer open/close cycles if filesystem latency is high; keep manageable for restarts.
- Match `time_chunk` to available memory; larger chunks reduce Python overhead but raise peak memory.
- Set `[remap_cmip] vert_time_chunk = 1` unless vertical interpolation becomes a bottleneck.
- Adjust `time_chunk_resample` if resampling memory or speed issues arise.
- Keep bias correction `time_chunk` aligned with extrapolated chunking to minimize rechunk cost.

Enable TEOS‑10 debug/profiling with:

```bash
export I7AOF_DEBUG_TEOS10=1
```

## Performance Considerations

- Prefer contiguous storage layouts (e.g., reorganize native NetCDFs so `time` is the slowest varying dimension).
- Use a fast parallel filesystem for `<workdir>` (scratch or burst buffer) and keep `<inputdir>` on reliable long-term storage.
- Avoid very small chunk sizes (< 3 months)—Python/Xarray overhead dominates.
- Monitor I/O wait with tools like `iostat` or HPC profiler to guide chunk adjustments.

## Validation Checklist

Before running the pipeline (or after Step 2):

- Temporal coverage: all months present; no duplicate timestamps.
- Units: `thetao` in degC or K? (must match expected TEOS‑10 conversion path); `so` should be dimensionless Practical Salinity (PSS‑78). Convert if necessary prior to use.
- Missing data: proportion of NaNs within cavity regions—large gaps may produce extensive extrapolation regions.
- Vertical coordinate monotonic and positive down (or pressure increasing). If not, preprocess.
- Global attributes: record source institution and experiment ID for provenance in downstream NetCDF outputs.

After bias correction (Step 5):

- Mean difference (model minus reference) over the climatology window should trend toward zero for `ct` and `sa` at most depths.
- Spot-check a few profiles for unrealistic gradients introduced by extrapolation.

## Common Pitfalls

- Mixing scenario names (`SSP585` vs `ssp585`) leading to separate directories.
- Supplying incomplete final year (e.g., 2024 missing later months) causing biased climatology.
- Extremely fine time chunks (1 month) causing poor throughput.
- Depth coordinate mislabeled (e.g., using `lev` that is actually layer number without physical meaning).
- Not cleaning temporary partial outputs after an interrupted run—reruns may skip steps with incomplete data.

## Minimal Programmatic Snippet (CMIP portion only)

```python
from i7aof.convert.split import split_cmip
from i7aof.convert.cmip_to_ct_sa import convert_cmip_to_ct_sa
from i7aof.remap.cmip import remap_cmip
from i7aof.extrap.cmip import extrap_cmip
from i7aof.biascorr.classic import biascorr_cmip

model = 'CESM2-WACCM'
future = 'ssp585'
clim = 'zhou_annual_06_nov'
cfg = 'my.cfg'

for scenario in ['historical', future]:
    split_cmip(model, scenario, user_config_filename=cfg)
    convert_cmip_to_ct_sa(model, scenario, user_config_filename=cfg)
    remap_cmip(model, scenario, user_config_filename=cfg)
    extrap_cmip(model, scenario, user_config_filename=cfg)

# Single bias correction invocation writes both scenarios
biascorr_cmip(model, future, clim_name=clim, user_config_filename=cfg)
```

For TF, annual averages, and back‑conversion see {doc}`workflows`.

## References

- TEOS‑10 Manual: https://teos-10.org
- CF Conventions: https://cfconventions.org
- ISMIP Documentation (grid definitions): see {doc}`remap`

## Next Steps

Once inputs validate, proceed to the full workflow or integrate with a
climatology (see {doc}`clim`). Consider adding automated QA checks using the
validation checklist above before large batch processing on HPC.
