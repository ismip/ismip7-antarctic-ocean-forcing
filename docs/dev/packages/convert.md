# i7aof.convert

Purpose: Conversion utilities and workflows to transform between ocean
variables used in the pipeline: CMIP thetao/so → CT/SA (TEOS‑10), CT/SA → TF,
CT/SA → thetao/so (back‑conversion), and splitting of native inputs.

## Public Python API (by module)

- Module: {py:mod}`i7aof.convert.split`
  - {py:func}`split_cmip() <i7aof.convert.split.split_cmip>`:
      Split CMIP monthly `thetao`/`so` files into N‑month chunks under
      ``split/{model}/{scenario}/Omon/{variable}``.

- Module: {py:mod}`i7aof.convert.cmip_to_ct_sa`
  - {py:func}`convert_cmip_to_ct_sa() <i7aof.convert.cmip_to_ct_sa.convert_cmip_to_ct_sa>`:
      Convert all split `thetao`/`so` pairs for a model/scenario to native‑grid
      CT/SA files under ``convert/{model}/{scenario}/Omon/ct_sa``.

- Module: {py:mod}`i7aof.convert.ct_sa_to_tf`
  - {py:func}`cmip_ct_sa_to_tf() <i7aof.convert.ct_sa_to_tf.cmip_ct_sa_to_tf>`:
      Compute monthly TF from bias‑corrected CT/SA under
      ``biascorr/{model}/{scenario}/{clim}/Omon/ct_sa`` and write to
      ``.../Omon/tf``.
  - {py:func}`clim_ct_sa_to_tf() <i7aof.convert.ct_sa_to_tf.clim_ct_sa_to_tf>`:
      Compute TF from extrapolated climatology CT/SA and write
      ``*_tf_extrap[_z].nc`` alongside the extrapolated files.

- Module: {py:mod}`i7aof.convert.ct_sa_to_thetao_so`
  - {py:func}`cmip_ct_sa_ann_to_thetao_so_tf() <i7aof.convert.ct_sa_to_thetao_so.cmip_ct_sa_ann_to_thetao_so_tf>`:
      Convert annual CT/SA to annual `thetao`/`so` and copy annual TF into
      ``Oyr/thetao_so_tf``.
  - {py:func}`clim_ct_sa_to_thetao_so() <i7aof.convert.ct_sa_to_thetao_so.clim_ct_sa_to_thetao_so>`:
      Convert extrapolated climatology CT/SA (no time) to static `thetao/so`.

- Module: {py:mod}`i7aof.convert.teos10`
  - {py:func}`compute_sa() <i7aof.convert.teos10.compute_sa>`: SA from SP + z/p + lon/lat.
  - {py:func}`compute_ct() <i7aof.convert.teos10.compute_ct>`: CT from PT + SA.
  - {py:func}`compute_ct_sa() <i7aof.convert.teos10.compute_ct_sa>`: Convenience SA+CT.
  - {py:func}`compute_ct_freezing() <i7aof.convert.teos10.compute_ct_freezing>`: CT at freezing (for TF).
  - {py:func}`convert_dataset_to_ct_sa() <i7aof.convert.teos10.convert_dataset_to_ct_sa>`: High‑level dataset converter.

## Required config options

- `[workdir] base_dir` — required unless ``workdir`` arg is provided.
- `[inputdir] base_dir` — required for splitting unless ``inputdir`` arg is provided.
- `[cmip_dataset]` (for conversion)
  - `lon_var`, `lat_var`: input coordinate names
- `[split_cmip]`
  - `months_per_file`: positive integer
- `[convert_cmip]`
  - `depth_var`: source depth coordinate name (default 'lev')
  - `time_chunk`: months per TEOS‑10 compute chunk (int or None)
- `[ct_sa_to_tf]` (optional)
  - `time_chunk`: months per TF chunk (int or None)
  - `use_poly`: bool; use `CT_freezing_poly` vs exact method
- `[ct_sa_to_thetao_so]` (optional)
  - `time_chunk_years`: years per back‑conversion chunk
- Scenario sections (e.g. `[historical_files]`, `[ssp585_files]`) define input lists for splitting:
  - `thetao`, `so`: expressions of relative paths

## Outputs

- Split monthly files:
  ``split/{model}/{scenario}/Omon/{thetao,so}/*_{YYYY}-{YYYY}.nc``
- Native‑grid CT/SA (monthly):
  ``convert/{model}/{scenario}/Omon/ct_sa/*_{ct,sa}_native.nc``
- Monthly TF (CMIP):
  ``biascorr/{model}/{scenario}/{clim}/Omon/tf/*_tf_*.nc``
- TF from climatology:
  ``extrap/climatology/<clim>/*_tf_extrap[_z].nc``
- Annual back‑conversion outputs:
  ``biascorr/{model}/{scenario}/{clim}/Oyr/thetao_so_tf/*_{thetao,so,tf}_ann.nc``
- Static thetao/so from climatology:
  ``extrap/climatology/<clim>/*_{thetao,so}_extrap.nc``

## Usage

Python
```python
from i7aof.convert.split import split_cmip
from i7aof.convert.cmip_to_ct_sa import convert_cmip_to_ct_sa
from i7aof.convert.ct_sa_to_tf import cmip_ct_sa_to_tf, clim_ct_sa_to_tf
from i7aof.convert.ct_sa_to_thetao_so import (
    cmip_ct_sa_ann_to_thetao_so_tf, clim_ct_sa_to_thetao_so,
)

split_cmip('CESM2-WACCM', 'historical', user_config_filename='my.cfg')
convert_cmip_to_ct_sa('CESM2-WACCM', 'historical', user_config_filename='my.cfg')
cmip_ct_sa_to_tf('CESM2-WACCM', 'historical', clim_name='zhou_annual_06_nov', user_config_filename='my.cfg')
cmip_ct_sa_ann_to_thetao_so_tf(model='CESM2-WACCM', scenario='historical', clim_name='zhou_annual_06_nov', user_config_filename='my.cfg')
clim_ct_sa_to_tf('zhou_annual_06_nov', user_config_filename='my.cfg')
clim_ct_sa_to_thetao_so('zhou_annual_06_nov', user_config_filename='my.cfg')
```

CLI
```text
ismip7-antarctic-split-cmip \
  --model CESM2-WACCM \
  --scenario historical \
  --config my.cfg

ismip7-antarctic-convert-cmip-to-ct-sa \
  --model CESM2-WACCM \
  --scenario historical \
  --config my.cfg

ismip7-antarctic-cmip-ct-sa-to-tf \
  --model CESM2-WACCM \
  --scenario historical \
  --clim zhou_annual_06_nov \
  --config my.cfg

ismip7-antarctic-clim-ct-sa-to-tf \
  --clim zhou_annual_06_nov \
  --config my.cfg

ismip7-antarctic-cmip-annual-ct-sa-to-thetao-so \
  --model CESM2-WACCM \
  --scenario historical \
  --clim zhou_annual_06_nov \
  --config my.cfg

ismip7-antarctic-clim-ct-sa-to-thetao-so \
  --clim zhou_annual_06_nov \
  --config my.cfg
```

## Internals (for maintainers)

- TEOS‑10 uses GSW with explicit broadcasting. Pressure is computed from
  ISMIP `z` and latitude when needed. Inputs are eagerly loaded per chunk to
  avoid large dask graphs.
- Splitting derives year ranges from input time bounds and writes compressed
  per‑variable chunks with clean `_YYYY-YYYY` suffixes.
- CT/SA conversion writes through a per‑file Zarr store for robustness, then
  finalizes to NetCDF once per output file.
- TF precomputes pressure for the ISMIP grid and loops over time chunks,
  appending to Zarr and finalizing to NetCDF once to reduce write overhead.
- Annual back‑conversion reads annual ct/sa/tf, computes thetao/so in year
  chunks, and passes through TF files; outputs are co‑located in
  `Oyr/thetao_so_tf`.
- Optional debug timings via `I7AOF_DEBUG_TEOS10=1`.

## Edge cases / validations

- Split: invalid or missing `[split_cmip] months_per_file` raises `ValueError`.
- Conversion: missing split inputs or mismatched `_YYYY-YYYY` ranges raise errors.
- TF: missing bias‑corrected ct/sa pairs raises `FileNotFoundError`.
- Back‑conversion: requires both annual CT/SA and TF; missing files raise errors.

