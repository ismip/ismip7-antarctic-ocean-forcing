# Conversion

This page groups all conversions performed by `i7aof.convert` to avoid
scattering details across multiple pages. It focuses on inputs, units, and
algorithm‑specific configuration without repeating the end‑to‑end ordering from
{doc}`workflows`.

Contents:

1) thetao/so → ct/sa (TEOS‑10)
2) ct/sa → TF (thermal forcing)
3) ct/sa → thetao/so (back‑conversion)

---

## 1) thetao/so → ct/sa (TEOS‑10)

Convert CMIP `thetao` and `so` on the native model grid to TEOS‑10 conservative
temperature (`ct`) and absolute salinity (`sa`).

- Inputs: monthly Omon `thetao`, `so` on the native grid
- Outputs: `ct`, `sa` on the same native grid and vertical coordinate
- CLI: `ismip7-antarctic-convert-cmip-to-ct-sa`
- Python: {py:mod}`i7aof.convert.cmip_to_ct_sa`

Units and conventions:

- `thetao`: potential temperature, units degC or K (CF‑compliant). Ensure
  attributes are correct for the conversion path.
- `so`: Practical Salinity (PSS‑78). If inputs are Absolute Salinity, convert to
  Practical or preprocess accordingly.
- Vertical: provide a monotonic depth or pressure coordinate.

Configuration:

```
[convert_cmip]
time_chunk = 12          # months per compute chunk
```

Tuning:

- Larger `time_chunk` reduces Python overhead but increases memory.
- For large 3D grids, 6–12 months per chunk is a good starting point.

Outputs:

```
<workdir>/convert/<model>/<scenario>/Omon/ct_sa/*_{ct,sa}_native.nc
```

Validation:

- Profiles show plausible `ct` and `sa` ranges.
- No time gaps or duplicates; dimensions/chunks match downstream expectations.

Troubleshooting:

```bash
export I7AOF_DEBUG_TEOS10=1
```

Minimal example:

```python
from i7aof.convert.cmip_to_ct_sa import convert_cmip_to_ct_sa

convert_cmip_to_ct_sa('CESM2-WACCM', 'historical', user_config_filename='my.cfg')
```

---

## 2) ct/sa → TF (thermal forcing)

Compute Thermal Forcing from CT/SA. For CMIP this uses bias‑corrected monthly
CT/SA; for climatology, the extrapolated fields.

- CMIP CLI: `ismip7-antarctic-cmip-ct-sa-to-tf`
- Climatology CLI: `ismip7-antarctic-clim-ct-sa-to-tf`
- Python: {py:mod}`i7aof.convert.ct_sa_to_tf`

Inputs and outputs:

- CMIP input: `<workdir>/biascorr/<model>/<scenario>/<clim>/Omon/ct_sa/*_biascorr_*.nc`
- CMIP output: `<workdir>/biascorr/<model>/<scenario>/<clim>/Omon/ct_sa_tf0/*_{ct,sa,tf}_*.nc`
- Climatology input: `<workdir>/extrap/climatology/<clim>/*_{ct,sa}_extrap.nc`
- Climatology output: `<workdir>/extrap/climatology/<clim>/*_tf_extrap.nc`

Notes:

- Internally computes the in‑situ freezing point from salinity/pressure and
  derives TF accordingly.
- Ensure bias correction is complete for both scenarios before running the CMIP
  TF step.

Validation:

- TF should generally be positive in warm waters and near zero in near‑freezing
  conditions; spot‑check profiles.

Minimal examples:

```python
from i7aof.convert.ct_sa_to_tf import cmip_ct_sa_to_tf, clim_ct_sa_to_tf

cmip_ct_sa_to_tf('CESM2-WACCM', 'ssp585', clim_name='zhou_annual_06_nov', user_config_filename='my.cfg')
clim_ct_sa_to_tf('zhou_annual_06_nov', user_config_filename='my.cfg')
```

---

## 3) ct/sa → thetao/so (back‑conversion)

Provide `thetao`/`so` from CT/SA for downstream consumers. For CMIP this is run
on annual means and also carries TF; for climatology it produces static
`thetao/so` from extrapolated CT/SA.

- CMIP CLI: `ismip7-antarctic-cmip-annual-ct-sa-to-thetao-so`
- Climatology CLI: `ismip7-antarctic-clim-ct-sa-to-thetao-so`
- Python: {py:mod}`i7aof.convert.ct_sa_to_thetao_so`

Inputs and outputs:

- CMIP input: `<workdir>/biascorr/<model>/<scenario>/<clim>/Oyr/ct_sa_tf/*_ann.nc`
- CMIP output: `<workdir>/biascorr/<model>/<scenario>/<clim>/Oyr/thetao_so_tf/*_{thetao,so,tf}_ann.nc`
- Climatology input: `<workdir>/extrap/climatology/<clim>/*_{ct,sa}_extrap.nc`
- Climatology output: `<workdir>/extrap/climatology/<clim>/*_{thetao,so}_extrap.nc`

Notes:

- Uses TEOS‑10 relationships to back‑convert; ensure metadata/units are
  consistent.
- CMIP path requires annual averages to exist (see {doc}`annual`).

Minimal examples:

```python
from i7aof.convert.ct_sa_to_thetao_so import (
    cmip_ct_sa_ann_to_thetao_so_tf, clim_ct_sa_to_thetao_so,
)

cmip_ct_sa_ann_to_thetao_so_tf(
    model='CESM2-WACCM',
    scenario='ssp585',
    clim_name='zhou_annual_06_nov',
    user_config_filename='my.cfg',
)

clim_ct_sa_to_thetao_so('zhou_annual_06_nov', user_config_filename='my.cfg')
```
