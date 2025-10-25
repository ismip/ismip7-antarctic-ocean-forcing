# Bias Correction

This page describes the classic bias-correction workflow that aligns CMIP
fields to an observational (or reanalysis) climatology on the ISMIP grid. It
operates on extrapolated model fields and writes bias-corrected CT/SA.

Two approaches exist in the codebase:

- A classic geographic-space bias correction (`i7aof.biascorr.classic`), with
	a simple two-step method on gridded fields (model climatology vs reference).
- A timeslice/projection approach (`i7aof.biascorr.timeslice`,
	`i7aof.biascorr.projection`) operating in T–S space by basin.

This page focuses on the classic approach.

## Classic workflow (CLI)

Run after remapping and extrapolating CMIP CT/SA to the ISMIP grid:

```text
ismip7-antarctic-biascorr-cmip \
	--model <MODEL> \
	--scenario <SCENARIO> \
	--clim <CLIM_NAME> \
	--workdir <WORKDIR> \
	--config <CONFIG>
```

Inputs expected under `<WORKDIR>`:

- Extrapolated CMIP: `extrap/<MODEL>/<scenario>/Omon/ct_sa/*_{ct,sa}_*.nc`
- Extrapolated reference climatology: `extrap/climatology/<CLIM_NAME>/*_extrap.nc`

Outputs:

- Bias-corrected CT/SA: `biascorr/<MODEL>/<scenario>/<CLIM_NAME>/Omon/ct_sa/*_{ct,sa}_*.nc`

## What it does

1. Compute model climatology over a configured historical window (e.g.,
	 1995–2015) from extrapolated CMIP monthly files.
2. Read extrapolated reference climatology on the ISMIP grid.
3. Form the bias: `bias = model_climatology - reference` for CT and SA.
4. Subtract the bias from each extrapolated monthly CT/SA file and write the
	 corrected outputs with ISMIP coordinates/bounds.

The `time_chunk` used for IO/chunking comes from `[biascorr] time_chunk` in
the config. Coordinates and bounds are copied from the canonical ISMIP grid.

## Python API

```python
from i7aof.biascorr.classic import biascorr_cmip

biascorr_cmip(
		model='CESM2-WACCM',
		scenario='ssp585',
		clim_name='OI_Climatology',
		workdir='/path/to/workdir',
		user_config_filename='my-config.cfg',
)
```

See also:

- {py:mod}`i7aof.biascorr.classic` — implementation details
- {doc}`../dev/packages/biascorr` — developer documentation with the
	timeslice/projection approach
