# CLI and Python API

This page lists the primary command-line tools and the top-level Python APIs exposed by this project.

## Command-line tools

- Split & Convert
	- `ismip7-antarctic-split-cmip` — optionally split long monthly CMIP inputs into smaller files.
	- `ismip7-antarctic-convert-cmip-to-ct-sa` — convert CMIP thetao/so to ct/sa on the native grid.

- Remap & Extrapolate
	- `ismip7-antarctic-remap-cmip` — remap CMIP ct/sa to the ISMIP grid (vertical → horizontal).
	- `ismip7-antarctic-extrap-cmip` — horizontally and vertically extrapolate remapped CMIP ct/sa; resample `z_extrap` → `z`.
	- `ismip7-antarctic-remap-clim` — remap observational climatology to the ISMIP grid.
	- `ismip7-antarctic-extrap-clim` — extrapolate remapped climatology (adds a dummy singleton time for Fortran; removed in final output).

- Bias correction & TF
	- `ismip7-antarctic-bias-corr-classic` — classic bias correction toward reference climatology; run once per model + future scenario (uses historical + future extrapolated inputs).
	- `ismip7-antarctic-cmip-ct-sa-to-tf` — compute TF from bias-corrected CMIP CT/SA.
	- `ismip7-antarctic-clim-ct-sa-to-tf` — compute TF from extrapolated climatology CT/SA (no time).

- Annual & back-conversion
	- `ismip7-antarctic-annual-average` — compute weighted annual means from monthly files.
	- `ismip7-antarctic-cmip-annual-averages` — CMIP driver: make annual means for bias-corrected CT/SA and TF.
	- `ismip7-antarctic-cmip-annual-ct-sa-to-thetao-so` — convert annual ct/sa/tf back to thetao/so (CMIP path).
	- `ismip7-antarctic-clim-ct-sa-to-thetao-so` — convert climatology ct/sa/tf to thetao/so (climatology path).

Each command supports `--help` for usage details. See the User Guide for end-to-end examples.

## Python API (entry points)

- Split & Convert
	- `i7aof.convert.split.main` — `ismip7-antarctic-split-cmip`
	- `i7aof.convert.cmip_to_ct_sa.main` — `ismip7-antarctic-convert-cmip-to-ct-sa`

- Remap & Extrapolate
	- `i7aof.remap.cmip.main` — `ismip7-antarctic-remap-cmip`
	- `i7aof.extrap.cmip.main` — `ismip7-antarctic-extrap-cmip`
	- `i7aof.remap.clim.main` — `ismip7-antarctic-remap-clim`
	- `i7aof.extrap.clim.main` — `ismip7-antarctic-extrap-clim`

- Bias correction & TF
	- `i7aof.biascorr.classic.main` — `ismip7-antarctic-bias-corr-classic`
	- `i7aof.convert.ct_sa_to_tf.main_cmip` — `ismip7-antarctic-cmip-ct-sa-to-tf`
	- `i7aof.convert.ct_sa_to_tf.main_clim` — `ismip7-antarctic-clim-ct-sa-to-tf`

- Annual & back-conversion
	- `i7aof.time.average.main` — `ismip7-antarctic-annual-average`
	- `i7aof.time.cmip.main` — `ismip7-antarctic-cmip-annual-averages`
	- `i7aof.convert.ct_sa_to_thetao_so.main_cmip` — `ismip7-antarctic-cmip-annual-ct-sa-to-thetao-so`
	- `i7aof.convert.ct_sa_to_thetao_so.main_clim` — `ismip7-antarctic-clim-ct-sa-to-thetao-so`

### Non-public internals

Helper modules `i7aof.remap.shared` and `i7aof.extrap.shared` provide the
factored logic reused by CMIP and climatology workflows. They are **not**
considered public API; do not import them directly in external scripts.

All public modules are covered in the API reference (see `docs/api/index.md`),
except the internal shared helpers noted above. When proposing new CLI entry
points add both: (1) the console script entry in `pyproject.toml` and (2) an
item in this document plus the relevant user guide page.
