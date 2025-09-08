# CLI and Python API

This page lists the primary command-line tools and the top-level Python APIs exposed by this project.

## Command-line tools

- `ismip7-antarctic-ocean-forcing` — main workflow driver.
- `ismip7-antarctic-convert-cmip` — convert CMIP thetao/so to ct/sa on the
	native grid.
- `ismip7-antarctic-remap-cmip` — remap CMIP data to the ISMIP grid.

Each command supports `--help` for usage details. See the User Guide for end-to-end examples.

## Python API (entry points)

- `i7aof.__main__.main` — implementation behind `ismip7-antarctic-ocean-forcing`.
- `i7aof.convert.cmip.main` — implementation behind
	`ismip7-antarctic-convert-cmip`.
- `i7aof.remap.cmip.main` — implementation behind `ismip7-antarctic-remap-cmip`.

More detailed API documentation (via autodoc/autosummary) will be added as interfaces stabilize.
