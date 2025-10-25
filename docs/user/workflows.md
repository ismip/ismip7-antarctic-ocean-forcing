# Workflows

Common end-to-end workflows for generating forcing datasets.

- Prepare configuration files (see examples in `scripts/*.cfg`).
- Run remapping, extrapolation, and bias-correction steps.
- Compute thermal forcing (TF) from CT/SA:
	- For CMIP bias-corrected outputs:
		`ismip7-antarctic-cmip-ct-sa-to-tf --model <m> --scenario <s> --clim <c>`
	- For extrapolated climatologies:
		`ismip7-antarctic-clim-ct-sa-to-tf --clim <c>`
- Validate and package outputs.
