# Bias Correction

The bias-correction approach (early draft):

1. Compute a climatology of CMIP data over a chosen baseline period (TBD).
2. Subtract the CMIP climatology, add the observational climatology.
3. Extrapolate into shelves, cavities, ice, and bathymetry as needed.

Modules: `i7aof.biascorr.timeslice`, `i7aof.biascorr.projection`.
