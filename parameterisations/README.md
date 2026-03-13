# parameterisations

A toolbox for parameter selection for Antarctic melt parameterisations for
the [ISMIP7]() activity

## Documentation

Parameter selection is specified in the [draft protocol](https://docs.google.com/document/d/17oYdWFC61RbDxzK2xMnnnUv4WrjgxoJz/edit?usp=sharing&ouid=106669563949845798274&rtpof=true&sd=true).

Contains the toolbox parameter_selection_toolbox.py with functions to help with parameter selection and two examples.

First example is the quadratic parameterisation (quadratic local with mean Antarctic slope as defined in Burgard et al., 2022). Note that here we calculate melt rates independently. In ISMIP7/meltMIP, the melt rate should be calculated through the ice-sheet model code and grid instead.
parameter_selection_quadratic_example.ipynb

Second example is PICO, based on simulations done with PISM-PICO. If you would like access to those, please contact Ronja.
parameter_selection_pico_example.ipynb

Toolbox:
- calculate_objective_function: calculates the optimal parameters specified for a number of samples
- calculate_term1, calculate_term2, calculate-term3 calculate the different terms
- optimise_deltaT: identifies optimal deltaT for a given fixed parameter value

## Data input requirements

Toolbox requires ensemble of modelled melt rates for a range of parameter values. These are given in xarray dataset called pd_ensemble, with parameter values indexing the melt rate field. Parameters are called "p1" and "p2" (if only one parameter set is used, just set p2=1). Melt rates are saved in variable called "melt_rate" given in kg/m2/a. This should contains polar stereographic x and y coordinates.

Equally, you will need to create a cold and warm ensemble for each tuning dataset (mathiot_cold_ensemble, mathiot_warm_ensemble, timmermann_cold_ensemble, timmermann_warm_ensemble,..) which are also indexed by p1 and p2 and are corresponding modelled melt rates.

Furthermore, you will need a
- basin mask, available through ismip for the 8km grid (used for observational data), and on your model tuning grid (only regular grids are suported at the moment).
- grounded, floating mask, which is 1 in floating regions/ice shelves and 0 otherwise, again on 8km for data (provided) and on your model tuning grid
- buttressing bins, on 8km for data and on yoour model tuning grid (provided)
