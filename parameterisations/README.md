# parameterisations

A toolbox for parameter selection for Antarctic melt parameterisations for
the [ISMIP7]() activity

## Documentation

Parameter selection is specified in the [draft protocol](https://docs.google.com/document/d/17oYdWFC61RbDxzK2xMnnnUv4WrjgxoJz/edit?usp=sharing&ouid=106669563949845798274&rtpof=true&sd=true).

Contains the toolbox parameter_selection_toolbox_v2.py with functions to help with parameter selection and two examples. Note that parameter_selection_toolbox.py is deprecated.

First example is the quadratic parameterisation (quadratic local with mean Antarctic slope as defined in Burgard et al., 2022). Note that here we calculate melt rates independently. In ISMIP7/meltMIP, the melt rate should be calculated through the ice-sheet model code and grid instead.
parameter_selection_quadratic_example_v2.ipynb

Second example is PICO, based on simulations done with PISM-PICO. If you would like access to those, please contact Ronja.
parameter_selection_pico_example_v2.ipynb

Toolbox:
- calculate_term1, calculate_term2, calculate_term3, calculate_term4 calculate the different terms
- calculate_objective_function: calculates the optimal parameters specified for a number of samples, with freely chosen weights (see below)
- optimise_deltaT: identifies optimal deltaT for a given fixed parameter value

Weights: You are free to adjust the weights of the different terms in the parameterisation, as these are simply inputs into the objective_function calculation. Please see the examples on how to do this.

## Data input requirements

1) Toolbox requires ensemble of modelled melt rates for a range of parameter values. These are given in xarray dataset called pd_ensemble, with parameter values indexing the melt rate field. Parameters are called "p1" and "p2" (if only one parameter set is used, just set p2=1). Melt rates are saved in variable called "melt_rate" given in kg/m2/a. This should contains polar stereographic x and y coordinates.

2) Equally, you will need to create a cold and warm ensemble (cold_ensemble, warm_ensemble) for each tuning dataset which are also indexed by p1 and p2 and the model ('mathiot', 'naughten_ais_1',..). They contain corresponding melt rates modelled with your melt module (variable called "melt_rates").
A range of cold and warm ocean states, as well as melt rates from 3D ocean dynamical models are available:
Circum-Antarctic
- Mathiot et al., 2023
- Naughten et al., 2018 (2 simulations)
Amundsen Sea
- Jourdain et al. 2022 (3 simulations)
- Naughten et al., 2023 (10 simulations)
Weddell Sea
- Timmermann and Goeller 2017
- Haid et al., 2023 (3 simulations)
- Naughten et al., 2021 (2 simulations)

Since doing all would be a considerable amount of work, we suggest using 2 circum-Antarctic simulations ("mathiot" and "naughten_ais_1"). However, if modellers want to focus on one particular region, e.g., the Amundsen Sea or the Weddell Sea, we suggest using the corresponding, higher resolution datasets in this case.

3) You will need to create also an ensemble for observational data (obs_ensemble) for observational datasets from the Amundsen Sea, which are again indexed by p1 and p2, the ice shelf ("pig" or "dotson") and the year of the observation. We provide dataset created from observations on the ismip 8km ocean grid for
- 1994 pig
- 2000 dotson
- 2006 dotson
- 2007 pig, dotson
- 2009 pig, dotson
- 2010 pig
- 2011 dotson
- 2012 pig, dotson
- 2014 pig, dotson
- 2016 pig, dotson
- 2018 dotson
- 2019 pig
- 2020 pig

Again, we suggest using only a subset of the datasets. As Dotson data is more complicated to interpret due to the connection between Dotson and Crosson ice shelves, we suggest using 2 years for Pine Island. Suggested years still need to be determined.

4)
Furthermore, you will need a
- basin mask, available through ismip for the 8km grid (used for observational data), and on your model tuning grid (only regular grids are suported at the moment).
- grounded, floating mask, which is 1 in floating regions/ice shelves and 0 otherwise, again on 8km for data (provided) and on your model tuning grid
- buttressing bins, on 8km for data and on yoour model tuning grid (provided)
