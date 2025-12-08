# meltmip

A toolbox for parameter selection for Antarctic melt parameterisations for
the [ISMIP7]() activity

## Documentation

Parameter selection is specified in the [draft protocol](https://docs.google.com/document/d/17oYdWFC61RbDxzK2xMnnnUv4WrjgxoJz/edit?usp=sharing&ouid=106669563949845798274&rtpof=true&sd=true).

Contains the toolbox parameter_selection_toolbox.py with functions to help with parameter selection and two examples.

First example is the quadratic parameterisation (quadratic local with mean Antarctic slope as defined in Burgard et al., 2022). Note that here we calculate melt rates independently. In ISMIP7/MeltMIP, the melt rate should be calculated through the ice-sheet model code and grid instead.
parameter_selection_quadratic_example.ipynb
(deltaT selection in parameter_selection_quadratic_deltaT_example.ipynb)

Second example is PICO, based on simulations done with PISM-PICO. If you would like access to those, please contact Ronja.
parameter_selection_pico_example.ipynb
(deltaT selection in parameter_selection_pico_deltaT_example.ipynb)

Toolbox:
- calculate_objective_function: calculates the optimal parameters specified for a number of samples
- select_optimal_deltaT: identifies optimal deltaT
- select_subensemble_using_optimal_deltaT: if optimal deltaT has been selected, this function applies this selection to any given ensemble, e.g., to the melt rates calculated using the ocean modelling data
- load_melt_rates_into_dataset: creates one dataset containing melt rates indexed by the parameter (s), this is used as input for calculate_objective_function

##
