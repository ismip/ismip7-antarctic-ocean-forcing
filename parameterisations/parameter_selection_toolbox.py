import os

import numpy as np
import xarray as xr


def calculate_objective_function(
    optimise,
    term2_spec,
    term3_spec,
    term3_opt,
    w3_spec,
    w3_only_basin,
    sample_size,
    average_as,
    basins,
    mask,
    bfrn,
    reso,
    ice_density,
    melt_obs,
    MeltData,
    data_path,
    pd_ensemble,
    mathiot_cold_ensemble,
    mathiot_warm_ensemble,
):
    """
    Input:
    -optimise: "all", "term1", "term2", "term3"
    -term2_spec: "aggregated", "average"
    -term3_spec:  "aggregated", "average"
    -term3_opt: "anomaly" "both"
    -w3_spec: "only_cold", "only_warm", "none" only works for term3_opt="both"
    -w3_only_basin: "false", or a basin number to be sampled
    -average_as: "true", "false"
    -sample_size: number of random samples
    -basins: basin mask to use
    -mask: mask that identified ice shelves (1) and otherwise (0)
    -bfrn: dataset containg bfrn bins and values for J2
    -reso: resolution in m
    -ice_density: used for conversion to Gt/a, in kg/m3
    -melt_obs: observational melt, in kg/m2/a
    -Melt_Data: basin-aggregated melt rates
    -data_path: path to ocean modelling melt rates
    -pd_ensemble: name of dataset containing present day melt rates for all
                  p1, p2 combinations, with optimised deltaT
    -mathiot_cold_ensemble: same, but ocean modelling data is the cold mathiot
    -mathiot_warm_ensemble: same, but with warm dataset
    Output:
    -min_p1: list of p1 values that minimise randomly sampled objective
             function, length of sample size
    -min_p2: list of cooresponding p2 values
    -min_coords: minimum values attained for the minimum p1,p2 values

    This function finds the pair of p1,p2 for which the parameterised melt
    optimises three terms:
    -J1: basin-integrated melt for present-day
    -J2: buttressing-bin integrated melt for present-day,
         weighted by buttressing
    -J3: basin-integrated melt for cold and warm cases of the
         ocean modelling datasets
    """

    nBasins = int(basins.max())
    cvt = reso**2 * ice_density / 1e12

    ########
    # TERM 1
    if optimise == 'term1' or optimise == 'all':
        print('Calculate Term 1')
        t1_model, t1_obs, t1_obs_mean, t1_obs_sigma = calculate_term1(
            pd_ensemble, mask, basins, nBasins, cvt, MeltData, sample_size
        )

    ########
    # TERM 2
    if optimise == 'term2' or optimise == 'all':
        print('Calculate Term 2')
        t2_model, t2_obs, t2_weights, t2_obs_mean, t2_obs_sigma = (
            calculate_term2(
                pd_ensemble,
                mask,
                bfrn,
                cvt,
                sample_size,
                term2_spec,
                ice_density,
                melt_obs,
            )
        )

    ########
    # TERM 3
    if optimise == 'term3' or optimise == 'all':
        print('Calculate Term 3')
        t3_model, t3_obs, t3_weights, t3_obs_mean, t3_obs_sigma = (
            calculate_term3(
                pd_ensemble,
                mathiot_cold_ensemble,
                mathiot_warm_ensemble,
                mask,
                basins,
                cvt,
                sample_size,
                term3_spec,
                term3_opt,
                w3_spec,
                ice_density,
                data_path,
                nBasins,
                w3_only_basin,
            )
        )

    ################################################
    # randomly sample the weights of the three terms
    a1 = np.random.uniform(0, 1, size=sample_size)
    a2 = np.random.uniform(0, 1, size=sample_size)
    a3 = np.random.uniform(0, 1, size=sample_size)

    if average_as == 'true':
        # make sure they add up to 1
        asum = a1 + a2 + a3
        a1 = a1 / asum
        a2 = a2 / asum
        a3 = a3 / asum

    #####################
    # Find optimal p1,p2
    min_p1 = []
    min_p2 = []
    print('Sampling, this might take a moment...')

    for s in range(sample_size):
        if optimise == 'all':
            term1 = mae(t1_model, t1_obs[:, s], 1, 'basins')
            term2 = mae(t2_model, t2_obs[:, s], t2_weights, ['BFRN_bins'])
            term3 = mae(t3_model, t3_obs[:, s], t3_weights[s, :], ['basins'])
            objective_function = (
                a1[s] * term1 / term1.median()
                + a2[s] * term2 / term2.median()
                + a3[s] * term3 / term3.median()
            )
        elif optimise == 'term1':
            term1 = mae(t1_model, t1_obs[:, s], 1, 'basins')
            objective_function = term1 / term1.median()
        elif optimise == 'term2':
            term2 = mae(t2_model, t2_obs[:, s], t2_weights, ['BFRN_bins'])
            objective_function = term2 / term2.median()
        elif optimise == 'term3':
            term3 = mae(t3_model, t3_obs[:, s], t3_weights[s, :], ['basins'])
            objective_function = term3 / term3.median()
        else:
            print('Specify term to optimise')
            return

        min_val = np.nanmin(objective_function)
        min_p1.append(
            objective_function.where(
                objective_function == min_val, drop=True
            ).p1.values[0]
        )
        min_p2.append(
            objective_function.where(
                objective_function == min_val, drop=True
            ).p2.values[0]
        )

    return min_p1, min_p2, a1, a2, a3  # , t3_weights


def mae(predicted=None, observed=None, weights=1, dims='basins'):
    """
    Calculates mean absolute error
    """
    return (
        abs(weights * (predicted - observed))
        .mean(dims, skipna=False)
        .rename('result')
    )


def calculate_term1(
    pd_ensemble, mask, basins, nBasins, cvt, MeltData, sample_size
):
    ########
    # TERM 1
    # parameterisaition melt, aggregate to Gt/a per basin
    t1_model = (
        pd_ensemble['melt_rate']
        .where(mask, np.nan)
        .groupby(basins)
        .sum(skipna=True)
        * cvt
    )  # convert to Gt/a
    # make sure to remove regions that do not have optimal dT for any basin
    t1_model = t1_model.where(t1_model != 0, np.nan)

    # Observed melt in Gt/a per basin, observed melt is "sample_size"-times
    # randomly sampled assuming normal distribution
    t1_obs_mean = MeltData['BMR (Gt/yr)'].values
    t1_obs_sigma = MeltData['BMR uncert (Gt/yr)'].values
    t1_obs = []
    for b in range(nBasins + 1):
        t1_obs = t1_obs + [
            np.random.normal(
                loc=t1_obs_mean[b], scale=t1_obs_sigma[b], size=sample_size
            )
        ]
    t1_obs = np.array(t1_obs)

    return t1_model, t1_obs, t1_obs_mean, t1_obs_sigma


def calculate_term2(
    pd_ensemble,
    mask,
    bfrn,
    cvt,
    sample_size,
    term2_spec,
    ice_density,
    melt_obs,
):
    ########
    # TERM 2
    if term2_spec == 'aggregate':
        # parameterisation melt aggregated per buttressing bin, in Gt/a
        t2_model = (
            pd_ensemble['melt_rate']
            .where(mask, np.nan)
            .groupby(bfrn['BFRN_bins'])
            .sum()
            * cvt
        )
        t2_model = t2_model.where(t2_model != 0, np.nan)

        # Observed melt in Gt/a per buttressing bin, observed melt is
        # "sample_size"-times randomly sampled assuming normal distribution
        t2_obs_mean = (
            melt_obs['melt_mean']
            .where(mask, np.nan)
            .groupby(bfrn['BFRN_bins'])
            .sum()
            * cvt
            / ice_density  # since in kg/m2/a
        )
        t2_obs_sigma = (
            melt_obs['melt_mean_err']
            .where(mask, np.nan)
            .groupby(bfrn['BFRN_bins'])
            .sum()
            * cvt
            / ice_density
        )
        t2_obs = []
        nBins = 10
        for b in range(nBins):
            t2_obs = t2_obs + [
                np.random.normal(
                    loc=t2_obs_mean[b], scale=t2_obs_sigma[b], size=sample_size
                )
            ]
        t2_obs = np.array(t2_obs)

    elif term2_spec == 'average':
        # parameterisation melt averaged per buttressing bin, in m/a -> kg/m2/a
        t2_model = (
            pd_ensemble['melt_rate']
            .where(mask, np.nan)
            .groupby(bfrn['BFRN_bins'])
            .mean()
        ) * ice_density
        t2_model = t2_model.where(t2_model != 0, np.nan)

        # Observed melt in kg/m2/a per buttressing bin, observed melt is
        # "sample_size"-times randomly sampled assuming normal distribution
        t2_obs_mean = (
            melt_obs['melt_mean']
            .where(mask, np.nan)
            .groupby(bfrn['BFRN_bins'])
            .mean()
        )
        # Note that to calculate uncertainty, we calculate for each bin
        # sqrt(uncert1**2 + uncert2**2 +...) / number of terms
        t2_obs_sigma = np.sqrt(
            ((melt_obs['melt_mean_err'].where(mask, np.nan)) ** 2)
            .groupby(bfrn['BFRN_bins'])
            .mean()
        )
        t2_obs = []
        nBins = 10
        for b in range(nBins):
            t2_obs = t2_obs + [
                np.random.normal(
                    loc=t2_obs_mean[b], scale=t2_obs_sigma[b], size=sample_size
                )
            ]
        t2_obs = np.array(t2_obs)
    else:
        print('Please specify term2_spec correctly')

    # important to only use values, as dim name does not match
    t2_weights = (bfrn['BFRN_medians'] / bfrn['BFRN_median']).values

    return t2_model, t2_obs, t2_weights, t2_obs_mean, t2_obs_sigma


def calculate_term3(
    pd_ensemble,
    mathiot_cold_ensemble,
    mathiot_warm_ensemble,
    mask,
    basins,
    cvt,
    sample_size,
    term3_spec,
    term3_opt,
    w3_spec,
    ice_density,
    data_path,
    nBasins,
    w3_only_basin,
):
    ########
    # TERM 3
    # prep data
    if term3_spec == 'aggregate':
        # parameterisation melt, aggregate to Gt/a per basin for cold ocean
        t3_model_mat_c = (
            mathiot_cold_ensemble['melt_rate'].where(mask, np.nan)
        ).groupby(basins).sum() * cvt
        t3_model_mat_c = t3_model_mat_c.where(t3_model_mat_c != 0, np.nan)

        # parameterisation melt, aggregate to Gt/a per basin for warm ocean
        t3_model_mat_w = (
            mathiot_warm_ensemble['melt_rate'].where(mask, np.nan)
        ).groupby(basins).sum() * cvt  # .where(t1_model != 0, np.nan)
        t3_model_mat_w = t3_model_mat_w.where(t3_model_mat_w != 0, np.nan)

        # Ocean modelling data that is target melt, aggregate to Gt/a per basin
        tmp = xr.load_dataset(
            os.path.join(
                data_path,
                'parameterisations',
                'Ocean_Modelling_Data',
                'Mathiot23_cold_m.nc',
            )
        )
        t3_obs_mat_mean_c = (
            tmp['melt_rate']
            .where(mask, np.nan)
            .groupby(basins)
            .sum(skipna=True)
            * cvt
            / ice_density  # as unit is kg/m2/a
        )
        t3_obs_mat_sigma_c = (
            tmp['melt_rate_uncert']
            .where(mask, np.nan)
            .groupby(basins)
            .sum(skipna=True)
            * cvt
            / ice_density  # as unit is kg/m2/a
        )
        tmp.close()

        # warm case
        tmp = xr.load_dataset(
            os.path.join(
                data_path,
                'parameterisations',
                'Ocean_Modelling_Data',
                'Mathiot23_warm_m.nc',
            )
        )
        t3_obs_mat_mean_w = (
            tmp['melt_rate']
            .where(mask, np.nan)
            .groupby(basins)
            .sum(skipna=True)
            * cvt
            / ice_density  # as unit is kg/m2/a
        )
        t3_obs_mat_sigma_w = (
            tmp['melt_rate_uncert']
            .where(mask, np.nan)
            .groupby(basins)
            .sum(skipna=True)
            * cvt
            / ice_density  # as unit is kg/m2/a
        )

    elif term3_spec == 'average':
        # parameterisation melt, average to kg/m2/a per basin for cold ocean
        t3_model_mat_c = (
            mathiot_cold_ensemble['melt_rate'].where(mask, np.nan)
        ).groupby(basins).mean() * ice_density
        t3_model_mat_c = t3_model_mat_c.where(t3_model_mat_c != 0, np.nan)

        # parameterisation melt, aggregate to Gt/a per basin for warm ocean
        t3_model_mat_w = (
            mathiot_warm_ensemble['melt_rate'].where(mask, np.nan)
        ).groupby(basins).mean() * ice_density  # where(t1_model != 0, np.nan)
        t3_model_mat_w = t3_model_mat_w.where(t3_model_mat_w != 0, np.nan)

        # Ocean modelling data that is target melt, average kg/m2/a per basin
        tmp = xr.load_dataset(
            os.path.join(
                data_path,
                'parameterisations',
                'Ocean_Modelling_Data',
                'Mathiot23_cold_m.nc',
            )
        )
        t3_obs_mat_mean_c = (
            tmp['melt_rate']
            .where(mask, np.nan)
            .groupby(basins)
            .mean(skipna=True)
            # as unit is kg/m2/a
        )
        t3_obs_mat_sigma_c = np.sqrt(
            (tmp['melt_rate_uncert'].where(mask, np.nan) ** 2)
            .groupby(basins)
            .mean(skipna=True)
        )
        tmp.close()

        # warm case
        tmp = xr.load_dataset(
            os.path.join(
                data_path,
                'parameterisations',
                'Ocean_Modelling_Data',
                'Mathiot23_warm_m.nc',
            )
        )
        t3_obs_mat_mean_w = (
            tmp['melt_rate']
            .where(mask, np.nan)
            .groupby(basins)
            .mean(skipna=True)
        )
        t3_obs_mat_sigma_w = np.sqrt(
            (tmp['melt_rate_uncert'].where(mask, np.nan) ** 2)
            .groupby(basins)
            .mean(skipna=True)
        )
    else:
        print('Make sure to specify correct term3_spec')

    #####################
    # Combine datasets

    if term3_opt == 'both':
        t3_model = xr.concat(
            [
                t3_model_mat_c.assign_coords(
                    {'basins': [str(i) + 'c' for i in range(nBasins + 1)]}
                ),
                t3_model_mat_w.assign_coords(
                    {'basins': [str(i) + 'w' for i in range(nBasins + 1)]}
                ),
            ],
            dim='basins',
        )
        t3_obs_mean = xr.concat(
            [
                t3_obs_mat_mean_c.assign_coords(
                    {'basins': [str(i) + 'c' for i in range(nBasins + 1)]}
                ),
                t3_obs_mat_mean_w.assign_coords(
                    {'basins': [str(i) + 'w' for i in range(nBasins + 1)]}
                ),
            ],
            dim='basins',
        )

        t3_obs_sigma = xr.concat(
            [
                t3_obs_mat_sigma_c.assign_coords(
                    {'basins': [str(i) + 'c' for i in range(nBasins + 1)]}
                ),
                t3_obs_mat_sigma_w.assign_coords(
                    {'basins': [str(i) + 'w' for i in range(nBasins + 1)]}
                ),
            ],
            dim='basins',
        )

        # For melt targets, randomly sample
        t3_obs = []
        for b in range(len(t3_obs_mean)):
            t3_obs = t3_obs + [
                np.random.normal(
                    loc=t3_obs_mean[b], scale=t3_obs_sigma[b], size=sample_size
                )
            ]
        t3_obs = np.array(t3_obs)

        t3_weights = []

        for _s in range(sample_size):
            # sample all basins, make sure to give same weight to cold and warm
            tmp = np.random.uniform(0, 1, size=int(len(t3_model.basins) / 2))
            tmp = np.tile(tmp, 2)
            tmp = tmp / tmp.sum()
            if w3_spec == 'only_warm':
                tmp[: len(tmp) // 2] = 0
            if w3_spec == 'only_cold':
                tmp[len(tmp) // 2 :] = 0
            t3_weights = t3_weights + [tmp]

        t3_weights = np.array(t3_weights)

    elif term3_opt == 'anomaly':
        t3_model = t3_model_mat_w - t3_model_mat_c

        t3_obs_mean = t3_obs_mat_mean_w - t3_obs_mat_mean_c

        # Note that this assumes errors to be uncorrelated
        # which is the most conservative approach
        t3_obs_sigma = np.sqrt(t3_obs_mat_sigma_w**2 + t3_obs_mat_sigma_c**2)

        # For melt targets, randomly sample
        t3_obs = []
        for b in range(len(t3_obs_mean)):
            t3_obs = t3_obs + [
                np.random.normal(
                    loc=t3_obs_mean[b], scale=t3_obs_sigma[b], size=sample_size
                )
            ]
        t3_obs = np.array(t3_obs)

        t3_weights = []

        for _s in range(sample_size):
            tmp = np.random.uniform(0, 1, size=int(len(t3_model.basins)))
            if w3_only_basin != 'false':
                tmp = np.zeros(len(t3_model.basins))
                tmp[w3_only_basin] = 1
            t3_weights = t3_weights + [tmp]

        t3_weights = np.array(t3_weights)

    else:
        print('Please specify correct term3_opt')

    return t3_model, t3_obs, t3_weights, t3_obs_mean, t3_obs_sigma


def optimise_deltaT(dT_ensemble, basins, reso, ice_density, MeltDataImbie):
    """
    Calculate optimal deltaT for basin-wide present-day melt.
    """

    number_of_basins = int(basins.max().values)
    cvt = reso**2 * ice_density / 1e12  # to convert to Gt/a

    optimal_deltaT_per_basin = []
    residual_per_basin = []
    sensitivity_per_basin = []

    param_melt_rate = dT_ensemble.sel(deltaT=0).copy(deep=True) * np.nan

    for basin_i in range(number_of_basins + 1):
        bmr = dT_ensemble.where(basins == basin_i, 0.0).sum(['x', 'y']) * cvt

        # only use deltaT between -2 and 2
        bmr = bmr.where(
            np.logical_and(bmr['deltaT'] <= 2, bmr['deltaT'] >= -2), np.nan
        )

        optimal_deltaT_per_basin.append(
            np.round(
                (abs(bmr - MeltDataImbie.loc[basin_i, 'BMR (Gt/yr)']))
                .idxmin()
                .item(),
                3,
            )
        )
        residual_per_basin.append(
            (abs(bmr - MeltDataImbie.loc[basin_i, 'BMR (Gt/yr)'])).min().item()
        )

        # if an optimal delatT exists, save melt and calc melt sensitivity
        if not np.isnan(optimal_deltaT_per_basin[-1]):
            param_melt_rate = param_melt_rate.where(
                basins != basin_i,
                dT_ensemble.sel(
                    deltaT=optimal_deltaT_per_basin[-1], method='nearest'
                ),
            )
            # Calc approx melt sensitivity.
            # otherwise approximate with higher values, ideally +1deg C
            # Note that this is only approximate
            sensitivity_per_basin.append(
                np.round(
                    (
                        (
                            dT_ensemble.sel(
                                deltaT=optimal_deltaT_per_basin[-1] + 1,
                                method='nearest',
                            )
                            .where(basins == basin_i, np.nan)
                            .mean()
                            - param_melt_rate.where(
                                basins == basin_i, np.nan
                            ).mean()
                        ).item()
                        / 1
                    ),
                    2,
                )
            )
        else:
            sensitivity_per_basin.append(np.nan)
    result_ds = xr.Dataset(
        data_vars=dict(
            melt_rate=(['y', 'x'], param_melt_rate.values),
            optimal_deltaT_per_basin=(
                ['basin'],
                np.array(optimal_deltaT_per_basin),
            ),
            sensitivity_per_basin=(['basin'], np.array(sensitivity_per_basin)),
            residual_per_basin=(['basin'], np.array(residual_per_basin)),
        ),
        coords=dict(
            x=(['x'], dT_ensemble['x'].values),
            y=(['y'], dT_ensemble['y'].values),
            basin=(['basin'], np.arange(0, number_of_basins + 1)),
        ),
    )

    return result_ds


'''
Deprecated

def select_optimal_deltaT(
    ds, basins, boxes, obs_data, param_type, outname, reso, ice_density
):
    """
    Input:
    - ds: xarray dataset containing melt rates (m.i.e/a), with a dimension
      deltaT that will be optimised over, on a regular grid
    - basins: xarray dataset containing basin numbers, starting at 1
    - obs_data: data frame containing basin-aggregated melt rates in Gt/a
    - param_type: pico, quadratic, ...
    - outname: output file to save to
    Output:
    - saves a netcdf file to "outname" containing the melt rates for each
    basin based on optimal deltaT, and arrays of optimal_deltaT, residuals
    and melt sensitivities
    """

    # print('Identifying optimal delta T for each basin...')

    number_of_basins = int(basins.max().values)
    cvt = reso**2 * ice_density / 1e12  # to convert to Gt/a

    optimal_deltaT_per_basin = []
    residual_per_basin = []
    sensitivity_per_basin = []

    param_melt_rate = ds['melt_rate'].sel(deltaT=0).copy(deep=True) * np.nan

    for basin_i in range(1, number_of_basins + 1):
        bmr = (
            ds['melt_rate'].where(basins == basin_i, 0.0).sum(['x', 'y']) * cvt
        )
        if param_type == 'pico':
            # Add physical constraints from Reese et al., 2018
            bmrBox1 = (
                ds['melt_rate']
                .where(np.logical_and(basins == basin_i, boxes == 1), 0.0)
                .sum(['x', 'y'])
                * cvt
            )
            bmrBox2 = (
                ds['melt_rate']
                .where(np.logical_and(basins == basin_i, boxes == 2), 0.0)
                .sum(['x', 'y'])
                * cvt
            )
            bmr = bmr.where(
                np.logical_and(bmrBox1 > 0, bmrBox1 > bmrBox2), np.nan
            )

        # only use deltaT between -2 and 2
        bmr = bmr.where(
            np.logical_and(bmr['deltaT'] <= 2, bmr['deltaT'] >= -2), np.nan
        )

        optimal_deltaT_per_basin.append(
            (abs(bmr - obs_data.loc[basin_i, 'BMR (Gt/yr)'])).idxmin()
        )
        residual_per_basin.append(
            (abs(bmr - obs_data.loc[basin_i, 'BMR (Gt/yr)'])).min()
        )

        # if an optimal delatT exists, save melt and calc melt sensitivity
        if not np.isnan(optimal_deltaT_per_basin[-1]):
            param_melt_rate = param_melt_rate.where(
                basins != basin_i,
                ds['melt_rate'].sel(deltaT=optimal_deltaT_per_basin[-1]),
            )
            # Calc approx melt sensitivity.
            # If this is the max deltaT, use half a degree colder,
            # otherwise approximate with higher values, ideally +1deg C
            # Note that this is only approximate
            if optimal_deltaT_per_basin[-1] == ds.deltaT.max():
                sensitivity_per_basin.append(
                    (
                        param_melt_rate.where(basins == basin_i, np.nan).mean()
                        - ds['melt_rate']
                        .sel(
                            deltaT=optimal_deltaT_per_basin[-1] - 0.5,
                            method='nearest',
                        )
                        .where(basins == basin_i, np.nan)
                        .mean()
                    )
                    / 0.5
                )
            else:
                sensitivity_per_basin.append(
                    (
                        ds['melt_rate']
                        .sel(
                            deltaT=optimal_deltaT_per_basin[-1] + 1,
                            method='nearest',
                        )
                        .where(basins == basin_i, np.nan)
                        .mean()
                        - param_melt_rate.where(
                            basins == basin_i, np.nan
                        ).mean()
                    )
                    / 1
                )
        else:
            sensitivity_per_basin.append(np.nan)

    result_ds = xr.Dataset(
        data_vars=dict(
            melt_rate=(['y', 'x'], param_melt_rate.values),
            optimal_deltaT_per_basin=(
                ['basin'],
                np.array(optimal_deltaT_per_basin),
            ),
            sensitivity_per_basin=(['basin'], np.array(sensitivity_per_basin)),
            residual_per_basin=(['basin'], np.array(residual_per_basin)),
        ),
        coords=dict(
            x=(['x'], ds['x'].values),
            y=(['y'], ds['y'].values),
            basin=(['basin'], np.arange(1, number_of_basins + 1)),
        ),
    )
    result_ds.to_netcdf(outname)
    ds.close()
    result_ds.close()
    return result_ds.drop_vars('melt_rate')
'''

''' Deprecated
def select_subensemble_using_optimal_deltaT(
    ds, basins, opt_ensemble, outname, p1, p2
):
    """
    Input:
    - ds: xarray dataset containing melt rates (m.i.e/a),
          wit dimension deltaT that will be selected from, on a regular grid
    - basins: xarray dataset containing basin numbers, starting at 1
    - opt_ensemble: array containing optimised deltaT's for present-day
    - outname: output file to save to
    Output:
    - saves a netcdf file to "outname" containing the melt rates for each
      basin based on optimal deltaT
    """

    # print('Select sub-ensemble...')

    number_of_basins = int(basins.max().values)

    # Create melt rate dataset based on optimal deltaT
    melt_rate = ds['melt_rate'].sel(deltaT=0).copy(deep=True) * np.nan

    for basin in range(1, number_of_basins + 1):
        optimal_deltaT = opt_ensemble['optimal_deltaT_per_basin'].loc[
            dict(p1=p1, p2=p2, basin=basin)
        ]

        if np.isnan(optimal_deltaT.values):
            melt_rate = melt_rate.where(
                basins != basin, ds['melt_rate'].sel(deltaT=0) * np.nan
            )
        else:
            melt_rate = melt_rate.where(
                basins != basin, ds['melt_rate'].sel(deltaT=optimal_deltaT)
            )

    result_ds = xr.Dataset(
        data_vars=dict(
            melt_rate=(['y', 'x'], melt_rate.values),
        ),
        coords=dict(x=(['x'], ds['x'].values), y=(['y'], ds['y'].values)),
    )

    result_ds.to_netcdf(outname)
'''


def load_melt_rates_into_dataset(
    ensemble_name, ensemble_table, ensemble_path, p1_name, p2_name
):
    print('Loading ' + ensemble_name + ' into one dataset...')
    members = []
    p1s = []
    p2s = []

    for _i, ehash in enumerate(ensemble_table.index):
        p1 = ensemble_table.loc[ehash, p1_name]
        p2 = ensemble_table.loc[ehash, p2_name]
        p1s.append(p1)
        p2s.append(p2)

        if os.path.isfile(
            os.path.join(
                ensemble_path,
                ensemble_name + '_' + str(ehash) + '/optimised.nc',
            )
        ):
            ds = xr.load_dataset(
                os.path.join(
                    ensemble_path,
                    ensemble_name + '_' + str(ehash) + '/optimised.nc',
                )
            )
        elif os.path.isfile(
            os.path.join(
                ensemble_path,
                ensemble_name + '_' + str(ehash) + '_optimised.nc',
            )
        ):
            ds = xr.load_dataset(
                os.path.join(
                    ensemble_path,
                    ensemble_name + '_' + str(ehash) + '_optimised.nc',
                )
            )
        else:
            print('Error: Cannot find dataset')

        ds = ds.assign_coords(ehash=ehash)
        members.append(ds)

    print('Combining datasets')
    ensemble = xr.concat(members, dim='ehash', coords='minimal')

    ensemble = (
        ensemble.assign_coords({'p1': ('ehash', p1s), 'p2': ('ehash', p2s)})
        .set_index(ehash=['p1', 'p2'])
        .unstack('ehash')
    )
    return ensemble
