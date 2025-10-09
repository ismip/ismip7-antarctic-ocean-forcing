"""
Bias correct the extrapolated CMIP data towards climatology.

This is the classical method with bias correction in geographic space.

Workflow
========

"""

import os
from typing import List

import xarray as xr
from mpas_tools.config import MpasConfigParser

from i7aof.cmip import get_model_prefix
from i7aof.grid.ismip import get_res_string
from i7aof.io import write_netcdf


def biascorr_cmip(
    model: str,
    scenario: str,
    clim_name: str,
    workdir: str | None = None,
    user_config_filename: str | None = None,
    variables: list[str, ...] | None = None,
):
    """
    Bias correct CMIP ct/sa in two stages:

    1) extract the bias in ct and sa
    2) apply the bias correction in ct and sa

    Parameters
    ----------
    model: str
        Name of the CMIP model to bias correct
    scenario: str
        The name of the scenario ('historical', 'ssp585', etc.)
    clim_name: str
        The name of the reference climatology
    workdir : str, optional
        The base work directory within which the bias corrected files will be
        placed
    user_config_filename : str, optional
        The path to a file with user config options that override the
        defaults
    variables : list of str, optional
        Variables to extrapolate (default: ``['ct', 'sa']``).
    """

    if variables is None:
        variables = ['ct', 'sa']

    # Read config
    (
        config,
        workdir,
        extrap_dir,
        outdir,
        ismip_res_str,
        model_prefix,
    ) = _load_config_and_paths(
        model=model,
        workdir=workdir,
        user_config_filename=user_config_filename,
        scenario=scenario,
        clim_name=clim_name,
    )

    # Collect files to bias correct
    in_files = _collect_extrap_outputs(extrap_dir, ismip_res_str)
    if not in_files:
        raise FileNotFoundError(
            'No extrapolated files found. Run: ismip7-antarctic-extrap-cmip'
        )

    # Compute bias over historical period
    _compute_biases(
        workdir=workdir,
        model=model,
        scenario=scenario,
        ismip_res_str=ismip_res_str,
        extrap_dir=extrap_dir,
        variables=variables,
        clim_name=clim_name,
    )

    # Apply actual correction
    _apply_biascorrection(
        workdir=workdir,
        model=model,
        scenario=scenario,
        ismip_res_str=ismip_res_str,
        extrap_dir=extrap_dir,
        variables=variables,
        clim_name=clim_name,
        in_files=in_files,
        outdir=outdir,
    )


# helper functions


def _load_config_and_paths(
    model,
    workdir,
    user_config_filename,
    scenario,
    clim_name,
):
    model_prefix = get_model_prefix(model)

    config = MpasConfigParser()
    config.add_from_package('i7aof', 'default.cfg')
    config.add_from_package('i7aof.cmip', f'{model_prefix}.cfg')
    config.add_from_package('i7aof.clim', f'{clim_name}.cfg')
    if user_config_filename is not None:
        config.add_user_config(user_config_filename)

    if workdir is None:
        if config.has_option('workdir', 'base_dir'):
            workdir = config.get('workdir', 'base_dir')
        else:
            raise ValueError(
                'Missing configuration option: [workdir] base_dir. '
                'Please supply a user config file that defines this option.'
            )
    assert workdir is not None, (
        'Internal error: workdir should be resolved to a string'
    )

    extrap_dir = os.path.join(
        workdir, 'extrap', model, scenario, 'Omon', 'ct_sa'
    )

    outdir = os.path.join(
        workdir, 'biascorr', model, scenario, clim_name, 'Omon', 'ct_sa'
    )

    os.makedirs(outdir, exist_ok=True)
    os.chdir(workdir)

    ismip_res_str = get_res_string(config, extrap=True)
    return config, workdir, extrap_dir, outdir, ismip_res_str, model_prefix


def _collect_extrap_outputs(extrap_dir: str, ismip_res_str: str) -> List[str]:
    """Collect all extrapolated ct and sa files"""
    if not os.path.isdir(extrap_dir):
        return []
    result: List[str] = []
    for name in sorted(os.listdir(extrap_dir)):
        if f'ismip{ismip_res_str}' in name and ('ct' in name or 'sa' in name):
            result.append(os.path.join(extrap_dir, name))
    return result


def _compute_biases(
    workdir, model, ismip_res_str, extrap_dir, scenario, variables, clim_name
):
    """Compute the bias if not already done"""

    biasdir = os.path.join(
        workdir, 'biascorr', model, 'intermediate', clim_name
    )
    os.makedirs(biasdir, exist_ok=True)

    climdir = os.path.join(workdir, 'extrap', 'climatology', clim_name)

    hist_dir = os.path.join(
        workdir, 'extrap', model, 'historical', 'Omon', 'ct_sa'
    )

    for var in variables:
        # Get historical files
        hist_files: List[str] = []
        for name in sorted(os.listdir(hist_dir)):
            if f'ismip{ismip_res_str}' in name and var in name:
                hist_files.append(os.path.join(extrap_dir, name))
        if not hist_files:
            raise FileNotFoundError(
                f'No historical extrapolated files available for {var}'
            )

        # Define filename for bias and skip if it's already present
        biasfile = os.path.join(biasdir, f'bias_{var}.nc')
        if os.path.exists(biasfile):
            print(f'Bias file already exists, skipping: {biasfile}')
            continue

        # Get climatology file for this variable
        climfile = os.path.join(
            climdir, f'OI_Climatology_ismip{ismip_res_str}_{var}_extrap.nc'
        )
        ds_clim = xr.open_dataset(climfile)

        # Get historical file(s)
        ds_hist = xr.open_mfdataset(hist_files, use_cftime=True)

        # Extract climatology period (only full annual for now)
        ds_hist = ds_hist.sel(time=slice('1995-01-01', '2014-12-31'))
        ds_hist = ds_hist.chunk({'time': 12})

        # Compute time-average over climatology period
        dpm = ds_hist.time.dt.days_in_month
        weightedsum = (ds_hist[var] * dpm).sum(dim='time')
        average = weightedsum / dpm.sum()

        bias = average - ds_clim[var]

        # Write out bias
        ds_out = xr.Dataset()
        for vvar in ['x', 'y', 'z_extrap']:
            ds_out[vvar] = ds_hist[vvar]
        ds_out[var] = bias
        write_netcdf(ds_out, biasfile, progress_bar=True)

        ds_clim.close()
        ds_hist.close()
        ds_out.close()


def _apply_biascorrection(
    workdir,
    model,
    ismip_res_str,
    extrap_dir,
    scenario,
    variables,
    clim_name,
    in_files,
    outdir,
):
    """Apply bias correction to all in_files"""

    biasdir = os.path.join(
        workdir, 'biascorr', model, 'intermediate', clim_name
    )

    for file in in_files:
        if f'ismip{ismip_res_str}' not in file:
            continue

        # Detect which variable is in this file
        if 'ct' in os.path.basename(file) and 'ct' in variables:
            var = 'ct'
        elif 'sa' in os.path.basename(file) and 'sa' in variables:
            var = 'sa'
        else:
            print(f'Skipping {file}')
            continue

        # Read bias
        biasfile = os.path.join(biasdir, f'bias_{var}.nc')
        # TODO check and compute here, remove computation from main
        # Only feed var and biasfile + whatever else necessary
        ds_bias = xr.open_dataset(biasfile)

        # Read CMIP file
        ds_cmip = xr.open_dataset(file)
        # TODO remove
        # ds_cmip = ds_cmip.sel(time=slice('1995-01-01', '2014-12-31'))

        ds_cmip = ds_cmip.chunk({'time': 12})

        var_corr = ds_cmip[var] - ds_bias[var]

        # Write out corrected field
        ds_out = xr.Dataset()
        for vvar in ['x', 'y', 'z_extrap', 'time']:
            ds_out[vvar] = ds_cmip[vvar]
        ds_out[var] = var_corr

        # Convert to yearly output
        ds_out = ds_out.resample(time='1YE').mean()
        ds_out['time'] = ds_out['time'].dt.year

        outfile = os.path.join(outdir, os.path.basename(file))
        write_netcdf(ds_out, outfile, progress_bar=True)

        # TODO if var == 'ct', also compute Tf and TF, write out TF

        ds_bias.close()
        ds_cmip.close()
        ds_out.close()
