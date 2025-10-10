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
    """

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
    ct_files, sa_files = _collect_extrap_outputs(extrap_dir, ismip_res_str)
    if not ct_files or not sa_files:
        raise FileNotFoundError(
            'No extrapolated files found. Run: ismip7-antarctic-extrap-cmip'
        )

    # Compute bias over historical period
    _compute_biases(
        config=config,
        workdir=workdir,
        model=model,
        scenario=scenario,
        ismip_res_str=ismip_res_str,
        extrap_dir=extrap_dir,
        clim_name=clim_name,
    )

    # Apply actual correction
    _apply_biascorrection(
        config=config,
        workdir=workdir,
        model=model,
        scenario=scenario,
        ismip_res_str=ismip_res_str,
        clim_name=clim_name,
        ct_files=ct_files,
        sa_files=sa_files,
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
    ct_files: List[str] = []
    sa_files: List[str] = []
    allfiles = sorted(os.listdir(extrap_dir))
    for name in allfiles:
        if f'ismip{ismip_res_str}' in name and 'ct' in name:
            ct_name = name
            sa_name = ct_name.replace('ct', 'sa')
            if sa_name in allfiles:
                ct_files.append(os.path.join(extrap_dir, ct_name))
                sa_files.append(os.path.join(extrap_dir, sa_name))

    return ct_files, sa_files


def _compute_biases(
    config, workdir, model, ismip_res_str, extrap_dir, scenario, clim_name
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

    time_chunk = config.get('biascorr', 'time_chunk')

    for var in ['ct', 'sa']:
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
        # TODO make dependent on clim
        ds_hist = ds_hist.sel(time=slice('1995-01-01', '2014-12-31'))
        ds_hist = ds_hist.chunk({'time': time_chunk})

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
    config,
    workdir,
    model,
    ismip_res_str,
    scenario,
    clim_name,
    ct_files,
    sa_files,
    outdir,
):
    """Apply bias correction to all in_files"""

    biasdir = os.path.join(
        workdir, 'biascorr', model, 'intermediate', clim_name
    )

    time_chunk = config.get('biascorr', 'time_chunk')

    for ct_file, sa_file in zip(ct_files, sa_files, strict=False):
        corrvar = {}

        ds_tf = xr.Dataset()

        for var, file in zip(['ct', 'sa'], [ct_file, sa_file], strict=False):
            # Read biases
            biasfile = os.path.join(biasdir, f'bias_{var}.nc')
            ds_bias = xr.open_dataset(biasfile)

            # Read CMIP files
            ds_cmip = xr.open_dataset(file)
            # TODO remove:
            ds_cmip = ds_cmip.sel(time=slice('1995-01-01', '2014-12-31'))
            ds_cmip = ds_cmip.chunk({'time': time_chunk})

            # Compute corrected variable
            corrvar[var] = ds_cmip[var] - ds_bias[var]

            # Write out corrected field
            ds_out = xr.Dataset()
            for vvar in ['x', 'y', 'z_extrap', 'time']:
                ds_out[vvar] = ds_cmip[vvar]
            ds_out[var] = corrvar[var]

            # Convert to yearly output
            ds_out = ds_out.resample(time='1YE').mean()
            ds_out['time'] = ds_out['time'].dt.year

            # Coarsen vertical resolution
            ds_out = ds_out.coarsen(z_extrap=3, boundary='trim').mean()

            outfile = os.path.join(outdir, os.path.basename(file))
            outfile = outfile.replace('20m', '60m')
            write_netcdf(ds_out, outfile, progress_bar=True)

            # Port variables to ds_tf (only once)
            if var == 'ct':
                for vvar in ['x', 'y', 'time']:
                    ds_tf[vvar] = ds_cmip[vvar]

            # Clean up
            ds_bias.close()
            ds_cmip.close()
            ds_out.close()

        # Read topography

        # Compute thermal forcing

        # Convert to yearly output
        ds_tf = ds_tf.resample(time='1YE').mean()
        ds_tf['time'] = ds_tf['time'].dt.year

        file = ct_file.replace('ct', 'tf')
        outfile = os.path.join(outdir, os.path.basename(file))
        outfile = outfile.replace('20m', '60m')
        write_netcdf(ds_tf, outfile, progress_bar=True)

        # Clean up
        ds_tf.close()
