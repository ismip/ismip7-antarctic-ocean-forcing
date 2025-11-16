"""
Bias correct the extrapolated CMIP data towards climatology.

This is the classical method with bias correction in geographic space.

Workflow
========

"""

import argparse
import os
from typing import Dict, List, Tuple

import xarray as xr
from xarray.coders import CFDatetimeCoder

from i7aof.config import load_config
from i7aof.coords import (
    attach_grid_coords,
    propagate_time_from,
    strip_fill_on_non_data,
)
from i7aof.grid.ismip import get_res_string
from i7aof.io import read_dataset, write_netcdf


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
        The name of the scenario in addition to 'historical', ('ssp585', etc.)
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
    bias_files = _compute_biases(
        config=config,
        workdir=workdir,
        model=model,
        scenario=scenario,
        ismip_res_str=ismip_res_str,
        clim_name=clim_name,
    )

    # Apply actual correction
    _apply_biascorrection(
        config=config,
        ct_files=ct_files,
        sa_files=sa_files,
        bias_files=bias_files,
        outdir=outdir,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Bias correct extrapolated CMIP ct/sa toward a reference '
            'climatology (classic method).'
        )
    )
    parser.add_argument(
        '-m',
        '--model',
        dest='model',
        required=True,
        help='CMIP model name (required).',
    )
    parser.add_argument(
        '-s',
        '--scenario',
        dest='scenario',
        required=True,
        help='Scenario in addition to historical (ssp585, ...: required).',
    )
    parser.add_argument(
        '-c',
        '--clim',
        dest='clim_name',
        required=True,
        help='Reference climatology name (required).',
    )
    parser.add_argument(
        '-w',
        '--workdir',
        dest='workdir',
        required=False,
        help='Base working directory (optional).',
    )
    parser.add_argument(
        '-C',
        '--config',
        dest='config',
        default=None,
        help='Path to user config file (optional).',
    )
    args = parser.parse_args()
    biascorr_cmip(
        model=args.model,
        scenario=args.scenario,
        clim_name=args.clim_name,
        workdir=args.workdir,
        user_config_filename=args.config,
    )


# helper functions


def _load_config_and_paths(
    model,
    workdir,
    user_config_filename,
    scenario,
    clim_name,
):
    config = load_config(
        model=model,
        clim_name=clim_name,
        workdir=workdir,
        user_config_filename=user_config_filename,
    )

    workdir_base: str = config.get('workdir', 'base_dir')

    extrap_dir = os.path.join(
        workdir_base, 'extrap', model, scenario, 'Omon', 'ct_sa'
    )

    outdir = os.path.join(
        workdir_base, 'biascorr', model, scenario, clim_name, 'Omon', 'ct_sa'
    )

    os.makedirs(outdir, exist_ok=True)
    os.chdir(workdir_base)

    ismip_res_str = get_res_string(config, extrap=False)
    return (
        config,
        workdir_base,
        extrap_dir,
        outdir,
        ismip_res_str,
    )


def _collect_extrap_outputs(
    extrap_dir: str, ismip_res_str: str
) -> Tuple[List[str], List[str]]:
    """Collect all extrapolated ct and sa files"""
    if not os.path.isdir(extrap_dir):
        return [], []
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
    config,
    workdir,
    model,
    scenario,
    ismip_res_str,
    clim_name,
):
    """Compute the bias if not already done"""

    biasdir = os.path.join(workdir, 'biascorr', model, 'bias', clim_name)
    os.makedirs(biasdir, exist_ok=True)

    modclimdir = os.path.join(workdir, 'biascorr', model, 'climatology')
    os.makedirs(modclimdir, exist_ok=True)

    climdir = os.path.join(workdir, 'extrap', 'climatology', clim_name)

    hist_dir = os.path.join(
        workdir, 'extrap', model, 'historical', 'Omon', 'ct_sa'
    )

    if not os.path.isdir(hist_dir):
        raise FileNotFoundError(
            f'No historical extrapolated files found: {hist_dir}'
        )

    # Add SSP directory to extend beyond historical if needed
    ssp_dir = os.path.join(workdir, 'extrap', model, scenario, 'Omon', 'ct_sa')

    if not os.path.isdir(ssp_dir):
        raise FileNotFoundError(
            f'No scenario extrapolated files found: {ssp_dir}'
        )

    time_chunk = config.get('biascorr', 'time_chunk')

    start_year = config.getint('climatology', 'start_year')
    end_year = config.getint('climatology', 'end_year')

    bias_files: Dict[str, str] = {}

    for var in ['ct', 'sa']:
        # Get historical files
        hist_files: List[str] = []
        for name in sorted(os.listdir(hist_dir)):
            if f'ismip{ismip_res_str}' in name and var in name:
                hist_files.append(os.path.join(hist_dir, name))
        if not hist_files:
            raise FileNotFoundError(
                f'No historical extrapolated files available for {var}'
            )

        # Gett SSP files (in case we need to extend end_year into SSP period)
        ssp_files: List[str] = []
        for name in sorted(os.listdir(ssp_dir)):
            if f'ismip{ismip_res_str}' in name and var in name:
                ssp_files.append(os.path.join(ssp_dir, name))

        basename = os.path.basename(hist_files[0])
        # remove "Omon" and years from filename
        basename = basename.replace('_Omon', '')[0 : basename.rfind('_')]
        # add climatology years and months
        ref_filename = f'{basename}_{start_year}01-{end_year}12.nc'

        # Define filename for bias and skip if it's already present
        biasfile = os.path.join(
            biasdir, ref_filename.replace('historical', 'bias')
        )
        bias_files[var] = biasfile
        if os.path.exists(biasfile):
            print(f'Bias file already exists, skipping: {biasfile}')
            continue
        modclimfile = os.path.join(
            modclimdir, ref_filename.replace('historical', 'climatology')
        )

        # Get climatology file for this variable
        climfile = os.path.join(
            climdir, f'OI_Climatology_ismip{ismip_res_str}_{var}_extrap.nc'
        )
        ds_clim = read_dataset(climfile)

        # Get historical file(s)
        # Open combined historical + SSP dataset
        files_to_open = hist_files + ssp_files
        ds_hist_ssp = xr.open_mfdataset(
            files_to_open,
            concat_dim='time',
            combine='nested',
            decode_times=CFDatetimeCoder(use_cftime=True),
        )

        # Extract climatology period (only full annual for now)
        ds_hist_ssp = ds_hist_ssp.sel(
            time=slice(f'{start_year}-01-01', f'{end_year + 1}-01-01')
        )
        # chunk just the variable because of issues chunking whole dataset
        da_hist = ds_hist_ssp[var].chunk({'time': time_chunk})

        # Compute time-average over climatology period
        dpm = ds_hist_ssp.time.dt.days_in_month
        weightedsum = (da_hist * dpm).sum(dim='time')
        modclim = weightedsum / dpm.sum()

        # Write out model climatology (preserve attrs) with ISMIP coords
        ds_out = xr.Dataset({var: modclim})
        ds_out[var].attrs = ds_hist_ssp[var].attrs
        ds_out = attach_grid_coords(ds_out, config)
        ds_out = strip_fill_on_non_data(ds_out, data_vars=(var,))
        write_netcdf(
            ds_out,
            modclimfile,
            progress_bar=True,
            has_fill_values=[var],
            compression=[var],
        )
        ds_out.close()

        # Compute bias in model climatology
        bias = modclim - ds_clim[var]

        # Write out bias (keep same attrs as variable) and coordinates
        ds_out = xr.Dataset({var: bias})
        ds_out[var].attrs = ds_hist_ssp[var].attrs
        ds_out = attach_grid_coords(ds_out, config)
        ds_out = strip_fill_on_non_data(ds_out, data_vars=(var,))
        write_netcdf(
            ds_out,
            biasfile,
            progress_bar=True,
            has_fill_values=[var],
            compression=[var],
        )
        ds_out.close()

        ds_clim.close()
        ds_hist_ssp.close()

    return bias_files


def _apply_biascorrection(
    config,
    ct_files,
    sa_files,
    bias_files,
    outdir,
):
    """Apply bias correction to all in_files"""

    time_chunk = config.get('biascorr', 'time_chunk')

    for ct_file, sa_file in zip(ct_files, sa_files, strict=True):
        for var, file in zip(['ct', 'sa'], [ct_file, sa_file], strict=True):
            # Read biases
            biasfile = bias_files[var]
            ds_bias = read_dataset(biasfile)

            # Read CMIP files
            ds_cmip = read_dataset(file)
            da_cmip = ds_cmip[var].chunk({'time': time_chunk})

            # Define output filename
            outfile = os.path.join(outdir, os.path.basename(file))
            if os.path.exists(outfile):
                print(f'Corrected files already exist: {outfile}')
            else:
                # Output file doesn't exist yet, write out

                # Build dataset with corrected variable first
                corrected = da_cmip - ds_bias[var]
                ds_out = xr.Dataset({var: corrected})
                ds_out[var].attrs = ds_cmip[var].attrs
                ds_out = attach_grid_coords(ds_out, config)
                # Propagate time coord and bounds from CMIP source
                ds_out = propagate_time_from(
                    ds_out,
                    ds_cmip,
                    apply_cf_encoding=True,
                    units='days since 1850-01-01 00:00:00',
                )
                # Ensure coords/bounds have no fill values
                ds_out = strip_fill_on_non_data(ds_out, data_vars=(var,))

                write_netcdf(
                    ds_out,
                    outfile,
                    progress_bar=True,
                    has_fill_values=[var],
                )
                ds_out.close()

            # Clean up
            ds_bias.close()
            ds_cmip.close()
