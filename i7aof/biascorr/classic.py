"""
Bias correct the extrapolated CMIP data towards climatology.

This is the classical method with bias correction in geographic space.

Workflow
========

"""

import argparse
import os
from typing import Dict, List

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
    future_scenario: str,
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
    future_scenario: str
        The name of the future scenario (e.g., 'ssp585'). This is in addition
        to 'historical', which is also used.
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
        ismip_res_str,
    ) = _load_config_and_paths(
        model=model,
        workdir=workdir,
        user_config_filename=user_config_filename,
        future_scenario=future_scenario,
        clim_name=clim_name,
    )

    # Collect extrapolated files (historical + future) to bias-correct
    var_files = _collect_extrap_outputs(
        workdir, model, future_scenario, ismip_res_str
    )
    ct_files = var_files.get('ct', [])
    sa_files = var_files.get('sa', [])
    if not ct_files or not sa_files:
        raise FileNotFoundError(
            'No extrapolated files found. Run: ismip7-antarctic-extrap-cmip'
        )

    # Compute bias over historical period
    bias_files = _compute_biases(
        config=config,
        workdir=workdir,
        model=model,
        future_scenario=future_scenario,
        ismip_res_str=ismip_res_str,
        clim_name=clim_name,
        var_files=var_files,
    )

    # Apply actual correction
    _apply_biascorrection(
        config=config,
        ct_files=ct_files,
        sa_files=sa_files,
        bias_files=bias_files,
        workdir=workdir,
        model=model,
        clim_name=clim_name,
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
        '--future_scenario',
        dest='future_scenario',
        required=True,
        help='Future scenario (ssp585, ...: required).',
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
        future_scenario=args.future_scenario,
        clim_name=args.clim_name,
        workdir=args.workdir,
        user_config_filename=args.config,
    )


# helper functions


def _load_config_and_paths(
    model,
    workdir,
    user_config_filename,
    future_scenario,
    clim_name,
):
    config = load_config(
        model=model,
        clim_name=clim_name,
        workdir=workdir,
        user_config_filename=user_config_filename,
    )

    workdir_base: str = config.get('workdir', 'base_dir')

    outdir = os.path.join(
        workdir_base,
        'biascorr',
        model,
        future_scenario,
        clim_name,
        'Omon',
        'ct_sa',
    )

    os.makedirs(outdir, exist_ok=True)
    os.chdir(workdir_base)

    ismip_res_str = get_res_string(config, extrap=False)
    return (
        config,
        workdir_base,
        ismip_res_str,
    )


def _collect_extrap_outputs(
    workdir: str, model: str, future_scenario: str, ismip_res_str: str
) -> Dict[str, List[str]]:
    """Collect all extrapolated ct and sa files from historical and future.

    Returns a dict: { 'ct': [...], 'sa': [...] } across both scenarios.
    """
    ct_files: List[str] = []
    sa_files: List[str] = []
    for scenario in ['historical', future_scenario]:
        extrap_dir = os.path.join(
            workdir, 'extrap', model, scenario, 'Omon', 'ct_sa'
        )
        if not os.path.isdir(extrap_dir):
            continue
        allfiles = sorted(os.listdir(extrap_dir))
        for name in allfiles:
            if f'ismip{ismip_res_str}' in name and name.endswith('.nc'):
                if 'ct' in name:
                    ct_name = name
                    sa_name = ct_name.replace('ct', 'sa')
                    if sa_name in allfiles:
                        ct_files.append(os.path.join(extrap_dir, ct_name))
                        sa_files.append(os.path.join(extrap_dir, sa_name))

    # Sort for stable processing order
    ct_files = sorted(ct_files)
    sa_files = sorted(sa_files)
    return {'ct': ct_files, 'sa': sa_files}


def _compute_biases(
    config,
    workdir,
    model,
    future_scenario,
    ismip_res_str,
    clim_name,
    var_files: Dict[str, List[str]],
):
    """Compute the bias if not already done"""

    biasdir = os.path.join(workdir, 'biascorr', model, 'bias', clim_name)
    os.makedirs(biasdir, exist_ok=True)

    modclimdir = os.path.join(workdir, 'biascorr', model, 'climatology')
    os.makedirs(modclimdir, exist_ok=True)

    climdir = os.path.join(workdir, 'extrap', 'climatology', clim_name)

    # Ensure at least some files exist for both historical and future
    hist_dir = os.path.join(
        workdir, 'extrap', model, 'historical', 'Omon', 'ct_sa'
    )
    ssp_dir = os.path.join(
        workdir, 'extrap', model, future_scenario, 'Omon', 'ct_sa'
    )
    if not os.path.isdir(hist_dir) or not os.path.isdir(ssp_dir):
        raise FileNotFoundError(
            'Missing extrapolated inputs for historical and/or future '
            f'scenarios: {hist_dir}, {ssp_dir}'
        )

    time_chunk = config.get('biascorr', 'time_chunk')

    start_year = config.getint('climatology', 'start_year')
    end_year = config.getint('climatology', 'end_year')

    bias_files: Dict[str, str] = {}

    for var in ['ct', 'sa']:
        all_files = var_files.get(var, [])
        if not all_files:
            raise FileNotFoundError(
                f'No extrapolated files available for {var}.'
            )

        # Determine a representative basename using a historical file
        hist_files_for_var = [
            f for f in all_files if os.sep + 'historical' + os.sep in f
        ]
        if not hist_files_for_var:
            raise FileNotFoundError(
                f'No historical extrapolated files available for {var}'
            )
        basename = os.path.basename(hist_files_for_var[0])
        # remove "Omon" and years from filename
        basename = basename.replace('_Omon', '')
        basename = basename[0 : basename.rfind('_')]
        # add climatology years
        ref_filename = f'{basename}_{start_year}-{end_year}.nc'

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
        if not os.path.exists(climfile):
            raise FileNotFoundError(
                f'Missing climatology file: {climfile}. Run '
                f'ismip7-antarctic-extrap-clim first'
            )

        ds_clim = read_dataset(climfile)

        # Open combined historical + future dataset (all files for variable)
        files_to_open = sorted(all_files)
        ds_hist_ssp = xr.open_mfdataset(
            files_to_open,
            combine='by_coords',
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
    workdir,
    model,
    clim_name,
):
    """Apply bias correction to all input files (historical and future).

    The output directory is derived per input file based on its scenario.
    """

    time_chunk = config.get('biascorr', 'time_chunk')

    for ct_file, sa_file in zip(ct_files, sa_files, strict=True):
        for var, file in zip(['ct', 'sa'], [ct_file, sa_file], strict=True):
            # Read biases
            biasfile = bias_files[var]
            ds_bias = read_dataset(biasfile)

            # Read CMIP files (extrapolated inputs prior to bias correction)
            ds_cmip = read_dataset(file)
            da_cmip = ds_cmip[var].chunk({'time': time_chunk})

            # Define per-file output directory and filename
            # Detect scenario from the input path
            parts = os.path.normpath(file).split(os.sep)
            # Expect structure: .../extrap/<model>/<scenario>/Omon/ct_sa/...
            try:
                model_idx = parts.index('extrap') + 1
                scenario_name = parts[model_idx + 1]
            except (ValueError, IndexError):
                # Fallback: default to future scenario layout if unexpected
                scenario_name = (
                    'historical' if 'historical' in file else 'unknown'
                )
            outdir = os.path.join(
                workdir,
                'biascorr',
                model,
                scenario_name,
                clim_name,
                'Omon',
                'ct_sa',
            )
            os.makedirs(outdir, exist_ok=True)
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
                # Copy time_bnds from the extrapolated source before
                # propagating CF-consistent time encodings, to mirror
                # other CMIP workflows.
                if 'time_bnds' not in ds_cmip:
                    raise ValueError(
                        f'Missing time_bnds in source file: {file}'
                    )
                ds_out['time_bnds'] = ds_cmip['time_bnds']
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
