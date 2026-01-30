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
)
from i7aof.grid.ismip import get_res_string
from i7aof.io import ensure_cf_time_encoding, read_dataset, write_netcdf
from i7aof.paths import (
    build_cmip_final_dir,
    build_cmip_final_filename,
    get_output_version,
    get_stage_dir,
)


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
        get_stage_dir(config, 'extrap'),
        model,
        future_scenario,
        ismip_res_str,
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
        get_stage_dir(config, 'biascorr'),
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
    extrap_root: str, model: str, future_scenario: str, ismip_res_str: str
) -> Dict[str, List[Dict[str, str]]]:
    """Collect all extrapolated ct and sa files from historical and future.

    Returns a dict: { 'ct': [{'path': ..., 'scenario': ...}],
                     'sa': [{'path': ..., 'scenario': ...}] }
    """
    ct_files: List[Dict[str, str]] = []
    sa_files: List[Dict[str, str]] = []
    for scenario in ['historical', future_scenario]:
        extrap_dir = os.path.join(
            extrap_root, model, scenario, 'Omon', 'ct_sa'
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
                        ct_files.append(
                            {
                                'path': os.path.join(extrap_dir, ct_name),
                                'scenario': scenario,
                            }
                        )
                        sa_files.append(
                            {
                                'path': os.path.join(extrap_dir, sa_name),
                                'scenario': scenario,
                            }
                        )

    # Sort for stable processing order
    ct_files = sorted(ct_files, key=lambda item: item['path'])
    sa_files = sorted(sa_files, key=lambda item: item['path'])
    return {'ct': ct_files, 'sa': sa_files}


def _compute_biases(
    config,
    workdir,
    model,
    future_scenario,
    ismip_res_str,
    clim_name,
    var_files: Dict[str, List[Dict[str, str]]],
):
    """Compute the bias if not already done"""

    version = get_output_version(config)

    climdir = os.path.join(
        get_stage_dir(config, 'extrap'), 'climatology', clim_name
    )

    # Ensure at least some files exist for both historical and future
    hist_dir = os.path.join(
        get_stage_dir(config, 'extrap'), model, 'historical', 'Omon', 'ct_sa'
    )
    ssp_dir = os.path.join(
        get_stage_dir(config, 'extrap'),
        model,
        future_scenario,
        'Omon',
        'ct_sa',
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

        year_range = f'{start_year}-{end_year}'

        clim_dir = build_cmip_final_dir(
            config,
            model=model,
            scenario='historical',
            variable=var,
            version=version,
            extras='climatology',
        )
        bias_dir = build_cmip_final_dir(
            config,
            model=model,
            scenario='historical',
            variable=var,
            version=version,
            extras='bias',
        )
        os.makedirs(clim_dir, exist_ok=True)
        os.makedirs(bias_dir, exist_ok=True)

        modclimfile = os.path.join(
            clim_dir,
            build_cmip_final_filename(
                variable=var,
                model=model,
                scenario='historical',
                version=version,
                year_range=year_range,
                extras='climatology',
            ),
        )
        biasfile = os.path.join(
            bias_dir,
            build_cmip_final_filename(
                variable=var,
                model=model,
                scenario='historical',
                version=version,
                year_range=year_range,
                extras='bias',
            ),
        )
        bias_files[var] = biasfile
        if os.path.exists(biasfile):
            print(f'Bias file already exists, skipping: {biasfile}')
            continue

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
        files_to_open = sorted(item['path'] for item in all_files)
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

    for ct_item, sa_item in zip(ct_files, sa_files, strict=True):
        for var, item in zip(['ct', 'sa'], [ct_item, sa_item], strict=True):
            # Read biases
            biasfile = bias_files[var]
            ds_bias = read_dataset(biasfile)

            # Read CMIP files (extrapolated inputs prior to bias correction)
            file_path = item['path']
            scenario_name = item['scenario']
            ds_cmip = read_dataset(file_path)
            da_cmip = ds_cmip[var].chunk({'time': time_chunk})

            # Define per-file output directory and filename
            outdir = os.path.join(
                get_stage_dir(config, 'biascorr'),
                model,
                scenario_name,
                clim_name,
                'Omon',
                'ct_sa',
            )
            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(outdir, os.path.basename(file_path))
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
                        f'Missing time_bnds in source file: {file_path}'
                    )
                ds_out['time_bnds'] = ds_cmip['time_bnds']
                # Propagate time coord and bounds from CMIP source
                ensure_cf_time_encoding(
                    ds=ds_out,
                    time_source=ds_cmip,
                )

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
