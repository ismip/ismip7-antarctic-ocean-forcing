"""
Bias correct the extrapolated CMIP data towards climatology.

This is the classical method with bias correction in geographic space.

Workflow
========

"""

import argparse
import os
from typing import List, Tuple

import xarray as xr
from mpas_tools.config import MpasConfigParser

from i7aof.cmip import get_model_prefix
from i7aof.grid.ismip import (
    get_ismip_grid_filename,
    get_res_string,
    write_ismip_grid,
)
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
        grid_filename,
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
        ismip_res_str=ismip_res_str,
        clim_name=clim_name,
        grid_filename=grid_filename,
    )

    # Apply actual correction
    _apply_biascorrection(
        config=config,
        workdir=workdir,
        model=model,
        clim_name=clim_name,
        ct_files=ct_files,
        sa_files=sa_files,
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
        help='Scenario (historical, ssp585, ...: required).',
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
    # Persist workdir into config for downstream consumers (path resolution)
    config.set('workdir', 'base_dir', workdir)

    extrap_dir = os.path.join(
        workdir, 'extrap', model, scenario, 'Omon', 'ct_sa'
    )

    outdir = os.path.join(
        workdir, 'biascorr', model, scenario, clim_name, 'Omon', 'ct_sa'
    )

    os.makedirs(outdir, exist_ok=True)
    os.chdir(workdir)

    grid_filename = _ensure_ismip_grid(config, workdir)

    ismip_res_str = get_res_string(config, extrap=False)
    return (
        config,
        workdir,
        extrap_dir,
        outdir,
        ismip_res_str,
        grid_filename,
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
    config, workdir, model, ismip_res_str, clim_name, grid_filename
):
    """Compute the bias if not already done"""

    biasdir = os.path.join(
        workdir, 'biascorr', model, 'intermediate', clim_name
    )
    os.makedirs(biasdir, exist_ok=True)

    modclimdir = os.path.join(workdir, 'biascorr', model, 'intermediate')

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
                hist_files.append(os.path.join(hist_dir, name))
        if not hist_files:
            raise FileNotFoundError(
                f'No historical extrapolated files available for {var}'
            )

        # Define filename for bias and skip if it's already present
        biasfile = os.path.join(biasdir, f'bias_{var}.nc')
        if os.path.exists(biasfile):
            print(f'Bias file already exists, skipping: {biasfile}')
            continue
        modclimfile = os.path.join(modclimdir, f'model_clim_{var}.nc')

        # Get climatology file for this variable
        climfile = os.path.join(
            climdir, f'OI_Climatology_ismip{ismip_res_str}_{var}_extrap.nc'
        )
        ds_clim = xr.open_dataset(climfile)

        # Get historical file(s)
        ds_hist = xr.open_mfdataset(hist_files, use_cftime=True)

        # Extract climatology period (only full annual for now)
        # TODO make dependent on clim
        ds_hist = ds_hist.sel(time=slice('1995-01-01', '2015-01-01'))
        # chunk just the variable because of issues chunking whole dataset
        da_hist = ds_hist[var].chunk({'time': time_chunk})

        # Compute time-average over climatology period
        dpm = ds_hist.time.dt.days_in_month
        weightedsum = (da_hist * dpm).sum(dim='time')
        modclim = weightedsum / dpm.sum()

        # Write out model climatology (preserve attrs) and overwrite
        # x/y/z (and bounds) from ISMIP grid
        ds_out = xr.Dataset()
        ds_grid = xr.open_dataset(
            grid_filename, decode_times=True, use_cftime=True
        )
        _assign_coord_with_bounds(ds_out, ds_grid, 'x')
        _assign_coord_with_bounds(ds_out, ds_grid, 'y')
        _assign_coord_with_bounds(ds_out, ds_grid, 'z')
        # Also include geodetic coordinates and their bounds
        for name in ['lat', 'lon', 'lat_bnds', 'lon_bnds']:
            if name in ds_grid:
                ds_out[name] = ds_grid[name]
        # data var with attrs
        ds_out[var] = modclim
        ds_out[var].attrs = ds_hist[var].attrs
        write_netcdf(
            ds_out,
            modclimfile,
            progress_bar=True,
            has_fill_values=lambda name, _v, v=var: name == v,
        )
        ds_out.close()
        ds_grid.close()

        # Compute bias in model climatology
        bias = modclim - ds_clim[var]

        # Write out bias (keep same attrs as variable) and coordinates
        ds_out = xr.Dataset()
        ds_grid = xr.open_dataset(
            get_ismip_grid_filename(config),
            decode_times=True,
            use_cftime=True,
        )
        _assign_coord_with_bounds(ds_out, ds_grid, 'x')
        _assign_coord_with_bounds(ds_out, ds_grid, 'y')
        _assign_coord_with_bounds(ds_out, ds_grid, 'z')
        for name in ['lat', 'lon', 'lat_bnds', 'lon_bnds']:
            if name in ds_grid:
                ds_out[name] = ds_grid[name]
        ds_out[var] = bias
        ds_out[var].attrs = ds_hist[var].attrs
        write_netcdf(
            ds_out,
            biasfile,
            progress_bar=True,
            has_fill_values=lambda name, _v, v=var: name == v,
        )
        ds_out.close()
        ds_grid.close()

        ds_clim.close()
        ds_hist.close()


def _apply_biascorrection(
    config,
    workdir,
    model,
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

    for ct_file, sa_file in zip(ct_files, sa_files, strict=True):
        for var, file in zip(['ct', 'sa'], [ct_file, sa_file], strict=True):
            # Read biases
            biasfile = os.path.join(biasdir, f'bias_{var}.nc')
            ds_bias = xr.open_dataset(biasfile)

            # Read CMIP files
            ds_cmip = xr.open_dataset(file)
            da_cmip = ds_cmip[var].chunk({'time': time_chunk})

            # Define output filename
            outfile = os.path.join(outdir, os.path.basename(file))
            if os.path.exists(outfile):
                print(f'Corrected files already exist: {outfile}')
            else:
                # Output file doesn't exist yet, write out

                # Build dataset with ISMIP coordinates (and bounds) first
                ds_out = xr.Dataset()
                ds_grid = xr.open_dataset(
                    get_ismip_grid_filename(config),
                    decode_times=True,
                    use_cftime=True,
                )
                _assign_coord_with_bounds(ds_out, ds_grid, 'x')
                _assign_coord_with_bounds(ds_out, ds_grid, 'y')
                _assign_coord_with_bounds(ds_out, ds_grid, 'z')
                for name in ['lat', 'lon', 'lat_bnds', 'lon_bnds']:
                    if name in ds_grid:
                        ds_out[name] = ds_grid[name]
                # time coord comes from source; bounds will be added
                # after resample
                ds_out['time'] = ds_cmip['time']

                # Corrected variable and preserve attrs
                ds_out[var] = da_cmip - ds_bias[var]
                ds_out[var].attrs = ds_cmip[var].attrs

                # Re-apply variable attrs after resample (may be dropped)
                ds_out[var].attrs = ds_cmip[var].attrs

                write_netcdf(
                    ds_out,
                    outfile,
                    progress_bar=True,
                    has_fill_values=lambda name, _v, v=var: name == v,
                )
                ds_out.close()
                ds_grid.close()

            # Clean up
            ds_bias.close()
            ds_cmip.close()


def _ensure_ismip_grid(config: MpasConfigParser, workdir: str) -> str:
    grid_rel = get_ismip_grid_filename(config)
    grid_abs = os.path.join(workdir, grid_rel)
    if not os.path.exists(grid_abs):
        cwd = os.getcwd()
        try:
            os.makedirs(os.path.dirname(grid_abs), exist_ok=True)
            os.chdir(workdir)
            write_ismip_grid(config)
        finally:
            os.chdir(cwd)
        if not os.path.exists(grid_abs):  # pragma: no cover
            raise FileNotFoundError(
                f'Failed to generate ISMIP grid file: {grid_abs}'
            )
    return grid_abs


# -----------------------------------------------------------------------------
# Helpers for attributes and CF bounds
# -----------------------------------------------------------------------------


def _assign_coord_with_bounds(
    ds_out: xr.Dataset, ds_grid: xr.Dataset, coord: str
) -> None:
    """Assign a 1D coordinate and its bounds to ds_out from ISMIP grid.

    - Copies the coordinate variable and its attributes from the canonical
      ISMIP grid
    - Copies the corresponding *_bnds variable and sets coord 'bounds'
    """
    if coord not in ds_grid:
        return
    ds_out[coord] = ds_grid[coord]
    # Determine bounds variable name and copy over
    bname = ds_grid[coord].attrs.get('bounds', f'{coord}_bnds')
    if bname in ds_grid:
        ds_out[bname] = ds_grid[bname]
        ds_out[bname].attrs = ds_grid[bname].attrs.copy()
    ds_out[coord].attrs['bounds'] = bname
