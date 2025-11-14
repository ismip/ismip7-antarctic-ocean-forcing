import argparse
import os
import shutil

import gsw
import numpy as np
import xarray as xr
from mpas_tools.logging import LoggingContext

from i7aof.config import load_config
from i7aof.grid.ismip import get_res_string, write_ismip_grid
from i7aof.io import read_dataset, write_netcdf
from i7aof.remap.shared import (
    _remap_horiz,
    _vert_mask_interp_norm_multi,
)


def remap_climatology(
    clim_name,
    inputdir=None,
    workdir=None,
    user_config_filename=None,
    overwrite=False,
):
    """
    Remap an observational climatology data to the ISMIP grid with two stages:

    1) vertical interpolation to ISMIP z_extrap levels, then

    2) horizontal remapping to the ISMIP lat/lon grid.

    This function orchestrates the basic flow per input file:

    - Prepare output dirs and ensure the ISMIP grid exists.

    - For each monthly file:

      * Vertical pipeline (see _vert_mask_interp_norm):
        mask invalid source points -> interpolate in z -> normalize.

      * Horizontal remap of the vertically processed data to ISMIP grid.

    Parameters
    ----------
    clim_name : str
        The name of the climatology to remap

    inputdir : str, optional
        The base input directory where the CMIP monthly input files are
        located

    workdir : str, optional
        The base work directory within which the remapped files will be placed

    user_config_filename : str, optional
        The path to a file with user config options that override the defaults

    overwrite : bool, optional
        Whether to overwrite the output file if it exists
    """
    config = load_config(
        model=None,
        clim_name=clim_name,
        inputdir=inputdir,
        workdir=workdir,
        user_config_filename=user_config_filename,
    )

    if not config.has_option('inputdir', 'base_dir'):
        raise ValueError(
            'Missing configuration option: [inputdir] base_dir. '
            'Please supply a user config file that defines this option.'
        )
    workdir = config.get('workdir', 'base_dir')
    inputdir = config.get('inputdir', 'base_dir')

    outdir = os.path.join(workdir, 'remap', 'climatology', clim_name)
    os.makedirs(outdir, exist_ok=True)

    os.chdir(workdir)

    ismip_res_str = get_res_string(config, extrap=True)

    rel_filename = config.get('climatology', 'filename')
    base_filename = os.path.basename(rel_filename)

    in_filename = os.path.join(inputdir, rel_filename)

    out_filename = base_filename.replace('.nc', f'_ismip{ismip_res_str}.nc')
    out_filename = os.path.join(outdir, out_filename)

    if os.path.exists(out_filename):
        print(f'Remapped file exists, skipping: {out_filename}')
        return

    # Ensure the destination ISMIP grid files exist (used by both steps)
    write_ismip_grid(config)

    if not overwrite and os.path.exists(out_filename):
        print(f'Remapped file exists, skipping: {out_filename}')
        return

    # Per-file tmp dirs for clarity and clean-up
    # Vertical stage tmp directory (mask -> interp -> normalize)
    vert_tmpdir = os.path.join(outdir, 'tmp_vert_interp')
    os.makedirs(vert_tmpdir, exist_ok=True)

    # Horizontal stage tmp directory (time-chunked remap + masks)
    horiz_tmpdir = os.path.join(outdir, 'tmp_horiz_remap')
    os.makedirs(horiz_tmpdir, exist_ok=True)

    # Preprocess input: drop SCALAR, rename dims/vars, convert pressure->lev,
    # ensure lev_bnds, reorder data vars to (lev, lat, lon)
    preprocessed = _preprocess_climatology_input(
        config, in_filename, vert_tmpdir
    )

    # 1) Vertical pipeline: masking -> vertical interpolation -> normalize
    vert_interp_filenames = _vert_mask_interp_norm_multi(
        config, preprocessed, outdir, ['ct', 'sa'], vert_tmpdir
    )

    with LoggingContext(__name__) as logger:
        # 2) Horizontal remap to ISMIP lat/lon grid
        # Requires a logger to capture output from ncremap calls (we use
        # stdout and stderr, rather than a log file here)
        _remap_horiz(
            config,
            vert_interp_filenames,
            out_filename,
            clim_name,
            horiz_tmpdir,
            logger,
            fill_and_compress=['ct', 'sa'],
            lat_var='lat',
            lon_var='lon',
            lon_dim='lon',
        )

    # Post-process dimension ordering so climatology fields follow
    # expected (z_extrap, y, x) prior to later extrapolation (where a
    # dummy time dimension will be added as the slowest axis). This
    # keeps consistency with CMIP workflow expectations.
    if os.path.exists(out_filename):
        ds_remap = read_dataset(out_filename)
        changed = False
        for var in ['ct', 'sa']:
            if var in ds_remap:
                dims = ds_remap[var].dims
                # target order (z_extrap, y, x) if all present
                target = tuple(d for d in ['z_extrap', 'y', 'x'] if d in dims)
                if dims != target and set(dims) == set(target):
                    ds_remap[var] = ds_remap[var].transpose(*target)
                    changed = True
        if changed:
            tmp_out = f'{out_filename}.tmp_reorder'
            write_netcdf(ds_remap, tmp_out, progress_bar=False)
            os.replace(tmp_out, out_filename)
        ds_remap.close()

    # Always clean up tmp dirs for this input file
    shutil.rmtree(vert_tmpdir)
    shutil.rmtree(horiz_tmpdir)


def main():
    parser = argparse.ArgumentParser(
        description='Remap climatology data to ISMIP grid.'
    )
    parser.add_argument(
        '-n',
        '--clim',
        dest='clim_name',
        type=str,
        required=True,
        help='Name of the climatology dataset to remap (required).',
    )
    parser.add_argument(
        '-i',
        '--inputdir',
        dest='inputdir',
        type=str,
        required=False,
        help='Path to the base input directory (optional).',
    )
    parser.add_argument(
        '-w',
        '--workdir',
        dest='workdir',
        type=str,
        required=False,
        help='Path to the base working directory (optional).',
    )
    parser.add_argument(
        '-c',
        '--config',
        dest='config',
        type=str,
        default=None,
        help='Path to user config file (optional).',
    )
    parser.add_argument(
        '--overwrite',
        dest='overwrite',
        action='store_true',
        help='Overwrite existing output files (optional).',
    )
    args = parser.parse_args()

    remap_climatology(
        clim_name=args.clim_name,
        inputdir=args.inputdir,
        workdir=args.workdir,
        user_config_filename=args.config,
        overwrite=args.overwrite,
    )


# preprocessing helper


def _preprocess_climatology_input(config, in_filename, tmpdir):
    """
    Prepare the climatology dataset for shared remapping pipeline:
    - Drop SCALAR dimension if present
    - Rename dims/variables to standard names: lon, lat, lev
    - Convert pressure (dbar) to height in meters (positive up) using GSW
    - Create lev_bnds from midpoints
    - Reorder variables to (lev, lat, lon)
    Returns path to a temporary preprocessed file.
    """
    ds = read_dataset(in_filename)

    lat_var = config.get('climatology', 'lat_var')
    lon_var = config.get('climatology', 'lon_var')
    lev_var = config.get('climatology', 'lev_var')
    lat_dim = config.get('climatology', 'lat_dim')
    lon_dim = config.get('climatology', 'lon_dim')
    lev_dim = config.get('climatology', 'lev_dim')

    ct_var = config.get('climatology', 'ct_var')
    sa_var = config.get('climatology', 'sa_var')

    ct_mse_var = config.get('climatology', 'ct_mse_var')
    sa_mse_var = config.get('climatology', 'sa_mse_var')

    threshold = config.getfloat('climatology', 'mse_threshold')

    # Drop SCALAR dim if present
    if 'SCALAR' in ds.dims:
        ds = ds.isel(SCALAR=0, drop=True)

    # Rename dims if present
    rename_dims = {lat_dim: 'lat', lon_dim: 'lon', lev_dim: 'lev'}
    ds = ds.rename(rename_dims)

    # some climatologies have an incorrect "unit" attribute instead of "units"
    for var in ds.data_vars:
        if 'unit' in ds[var].attrs and 'units' not in ds[var].attrs:
            ds[var].attrs['units'] = ds[var].attrs['unit']

    lev = ds[lev_var]
    if 'dbar' in lev.units:
        # pressure is in dbar; convert to height (m, positive up)
        lev_vals = gsw.z_from_p(ds[lev_var], lat=-75.0)
        lev = xr.DataArray(lev_vals.values, dims=['lev'])
        lev.attrs = {'units': 'm', 'positive': 'up'}
    ds = ds.drop_vars(lev_var)
    ds = ds.assign_coords({'lev': ('lev', lev.data)})

    # Create lev_bnds from midpoints
    lev_np = np.asarray(lev.values)
    mid = 0.5 * (lev_np[1:] + lev_np[:-1])
    first = lev_np[0] - (mid[0] - lev_np[0])
    last = lev_np[-1] + (lev_np[-1] - mid[-1])
    edges = np.concatenate([[first], mid, [last]])
    lev_bnds = np.column_stack([edges[:-1], edges[1:]])
    ds['lev_bnds'] = (('lev', 'd2'), lev_bnds)

    lat = ds[lat_var]
    ds = ds.drop_vars(lat_var)
    ds = ds.assign_coords({'lat': ('lat', lat.data)})
    lon = ds[lon_var]
    ds = ds.drop_vars(lon_var)
    ds = ds.assign_coords({'lon': ('lon', lon.data)})

    # Ensure coords have expected attrs
    ds['lev'].attrs = {'units': 'm', 'positive': 'up'}
    ds['lon'].attrs.setdefault('units', 'degrees_east')
    ds['lat'].attrs.setdefault('units', 'degrees_north')

    vars_to_transpose = [ct_var, sa_var]
    has_mse = ct_mse_var in ds.data_vars and sa_mse_var in ds.data_vars
    if has_mse:
        vars_to_transpose += [ct_mse_var, sa_mse_var]

    # Reorder ct/sa to (lev, lat, lon)
    for var in vars_to_transpose:
        if var in ds:
            dims = ds[var].dims
            target = tuple([d for d in ['lev', 'lat', 'lon'] if d in dims])
            if dims != target:
                ds[var] = ds[var].transpose(*target)

    if has_mse:
        # Omit values with too high mse
        ds[ct_var] = ds[ct_var].where(ds[ct_mse_var] < threshold, other=np.nan)
        ds[sa_var] = ds[sa_var].where(ds[sa_mse_var] < threshold, other=np.nan)

    out_path = os.path.join(tmpdir, 'preprocessed.nc')
    write_netcdf(ds, out_path, progress_bar=True)
    return out_path
