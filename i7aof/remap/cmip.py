#!/usr/bin/env python
import argparse
import os
import shutil

import numpy as np
import xarray as xr
from mpas_tools.config import MpasConfigParser
from mpas_tools.logging import LoggingContext

from i7aof.cmip import get_model_prefix
from i7aof.grid.ismip import (
    get_ismip_grid_filename,
    get_res_string,
    write_ismip_grid,
)
from i7aof.io import write_netcdf
from i7aof.remap import add_periodic_lon, remap_lat_lon_to_ismip
from i7aof.vert.interp import VerticalInterpolator, fix_src_z_coord


def remap_cmip(
    model,
    variable,
    scenario,
    inputdir=None,
    workdir=None,
    user_config_filename=None,
):
    """
    Remap CMIP data both vertically and horizontally to the ISMIP grid.

    Parameters
    ----------
    model : str
        Name of the CMIP model to remap

    variable : {'thetao', 'so'}
        The name of the variable to remap

    scenario : str
        The name of the scenario ('historical', 'ssp585', etc.)

    inputdir : str, optional
        The base input directory where the CMIP monthly input files are
        located

    workdir : str, optional
        The base work directory within which the remapped files will be placed

    user_config_filename : str, optional
        The path to a file with user config options that override the defaults
    """

    model_prefix = get_model_prefix(model)

    config = MpasConfigParser()
    config.add_from_package('i7aof', 'default.cfg')
    config.add_from_package('i7aof.cmip', f'{model_prefix}.cfg')
    if user_config_filename is not None:
        config.add_user_config(user_config_filename)

    if workdir is None:
        if config.has_option('workdir', 'base_dir'):
            workdir = config.get('workdir', 'base_dir')
        else:
            raise ValueError(
                'Missing configuration option: [workdir] base_dir. '
                'Please supply a user config file that '
                'defines this option.'
            )

    if inputdir is None:
        if config.has_option('inputdir', 'base_dir'):
            inputdir = config.get('inputdir', 'base_dir')
        else:
            raise ValueError(
                'Missing configuration option: [inputdir] base_dir. '
                'Please supply a user config file that '
                'defines this option.'
            )

    outdir = os.path.join(workdir, 'remap', model, scenario, 'Omon', variable)
    os.makedirs(outdir, exist_ok=True)

    os.chdir(workdir)

    ismip_res_str = get_res_string(config)

    in_rel_paths = config.getexpression(f'{scenario}_files', variable)

    in_files = []
    out_files = []

    for rel_filename in in_rel_paths:
        base_filename = os.path.basename(rel_filename)
        if 'gn' not in base_filename:
            raise ValueError(
                f'Expected input to be on native grid (gn): {base_filename}'
            )

        out_filename = base_filename.replace('gn', f'ismip{ismip_res_str}')
        out_filename = os.path.join(outdir, out_filename)

        in_filename = os.path.join(inputdir, rel_filename)
        in_files.append(in_filename)
        out_files.append(out_filename)

    write_ismip_grid(config)

    for index, (in_filename, out_filename) in enumerate(
        zip(in_files, out_files, strict=True)
    ):
        if os.path.exists(out_filename):
            print(f'Remapped file exists, skipping: {out_filename}')
            continue

        vert_tmpdir = os.path.join(
            outdir, f'tmp_vert_interp_{variable}_{index}'
        )
        os.makedirs(vert_tmpdir, exist_ok=True)

        horiz_tmpdir = os.path.join(
            outdir, f'tmp_horiz_remap_{variable}_{index}'
        )
        os.makedirs(horiz_tmpdir, exist_ok=True)

        vert_interp_filenames = _vert_mask_interp_norm(
            config, in_filename, outdir, variable, vert_tmpdir
        )

        with LoggingContext(__name__) as logger:
            _remap_horiz(
                config,
                vert_interp_filenames,
                out_filename,
                variable,
                model_prefix,
                horiz_tmpdir,
                logger,
            )

        shutil.rmtree(vert_tmpdir)
        shutil.rmtree(horiz_tmpdir)


def main():
    parser = argparse.ArgumentParser(
        description='Remap CMIP model data to ISMIP grid.'
    )
    parser.add_argument(
        '-m',
        '--model',
        dest='model',
        type=str,
        required=True,
        help='Name of the CMIP model to remap (required).',
    )
    parser.add_argument(
        '-v',
        '--variable',
        dest='variable',
        type=str,
        required=True,
        help='Name of the variable to remap ("thetao" or "so": required).',
    )
    parser.add_argument(
        '-s',
        '--scenario',
        dest='scenario',
        type=str,
        required=True,
        help=(
            'Name of the scenario to remap ("historical", "ssp585", etc.: '
            'required).'
        ),
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
    args = parser.parse_args()

    remap_cmip(
        model=args.model,
        variable=args.variable,
        scenario=args.scenario,
        inputdir=args.inputdir,
        workdir=args.workdir,
        user_config_filename=args.config,
    )


def _vert_mask(
    interpolator,
    src_filename,
    mask_filename,
    time_chunk,
    variable,
    lev,
    lev_bnds,
):
    """Mask the dataset before vertical interpolation"""

    ds = xr.open_dataset(src_filename, decode_times=False)
    ds = ds.chunk({'time': time_chunk})

    ds_out = xr.Dataset()

    ds_out = ds_out.assign_coords(
        {
            'lev': ('lev', lev.data),
            'lev_bnds': (('lev', 'd2'), lev_bnds.data),
        }
    )
    ds_out['lev'].attrs = lev.attrs

    da = ds[variable]

    da_masked = interpolator.mask_and_sort(da)
    ds_out[variable] = da_masked.astype(np.float32)

    ds_out['src_valid'] = interpolator.src_valid.astype(np.float32)
    for var in ['lat', 'lon', 'time']:
        ds_out[var] = ds[var]
        var_bnds = f'{var}_bnds'
        ds_out[var_bnds] = ds[var_bnds]

    write_netcdf(ds_out, mask_filename, progress_bar=True)


def _vert_interp(
    interpolator,
    ds_ismip,
    time_chunk,
    mask_filename,
    interp_filename,
    variable,
):
    """Vertically interpolate the dataset"""
    # open it again to get a clean dataset
    ds = xr.open_dataset(mask_filename, decode_times=False)
    ds = ds.chunk({'time': time_chunk})
    da_masked = ds[variable]

    da_interp = interpolator.interp(da_masked)

    ds_out = xr.Dataset()
    ds_out[variable] = da_interp.astype(np.float32)
    ds_out['src_frac_interp'] = interpolator.src_frac_interp.astype(np.float32)
    for var in ['lat', 'lon', 'time']:
        ds_out[var] = ds[var]
        var_bnds = f'{var}_bnds'
        ds_out[var_bnds] = ds[var_bnds]
    ds_out['z_extrap_bnds'] = ds_ismip['z_extrap_bnds']
    ds_out = ds_out.drop_vars(['lev'])

    write_netcdf(ds_out, interp_filename, progress_bar=True)


def _vert_normalize(
    interpolator,
    ds_ismip,
    time_chunk,
    interp_filename,
    normalize_filename,
    variable,
):
    """Normalize the dataset following vertical interpolation"""
    # open it again to get a clean dataset
    ds = xr.open_dataset(interp_filename, decode_times=False)
    ds = ds.chunk({'time': time_chunk})
    da_interp = ds[variable]

    da_normalized = interpolator.normalize(da_interp)

    ds_out = xr.Dataset()
    ds_out[variable] = da_normalized.astype(np.float32)
    ds_out['src_frac_interp'] = interpolator.src_frac_interp.astype(np.float32)
    for var in ['lat', 'lon', 'time']:
        ds_out[var] = ds[var]
        var_bnds = f'{var}_bnds'
        ds_out[var_bnds] = ds[var_bnds]
    ds_out['z_extrap_bnds'] = ds_ismip['z_extrap_bnds']

    write_netcdf(ds_out, normalize_filename, progress_bar=True)


def _vert_mask_interp_norm(config, in_filename, outdir, variable, tmpdir):
    """Mask, vertically interpolate and normalize the dataset"""

    mask_filename = os.path.join(tmpdir, f'{variable}_masked.nc')
    interp_filename = os.path.join(tmpdir, f'{variable}_interp.nc')
    normalized_filename = os.path.join(tmpdir, f'{variable}_normalized.nc')

    if os.path.exists(normalized_filename):
        print(
            f'Vertically interpolated file exists, skipping: '
            f'{normalized_filename}'
        )
        return normalized_filename

    ds_ismip = xr.open_dataset(get_ismip_grid_filename(config))

    with xr.open_dataset(in_filename, decode_times=False) as ds:
        lev, lev_bnds = fix_src_z_coord(ds, 'lev', 'lev_bnds')
        ds = ds.assign_coords({'lev': ('lev', lev.data)})
        ds['lev'].attrs = lev.attrs
        ds['lev_bnds'] = lev_bnds

        src_valid = ds[variable].isel(time=0).notnull()
        src_valid = src_valid.drop_vars(['time'])

    time_chunk = config.getint('remap_cmip', 'vert_time_chunk')

    interpolator = VerticalInterpolator(
        src_valid=src_valid,
        src_coord='lev',
        dst_coord='z_extrap',
        config=config,
    )

    _vert_mask(
        interpolator,
        in_filename,
        mask_filename,
        time_chunk,
        variable,
        lev,
        lev_bnds,
    )

    _vert_interp(
        interpolator,
        ds_ismip,
        time_chunk,
        mask_filename,
        interp_filename,
        variable,
    )

    _vert_normalize(
        interpolator,
        ds_ismip,
        time_chunk,
        interp_filename,
        normalized_filename,
        variable,
    )

    print(
        f'Vertical interpolation completed and saved to '
        f"'{normalized_filename}'."
    )

    return normalized_filename


def _remap_horiz(
    config,
    in_filename,
    out_filename,
    variable,
    model_prefix,
    tmpdir,
    logger,
):
    """Horizontally remap the dataset"""

    method = config.get('remap', 'method')
    renorm_threshold = config.getfloat('remap', 'threshold')
    lat_var = config.get('remap_cmip', 'lat_var')
    lon_var = config.get('remap_cmip', 'lon_var')
    lon_dim = config.get('remap_cmip', 'lon_dim')
    in_grid_name = model_prefix

    # Open dataset (but do not load into memory)
    ds = xr.open_dataset(in_filename, chunks={'time': 1}, decode_times=False)

    if method == 'bilinear':
        # we need to add a periodic longitude value or remapping will have a
        # seam
        ds = add_periodic_lon(ds, lon_var=lon_var, periodic_dim=lon_dim)

    input_mask_path = os.path.join(tmpdir, 'input_mask.nc')
    output_mask_path = os.path.join(tmpdir, 'output_mask.nc')
    if os.path.exists(output_mask_path):
        ds_mask = xr.open_dataset(output_mask_path, decode_times=False)
    else:
        ds_mask = ds.copy()
        ds_mask = ds_mask.drop_vars([variable, 'time', 'time_bnds'])
        write_netcdf(ds_mask, input_mask_path, progress_bar=True)

        # remap the mask without renormalizing
        remap_lat_lon_to_ismip(
            in_filename=input_mask_path,
            in_grid_name=in_grid_name,
            out_filename=output_mask_path,
            map_dir=tmpdir,
            method=method,
            config=config,
            logger=logger,
            lon_var=lon_var,
            lat_var=lat_var,
        )
        ds_mask = xr.open_dataset(output_mask_path, decode_times=False)

    # remap in 10-year chunks (by default)
    chunk_size = config.getint('remap_cmip', 'horiz_time_chunk')
    n_time = ds.sizes['time']
    time_indices = np.arange(0, n_time, chunk_size)

    remapped_chunks = []

    for i_start in time_indices:
        i_end = min(i_start + chunk_size, n_time)

        input_chunk_path = os.path.join(tmpdir, f'input_{i_start}_{i_end}.nc')

        # Remapped output path
        output_chunk_path = os.path.join(
            tmpdir, f'output_{i_start}_{i_end}.nc'
        )
        if os.path.exists(output_chunk_path):
            print(
                f'Skipping remapping for chunk {i_start}-{i_end} '
                f'(already exists).'
            )
            # Load remapped chunk
            remapped_chunk = xr.open_dataset(
                output_chunk_path, chunks={'time': 1}, decode_times=False
            )
            remapped_chunks.append(remapped_chunk)
            continue

        # Slice dataset
        subset = ds.isel(time=slice(i_start, i_end))
        subset = subset.drop_vars(['src_frac_interp'])

        # Write temporary input chunk
        write_netcdf(subset, input_chunk_path, progress_bar=True)

        # Run remapping
        remap_lat_lon_to_ismip(
            in_filename=input_chunk_path,
            in_grid_name=in_grid_name,
            out_filename=output_chunk_path,
            map_dir='.',
            method=method,
            config=config,
            logger=logger,
            lon_var=lon_var,
            lat_var=lat_var,
            renormalize=renorm_threshold,
        )

        # Load remapped chunk
        remapped_chunk = xr.open_dataset(
            output_chunk_path, chunks={'time': 1}, decode_times=False
        )
        remapped_chunks.append(remapped_chunk)

    # Concatenate all remapped chunks along time
    ds_final = xr.concat(remapped_chunks, dim='time')
    ds_final['src_frac_interp'] = ds_mask['src_frac_interp']

    # Save final output
    write_netcdf(ds_final, out_filename, progress_bar=True)
