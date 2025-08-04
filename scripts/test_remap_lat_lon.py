#!/usr/bin/env python
import os

import numpy as np
import xarray as xr
from mpas_tools.config import MpasConfigParser
from mpas_tools.logging import LoggingContext

from i7aof.grid.ismip import (
    get_ismip_grid_filename,
    get_res_string,
    write_ismip_grid,
)
from i7aof.io import write_netcdf
from i7aof.remap import add_periodic_lon, remap_lat_lon_to_ismip
from i7aof.vert.interp import VerticalInterpolator


def mask(interpolator, src_filename, time_chunk):
    ds_raw = xr.open_dataset(src_filename, decode_cf=False, decode_times=False)
    # switch to positive up
    lev_bnds = -ds_raw['lev_bnds']

    ds = xr.open_dataset(src_filename, decode_times=False)
    ds = ds.chunk({'time': time_chunk})

    da = ds['thetao']

    da_masked = interpolator.mask_and_sort(da)

    ds_out = xr.Dataset()

    ds_out = ds_out.assign_coords(
        {
            'lev': ('lev', interpolator.z_src.data),
            'lev_bnds': (('lev', 'd2'), lev_bnds.data),
        }
    )
    ds_out['lev'].attrs = interpolator.z_src.attrs
    ds_out['lev_bnds'].attrs = lev_bnds.attrs

    ds_out['thetao'] = da_masked.astype(np.float32)
    ds_out['src_valid'] = interpolator.src_valid.astype(np.float32)
    for var in ['lat', 'lon', 'time']:
        ds_out[var] = ds[var]
        var_bnds = f'{var}_bnds'
        ds_out[var_bnds] = ds[var_bnds]

    write_netcdf(ds_out, 'thetao_masked.nc', progress_bar=True)

    ds_out = ds_out.isel(time=0)
    write_netcdf(ds_out, 'thetao_time0_masked.nc', progress_bar=True)


def interp(interpolator, ds_ismip, time_chunk):
    # open it again to get a clean dataset
    ds = xr.open_dataset('thetao_masked.nc', decode_times=False)
    ds = ds.chunk({'time': time_chunk})
    da_masked = ds['thetao']

    da_interp = interpolator.interp(da_masked)

    ds_out = xr.Dataset()
    ds_out['thetao'] = da_interp.astype(np.float32)
    ds_out['src_frac_interp'] = interpolator.src_frac_interp.astype(np.float32)
    for var in ['lat', 'lon', 'time']:
        ds_out[var] = ds[var]
        var_bnds = f'{var}_bnds'
        ds_out[var_bnds] = ds[var_bnds]
    ds_out['z_extrap_bnds'] = ds_ismip['z_extrap_bnds']
    ds_out = ds_out.drop_vars(['lev'])

    write_netcdf(ds_out, 'thetao_interp.nc', progress_bar=True)

    ds_out = ds_out.isel(time=0)
    write_netcdf(ds_out, 'thetao_time0_interp.nc', progress_bar=True)


def normalize(interpolator, ds_ismip, time_chunk):
    # open it again to get a clean dataset
    ds = xr.open_dataset('thetao_interp.nc', decode_times=False)
    ds = ds.chunk({'time': time_chunk})
    da_interp = ds['thetao']

    da_normalized = interpolator.normalize(da_interp)

    ds_out = xr.Dataset()
    ds_out['thetao'] = da_normalized.astype(np.float32)
    ds_out['src_frac_interp'] = interpolator.src_frac_interp.astype(np.float32)
    for var in ['lat', 'lon', 'time']:
        ds_out[var] = ds[var]
        var_bnds = f'{var}_bnds'
        ds_out[var_bnds] = ds[var_bnds]
    ds_out['z_extrap_bnds'] = ds_ismip['z_extrap_bnds']

    write_netcdf(ds_out, 'thetao_normalized.nc', progress_bar=True)

    ds_out = ds_out.isel(time=0)
    write_netcdf(ds_out, 'thetao_time0_normalized.nc', progress_bar=True)


def vert_interp(config):
    ds_ismip = xr.open_dataset(get_ismip_grid_filename(config))

    src_filename = (
        '/lcrc/group/e3sm/ac.xylar/ismip7/CMIP6_test_protocol/CESM2-WACCM/'
        'historical/Omon/'
        'thetao_Omon_CESM2-WACCM_historical_r1i1p1f1_gn_185001-201412.nc'
    )

    with xr.open_dataset(src_filename, decode_times=False) as ds:
        src_valid = ds['thetao'].isel(time=0).notnull()
        src_valid = src_valid.drop_vars(['time'])

    time_chunk = config.getint('cesm2_waccm', 'vert_time_chunk')

    interpolator = VerticalInterpolator(
        src_valid=src_valid,
        src_coord='lev',
        dst_coord='z_extrap',
        config=config,
    )

    mask(interpolator, src_filename, time_chunk)

    interp(interpolator, ds_ismip, time_chunk)

    normalize(interpolator, ds_ismip, time_chunk)

    print(
        "Vertical interpolation completed and saved to 'thetao_normalized.nc'."
    )


def remap_horiz(config, in_filename, logger):
    ismip_res_str = get_res_string(config)

    out_filename = (
        f'thetao_Omon_CESM2-WACCM_historical_r1i1p1f1_185001-201412_'
        f'ismip_{ismip_res_str}.nc'
    )

    tmpdir = 'tmp_remap_lat_lon'
    os.makedirs(tmpdir, exist_ok=True)

    method = config.get('cesm2_waccm', 'remap_method')
    lat_var = config.get('cesm2_waccm', 'lat_var')
    lon_var = config.get('cesm2_waccm', 'lon_var')

    # Open dataset (but do not load into memory)
    ds = xr.open_dataset(in_filename, chunks={'time': 1}, decode_times=False)

    if method == 'bilinear':
        # we need to add a periodic longitude value or remapping will have a
        # seam
        ds = add_periodic_lon(ds, threshold=1e-10)

    input_mask_path = os.path.join(tmpdir, 'input_mask.nc')
    output_mask_path = os.path.join(tmpdir, 'output_mask.nc')
    if os.path.exists(output_mask_path):
        ds_mask = xr.open_dataset(output_mask_path, decode_times=False)
    else:
        ds_mask = ds.copy()
        ds_mask = ds_mask.drop_vars(['thetao', 'time', 'time_bnds'])
        write_netcdf(ds_mask, input_mask_path, progress_bar=True)

        # remap the mask without renormalizing
        remap_lat_lon_to_ismip(
            in_filename=input_mask_path,
            in_grid_name='cesm2_waccm',
            out_filename=output_mask_path,
            map_dir='.',
            method=method,
            config=config,
            logger=logger,
            lon_var=lon_var,
            lat_var=lat_var,
        )
        ds_mask = xr.open_dataset(output_mask_path, decode_times=False)

    renorm_threshold = config.getfloat('cesm2_waccm', 'renorm_threshold')

    # remap in 10-year chunks (by default)
    chunk_size = config.getint('cesm2_waccm', 'horiz_time_chunk')
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
            in_grid_name='cesm2_waccm',
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

    ds_final = ds_final.isel(time=0)
    write_netcdf(ds_final, 'thetao_time0_remapped.nc', progress_bar=True)


def main():
    config = MpasConfigParser()
    config.add_from_package('i7aof', 'default.cfg')
    config.add_user_config('test_remap_lat_lon.cfg')

    work_base_dir = config.get('workdir', 'base_dir')
    os.makedirs(work_base_dir, exist_ok=True)
    os.chdir(work_base_dir)

    write_ismip_grid(config)

    vert_interp_filename = 'thetao_normalized.nc'
    if not os.path.exists(vert_interp_filename):
        print(
            f"Vertical interpolation file '{vert_interp_filename}' does not "
            'exist. Running vertical interpolation...'
        )
        vert_interp(config)
    else:
        print(
            f"Vertical interpolation file '{vert_interp_filename}' already "
            'exists. Skipping vertical interpolation.'
        )

    with LoggingContext(__name__) as logger:
        remap_horiz(config, vert_interp_filename, logger)


if __name__ == '__main__':
    main()
