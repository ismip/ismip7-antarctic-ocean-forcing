#!/usr/bin/env python
import os

import numpy as np
import xarray as xr
from mpas_tools.config import MpasConfigParser

from i7aof.grid.ismip import get_ismip_grid_filename, write_ismip_grid
from i7aof.io import write_netcdf
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
    for var in ['lat', 'lon']:
        ds_out[var] = ds[var]
        var_bnds = f'{var}_bnds'
        ds_out[var_bnds] = ds[var_bnds]
    ds_out['z_extrap_bnds'] = ds_ismip['z_extrap_bnds']

    write_netcdf(ds_out, 'thetao_normalized.nc', progress_bar=True)

    ds_out = ds_out.isel(time=0)
    write_netcdf(ds_out, 'thetao_time0_normalized.nc', progress_bar=True)


def main():
    config = MpasConfigParser()
    config.add_from_package('i7aof', 'default.cfg')
    config.add_user_config('test_vert_interp.cfg')

    work_base_dir = config.get('workdir', 'base_dir')
    os.makedirs(work_base_dir, exist_ok=True)
    os.chdir(work_base_dir)

    write_ismip_grid(config)

    ds_ismip = xr.open_dataset(get_ismip_grid_filename(config))

    src_filename = (
        '/lcrc/group/e3sm/ac.xylar/ismip7/CMIP6_test_protocol/CESM2-WACCM/'
        'historical/Omon/'
        'thetao_Omon_CESM2-WACCM_historical_r1i1p1f1_gn_185001-201412.nc'
    )

    with xr.open_dataset(src_filename, decode_times=False) as ds:
        src_valid = ds['thetao'].isel(time=0).notnull()
        src_valid = src_valid.drop_vars(['time'])

    time_chunk = 1

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


if __name__ == '__main__':
    main()
