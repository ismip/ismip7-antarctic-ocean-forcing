import os

import numpy as np
import xarray as xr

# size of the ISMIP grid in meters
ismip_lx = 6088e3
ismip_ly = 6088e3

ismip_proj4 = 'epsg:3031'


def write_ismip_grid(config):
    """
    Write the ISMIP grid to a NetCDF file.

    Parameters
    ----------
    config : ConfigParser
        Configuration object with grid parameters.
    """
    out_filename = get_ismip_grid_filename(config)
    if os.path.exists(out_filename):
        return

    path = os.path.dirname(out_filename)
    os.makedirs(path, exist_ok=True)

    ds = xr.Dataset()
    section = config['ismip_grid']
    dx = section.getfloat('dx')
    dy = section.getfloat('dy')
    nx = np.round(ismip_lx / dx)
    ny = np.round(ismip_ly / dy)
    dx = ismip_lx / nx
    dy = ismip_ly / ny
    x = dx * np.arange(-(nx - 1) // 2, (nx - 1) // 2 + 1)
    y = dy * np.arange(-(ny - 1) // 2, (ny - 1) // 2 + 1)
    ds['x'] = ('x', x)
    ds.x.attrs['units'] = 'meters'
    ds.x.attrs['standard_name'] = 'projection_x_coordinate'
    ds.x.attrs['long_name'] = 'x coordinate of projection'
    ds['y'] = ('y', y)
    ds.y.attrs['units'] = 'meters'
    ds.y.attrs['standard_name'] = 'projection_y_coordinate'
    ds.y.attrs['long_name'] = 'y coordinate of projection'
    ds.attrs['Grid'] = (
        'Datum = WGS84, earth_radius = 6378137., '
        'earth_eccentricity = 0.081819190842621, '
        'falseeasting = -3040000., '
        'falsenorthing = -3040000., '
        'standard_parallel = -71., central_meridien = 0, '
        'EPSG=3031'
    )
    ds.attrs['proj'] = (
        '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 '
        '+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
    )
    ds.attrs['proj4'] = ismip_proj4
    ds.to_netcdf(out_filename)


def get_ismip_grid_filename(config):
    """
    Get the ISMIP grid filename from the configuration.

    Parameters
    ----------
    config : ConfigParser
        Configuration object with grid parameters.

    Returns
    -------
    str
        The ISMIP grid filename.
    """
    horiz_res = get_horiz_res_string(config)
    return os.path.join('ismip', f'ismip_{horiz_res}_grid.nc')


def get_horiz_res_string(config):
    """
    Get the horizontal resolution string from the configuration.

    Parameters
    ----------
    config : ConfigParser
        Configuration object with grid parameters.

    Returns
    -------
    str
        The horizontal resolution as a string.
    """
    section = config['ismip_grid']
    hres = 1e-3 * section.getfloat('dx')
    if hres == int(hres):
        hres = int(hres)
    res = f'{hres}km'
    return res
