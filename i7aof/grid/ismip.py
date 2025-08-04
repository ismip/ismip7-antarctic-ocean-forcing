import os

import numpy as np
import xarray as xr
from pyproj import Proj

# size of the ISMIP grid in meters
ismip_lx = 6080e3
ismip_ly = 6080e3

ismip_proj4 = 'epsg:3031'

# Reference grid metadata
falseeasting = -3040000.0
falsenorthing = -3040000.0
nx_base = 6081
ny_base = 6081


def write_ismip_grid(config):
    """
    Write the ISMIP grid to a NetCDF file.

    Parameters
    ----------
    config : mpas_tools.config.MpasConfigParser
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
    # Compute nx, ny as in reference: ((nx_base-1)/r)+1
    nx = int(((nx_base - 1) * 1000 / dx) + 1)
    ny = int(((ny_base - 1) * 1000 / dy) + 1)
    dx = ismip_lx / (nx - 1)
    dy = ismip_ly / (ny - 1)
    # x/y start at falseeasting/falsenorthing
    x = falseeasting + dx * np.arange(nx)
    y = falsenorthing + dy * np.arange(ny)
    ds['x'] = ('x', x)
    ds.x.attrs['units'] = 'meters'
    ds.x.attrs['standard_name'] = 'projection_x_coordinate'
    ds.x.attrs['long_name'] = 'x coordinate of projection'
    ds['y'] = ('y', y)
    ds.y.attrs['units'] = 'meters'
    ds.y.attrs['standard_name'] = 'projection_y_coordinate'
    ds.y.attrs['long_name'] = 'y coordinate of projection'

    # Compute lat/lon using pyproj
    proj = Proj(ismip_proj4)
    x_bcast, y_bcast = np.meshgrid(x, y)
    lon, lat = proj(x_bcast, y_bcast, inverse=True)
    # Ensure lon in [-180, 180)
    lon = np.mod(lon + 180, 360) - 180
    ds['lat'] = (('y', 'x'), lat)
    ds.lat.attrs['units'] = 'degrees_north'
    ds.lat.attrs['standard_name'] = 'latitude'
    ds.lat.attrs['long_name'] = 'latitude coordinate'
    ds['lon'] = (('y', 'x'), lon)
    ds.lon.attrs['units'] = 'degrees_east'
    ds.lon.attrs['standard_name'] = 'longitude'
    ds.lon.attrs['long_name'] = 'longitude coordinate'

    x_bnds = np.zeros((nx, 2))
    y_bnds = np.zeros((ny, 2))
    x_bnds[:, 0] = x - 0.5 * dx
    x_bnds[:, 1] = x + 0.5 * dx
    y_bnds[:, 0] = y - 0.5 * dy
    y_bnds[:, 1] = y + 0.5 * dy
    ds['x_bnds'] = (('x', 'nbounds'), x_bnds)
    ds['y_bnds'] = (('y', 'nbounds'), y_bnds)
    ds.x_bnds.attrs['units'] = 'meters'
    ds.x_bnds.attrs['standard_name'] = 'projection_x_coordinate_bounds'
    ds.x_bnds.attrs['long_name'] = 'x coordinate bounds of projection'
    ds.y_bnds.attrs['units'] = 'meters'
    ds.y_bnds.attrs['standard_name'] = 'projection_y_coordinate_bounds'
    ds.y_bnds.attrs['long_name'] = 'y coordinate bounds of projection'

    # Compute lat_bnds and lon_bnds using numpy array math
    x_offsets = np.array([0.5, 0.5, -0.5, -0.5]) * dx
    y_offsets = np.array([-0.5, 0.5, 0.5, -0.5]) * dy
    x_corners = np.zeros((ny, nx, 4))
    y_corners = np.zeros((ny, nx, 4))
    for i in range(4):
        x_corners[:, :, i] = x_bcast + x_offsets[i]
        y_corners[:, :, i] = y_bcast + y_offsets[i]
    lon_bnds, lat_bnds = proj(x_corners, y_corners, inverse=True)
    # Ensure lon_bnds in [-180, 180)
    lon_bnds = np.mod(lon_bnds + 180, 360) - 180
    ds['lat_bnds'] = (('y', 'x', 'nv'), lat_bnds)
    ds.lat_bnds.attrs['units'] = 'degrees_north'
    ds.lat_bnds.attrs['standard_name'] = 'latitude_bounds'
    ds.lat_bnds.attrs['long_name'] = 'latitude bounds'
    ds['lon_bnds'] = (('y', 'x', 'nv'), lon_bnds)
    ds.lon_bnds.attrs['units'] = 'degrees_east'
    ds.lon_bnds.attrs['standard_name'] = 'longitude_bounds'
    ds.lon_bnds.attrs['long_name'] = 'longitude bounds'

    ds.attrs['Grid'] = (
        'Datum = WGS84, earth_radius = 6378137., '
        'earth_eccentricity = 0.081819190842621, '
        f'falseeasting = {falseeasting}, '
        f'falsenorthing = {falsenorthing}, '
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
    config : mpas_tools.config.MpasConfigParser
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
    config : mpas_tools.config.MpasConfigParser
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
