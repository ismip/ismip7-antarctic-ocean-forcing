import os

import numpy as np
import xarray as xr
from pyproj import Proj

from i7aof.io import write_netcdf

# size of the ISMIP grid in meters
ismip_lx = 6080e3
ismip_ly = 6080e3
ismip_lz = 1800.0

ismip_proj4 = 'epsg:3031'

# Reference grid metadata
min_x = -3040000.0
min_y = -3040000.0
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

    _add_horiz_grid(ds, config)

    section = config['ismip_grid']
    dz = section.getfloat('dz')
    dz_extrap = section.getfloat('dz_extrap')

    _add_vert_levels(ds, 'z', dz)
    _add_vert_levels(ds, 'z_extrap', dz_extrap)

    # Grid variables should not have _FillValue in outputs
    write_netcdf(ds, out_filename, has_fill_values=False)


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
    res = get_res_string(config)
    return os.path.join('ismip', f'ismip_{res}_grid.nc')


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


def get_ver_res_string(config, extrap: bool = False):
    """
    Get the vertical resolution string from the configuration.

    Parameters
    ----------
    config : mpas_tools.config.MpasConfigParser
        Configuration object with grid parameters.
    extrap : bool, optional
        If True, use ``dz_extrap`` instead of ``dz`` for the resolution.

    Returns
    -------
    str
        The vertical resolution as a string.
    """
    section = config['ismip_grid']
    key = 'dz_extrap' if extrap else 'dz'
    vres = section.getfloat(key)
    if vres == int(vres):
        vres = int(vres)
    res = f'{vres}m'
    return res


def get_res_string(config, extrap: bool = False):
    """
    Get the resolution string from the configuration.

    Parameters
    ----------
    config : mpas_tools.config.MpasConfigParser
        Configuration object with grid parameters.
    extrap : bool, optional
        If True, use ``dz_extrap`` for the vertical component of the
        resolution string.

    Returns
    -------
    str
        The resolution as a string combining horizontal and vertical
        resolutions.
    """
    hres = get_horiz_res_string(config)
    vres = get_ver_res_string(config, extrap=extrap)
    return f'{hres}_{vres}'


def _add_horiz_grid(ds, config):
    """
    Add horizontal grid variables to the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to which the horizontal grid variables will be added.
    config : mpas_tools.config.MpasConfigParser
        Configuration object with grid parameters.
    """

    section = config['ismip_grid']
    dx = section.getfloat('dx')
    dy = section.getfloat('dy')
    # Compute nx, ny as in reference: ((nx_base-1)/r)+1
    nx = int(((nx_base - 1) * 1000 / dx) + 1)
    ny = int(((ny_base - 1) * 1000 / dy) + 1)
    dx = ismip_lx / (nx - 1)
    dy = ismip_ly / (ny - 1)
    # x/y start at min_x, min_y
    x = min_x + dx * np.arange(nx)
    y = min_y + dy * np.arange(ny)
    ds['x'] = ('x', x)
    ds.x.attrs['units'] = 'm'
    ds.x.attrs['standard_name'] = 'projection_x_coordinate'
    ds.x.attrs['long_name'] = 'x coordinate of projection'
    ds.x.attrs['axis'] = 'X'
    ds.x.attrs['bounds'] = 'x_bnds'
    ds['y'] = ('y', y)
    ds.y.attrs['units'] = 'm'
    ds.y.attrs['standard_name'] = 'projection_y_coordinate'
    ds.y.attrs['long_name'] = 'y coordinate of projection'
    ds.y.attrs['axis'] = 'Y'
    ds.y.attrs['bounds'] = 'y_bnds'

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
    ds.lat.attrs['bounds'] = 'lat_bnds'
    ds['lon'] = (('y', 'x'), lon)
    ds.lon.attrs['units'] = 'degrees_east'
    ds.lon.attrs['standard_name'] = 'longitude'
    ds.lon.attrs['long_name'] = 'longitude coordinate'
    ds.lon.attrs['bounds'] = 'lon_bnds'

    x_bnds = np.zeros((nx, 2))
    y_bnds = np.zeros((ny, 2))
    x_bnds[:, 0] = x - 0.5 * dx
    x_bnds[:, 1] = x + 0.5 * dx
    y_bnds[:, 0] = y - 0.5 * dy
    y_bnds[:, 1] = y + 0.5 * dy
    ds['x_bnds'] = (('x', 'bnds'), x_bnds)
    ds['y_bnds'] = (('y', 'bnds'), y_bnds)

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
    ds['lon_bnds'] = (('y', 'x', 'nv'), lon_bnds)

    ds['crs'] = xr.DataArray(
        np.array(0, dtype=np.int32),
        dims=(),
        attrs={
            'grid_mapping_name': 'polar_stereographic',
            'latitude_of_projection_origin': -90.0,
            'standard_parallel': -71.0,
            'straight_vertical_longitude_from_pole': 0.0,
            'false_easting': 0.0,
            'false_northing': 0.0,
            'semi_major_axis': 6378137.0,
            'inverse_flattening': 298.257223563,
            'epsg_code': 'EPSG:3031',
        },
    )

    ds.attrs['Grid'] = (
        'Datum = WGS84, earth_radius = 6378137., '
        'earth_eccentricity = 0.081819190842621, '
        'falseeasting = 0, '
        'falsenorthing = 0, '
        'standard_parallel = -71., '
        'central_meridian = 0, '
        'EPSG=3031'
    )
    ds.attrs['proj'] = (
        '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 '
        '+x_0=0 +y_0=0 +datum=WGS84 +units=m '
        '+no_defs'
    )
    ds.attrs['proj4'] = ismip_proj4
    ds.attrs['Conventions'] = 'CF-1.10'


def _add_vert_levels(ds, coord_name, dz):
    """
    Add vertical levels to the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to which the vertical levels will be added.

    coord_name : str
        The name of the vertical coordinate (e.g., 'z').

    dz : float
        The vertical grid spacing.
    """
    nz = int(np.ceil(ismip_lz / dz))
    z = -np.arange(nz + 1) * dz

    lev = 0.5 * (z[:-1] + z[1:])
    lev_bnds = np.zeros((nz, 2))
    lev_bnds[:, 0] = z[:-1]
    lev_bnds[:, 1] = z[1:]

    coord_bnds_name = f'{coord_name}_bnds'

    ds[coord_name] = ((coord_name), lev)
    ds[coord_name].attrs['units'] = 'm'
    ds[coord_name].attrs['standard_name'] = 'height'
    ds[coord_name].attrs['long_name'] = (
        'height relative to sea surface (positive up)'
    )
    ds[coord_name].attrs['positive'] = 'up'
    ds[coord_name].attrs['axis'] = 'Z'
    ds[coord_name].attrs['bounds'] = coord_bnds_name

    ds[coord_bnds_name] = ((coord_name, 'bnds'), lev_bnds)
