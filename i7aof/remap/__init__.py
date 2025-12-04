import os

import numpy as np
from pyremap import Remapper

from i7aof.grid.ismip import (
    ensure_ismip_grid,
    get_horiz_res_string,
    ismip_proj4,
)


def remap_projection_to_ismip(
    in_filename,
    in_grid_name,
    out_filename,
    map_dir,
    method,
    config,
    logger,
    in_proj4='epsg:3031',
    renormalize=None,
):
    """
    Remap a dataset to the ISMIP grid, creating a mapping file if one does
    not already exist.

    Parameters
    ----------
    in_filename : str
        The input dataset filename.
    in_grid_name : str
        The name of the input grid.
    out_filename : str
        The output dataset filename.
    map_dir : str
        The directory where the mapping file will be stored.
    method : {'bilinear', 'neareststod', 'conserve'}
        The remapping method to use.
    config : mpas_tools.config.MpasConfigParser
        Configuration object with remapping parameters.
    logger : logging.Logger
        Logger object for logging messages.
    in_proj4 : str, optional
        The projection string for the input projection (default is
        'epsg:3031'). This can be any string that `pyproj.Proj` accepts.
    renormalize : float, optional
        If provided, a threshold to use to renormalize the data
    """
    if os.path.exists(out_filename):
        return

    (
        ismip_grid_filename,
        horiz_res_str,
        out_mesh_name,
        cores,
        remap_tool,
        esmf_path,
        moab_path,
        parallel_exec,
    ) = _get_remap_config(config)
    map_filename = os.path.join(
        map_dir, f'map_{in_grid_name}_to_{out_mesh_name}_{method}.nc'
    )

    print(f'Remapping from {in_grid_name} to ISMIP {horiz_res_str} grid...')

    remapper = _get_remapper(
        cores,
        method,
        map_filename,
        parallel_exec,
        remap_tool,
        esmf_path,
        moab_path,
    )
    remapper.src_from_proj(
        filename=in_filename,
        mesh_name=in_grid_name,
        proj_str=in_proj4,
    )
    remapper.dst_from_proj(
        filename=ismip_grid_filename,
        mesh_name=out_mesh_name,
        proj_str=ismip_proj4,
    )
    _remap_common(
        remapper,
        in_filename,
        ismip_grid_filename,
        out_filename,
        map_filename,
        logger,
        renormalize,
    )


def remap_lat_lon_to_ismip(
    in_filename,
    in_grid_name,
    out_filename,
    map_dir,
    method,
    config,
    logger,
    lon_var='lon',
    lat_var='lat',
    renormalize=None,
):
    """
    Remap a dataset on a lat-lon grid to the ISMIP grid, creating a mapping
    file if one does not already exist. Latitude and longitude can be 1D or
    2D arrays.

    Parameters
    ----------
    in_filename : str
        The input dataset filename.
    in_grid_name : str
        The name of the input grid.
    out_filename : str
        The output dataset filename.
    map_dir : str
        The directory where the mapping file will be stored.
    method : {'bilinear', 'neareststod', 'conserve'}
        The remapping method to use.
    config : mpas_tools.config.MpasConfigParser
        Configuration object with remapping parameters.
    logger : logging.Logger
        Logger object for logging messages.
    lon_var : str, optional
        The name of the longitude variable in the input dataset (default is
        'lon').
    lat_var : str, optional
        The name of the latitude variable in the input dataset (default is
        'lat').
    renormalize : float, optional
        If provided, a threshold to use to renormalize the data
    """
    if os.path.exists(out_filename):
        return

    (
        ismip_grid_filename,
        horiz_res_str,
        out_mesh_name,
        cores,
        remap_tool,
        esmf_path,
        moab_path,
        parallel_exec,
    ) = _get_remap_config(config)
    map_filename = os.path.join(
        map_dir, f'map_{in_grid_name}_to_{out_mesh_name}_{method}.nc'
    )

    print(f'Remapping from {in_grid_name} to ISMIP {horiz_res_str} grid...')

    remapper = _get_remapper(
        cores,
        method,
        map_filename,
        parallel_exec,
        remap_tool,
        esmf_path,
        moab_path,
    )
    remapper.src_from_lon_lat(
        filename=in_filename,
        mesh_name=in_grid_name,
        lon_var=lon_var,
        lat_var=lat_var,
    )
    remapper.dst_from_proj(
        filename=ismip_grid_filename,
        mesh_name=out_mesh_name,
        proj_str=ismip_proj4,
    )
    _remap_common(
        remapper,
        in_filename,
        ismip_grid_filename,
        out_filename,
        map_filename,
        logger,
        renormalize,
    )


def add_periodic_lon(ds, threshold=1e-10, lon_var='lon', periodic_dim=None):
    """
    Add a periodic longitude to a dataset if the longitude range is not
    approximately 360 degrees. This is typically needed for bilinear remapping
    to prevent there from being a seam between the first and last longitude
    values.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the longitude variable.

    threshold : float, optional
        The threshold to determine if the longitude range is approximately
        360 degrees. Default is 1e-10.

    lon_var : str, optional
        The name of the longitude variable in the dataset. Default is 'lon'.

    periodic_dim : str, optional
        The name of the dimension along which to add periodicity. For 1D
        longitude, `periodic_dim` is found automatically and this parameter is
        ignored.  For 2D longitude, the default is the same as `lon_var`.

    Returns
    -------
    xarray.Dataset
        The dataset with periodic longitude added if necessary.
    """
    if len(ds[lon_var].dims) == 1:
        lon = ds[lon_var].values
        lon_gap = np.abs(lon[-1] - lon[0] - 360.0)
        periodic_dim = ds[lon_var].dims[0]
    elif len(ds[lon_var].dims) == 2:
        if periodic_dim is None:
            periodic_dim = lon_var

        lon_min = ds[lon_var].isel({periodic_dim: 0})
        lon_max = ds[lon_var].isel({periodic_dim: -1})
        # the maximum gap
        lon_gap = np.abs(lon_max - lon_min - 360.0).max().values
    else:
        raise ValueError(
            f'Expected longitude variable "{lon_var}" to have 1 or 2 '
            f'dimensions, but got {len(ds[lon_var].dims)}.'
        )

    rad = 'rad' in ds[lon_var].attrs.get('units', '')
    if rad:
        lon_gap = np.rad2deg(lon_gap)

    if lon_gap > threshold:
        nperiodic_dim = ds.sizes[periodic_dim]
        ds = ds.isel({periodic_dim: np.append(np.arange(nperiodic_dim), [0])})

        if len(ds[lon_var].dims) == 1:
            # keep 1D longitude monotonic
            attrs = ds[lon_var].attrs
            lon = ds[lon_var].values
            lon[-1] = lon[0] + (2 * np.pi if rad else 360.0)
            ds[lon_var] = (periodic_dim, lon)
            ds[lon_var].attrs = attrs

    return ds


# --- Private helper functions below ---


def _get_remap_config(config):
    """
    Extract common remapping configuration values from config.
    """
    ismip_grid_filename = ensure_ismip_grid(config)
    horiz_res_str = get_horiz_res_string(config)
    out_mesh_name = f'ismip_{horiz_res_str}'

    cores = config.getint('remap', 'cores')
    remap_tool = config.get('remap', 'tool')

    esmf_path = config.get('remap', 'esmf_path')
    if esmf_path.lower() == 'none':
        esmf_path = None

    moab_path = config.get('remap', 'moab_path')
    if moab_path.lower() == 'none':
        moab_path = None

    parallel_exec = config.get('remap', 'parallel_exec')
    if parallel_exec.lower() == 'none':
        parallel_exec = None

    return (
        ismip_grid_filename,
        horiz_res_str,
        out_mesh_name,
        cores,
        remap_tool,
        esmf_path,
        moab_path,
        parallel_exec,
    )


def _get_remapper(
    cores,
    method,
    map_filename,
    parallel_exec,
    remap_tool,
    esmf_path,
    moab_path,
):
    """
    Create and configure a Remapper instance with common settings.
    """
    remapper = Remapper(
        ntasks=cores,
        method=method,
        map_filename=map_filename,
        parallel_exec=parallel_exec,
        map_tool=remap_tool,
        use_tmp=False,
    )
    remapper.esmf_path = esmf_path
    remapper.moab_path = moab_path
    return remapper


def _remap_common(
    remapper,
    in_filename,
    ismip_grid_filename,
    out_filename,
    map_filename,
    logger,
    renormalize,
):
    """
    Common logic for building the map and remapping fields.
    """
    remapper.src_scrip_filename = in_filename.replace('.nc', '.scrip.nc')
    remapper.dst_scrip_filename = ismip_grid_filename.replace(
        '.nc', '.scrip.nc'
    )
    if not os.path.exists(map_filename):
        print('  Computing remapping weights...')
        remapper.build_map(logger=logger)
    print('  Remapping fields...')
    remapper.ncremap(in_filename, out_filename, renormalize=renormalize)
    print('  Done.')
