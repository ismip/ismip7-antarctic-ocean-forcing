import os

from i7aof.grid.ismip import (
    get_horiz_res_string,
    get_ismip_grid_filename,
    ismip_proj4,
)
from i7aof.remap.mapping import Mapping


def remap_projection_to_ismip(
    in_filename,
    in_mesh_name,
    out_filename,
    map_dir,
    method,
    config,
    in_proj4='epsg:3031',
):
    """
    Remap a dataset to the ISMIP grid, creating a mapping file if one does
    not already exist.

    Parameters
    ----------
    in_filename : str
        The input dataset filename.
    in_mesh_name : str
        The name of the input mesh.
    out_filename : str
        The output dataset filename.
    map_dir : str
        The directory where the mapping file will be stored.
    method : str
        The remapping method to use (e.g., 'bilinear', 'nearest_s2d').
    config : ConfigParser
        Configuration object with remapping parameters.
    in_proj4 : str, optional
        The projection string for the input projection (default is
        'epsg:3031'). This can be any string that `pyproj.Proj` accepts.
    """
    if os.path.exists(out_filename):
        return

    in_scrip_filename = in_filename.replace('.nc', '.scrip.nc')
    ismip_grid_filename = get_ismip_grid_filename(config)
    ismip_scrip_filename = ismip_grid_filename.replace('.nc', '.scrip.nc')
    horiz_res_str = get_horiz_res_string(config)
    out_mesh_name = f'ismip_{horiz_res_str}'

    map_filename = os.path.join(
        map_dir, f'map_{in_mesh_name}_to_{out_mesh_name}_{method}.nc'
    )
    cores = config.getint('remap', 'cores')
    mapping = Mapping(
        config=config,
        ntasks=cores,
        map_filename=map_filename,
        method=method,
        src_mesh_filename=in_scrip_filename,
        dst_mesh_filename=ismip_scrip_filename,
    )
    mapping.src_from_proj(
        filename=in_filename,
        mesh_name=in_mesh_name,
        x_var='x',
        y_var='y',
        proj_str=in_proj4,
    )
    mapping.dst_from_proj(
        filename=ismip_grid_filename,
        mesh_name=out_mesh_name,
        x_var='x',
        y_var='y',
        proj_str=ismip_proj4,
    )

    print(f'Remapping from {in_mesh_name} to ISMIP {horiz_res_str} grid...')

    print('  Computing remapping weights...')
    remapper = mapping.build_map()

    print('  Remapping fields...')
    remapper.remap_file(in_filename, out_filename)
    print('  Done.')
