import os

from pyremap import Remapper

from i7aof.grid.ismip import (
    get_horiz_res_string,
    get_ismip_grid_filename,
    ismip_proj4,
)


def remap_projection_to_ismip(
    in_filename,
    in_mesh_name,
    out_filename,
    map_dir,
    method,
    config,
    logger,
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
    logger : logging.Logger
        Logger object for logging messages.
    in_proj4 : str, optional
        The projection string for the input projection (default is
        'epsg:3031'). This can be any string that `pyproj.Proj` accepts.
    """
    if os.path.exists(out_filename):
        return

    ismip_grid_filename = get_ismip_grid_filename(config)
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

    map_filename = os.path.join(
        map_dir, f'map_{in_mesh_name}_to_{out_mesh_name}_{method}.nc'
    )

    print(f'Remapping from {in_mesh_name} to ISMIP {horiz_res_str} grid...')

    remapper = Remapper(
        ntasks=cores,
        method=method,
        map_filename=map_filename,
        parallel_exec=parallel_exec,
        map_tool=remap_tool,
        use_tmp=False,
    )
    remapper.src_scrip_filename = in_filename.replace('.nc', '.scrip.nc')
    remapper.dst_scrip_filename = ismip_grid_filename.replace(
        '.nc', '.scrip.nc'
    )
    remapper.esmf_path = esmf_path
    remapper.moab_path = moab_path
    remapper.src_from_proj(
        filename=in_filename,
        mesh_name=in_mesh_name,
        proj_str=in_proj4,
    )
    remapper.dst_from_proj(
        filename=ismip_grid_filename,
        mesh_name=out_mesh_name,
        proj_str=ismip_proj4,
    )
    if not os.path.exists(map_filename):
        print('  Computing remapping weights...')
        remapper.build_map(logger=logger)
    print('  Remapping fields...')
    remapper.ncremap(in_filename, out_filename)
    print('  Done.')
