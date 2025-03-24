import os

import pyproj
from pyremap import Remapper
from pyremap.descriptor import ProjectionGridDescriptor

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

    ismip_grid_filename = get_ismip_grid_filename(config)
    horiz_res_str = get_horiz_res_string(config)
    out_mesh_name = f'ismip_{horiz_res_str}'

    cores = config.get('remap', 'cores')

    esmf_path = config.get('remap', 'esmf_path')
    if esmf_path.lower() == 'none':
        esmf_path = None

    parallel_exec = config.get('remap', 'parallel_exec')
    if parallel_exec.lower() == 'none':
        parallel_exec = None

    include_logs = config.getboolean('remap', 'include_logs')

    map_filename = os.path.join(
        map_dir, f'map_{in_mesh_name}_to_{out_mesh_name}_{method}.nc'
    )

    print(f'Remapping from {in_mesh_name} to ISMIP {horiz_res_str} grid...')

    in_proj = pyproj.Proj(in_proj4)
    out_proj = pyproj.Proj(ismip_proj4)
    in_descriptor = ProjectionGridDescriptor.read(
        projection=in_proj, fileName=in_filename, meshName=in_mesh_name
    )
    out_descriptor = ProjectionGridDescriptor.read(
        projection=out_proj,
        fileName=ismip_grid_filename,
        meshName=out_mesh_name,
    )

    remapper = Remapper(
        sourceDescriptor=in_descriptor,
        destinationDescriptor=out_descriptor,
        mappingFileName=map_filename,
    )
    print('  Computing remapping weights...')
    remapper.build_mapping_file(
        method=method,
        mpiTasks=cores,
        esmf_parallel_exec=parallel_exec,
        esmf_path=esmf_path,
        include_logs=include_logs,
    )
    print('  Remapping fields...')
    remapper.remap_file(in_filename, out_filename)
    print('  Done.')
