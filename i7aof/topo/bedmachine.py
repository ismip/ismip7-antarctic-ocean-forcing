import os

import pyproj
from pyremap import Remapper
from pyremap.descriptor import ProjectionGridDescriptor

from i7aof.grid.ismip import get_ismip_grid_filename, ismip_proj4
from i7aof.topo.topo_base import TopoBase

data_url = 'https://nsidc.org/data/nsidc-0756/versions/3'

data_filename = 'BedMachineAntarctica-v3.nc'


class BedMachineAntarcticaV3(TopoBase):
    """
    A class for remapping and reading Bedmachine Antarctica v3 data

    See https://nsidc.org/data/nsidc-0756/versions/3 for more information
    about Bedmachine Antarctica v3 topography data.
    """

    def download_topo(self):
        """
        Download the original topography file.
        """
        self.get_orig_topo_path()

    def get_orig_topo_path(self):
        """
        Get the path to the original topography file before remapping
        """
        filename = os.path.join('topo', data_filename)
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f'File {filename} not found. Please download manually from '
                f'{data_url}, as we are not authorized to download it for '
                f'you.'
            )
        return filename

    def get_topo_on_ismip_path(self):
        """
        Get the path to the topography file.
        """
        filename = data_filename.replace(
            '.nc', f'_ismip_{self.horiz_res_str}.nc'
        )
        path = os.path.join('topo', filename)
        return path

    def remap_topo_to_ismip(self):
        """
        Remap the topography to the ISMIP grid."
        """
        config = self.config
        horiz_res_str = self.horiz_res_str
        in_filename = self.get_orig_topo_path()
        out_filename = self.get_topo_on_ismip_path()
        ismip_grid_filename = get_ismip_grid_filename(config)
        if os.path.exists(out_filename):
            return

        in_mesh_name = 'bedmachine_antarctica_v3'
        out_mesh_name = f'ismip_{horiz_res_str}'
        method = config.get('topo', 'remap_method')
        cores = config.get('topo', 'remap_cores')

        map_filename = os.path.join(
            'topo', f'map_{in_mesh_name}_to_{out_mesh_name}_{method}.nc'
        )

        print(
            f'Remapping Bedmachine Antarctica v3 to ISMIP '
            f'{horiz_res_str} grid...'
        )

        in_proj = pyproj.Proj('epsg:3031')
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
        remapper.build_mapping_file(method=method, mpiTasks=cores)
        print('  Remapping fields...')
        remapper.remap_file(in_filename, out_filename)
        print('  Done.')
