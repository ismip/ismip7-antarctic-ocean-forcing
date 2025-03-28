import os

from i7aof.remap.remap import remap_projection_to_ismip
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
        remap_projection_to_ismip(
            in_filename=self.get_orig_topo_path(),
            in_mesh_name='bedmachine_antarctica_v3',
            in_proj4='epsg:3031',
            out_filename=self.get_topo_on_ismip_path(),
            map_dir='topo',
            method=self.config.get('topo', 'remap_method'),
            config=self.config,
        )
