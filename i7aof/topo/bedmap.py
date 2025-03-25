import os

from i7aof.download import download_file
from i7aof.remap import remap_projection_to_ismip
from i7aof.topo.topo_base import TopoBase

data_url = (
    'https://ramadda.data.bas.ac.uk/repository/entry/get/bedmap3.nc'
    '?entryid=synth%3A2d0e4791-8e20-46a3-80e4-f5f6716025d2%3AL2JlZG1hcDMubmM%3D'  # noqa: E501
)

data_filename = 'bedmap3.nc'


class Bedmap3(TopoBase):
    """
    A class for remapping and reading Bedmap3 data

    See https://doi.org/10.5285/2d0e4791-8e20-46a3-80e4-f5f6716025d2 and
    https://doi.org/10.1038/s41597-025-04672-y for more information about
    Bedmap3 topography data.
    """

    def download_topo(self):
        """
        Download the original topography file.
        """
        download_file(
            url=data_url,
            dest_path='topo/bedmap3.nc',
            quiet=self.config.getboolean('download', 'quiet'),
            overwrite=False,
        )

    def get_orig_topo_path(self):
        """
        Get the path to the original topography file before remapping
        """
        filename = os.path.join('topo', data_filename)
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
            in_mesh_name='bedmap3',
            in_proj4='epsg:3031',
            out_filename=self.get_topo_on_ismip_path(),
            map_dir='topo',
            method=self.config.get('topo', 'remap_method'),
            config=self.config,
            logger=self.logger,
        )
