import os

import numpy as np
import xarray as xr

from i7aof.remap import remap_projection_to_ismip
from i7aof.topo.topo_base import TopoBase

data_url = 'https://nsidc.org/data/nsidc-0756/versions/3'

data_filename = 'BedMachineAntarctica-v3.nc'


class BedMachineAntarcticaV3(TopoBase):
    """
    A class for remapping and reading Bedmachine Antarctica v3 data

    See https://nsidc.org/data/nsidc-0756/versions/3 for more information
    about Bedmachine Antarctica v3 topography data.
    """

    def download_and_preprocess_topo(self):
        """
        Download the original topography file.
        """
        download_filename = os.path.join('topo', data_filename)
        if not os.path.exists(download_filename):
            raise FileNotFoundError(
                f'File {download_filename} not found. Please download '
                f'manually from {data_url}, as we are not authorized to '
                f'download it for you.'
            )

        out_filename = self.get_preprocessed_topo_path()
        self._preprocess_topo(download_filename, out_filename)
        super().download_and_preprocess_topo()

    def get_preprocessed_topo_path(self):
        """
        Get the path to the preprocessed topography file before remapping
        """
        filename = os.path.join(
            'topo', data_filename.replace('.nc', '_processed.nc')
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
            in_filename=self.get_preprocessed_topo_path(),
            in_mesh_name='bedmachine_antarctica_v3',
            in_proj4='epsg:3031',
            out_filename=self.get_topo_on_ismip_path(),
            map_dir='topo',
            method=self.config.get('topo', 'remap_method'),
            config=self.config,
            logger=self.logger,
        )

    def _preprocess_topo(self, in_filename, out_filename):
        """
        Preprocess the topography file before remapping.

        Parameters
        ----------
        in_filename : str
            The input filename to preprocess.
        out_filename : str
            The output filename after preprocessing.
        """
        ds_in = xr.open_dataset(in_filename)
        vars_to_copy = ['bed', 'surface', 'thickness']

        ds_out = ds_in[vars_to_copy]
        ds_out.attrs = ds_in.attrs

        ds_out['draft'] = ds_out['surface'] - ds_out['thickness']
        ds_out.draft.attrs = {
            'long_name': 'Ice draft',
            'units': 'meters',
        }

        # mask:
        # 0: open ocean
        # 1: ice_free_land
        # 2: grounded_ice
        # 3: floating_ice
        # 4: lake_vostok
        mask = ds_in.mask

        ds_out['ice_frac'] = np.logical_and(mask != 0, mask != 1).astype(float)
        ds_out.ice_frac.attrs = {
            'long_name': 'Area Fraction of Ice',
            'units': '1',
        }
        ds_out['ocean_frac'] = np.logical_or(mask == 0, mask == 3).astype(
            float
        )
        ds_out.ocean_frac.attrs = {
            'long_name': 'Area Fraction of Ocean',
            'units': '1',
        }
        ds_out['grounded_frac'] = (mask == 2).astype(float)
        ds_out.grounded_frac.attrs = {
            'long_name': 'Area Fraction of Grounded Ice',
            'units': '1',
        }
        ds_out['floating_frac'] = (mask == 3).astype(float)
        ds_out.floating_frac.attrs = {
            'long_name': 'Area Fraction of Floating Ice',
            'units': '1',
        }
        ds_out['rock_frac'] = (mask == 1).astype(float)
        ds_out.rock_frac.attrs = {
            'long_name': 'Area Fraction of Bare Rock',
            'units': '1',
        }

        ds_out.to_netcdf(out_filename)
