import os

import numpy as np
import xarray as xr

from i7aof.download import download_file
from i7aof.io import write_netcdf
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

    def download_and_preprocess_topo(self):
        """
        Download the original topography file.
        """
        download_filename = os.path.join('topo', data_filename)
        download_file(
            url=data_url,
            dest_path=download_filename,
            quiet=self.config.getboolean('download', 'quiet'),
            overwrite=False,
        )

        out_filename = self.get_preprocessed_topo_path()
        os.makedirs(os.path.join('topo', 'intermediate'), exist_ok=True)
        self._preprocess_topo(download_filename, out_filename)
        super().download_and_preprocess_topo()

    def get_preprocessed_topo_path(self):
        """
        Get the path to the preprocessed topography file before remapping
        """
        filename = os.path.join(
            'topo',
            'intermediate',
            data_filename.replace('.nc', '_processed.nc'),
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
        os.makedirs(os.path.join('topo', 'intermediate'), exist_ok=True)
        remapped_filename = os.path.join(
            'topo',
            'intermediate',
            data_filename.replace('.nc', '_remapped.nc'),
        )
        remap_projection_to_ismip(
            in_filename=self.get_preprocessed_topo_path(),
            in_grid_name='bedmap3',
            in_proj4='epsg:3031',
            out_filename=remapped_filename,
            map_dir='topo',
            method=self.config.get('topo', 'remap_method'),
            config=self.config,
            logger=self.logger,
        )
        self.renormalize_topo_fields(
            remapped_filename,
            self.get_topo_on_ismip_path(),
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
        rename = {
            'bed_topography': 'bed',
            'surface_topography': 'surface',
            'ice_thickness': 'thickness',
        }

        ds_out = xr.Dataset()
        ds_out.attrs = ds_in.attrs

        mask = ds_in.mask
        ice_mask = np.logical_and(mask.notnull(), mask != 4)

        for var_name, new_var_name in rename.items():
            if var_name in ds_in:
                var_data = ds_in[var_name].astype(float)
                if new_var_name in ['surface', 'thickness']:
                    # use zero instead of NaN where ice is absent
                    var_data = var_data.where(ice_mask, 0.0)
                ds_out[new_var_name] = var_data
                ds_out[new_var_name].attrs = ds_in[var_name].attrs
            else:
                self.logger.warning(
                    f'Variable {var_name} not found in input dataset.'
                )

        ds_out['draft'] = ds_out['surface'] - ds_out['thickness']
        ds_out.draft.attrs = {
            'long_name': 'Ice draft',
            'units': 'meters',
        }

        # mask:
        # _FillValue (-9999): open ocean
        # 1: grounded_ice
        # 2: transiently_grounded_ice_shelf
        # 3: floating_ice_shelf
        # 4: rock

        ds_out['ice_frac'] = ice_mask.astype(float)
        ds_out.ice_frac.attrs = {
            'long_name': 'Area Fraction of Ice',
            'units': '1',
        }
        ds_out['ocean_frac'] = np.logical_or(mask.isnull(), mask == 3).astype(
            float
        )
        ds_out.ocean_frac.attrs = {
            'long_name': 'Area Fraction of Ocean',
            'units': '1',
        }
        ds_out['grounded_frac'] = np.logical_or(mask == 1, mask == 2).astype(
            float
        )
        ds_out.grounded_frac.attrs = {
            'long_name': 'Area Fraction of Grounded Ice',
            'units': '1',
        }
        ds_out['floating_frac'] = (mask == 3).astype(float)
        ds_out.floating_frac.attrs = {
            'long_name': 'Area Fraction of Floating Ice',
            'units': '1',
        }
        ds_out['rock_frac'] = (mask == 4).astype(float)
        ds_out.rock_frac.attrs = {
            'long_name': 'Area Fraction of Bare Rock',
            'units': '1',
        }

        topo_vars = [
            'bed',
            'draft',
            'surface',
            'thickness',
        ]
        for var in topo_vars:
            masked_var = f'ocean_masked_{var}'
            ds_out[masked_var] = ds_out[var].where(
                ds_out['ocean_frac'] > 0.0, 0.0
            )
            ds_out[masked_var].attrs = ds_out[var].attrs
            ds_out[masked_var].attrs['long_name'] = (
                f'Ocean-masked {ds_out[var].attrs["long_name"]}'
            )

        write_netcdf(ds_out, out_filename)
