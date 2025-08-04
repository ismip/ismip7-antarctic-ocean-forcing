import numpy as np
import xarray as xr

from i7aof.grid.ismip import get_horiz_res_string
from i7aof.io import write_netcdf


class TopoBase:
    """
    A base class for a topography data.

    Attributes
    ----------
    config : mpas_tools.config.MpasConfigParser
        Configuration options.

    logger : logging.Logger
        Logger for the class.

    horiz_res_str : str
        The horizontal resolution string (e.g. '1km', '2km', '4km', '8km').
    """

    def __init__(self, config, logger):
        """
        Create a topography object.

        Parameters
        ----------
        config : mpas_tools.config.MpasConfigParser
            Configuration options.
        logger : logging.Logger
            Logger for the class.
        """
        self.config = config
        self.logger = logger
        self.horiz_res_str = get_horiz_res_string(config)

    def download_and_preprocess_topo(self):
        """
        Download the original topography file, preprocess and check
        that the required fields are present. Subclasses should
        implement this method to download the data and preprocess it (e.g.
        rename variables and define fraction fields). Then, they should call
        this method to check the results.
        """
        ds = xr.open_dataset(self.get_preprocessed_topo_path())
        self.check(ds)

    def get_preprocessed_topo_path(self):
        """
        Get the path to the preprocessed topography file before remapping

        Returns
        -------
        str
            The path to the preprocessed topography
        """
        raise NotImplementedError(
            'get_preprocessed_topo_path must be implemented in a subclass'
        )

    def get_topo_on_ismip_path(self):
        """
        Get the path to the topography file.

        Returns
        -------
        str
            The path to the topography on the ISMIP grid
        """
        raise NotImplementedError(
            'get_topo_on_ismip_path must be implemented in a subclass'
        )

    def remap_topo_to_ismip(self):
        """
        Remap the topography to the ISMIP grid."
        """
        raise NotImplementedError(
            'remap_topo_to_ismip must be implemented in a subclass'
        )

    def renormalize_topo_fields(self, in_filename, out_filename):
        """
        Renormalize the topography fields (those that aren't fractions) by
        the appropriate area fractions.

        Parameters
        ----------
        in_filename : str
            The input filename to preprocess.
        out_filename : str
            The output filename after preprocessing.
        """
        renorm_threshold = self.config.getfloat('topo', 'renorm_threshold')

        renorm_fields = {
            'draft': 'ice_frac',
            'surface': 'ice_frac',
            'thickness': 'ice_frac',
            'ocean_masked_bed': 'ocean_frac',
            'ocean_masked_draft': 'floating_frac',
            'ocean_masked_surface': 'floating_frac',
            'ocean_masked_thickness': 'floating_frac',
        }

        ds = xr.open_dataset(in_filename)

        for field, frac in renorm_fields.items():
            attrs = ds[field].attrs
            mask = ds[frac] > renorm_threshold
            ds[field] = xr.where(mask, ds[field] / ds[frac], np.nan)
            ds[field].attrs = attrs

        write_netcdf(ds, out_filename)

    def check(self, ds):
        """
        Check that the fields in the topography dataset are as expected.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset to check.

        Raises
        ------
        ValueError
            If the dataset does not contain the expected fields.
        """
        expected_fields = [
            'bed',
            'draft',
            'surface',
            'thickness',
            'ocean_masked_bed',
            'ocean_masked_draft',
            'ocean_masked_surface',
            'ocean_masked_thickness',
            'ice_frac',
            'ocean_frac',
            'grounded_frac',
            'floating_frac',
            'rock_frac',
        ]
        for field in expected_fields:
            if field not in ds:
                raise ValueError(
                    f'The dataset does not contain the expected field: {field}'
                )
        self.logger.info('Topography dataset contains all expected fields.')
