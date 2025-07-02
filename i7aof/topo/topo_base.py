import xarray as xr

from i7aof.grid.ismip import get_horiz_res_string


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
