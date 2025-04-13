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

    def download_topo(self):
        """
        Download the original topography file.
        """
        raise NotImplementedError(
            'download_topo must be implemented in a subclass'
        )

    def get_orig_topo_path(self):
        """
        Get the path to the original topography file before remapping

        Returns
        -------
        str
            The path to the original topography
        """
        raise NotImplementedError(
            'get_orig_topo_path must be implemented in a subclass'
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
