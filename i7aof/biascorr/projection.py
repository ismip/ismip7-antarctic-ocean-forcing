import numpy as np

from i7aof.biascorr.timeslice import Timeslice


class Projection:
    """
    A main class for projections

    Attributes
    ----------
    config : mpas_tools.config.MpasConfigParser
        Configuration options.

    logger : logging.Logger
        Logger for the class.
    """

    def __init__(self, config, logger):
        """
        Create a bias-corrected projection

        Parameters
        ----------

        config : mpas_tools.config.MpasConfigParser
            Configuration options.

        logger : logging.Logger
            Logger for the class.
        """
        self.config = config
        self.logger = logger

        self.get_model_info()

    def get_model_info(self):
        """
        Extract model info from config file
        """

        section = self.config['biascorr']
        self.thetao_ref = section.get('thetao_ref')
        self.so_ref = section.get('so_ref')
        self.thetao_modref = section.get('thetao_modref')
        self.so_modref = section.get('so_modref')
        self.thetao_mod = section.get('thetao_mod')
        self.so_mod = section.get('so_mod')

    def read_reference(self):
        """
        Read the reference period
        """

        self.ref = Timeslice(self.config, self.thetao_ref, self.so_ref)
        self.ref.get_all_data()
        self.modref = Timeslice(
            self.config, self.thetao_modref, self.so_modref
        )
        self.modref.get_all_data()

        print(np.nanmean(self.modref.T))
