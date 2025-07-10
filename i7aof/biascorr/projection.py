import numpy as np
import xarray as xr

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

        self.create_basin_mask()

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

        self.mod_ystart = section.getint('mod_ystart')
        self.mod_yend = section.getint('mod_yend')
        self.mod_ystep = section.getint('mod_ystep')

        self.z_shelf = section.getfloat('z_shelf')
        self.filename_topo = section.get('filename_topo')
        self.filename_imbie = section.get('filename_imbie')

    def read_reference(self):
        """
        Read the reference period
        """

        self.ref = Timeslice(
            self.config, self.thetao_ref, self.so_ref, self.basinmask
        )
        self.ref.get_all_data()
        self.modref = Timeslice(
            self.config, self.thetao_modref, self.so_modref, self.basinmask
        )
        self.modref.get_all_data()

    def read_model(self):
        """
        Read whole model period
        """

        self.years = range(self.mod_ystart, self.mod_yend)

        for year in self.years:
            _ = self.read_model_timeslice(year)
            print(f'Read year {year}')

    def read_model_timeslice(self, year):
        """
        Read a timeslice from the future period
        """

        out = Timeslice(
            self.config,
            self.thetao_mod,
            self.so_mod,
            self.basinmask,
            year=year,
        )
        out.get_all_data()

        return out

    def create_basin_mask(self):
        """
        Create a mask per IMBIE basin
        over the continental shelf
        """

        ds_topo = xr.open_dataset(self.filename_topo)
        ds_topo.close()

        ds_imbie = xr.open_dataset(self.filename_imbie)
        self.basins = np.unique(ds_imbie.basinNumber.values)
        self.basinmask = np.zeros(
            (len(self.basins), len(ds_imbie.x), len(ds_imbie.y))
        )
        for b, basin in enumerate(self.basins):
            self.basinmask[b, :, :] = np.where(
                ds_imbie.basinNumber.values == basin, 1, 0
            )
            self.basinmask[b, :, :] = np.where(
                ds_topo.bed.values > self.z_shelf, self.basinmask[b, :, :], 0
            )
        ds_imbie.close()
