import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from i7aof.biascorr.timeslice import Timeslice


def status(process_name):
    def inner_decorator(func):
        def wrapper(*args, **kwargs):
            statement = process_name
            for _key, value in kwargs.items():
                statement += f'{value}'

            Ndots = max(3, 40 - len(statement))
            print(f'{statement} {"." * Ndots} \033[033mRunning\033[0m')
            result = func(*args, **kwargs)
            # Code to execute after the function call
            print(f'\033[F{statement} {"." * Ndots} \033[032mFinished\033[0m')
            return result

        return wrapper

    return inner_decorator


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

        self.Nbins = section.getint('Nbins')

    @status('Reading reference data')
    def read_reference(self):
        """
        Read the reference period of
        reference data set (ref)
        and model (modref)
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
            _ = self.read_model_timeslice(year=year)

        return

    @status('Reading model timeslice year ')
    def read_model_timeslice(self, year):
        """
        Read a timeslice from the future period
        """

        timeslice = Timeslice(
            self.config,
            self.thetao_mod,
            self.so_mod,
            self.basinmask,
            year=year,
        )
        timeslice.get_all_data()
        timeslice.compute_delta(self.modref)

        return timeslice

    @status('Creating basin mask')
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

    def compute_bias(self):
        """
        Compute the bias of the model T and S
        with respect to the reference.
        This will create corrected T and S bins (Tc and Sc)
        """

        self.compute_S_bias()

        self.compute_T_bias()

        return

    @status('Computing salinity bias')
    def compute_S_bias(self, perc=99):
        """
        Compute the salinity bias
        """

        self.modref.Sc = 0.0 * self.modref.Sb
        self.ref.Sperc = np.zeros(len(self.basins))
        self.modref.Sperc = np.zeros(len(self.basins))
        self.Sscaling = np.zeros(len(self.basins))

        for b, bmask in enumerate(self.basinmask):
            # Determine the percentile of reference salinity
            volume = (self.ref.V.values * bmask).flatten()
            self.ref.Sperc[b] = np.percentile(
                self.ref.S.values.flatten()[volume > 0],
                perc,
                method='inverted_cdf',
                weights=volume[volume > 0],
            )

            # Determine the PDF of model salinity
            volume = (self.modref.V.values * bmask).flatten()
            self.modref.Sperc[b] = np.percentile(
                self.modref.S.values.flatten()[volume > 0],
                perc,
                method='inverted_cdf',
                weights=volume[volume > 0],
            )

            # Try various scalings
            scalings = np.arange(0.5, 1.5, 0.1)
            rmse = np.zeros((len(scalings)))

            # Get binned histogram of reference salinity
            ref, _ = np.histogram(
                self.ref.S, bins=self.ref.Sb[b, :], weights=self.ref.V * bmask
            )
            # Determine rmse for each scaling
            for s, scaling in enumerate(scalings):
                modref, _ = np.histogram(
                    scaling * (self.modref.S - self.modref.Sperc[b])
                    + self.ref.Sperc[b],
                    bins=self.ref.Sb[b, :],
                    weights=self.modref.V * bmask,
                )
                rmse[s] = (
                    np.sum((modref / np.sum(modref) - ref / np.sum(ref)) ** 2)
                    ** 0.5
                )

            # Get the scaling with the lowest rmse
            idx = np.unravel_index(np.nanargmin(rmse, axis=None), rmse.shape)
            self.Sscaling[b] = scalings[idx[0]]

            # Extract corrected bins
            self.modref.Sc[b, :] = (
                self.Sscaling[b]
                * (self.modref.Sb[b, :] - self.modref.Sperc[b])
                + self.ref.Sperc[b]
            )

        return

    @status('Computing temperature bias')
    def compute_T_bias(self, perc=99):
        """
        Compute the temperature bias
        """

        self.modref.Tc = 0.0 * self.modref.Tb
        self.ref.Tperc = np.zeros(len(self.basins))
        self.modref.Tperc = np.zeros(len(self.basins))
        self.Tscaling = np.zeros(len(self.basins))

        for b, bmask in enumerate(self.basinmask):
            # Determine the percentile of reference T - Tmin
            volume = (self.ref.V.values * bmask).flatten()
            self.ref.Tperc[b] = np.percentile(
                (self.ref.T - self.ref.Tb[b, 0]).values.flatten()[volume > 0],
                perc,
                method='inverted_cdf',
                weights=volume[volume > 0],
            )

            # Determine the percentile of model T - Tmin
            volume = (self.modref.V.values * bmask).flatten()
            self.modref.Tperc[b] = np.percentile(
                (self.modref.T - self.modref.Tb[b, 0]).values.flatten()[
                    volume > 0
                ],
                perc,
                method='inverted_cdf',
                weights=volume[volume > 0],
            )

            # Determine scaling
            self.Tscaling[b] = self.ref.Tperc[b] / self.modref.Tperc[b]

            # Extract corrected bins
            self.modref.Tc[b, :] = (
                self.Tscaling[b]
                * (self.modref.Tb[b, :] - self.modref.Tb[b, 0])
                + self.ref.Tb[b, 0]
            )

    @status('Plotting TS diagrams')
    def plot_TS_diagrams(self, filename):
        """
        Create a figure with T-S diagrams per basin
        of reference (blue), model reference (orange)
        and bias-corrected model (green)
        """

        fig = plt.figure(figsize=(7, 7), constrained_layout=True)

        for b, _bmask in enumerate(self.basinmask):
            ax = fig.add_subplot(4, 4, b + 1)

            ax.pcolormesh(
                self.ref.Sb[b, :],
                self.ref.Tb[b, :],
                self.ref.Vb[b, :, :].T,
                cmap='Blues',
                norm=mpl.colors.LogNorm(),
                alpha=0.8,
            )
            ax.pcolormesh(
                self.modref.Sb[b, :],
                self.modref.Tb[b, :],
                self.modref.Vb[b, :, :].T,
                cmap='Oranges',
                norm=mpl.colors.LogNorm(),
                alpha=0.8,
            )
            ax.pcolormesh(
                self.modref.Sc[b, :],
                self.modref.Tc[b, :],
                self.modref.Vb[b, :, :].T,
                cmap='Greens',
                norm=mpl.colors.LogNorm(),
                alpha=0.8,
            )

        plt.savefig(filename)
