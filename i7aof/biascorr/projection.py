import os
from glob import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tqdm import tqdm

from i7aof.biascorr.timeslice import Timeslice
from i7aof.io import read_dataset


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

    Public
    ------
    ref: Timeslice
        Reference timeslice to which a model will be bias-corrected.
        This should usually be an observational dataset
    modref: Timeslice
        Model reference timeslice. This should be a model Timeslice
        averaged over the reference period. The input fields (T,S)
        should not be extrapolated, to prevent biases in the volume
        of certain water masses.
    base: Timeslice
        Base timeslice to which anomalies are added. This should be
        a model Timeslice over the reference period, similar to modref.
        Usually, the base Timeslice should contain extrapolated T, S
        fields though, to ensure the output bias-corrected fields are
        also extrapolated.
    basinNumber: int([ny, nx])
        Gridded product denoting the individual basins by integer values
        Usually, this should be the 16 IMBIE basins, though a modification
        of these is required to ensure each contains a finite volume over
        the continental shelf. This same modification is required for the
        basic extrapolation of T and S.
    basinmask: int([Nbasins, ny, nx])
        Mask containing 0s and 1s, denoting which horizontal grid cell
        is part of which basin. 1: True, 0: False
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

        self.create_basin_masks()

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
        self.thetao_base = section.get('thetao_base')
        self.so_base = section.get('so_base')

        self.z_shelf = section.getfloat('z_shelf')
        self.filename_topo = section.get('filename_topo')
        self.filename_imbie = section.get('filename_imbie')

        self.Nbins = section.getint('Nbins')

    @status('Reading reference data')
    def read_reference(self):
        """
        Read the reference period of
        reference data set (ref),
        model (modref),
        and base, extrapolated reference (base)
        """

        self.ref = Timeslice(
            self.config,
            self.thetao_ref,
            self.so_ref,
            self.basinmask,
            self.basinNumber,
        )
        self.ref.get_all_data()

        self.modref = Timeslice(
            self.config,
            self.thetao_modref,
            self.so_modref,
            self.basinmask,
            self.basinNumber,
        )
        self.modref.get_all_data()

        self.base = Timeslice(
            self.config,
            self.thetao_base,
            self.so_base,
            self.basinmask_extrap,
            self.basinNumber,
        )
        self.base.get_all_data()

    def read_model(self):
        """
        Read whole model period
        """

        # Get all thetao files
        thetao_files = sorted(glob(self.thetao_mod))

        for thetao_file in thetao_files:
            # Check whether similar so file exists
            so_file = thetao_file.replace('thetao', 'so')
            assert os.path.isfile(so_file), f'No so file for {thetao_file}'

            # Create datasets for output
            ds = read_dataset(thetao_file)
            dsT = ds.copy()
            ds.close()
            dsT.thetao[:] = np.nan
            ds = read_dataset(so_file)
            dsS = ds.copy()
            dsS.so[:] = np.nan
            ds.close()

            # Make sure time-dimensions are equal
            xr.testing.assert_equal(dsT.time, dsS.time)
            years = [times.year for times in dsT.time.values]

            # Do actual bias correction
            desc = f'Processing {years[0]} to {years[-1]}'
            for y in tqdm(range(len(years)), desc=desc, colour='yellow'):
                ts = self.read_model_timeslice(thetao_file, so_file, yidx=y)
                dsT.thetao[y, :, :, :] = ts.T_corrected
                dsS.so[y, :, :, :] = ts.S_corrected
            dsT.to_netcdf(f'thetao_corrected_{years[0]}_{years[-1]}.nc')
            dsS.to_netcdf(f'so_corrected_{years[0]}_{years[-1]}.nc')
            dsT.close()
            dsS.close()

        return

    def read_model_timeslice(self, thetao_file, so_file, yidx):
        """
        Read a timeslice from the future period
        """

        timeslice = Timeslice(
            self.config,
            thetao_file,
            so_file,
            self.basinmask,
            self.basinNumber,
            yidx=yidx,
        )
        timeslice.get_T_and_S()
        timeslice.compute_delta(self.modref)
        timeslice.apply_anomaly(self.base)

        return timeslice

    @status('Creating basin mask')
    def create_basin_masks(self):
        """
        Create a mask per IMBIE basin
        over the continental shelf
        """

        ds_topo = read_dataset(self.filename_topo)
        ds_topo.close()

        ds_imbie = read_dataset(self.filename_imbie)
        self.basinNumber = ds_imbie.basinNumber.values
        self.basins = np.unique(ds_imbie.basinNumber.values)

        self.basinmask = np.zeros(
            (len(self.basins), len(ds_imbie.x), len(ds_imbie.y))
        )
        self.basinmask_extrap = np.zeros(
            (len(self.basins), len(ds_imbie.x), len(ds_imbie.y))
        )
        for b, basin in enumerate(self.basins):
            # Basin mask for extrapolated values
            self.basinmask_extrap[b, :, :] = np.where(
                ds_imbie.basinNumber.values == basin, 1, 0
            )
            # Basin mask with ice, bed, and deep ocean masked
            self.basinmask[b, :, :] = np.where(
                ds_topo.bed.values > self.z_shelf,
                self.basinmask_extrap[b, :, :],
                0,
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
