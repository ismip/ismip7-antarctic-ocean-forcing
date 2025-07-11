import numpy as np
import xarray as xr
from cftime import DatetimeNoLeap


class Timeslice:
    """
    A class for a time slice containing T and S

    Attributes
    ----------
    config : mpas_tools.config.MpasConfigParser
        Configuration options.
    thetao: str
        name of file containing thetao
    so: str
        name of file containing so
    """

    def __init__(self, config, thetao, so, basinmask, year=None):
        """
        Extract T and S data from a timeslice

        Parameters
        ----------
        config : mpas_tools.config.MpasConfigParser
            Configuration options.
        thetao: str
            name of file containing thetao
        so: str
            name of file containing so
        basinmask: np.array(16, len(y), len(x))
            mask to multiply volume with
        year: int
            year to select. If None, taking time-mean
            Use None for dataset with single time value
        """

        self.config = config
        self.thetao = thetao
        self.so = so
        self.basinmask = basinmask
        self.year = year

        section = self.config['biascorr']
        self.Nbins = section.getint('Nbins')
        self.Nbasins = self.basinmask.shape[0]

    def get_all_data(self):
        """
        Extract all data from time slice
        """

        self.get_T_and_S()
        self.get_volume()
        self.get_bins()

    def get_volume(self):
        """
        Extract volume per grid cell
        """

        # TODO: apply grid cell fractions
        ds = xr.open_mfdataset(self.thetao, use_cftime=True)
        dz = abs(ds.z[1] - ds.z[0]).values
        dx = abs(ds.x[1] - ds.x[0]).values
        dy = abs(ds.y[1] - ds.y[0]).values
        ds.close()

        # Set volume to zero where temperature data is missing
        self.V = xr.where(np.isnan(self.T), 0, dz * dy * dx)

    def get_T_and_S(self):
        """
        Extract T and S data
        """

        ds = xr.open_mfdataset(self.thetao, use_cftime=True)
        if self.year is not None:
            ds = ds.sel(
                time=slice(
                    DatetimeNoLeap(self.year, 1, 1),
                    DatetimeNoLeap(self.year + 1, 1, 1),
                )
            )
        self.T = ds.thetao.mean(dim='time')
        ds.close()

        ds = xr.open_mfdataset(self.so, use_cftime=True)
        if self.year is not None:
            ds = ds.sel(
                time=slice(
                    DatetimeNoLeap(self.year, 1, 1),
                    DatetimeNoLeap(self.year + 1, 1, 1),
                )
            )
        self.S = ds.so.mean(dim='time')
        ds.close()

    def get_bins(self):
        """
        Get 2D histogram of volume Vb
        as a function of binned salinity (Sb) and temperature (Tb)
        """

        self.Vb = np.zeros((self.Nbasins, self.Nbins, self.Nbins))
        self.Sb = np.zeros((self.Nbasins, self.Nbins + 1))
        self.Tb = np.zeros((self.Nbasins, self.Nbins + 1))

        for b, bmask in enumerate(self.basinmask):
            volume = self.V.values * bmask
            self.Vb[b, :, :], self.Sb[b, :], self.Tb[b, :] = np.histogram2d(
                self.S.values.flatten()[volume.flatten() > 0],
                self.T.values.flatten()[volume.flatten() > 0],
                bins=self.Nbins,
                weights=volume.flatten()[volume.flatten() > 0],
            )
            print(b, sum(sum(self.Vb[b, :, :])))
