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

        # TODO apply per basin

        self.Vb, self.Sb, self.Tb = np.histogram2d(
            self.S.values.flatten()[self.V.values.flatten() > 0],
            self.T.values.flatten()[self.V.values.flatten() > 0],
            bins=self.Nbins,
            weights=self.V.values.flatten()[self.V.values.flatten() > 0],
        )
