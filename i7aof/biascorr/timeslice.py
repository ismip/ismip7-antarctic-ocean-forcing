import numpy as np
import xarray as xr


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

    def __init__(self, config, thetao, so, y0=None, y1=None):
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
        y0: int
            start year
        y1: int
            end year
        """

        self.config = config
        self.thetao = thetao
        self.so = so
        self.y0 = y0
        self.y1 = y1

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
        ds = xr.open_dataset(self.thetao)
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

        ds = xr.open_dataset(self.thetao)
        if not (self.y0 is None and self.y1 is None):
            ds = ds.sel(time=slice(self.y0, self.y1))
        self.T = ds.thetao.mean(dim='time')
        ds.close()

        ds = xr.open_dataset(self.so)
        if not (self.y0 is None and self.y1 is None):
            ds = ds.sel(time=slice(self.y0, self.y1))
        self.S = ds.so.mean(dim='time')
        ds.close()

    def get_bins(self):
        """
        Get 2D histogram of volume Vb
        as a function of binned salinity (Sb) and temperature (Tb)
        """

        self.Vb, self.Sb, self.Tb = np.histogram2d(
            self.S.values.flatten()[self.V.values.flatten() > 0],
            self.T.values.flatten()[self.V.values.flatten() > 0],
            bins=self.Nbins,
            weights=self.V.values.flatten()[self.V.values.flatten() > 0],
        )

        print(self.Vb.shape)
