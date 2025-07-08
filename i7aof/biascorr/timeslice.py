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

    def get_all_data(self):
        """
        Extract all data from time slice
        """

        self.get_volume()
        self.get_T_and_S()
        self.get_bins()

    def get_volume(self):
        """
        Extract volume per grid cell
        """

        # TODO

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
        Get binned volume, temperature and salinity
        """
        # Create 2D histogram of volume Vb in terms of binned salinity Sb and
        # temperature Tb

        # self.Vb, self.Sb, self.Tb = np.histogram2d(
        #    self.S.flatten()[self.V.flatten() > 0],
        #    self.T.flatten()[self.V.flatten() > 0],
        #    bins=100,
        #    weights=self.V.flatten()[self.V.flatten() > 0]
        # )

        # TODO
