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

    Public
    ------
    V: array([nz, ny, nx])
        gridded volume in 3D space
    T: array([nz, ny, nx])
        gridded temperature in 3D space
    S: array([nz, ny, nx])
        gridded salinity in 3D space
    Vb: array([Nbasins, Nbins, Nbins])
        binned volume in S, T space per basin
    Sb: array([Nbasins, Nbins + 1])
        Evenly spaced salinity bins per basin
    Tb: array([Nbasins, Nbins + 1])
        Evenly spaced temperature bins per basin
    deltaSf: array([Nbasins, Nbins, Nbins))
        Filled delta S between this time slice and
        the model reference, in S, T space per basin
    deltaTf: array([Nbasins, Nbins, Nbins))
        Filled delta T between this time slice and
        the model reference, in S, T space per basin
    S_corrected: array([nz, ny, nx])
        Bias-corrected salinity in 3D space
    T_corrected: array([nz, ny, nx])
        Bias-corrected temperature in 3D space
    """

    def __init__(self, config, thetao, so, basinmask, basinNumber, yidx=None):
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
        basinNumber: np.array(len(y), len(x))
            gridded basin numbers
        year: int
            year to select. If None, taking time-mean
            Use None for dataset with single time value
        """

        self.config = config
        self.thetao = thetao
        self.so = so
        self.basinmask = basinmask
        self.basinNumber = basinNumber
        self.yidx = yidx

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
        if self.yidx is None:
            ds = ds.isel(time=0)
        else:
            ds = ds.isel(time=self.yidx)
        self.T = ds.thetao
        ds.close()

        ds = xr.open_mfdataset(self.so, use_cftime=True)
        if self.yidx is None:
            ds = ds.isel(time=0)
        else:
            ds = ds.isel(time=self.yidx)
        self.S = ds.so
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

    def compute_delta(self, modref):
        """
        Compute difference between timeslice and modref,
        binned on the modelref bins

        Parameters
        ----------
        modref: Timeslice
            Timeslice of the same model as this Timeslice
            over the reference period.
            This should usually be the non-extrapolated fields
        """

        # Get the raw, gridded difference in T and S
        self.dTraw = (self.T - modref.T).values
        self.dSraw = (self.S - modref.S).values

        # Inherit bias-corrected bins from modref
        self.Tc = modref.Tc
        self.Sc = modref.Sc

        self.deltaTf = np.zeros((self.Nbasins, self.Nbins, self.Nbins))
        self.deltaSf = np.zeros((self.Nbasins, self.Nbins, self.Nbins))

        for b in range(self.Nbasins):
            bmask = self.basinmask[b, :, :]
            volume = modref.V.values * bmask

            # Compute the binned deltaT and deltaS
            VTdel, Sdel, Tdel = np.histogram2d(
                modref.S.values.flatten()[volume.flatten() > 0],
                modref.T.values.flatten()[volume.flatten() > 0],
                bins=self.Nbins,
                weights=(volume * self.dTraw).flatten()[volume.flatten() > 0],
            )
            deltaT = np.divide(
                VTdel,
                modref.Vb[b, :, :],
                out=np.nan * np.ones((self.Nbins, self.Nbins)),
                where=modref.Vb[b, :, :] > 0,
            )

            VSdel, Sdel, Tdel = np.histogram2d(
                modref.S.values.flatten()[volume.flatten() > 0],
                modref.T.values.flatten()[volume.flatten() > 0],
                bins=self.Nbins,
                weights=(volume * self.dSraw).flatten()[volume.flatten() > 0],
            )
            deltaS = np.divide(
                VSdel,
                modref.Vb[b, :, :],
                out=np.nan * np.ones((self.Nbins, self.Nbins)),
                where=modref.Vb[b, :, :] > 0,
            )

            # Create filled binned deltaT and deltaS through extrapolation
            self.deltaTf[b, :, :] = self.fill_delta(deltaT)
            self.deltaSf[b, :, :] = self.fill_delta(deltaS)

    def fill_delta(self, deltavar):
        """
        Fill deltavar (either deltaT or deltaS)
        in full normalised T,S space

        Parameters
        ----------
        deltavar: array([Nbins, Nbins])
            deltaT or deltaS in S,T space
        """

        newval = np.nan * np.ones((self.Nbins + 1, self.Nbins + 1))
        out = np.nan * np.ones((self.Nbins + 1, self.Nbins + 1))

        newval[1:, 1:] = deltavar.copy()
        Nleft = np.sum(np.isnan(newval[1:, 1:]))
        while Nleft > 0:
            mask = np.where(np.isnan(newval), 0, 1)
            newval2 = np.where(np.isnan(newval), 0, newval)
            AA = newval2 * mask
            AAm1 = np.roll(AA, -1, axis=0)
            m1 = np.roll(mask, -1, axis=0)
            AAp1 = np.roll(AA, 1, axis=0)
            p1 = np.roll(mask, 1, axis=0)

            num = (
                np.roll(AAm1, -1, axis=1)
                + AAm1
                + np.roll(AAm1, 1, axis=1)
                + np.roll(AA, -1, axis=1)
                + np.roll(AA, 1, axis=1)
                + np.roll(AAp1, -1, axis=1)
                + AAp1
                + np.roll(AAp1, 1, axis=1)
            )
            denom = (
                np.roll(m1, -1, axis=1)
                + m1
                + np.roll(m1, 1, axis=1)
                + np.roll(mask, -1, axis=1)
                + np.roll(mask, 1, axis=1)
                + np.roll(p1, -1, axis=1)
                + p1
                + np.roll(p1, 1, axis=1)
            )
            newval3 = np.divide(num, denom, out=out, where=denom > 0)
            newval4 = np.where(mask, newval, newval3)
            newval = newval4
            newval[0, :] = np.nan
            newval[:, 0] = np.nan

            Nleft = np.sum(np.isnan(newval[1:, 1:]))

        return newval[1:, 1:]

    def apply_anomaly(self, base):
        """
        Determine model anomaly with respect to reference period
        and apply anomaly to the base T and S to get the corrected
        values

        Parameters
        ----------
        base: Timeslice
            base timeslice on which to add T- and S-anomalies.
            This should usually be the timeslice of the same model
            over the reference period (same as modref), but with
            the extrapolated fields, so that T_corrected and
            S_corrected are also extrapolated over the full domain
        """

        # Get basin per 3D grid cell
        b = np.repeat(
            self.basinNumber[np.newaxis, :, :] - 1, self.T.shape[0], axis=0
        )

        imax = self.Nbins - 1
        Smin = self.Sc[b, 0]
        Smax = self.Sc[b, -1]

        Tmin = self.Tc[b, 0]
        Tmax = self.Tc[b, -1]

        isref = np.minimum(
            imax,
            np.maximum(
                0, (imax * (base.S - Smin) / (Smax - Smin)).astype(int)
            ),
        )
        jtref = np.minimum(
            imax,
            np.maximum(
                0, (imax * (base.T - Tmin) / (Tmax - Tmin)).astype(int)
            ),
        )

        self.S_corrected = base.S + self.deltaSf[b, isref, jtref]
        self.T_corrected = base.T + self.deltaTf[b, isref, jtref]

        return
