import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from i7aof.biascorr.run import Run


# Region class with functions for processing and plotting
class Region:
    def __init__(self, region, model, refmodel='UKESM1-0-LL', k0=0, k1=6000):
        # Get volume CMIP
        self.region = region
        self.model = model  # Reference model considered 'truth'
        self.refmodel = refmodel  # Model to be bias corrected
        self.k0 = k0  # Start depth (top)
        self.k1 = k1  # End depth (bottom)

        # Read historical period of both refmodel and model to get started
        self.get_historical()

    def get_historical(self):
        # Get data for refmodel over historical period
        self.ref = Run(self.refmodel, self.region, 'historical', 1995, 2015,
                       k0=self.k0, k1=self.k1)

        # Get data for model over historical period
        self.prd = Run(self.model, self.region, 'historical', 1995, 2015,
                       k0=self.k0, k1=self.k1)

        # Define the bias in the model with respect to the reference model
        self.get_bias()

    def get_future(self, run, y0, y1):
        # Get data for model over some future period y0 - y1
        self.fut = Run(self.model, self.region, run, y0, y1, k0=self.k0,
                       k1=self.k1)

        # Compute change in T and S between future and historical in T,S space
        # of model
        # Example: between y0 and y1, the water mass that historically had a
        # temperature of X degrees and a salinity of Y psu has warmed by an
        # amount of Z degrees.
        self.get_delta()

        # Construct deltaT and deltaS on the grid of the reference model for
        # this future period
        self.get_anom()

    def plot_region_cmip(self, ii, jj):
        plt.pcolormesh(self.lonc, self.latc, np.nansum(self.thkcello, axis=0),
                       cmap='Blues', norm=mpl.colors.LogNorm(vmin=10,
                                                             vmax=1e4))
        plt.scatter(self.lonc[jj, ii], self.latc[jj, ii], 50, c='tab:red')

    def plot_volumes(self):
        fig, ax = plt.subplots(1, 3, figsize=(3 * 5, 6), sharex=True,
                               sharey=True)

        im = ax[0].pcolormesh(self.ref.Sb, self.ref.Tb, self.prd.Vb.T,
                         norm=mpl.colors.LogNorm(vmin=1e9, vmax=1e13),
                         cmap='Greys')
        plt.colorbar(im, ax=ax[0], orientation='horizontal')
        ax[0].set_title(f'Reference {self.ref.model} '
                        + f'({self.ref.y0}-{self.ref.y1})')
        ax[0].set_ylabel(self.region)

        im = ax[1].pcolormesh(self.prd.Sb, self.prd.Tb, self.prd.Vb.T,
                              norm=mpl.colors.LogNorm(vmin=1e9, vmax=1e13))
        plt.colorbar(im, ax=ax[1], orientation='horizontal')
        ax[1].set_title(f'Present day {self.model} '
                        + f'({self.prd.y0}-{self.prd.y1})')
        # ax[1].plot(self.prd.S[:,jj,ii],self.prd.T[:,jj,ii],c='tab:red')

        im = ax[2].pcolormesh(self.fut.Sb, self.fut.Tb, self.fut.Vb.T,
                              norm=mpl.colors.LogNorm(vmin=1e9, vmax=1e13),
                              cmap='Oranges')
        plt.colorbar(im, ax=ax[2], orientation='horizontal')
        ax[2].set_title(f'{self.fut.run} ({self.fut.y0}-{self.fut.y1})')
        # ax[2].plot(self.fut.S[:,jj,ii],self.fut.T[:,jj,ii],c='tab:red')

    def plot_delta(self):
        fig, ax = plt.subplots(1, 5, figsize=(5 * 3.5, 4.5), sharex=True,
                               sharey=True)

        im = ax[0].pcolormesh(self.ref.Sb, self.ref.Tb, self.ref.Vb.T,
                              norm=mpl.colors.LogNorm(vmin=1e9, vmax=1e13))
        plt.colorbar(im, ax=ax[0], orientation='horizontal')
        ax[0].set_title(f'Reference {self.ref.model} '
                        + f'({self.ref.y0}-{self.ref.y1})')
        ax[0].set_ylabel(self.region)

        im = ax[1].pcolormesh(self.prd.Sb, self.prd.Tb, self.prd.Vb.T,
                              norm=mpl.colors.LogNorm(vmin=1e9, vmax=1e13))
        plt.colorbar(im, ax=ax[1], orientation='horizontal')
        ax[1].set_title(f'present day {self.model} '
                        + f'({self.prd.y0}-{self.prd.y1})')
        # ax[0].plot(self.prd.S[:,jj,ii],self.prd.T[:,jj,ii],c='tab:red')

        im = ax[2].pcolormesh(self.prd.Sb, self.prd.Tb, self.fut.deltaT.T,
                              cmap='cmo.balance', vmin=-3, vmax=3)
        plt.colorbar(im, ax=ax[2], orientation='horizontal')
        ax[2].set_title(f'{self.fut.run} ({self.fut.y0}-{self.fut.y1})')

        im = ax[3].pcolormesh(self.prd.Sb, self.prd.Tb, self.fut.deltaTf.T,
                              cmap='cmo.balance', vmin=-3, vmax=3)

        plt.colorbar(im, ax=ax[3], orientation='horizontal')
        ax[3].set_title('Filled')
        # ax[1].plot(self.fut.S[:,jj,ii],self.fut.T[:,jj,ii],c='tab:red')

        im = ax[4].pcolormesh(self.ref.Sb, self.ref.Tb, self.fut.dTcorb.T,
                              cmap='cmo.balance', vmin=-3, vmax=3)
        plt.colorbar(im, ax=ax[4], orientation='horizontal')
        # ax[3].set_title(f'{self.fut.run} ({self.fut.y0}-{self.fut.y1})')

    def plot_profiles(self, vlim=-1):
        fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharey=True)

        Tav = np.average(np.where(np.isnan(self.ref.T[:vlim, :, :]), 0,
                                  self.ref.T[:vlim, :, :]), axis=(1, 2),
                         weights=self.ref.V[:vlim, :, :])
        ax[1].plot(Tav, self.ref.depth[:vlim], label='optimised', c='tab:red',
                   ls='--')

        Tav = np.average(np.where(np.isnan(self.prd.T[:vlim, :, :]), 0,
                                  self.prd.T[:vlim, :, :]), axis=(1, 2),
                         weights=self.prd.V[:vlim, :, :])
        ax[0].plot(Tav, self.prd.depth[:vlim], label='raw', c='.5', ls='--')

        Tav = np.average(np.where(np.isnan(self.ref.T[:vlim, :, :]), 0,
                                  self.fut.dTcor[:vlim, :, :]), axis=(1, 2),
                         weights=self.ref.V[:vlim, :, :])
        ax[2].plot(Tav, self.ref.depth[:vlim], c='tab:red')

        Tav = np.average(np.where(np.isnan(self.prd.T[:vlim, :, :]), 0,
                                  self.fut.dTraw[:vlim, :, :]), axis=(1, 2),
                         weights=self.prd.V[:vlim, :, :])
        ax[2].plot(Tav, self.prd.depth[:vlim], c='.5')

        Tav = np.average(np.where(np.isnan(self.ref.T[:vlim, :, :]), 0,
                                  (self.ref.T + self.fut.dTcor)[:vlim, :, :]),
                         axis=(1, 2), weights=self.ref.V[:vlim, :, :])
        ax[1].plot(Tav, self.ref.depth[:vlim], c='tab:red')

        Tav = np.average(np.where(np.isnan(self.prd.T[:vlim, :, :]), 0,
                                  self.fut.T[:vlim, :, :]), axis=(1, 2),
                         weights=self.prd.V[:vlim, :, :])
        ax[0].plot(Tav, self.prd.depth[:vlim], c='.5')

        ax[0].set_title('Raw model')
        ax[1].set_title('Optimised')
        ax[2].set_xlabel('Warming T')
        # ax[0].legend()

        for Ax in ax:
            Ax.set_ylim([-1500, 0])
            # Ax.invert_yaxis()

    def get_bias(self, Tperc=.99, Sperc=.99):

        # S-bias constrain 95th percentile
        A, B = np.histogram(
            (self.ref.S).flatten()[self.ref.V.flatten() > 0],
            weights=self.ref.V.flatten()[self.ref.V.flatten() > 0],
            bins=1000, density=True)
        AA = np.cumsum(A) / np.cumsum(A)[-1]
        aa = np.argmin((AA - Sperc) ** 2)
        self.ref.S95 = B[aa]

        A, B = np.histogram(
            (self.prd.S).flatten()[self.prd.V.flatten() > 0],
            weights=self.prd.V.flatten()[self.prd.V.flatten() > 0],
            bins=1000, density=True)
        AA = np.cumsum(A) / np.cumsum(A)[-1]
        aa = np.argmin((AA - Sperc) ** 2)
        self.prd.S95 = B[aa]

        # S-bias
        a = np.arange(.7, 1.3, .01)
        ref, dum = np.histogram(self.ref.S, bins=self.ref.Sb,
                                weights=self.ref.V)
        rmse = np.zeros((len(a)))
        for A, AA in enumerate(a):
            prd, dum = np.histogram(AA * (self.prd.S - self.prd.S95) +
                                    self.ref.S95, bins=self.ref.Sb,
                                    weights=self.prd.V)
            rmse[A] = (np.sum(np.where(prd == 0, 0,
                       (np.log10(prd / np.sum(prd))
                        - np.log10(ref / np.sum(ref))) ** 2)) ** .5
                      / np.sum(np.where(prd == 0, 0, 1)))

            # print(AA,rmse[A])
        out = np.unravel_index(np.nanargmin(rmse, axis=None), rmse.shape)
        self.dSa = a[out[0]]
        self.prd.Sc = self.dSa * (self.prd.Sb - self.prd.S95) + self.ref.S95

        # T-bias (scale 95th percentile of T-Tmin)
        A, B = np.histogram(
            (self.ref.T - self.ref.Tb[0]).flatten()[self.ref.V.flatten() > 0],
            weights=self.ref.V.flatten()[self.ref.V.flatten() > 0],
            bins=1000, density=True)
        AA = np.cumsum(A) / np.cumsum(A)[-1]
        aa = np.argmin((AA - Tperc) ** 2)
        self.ref.TF95 = B[aa]

        A, B = np.histogram(
            (self.prd.T - self.prd.Tb[0]).flatten()[self.prd.V.flatten() > 0],
            weights=self.prd.V.flatten()[self.prd.V.flatten() > 0],
            bins=1000, density=True)
        AA = np.cumsum(A) / np.cumsum(A)[-1]
        aa = np.argmin((AA - Tperc) ** 2)
        self.prd.TF95 = B[aa]
        self.dTa = self.ref.TF95 / self.prd.TF95
        self.prd.Tc = (
            self.dTa * (self.prd.Tb - self.prd.Tb[0]) + self.ref.Tb[0])
        # self.fut.Tc = self.dTa*(self.fut.Tb-self.fut.Tb[0])+self.ref.Tb[0]

    def get_delta(self):

        # Get difference in T and S between future and historical period
        self.fut.dTraw = self.fut.T - self.prd.T
        self.fut.dSraw = self.fut.S - self.prd.S

        # Compute binned deltaT and deltaS on historical S- and T- bins
        VTdel, Sdel, Tdel = np.histogram2d(
            self.prd.S.flatten()[self.prd.V.flatten() > 0],
            self.prd.T.flatten()[self.prd.V.flatten() > 0],
            bins=100,
            weights=(self.prd.V * self.fut.dTraw).flatten()
                [self.prd.V.flatten() > 0])
        self.fut.deltaT = VTdel / self.prd.Vb
        VSdel, Sdel, Tdel = np.histogram2d(
            self.prd.S.flatten()[self.prd.V.flatten() > 0],
            self.prd.T.flatten()[self.prd.V.flatten() > 0],
            bins=100,
            weights=(self.prd.V * self.fut.dSraw).flatten()
                [self.prd.V.flatten() > 0])
        self.fut.deltaS = VSdel / self.prd.Vb

        # Create filled binned deltaT and deltaS through extrapolation
        self.fut.deltaTf = self.fill_delta(self.fut.deltaT)
        self.fut.deltaSf = self.fill_delta(self.fut.deltaS)

    def fill_delta(self, deltavar):
        """Routine to fill deltaT and deltaS in full normalised T,S space"""

        newval = np.nan * np.ones((len(self.prd.Sc), len(self.prd.Tc)))

        newval[1:, 1:] = deltavar
        Nleft = np.sum(np.isnan(newval[1:, 1:]))
        while Nleft > 0:
            mask = np.where(np.isnan(newval), 0, 1)
            newval2 = np.where(np.isnan(newval), 0, newval)
            AA = newval2 * mask
            AAm1 = np.roll(AA, -1, axis=0)
            m1 = np.roll(mask, -1, axis=0)
            AAp1 = np.roll(AA, 1, axis=0)
            p1 = np.roll(mask, 1, axis=0)
            newval3 = (
                (np.roll(AAm1, -1, axis=1) + AAm1 + np.roll(AAm1, 1, axis=1) +
                 np.roll(AA, -1, axis=1) + np.roll(AA, 1, axis=1) +
                 np.roll(AAp1, -1, axis=1) + AAp1 + np.roll(AAp1, 1, axis=1)) /
                (np.roll(m1, -1, axis=1) + m1 + np.roll(m1, 1, axis=1) +
                 np.roll(mask, -1, axis=1) + np.roll(mask, 1, axis=1) +
                 np.roll(p1, -1, axis=1) + p1 + np.roll(p1, 1, axis=1)))
            newval4 = np.where(mask, newval, newval3)
            newval = newval4
            newval[0, :] = np.nan
            newval[:, 0] = np.nan

            Nleft = np.sum(np.isnan(newval[1:, 1:]))

        return newval[1:, 1:]

    def get_anom(self):
        """Get anomalies of reference T,S values"""

        self.fut.dTcor = np.zeros(self.ref.T.shape)
        self.fut.dScor = np.zeros(self.ref.S.shape)

        for i in range(self.fut.dTcor.shape[0]):
            for j in range(self.fut.dTcor.shape[1]):
                for k in range(self.fut.dTcor.shape[2]):
                    if self.ref.V[i, j, k] == 0:
                        continue
                    else:
                        isref = min(99, max(0, int(
                            99 * (self.ref.S[i, j, k] - self.prd.Sc[0]) /
                            (self.prd.Sc[-1] - self.prd.Sc[0]))))
                        jtref = min(99, max(0, int(
                            99 * (self.ref.T[i, j, k] - self.prd.Tc[0]) /
                            (self.prd.Tc[-1] - self.prd.Tc[0]))))
                        self.fut.dScor[i, j, k] = \
                            self.fut.deltaSf[isref, jtref]
                        self.fut.dTcor[i, j, k] = \
                            self.fut.deltaTf[isref, jtref]

        # Distributions of dS and dT, only for plotting purposes
        VTdel, Sdel, Tdel = np.histogram2d(
            self.ref.S.flatten()[self.ref.V.flatten() > 0],
            self.ref.T.flatten()[self.ref.V.flatten() > 0],
            bins=100,
            range=[[self.ref.Sb[0], self.ref.Sb[-1]],
                   [self.ref.Tb[0], self.ref.Tb[-1]]],
            weights=(self.ref.V * self.fut.dTcor).flatten()
                [self.ref.V.flatten() > 0])
        self.fut.dTcorb = VTdel / self.ref.Vb
        VSdel, Sdel, Tdel = np.histogram2d(
            self.ref.S.flatten()[self.ref.V.flatten() > 0],
            self.ref.S.flatten()[self.ref.V.flatten() > 0],
            bins=100,
            range=[[self.ref.Sb[0], self.ref.Sb[-1]],
                   [self.ref.Tb[0], self.ref.Tb[-1]]],
            weights=(self.ref.V * self.fut.dScor).flatten()
                [self.ref.V.flatten() > 0])
        self.fut.dScorb = VSdel / self.ref.Vb

    def get_profiles(self):
        # Construct vertical profiles of reference T and S

        self.ref.Tz = np.nan * np.ones(len(self.ref.depth))
        self.ref.Sz = np.nan * np.ones(len(self.ref.depth))

        kmax = np.where(np.sum(self.ref.V, axis=(1, 2)) == 0)[0][0]
        self.ref.Tz[:kmax] = np.average(
            np.where(
                np.isnan(self.ref.T[:kmax, :, :]),
                0,
                self.ref.T[:kmax, :, :]),
            axis=(1, 2),
            weights=self.ref.V[:kmax, :, :])
        self.ref.Sz[:kmax] = np.average(
            np.where(
                np.isnan(self.ref.S[:kmax, :, :]),
                0,
                self.ref.S[:kmax, :, :]),
            axis=(1, 2),
            weights=self.ref.V[:kmax, :, :])

    def calc_horav(self, var, run):
        """Calculate horizontally averaged vertical profile"""

        try:
            out = np.average(np.where(np.isnan(var), 0, var), axis=(1, 2),
                             weights=run.V)
        except ZeroDivisionError:
            out = np.nan * np.ones(var.shape[0])
            kmax = np.where(np.sum(run.V, axis=(1, 2)) == 0)[0][0]
            out[:kmax] = np.average(np.where(np.isnan(var[:kmax, :, :]), 0,
                                             var[:kmax, :, :]), axis=(1, 2),
                                    weights=run.V[:kmax, :, :])

        return out
