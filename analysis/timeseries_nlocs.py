import matplotlib
import xarray as xr

matplotlib.use('Agg')
import matplotlib.pyplot as plt

time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)

# These are four handpicked locations

# Arrays of indices per location
ii = [190, 245, 380, 665]  # Indices on x-dimension
jj = [334, 460, 265, 260]  # Indices on y-dimension
kk = [10, 10, 10, 10]  # Indices on z-dimension

# Names of the locations
titles = ['Thwaites', 'FRIS', 'Ross', 'Totten']

# Array of variable names
varr = ['thetao', 'tf', 'so']

clim = 'zhou_annual_06_nov'
model = 'MRI-ESM2-0'
ssp = 'ssp585'
ver = 'v3'
clim_years = '1972-2024'


# Prepare figure and subplots
fig, ax = plt.subplots(
    3, 4, figsize=(7, 7), sharey='row', sharex=True, constrained_layout=True
)

# Loop over requested variables
for v, var in enumerate(varr):
    # Open the historical files into 1 dataset
    hist_file_pattern = (
        f'final/AIS/{model}/historical/ocean/{var}/{ver}/'
        f'{var}_AIS_{model}_historical_ocean_{ver}_*.nc'
    )
    ds_hist = xr.open_mfdataset(
        hist_file_pattern,
        decode_times=time_coder,
        data_vars=None,
        compat='override',
        coords='minimal',
    )

    # Open the SSP files into 1 dataset
    ssp_file_pattern = (
        f'final/AIS/{model}/{ssp}/ocean/{var}/{ver}/'
        f'{var}_AIS_{model}_{ssp}_ocean_{ver}_*.nc'
    )
    ds_ssp = xr.open_mfdataset(
        ssp_file_pattern,
        decode_times=time_coder,
        data_vars=None,
        compat='override',
        coords='minimal',
    )

    # Open the climatology file
    clim_filename = (
        f'final/AIS/obs/ocean/climatology/{clim}/{var}/{ver}/'
        f'{var}_AIS_obs_ocean_climatology_{clim}_{ver}_{clim_years}.nc'
    )

    dsc = xr.open_dataset(clim_filename)

    # Loop over requested locations
    for n, (i, j, k, title) in enumerate(zip(ii, jj, kk, titles, strict=True)):
        # Plot the historical period
        ax[v, n].plot(ds_hist.time, ds_hist[var][:, k, j, i], c='tab:blue')

        # Plot the projection period
        ax[v, n].plot(ds_ssp.time, ds_ssp[var][:, k, j, i], c='tab:red')

        # Plot the climatology
        ax[v, n].axhline(dsc[var][k, j, i], 0, 1, c='k')

        # Add some attributes to the panels
        if v == 1:
            ax[v, n].axhline(0, 0, 1, c='k', ls=':')
        if v == 0:
            ax[v, n].set_title(title)
        if n == 0:
            ax[v, n].set_ylabel(var)

        # Comment out to zoom into a specific time period
        # ax[v,n].set_xlim([ds.time[190].values,ds.time[230].values])

plt.savefig('timeseries_nlocs.png', dpi=200)
plt.close()
