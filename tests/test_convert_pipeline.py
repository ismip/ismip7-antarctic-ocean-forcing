import os

import cftime
import netCDF4
import numpy as np
import xarray as xr

from i7aof.io import ensure_cf_time_encoding, read_dataset
from i7aof.io_zarr import append_to_zarr, finalize_zarr_to_netcdf


def _make_mock_cmip_ds(n_time: int = 6) -> xr.Dataset:
    """Create a small CMIP-like dataset mirroring key metadata of CESM2-WACCM.

    Dimensions (small for speed):
      - time = n_time (UNLIMITED later)
      - lev = 3
      - nlat = 4
      - nlon = 3
      - d2 = 2 (bounds)
      - vertices = 4 (lat/lon cell corners)
    """
    nlat, nlon, lev = 4, 3, 3

    # Synthetic coordinates
    lat = np.linspace(-80, -20, nlat).reshape(nlat, 1) * np.ones((1, nlon))
    lon = np.linspace(0, 40, nlon).reshape(1, nlon) * np.ones((nlat, 1))
    lev_vals = np.array([5.0, 50.0, 200.0])  # cm

    # Bounds (simple placeholders)
    lat_bnds = np.stack([lat - 0.5, lat + 0.5, lat + 0.5, lat - 0.5], axis=2)
    lon_bnds = np.stack([lon - 0.5, lon + 0.5, lon + 0.5, lon - 0.5], axis=2)
    lev_bnds = np.vstack([lev_vals - 5.0, lev_vals + 5.0]).T  # meters later

    # Time values as numeric days since 0001-01-01, calendar 365_day
    # Months starting Jan 1850
    time_dates = [
        cftime.DatetimeNoLeap(1850, m, 1) for m in range(1, n_time + 1)
    ]
    time_bnds_dates = [
        (
            cftime.DatetimeNoLeap(1850, m, 1),
            cftime.DatetimeNoLeap(1850, m + 1 if m < 12 else 12, 1)
            if m < 12
            else cftime.DatetimeNoLeap(1851, 1, 1),
        )
        for m in range(1, n_time + 1)
    ]

    units_0001 = 'days since 0001-01-01 00:00:00'
    calendar_time = '365_day'
    calendar_bnds = 'noleap'  # matches the real file header
    t_vals = cftime.date2num(time_dates, units_0001, calendar=calendar_time)
    tb_vals = np.array(
        [
            cftime.date2num(list(pair), units_0001, calendar=calendar_bnds)
            for pair in time_bnds_dates
        ]
    )

    # Data variable
    thetao = np.zeros((n_time, lev, nlat, nlon), dtype=np.float32)

    ds = xr.Dataset(
        {
            'thetao': (('time', 'lev', 'nlat', 'nlon'), thetao),
            'lat': (('nlat', 'nlon'), lat.astype(np.float64)),
            'lon': (('nlat', 'nlon'), lon.astype(np.float64)),
            'lev': (('lev',), lev_vals.astype(np.float64)),
            'time': (('time',), t_vals.astype(np.float64)),
            'time_bnds': (('time', 'd2'), tb_vals.astype(np.float64)),
            'lat_bnds': (
                ('nlat', 'nlon', 'vertices'),
                lat_bnds.astype(np.float32),
            ),
            'lon_bnds': (
                ('nlat', 'nlon', 'vertices'),
                lon_bnds.astype(np.float32),
            ),
            'lev_bnds': (('lev', 'd2'), lev_bnds.astype(np.float32)),
        }
    )

    # Attributes mirroring the real file
    ds['thetao'].attrs.update(
        {
            'units': 'degC',
            'long_name': 'Sea Water Potential Temperature',
            'coordinates': 'time lev lat lon',
            'missing_value': 1.0e20,
        }
    )
    # Ensure Zarr encoding doesn't conflict: set _FillValue only in encoding
    ds['thetao'].encoding['_FillValue'] = np.float32(1.0e20)
    ds['thetao'].attrs.pop('_FillValue', None)
    ds['lat'].attrs.update(
        {'axis': 'Y', 'bounds': 'lat_bnds', 'units': 'degrees_north'}
    )
    ds['lon'].attrs.update(
        {'axis': 'X', 'bounds': 'lon_bnds', 'units': 'degrees_east'}
    )
    ds['lev'].attrs.update(
        {
            'axis': 'Z',
            'bounds': 'lev_bnds',
            'units': 'centimeters',
            'positive': 'down',
        }
    )
    ds['lat_bnds'].attrs['units'] = 'degrees_north'
    ds['lon_bnds'].attrs['units'] = 'degrees_east'
    ds['lev_bnds'].attrs['units'] = 'm'

    ds['time'].attrs.update(
        {
            'axis': 'T',
            'bounds': 'time_bnds',
            'standard_name': 'time',
            'title': 'time',
            'type': 'double',
            'units': units_0001,
            'calendar': calendar_time,
        }
    )
    ds['time_bnds'].attrs.update(
        {
            'units': units_0001,
            'calendar': calendar_bnds,
        }
    )
    # Keep numeric time values; attrs carry units/calendar like the real file
    ds['time'].encoding.clear()

    return ds


def _write_mock_input_nc(ds: xr.Dataset, path: str) -> None:
    # Write as NetCDF4 preserving time encodings
    ds.to_netcdf(path, format='NETCDF4', engine='netcdf4')


def test_opening_mock_cmip_has_cf_time(tmp_path) -> None:
    """Opening the CMIP-like file should yield cftime time/time_bnds."""
    in_nc = os.path.join(tmp_path, 'thetao_mock.nc')
    _write_mock_input_nc(_make_mock_cmip_ds(n_time=6), in_nc)

    ds = read_dataset(in_nc)
    assert 'time' in ds and 'time_bnds' in ds
    assert ds['time'].dtype == object and isinstance(
        ds['time'].values[0], cftime.datetime
    )
    assert ds['time_bnds'].dtype == object and isinstance(
        ds['time_bnds'].values[0, 0], cftime.datetime
    )


def test_zarr_to_netcdf_preserves_time_and_units(tmp_path) -> None:
    """Chunk through Zarr and finalize to NetCDF with days-based time units."""
    base = os.path.join(tmp_path, 'cmip_mock')
    in_nc = base + '_in.nc'
    out_nc = base + '_out.nc'
    zarr_store = base + '.zarr'
    ds_in = _make_mock_cmip_ds(n_time=6)
    _write_mock_input_nc(ds_in, in_nc)

    # Open using package defaults (cftime decoding)
    ds = read_dataset(in_nc)

    # Append chunks (2-month chunks) to Zarr store
    first = True
    for t0 in range(0, ds.sizes['time'], 2):
        t1 = min(t0 + 2, ds.sizes['time'])
        chunk = ds.isel(time=slice(t0, t1))
        # No-op transform on data; preserve coords
        first = append_to_zarr(
            ds=chunk,
            zarr_store=zarr_store,
            first=first,
            append_dim='time',
        )

    # Postprocess to ensure CF-consistent time encoding
    def _post(d: xr.Dataset) -> xr.Dataset:
        ensure_cf_time_encoding(
            ds=d,
            time_source=ds,
        )
        return d

    finalize_zarr_to_netcdf(
        zarr_store=zarr_store,
        out_nc=out_nc,
        has_fill_values=lambda *_: False,
        progress_bar=False,
        postprocess=_post,
    )

    # Validate NetCDF output
    with netCDF4.Dataset(out_nc) as nc:
        # time should be unlimited
        assert nc.dimensions['time'].isunlimited()
        tvar = nc.variables['time']
        tbvar = nc.variables['time_bnds']
        # No microseconds allowed
        assert 'units' in tvar.ncattrs()
        tunits = tvar.getncattr('units')
        assert tunits.startswith('days since 1850-01-01')
        assert 'microsecond' not in tunits
        # time_bnds should match time units
        if 'units' in tbvar.ncattrs():
            assert tbvar.getncattr('units') == tunits
        # Numeric alignment lower bound == time
        t_vals = np.array(tvar[:])
        tb0 = np.array(tbvar[:, 0])
        assert np.allclose(t_vals, tb0)
        # Expected coords and bounds present
        for name in ('lat', 'lon', 'lev'):
            assert name in nc.variables
        for bname in ('lat_bnds', 'lon_bnds', 'lev_bnds'):
            assert bname in nc.variables
