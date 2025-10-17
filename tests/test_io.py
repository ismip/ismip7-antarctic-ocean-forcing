import cftime
import netCDF4
import numpy as np
import pytest
import xarray as xr

from i7aof.io import write_netcdf


def _make_ds(with_time: bool = True):
    data = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    ds = xr.Dataset(
        {
            'varf': ('x', data.astype(np.float32)),
            'vari': ('x', np.array([1, 2, 3], dtype=np.int32)),
        },
        coords={'x': ('x', np.arange(3, dtype=np.int32))},
    )
    if with_time:
        ds = ds.expand_dims({'time': 1})
        ds['time'] = xr.DataArray(
            np.array([0], dtype=np.int32), dims=('time',)
        )
    return ds


def test_write_netcdf_tmpdir(tmp_path):
    # Create simple dataset with a NaN in varf (float) and no NaN in vari (int)
    ds = _make_ds(with_time=True)

    out_file = tmp_path / 'test_write.nc'

    # Write using default settings
    write_netcdf(ds, str(out_file), format='NETCDF4', engine='netcdf4')

    # Inspect with netCDF4 to ensure _FillValue is present for float var only
    with netCDF4.Dataset(out_file, mode='r') as nc:
        # time should be unlimited
        dim = nc.dimensions['time']
        assert dim.isunlimited(), 'time dimension should be unlimited'

        # varf should have a _FillValue because it contains NaN
        varf = nc.variables['varf']
        assert '_FillValue' in varf.ncattrs()

        # vari is int with no NaNs; default behavior should avoid _FillValue
        vari = nc.variables['vari']
        assert '_FillValue' not in vari.ncattrs()


def test_write_netcdf_has_fill_values_override(tmp_path):
    ds = _make_ds(with_time=False)
    out_file = tmp_path / 'override.nc'

    # Force _FillValue on both variables
    write_netcdf(
        ds,
        str(out_file),
        format='NETCDF4',
        engine='netcdf4',
        has_fill_values=True,
    )

    with netCDF4.Dataset(out_file, mode='r') as nc:
        varf = nc.variables['varf']
        vari = nc.variables['vari']
        assert '_FillValue' in varf.ncattrs()
        assert '_FillValue' in vari.ncattrs()


def test_write_netcdf_disable_fill_values(tmp_path):
    ds = _make_ds(with_time=False)
    out_file = tmp_path / 'no_fill.nc'

    # Disable _FillValue on all variables
    write_netcdf(
        ds,
        str(out_file),
        format='NETCDF4',
        engine='netcdf4',
        has_fill_values=False,
    )

    with netCDF4.Dataset(out_file, mode='r') as nc:
        for name in ('varf', 'vari'):
            assert '_FillValue' not in nc.variables[name].ncattrs()


def test_write_netcdf_has_fill_values_dict(tmp_path):
    ds = _make_ds(with_time=False)
    out_file = tmp_path / 'dict_fill.nc'

    # Set per-variable choices: force only the integer variable
    # to have a fill value
    write_netcdf(
        ds,
        str(out_file),
        format='NETCDF4',
        engine='netcdf4',
        has_fill_values={'varf': False, 'vari': True},
    )

    with netCDF4.Dataset(out_file, mode='r') as nc:
        # varf should not contain _FillValue
        assert '_FillValue' not in nc.variables['varf'].ncattrs()
        # vari should contain _FillValue even though it has no NaNs
        assert '_FillValue' in nc.variables['vari'].ncattrs()


def test_time_and_time_bnds_units_alignment(tmp_path):
    # Build a dataset with cftime noleap times covering 2000, 2001, 2002,
    # and yearly bounds [start_of_year, start_of_next_year]
    times = [
        cftime.DatetimeNoLeap(2000, 1, 1),
        cftime.DatetimeNoLeap(2001, 1, 1),
        cftime.DatetimeNoLeap(2002, 1, 1),
    ]

    time_bnds = [
        [cftime.DatetimeNoLeap(2000, 1, 1), cftime.DatetimeNoLeap(2001, 1, 1)],
        [cftime.DatetimeNoLeap(2001, 1, 1), cftime.DatetimeNoLeap(2002, 1, 1)],
        [cftime.DatetimeNoLeap(2002, 1, 1), cftime.DatetimeNoLeap(2003, 1, 1)],
    ]

    ds = xr.Dataset()
    ds = ds.assign_coords(time=xr.DataArray(times, dims=('time',)))
    ds['time'].attrs['bounds'] = 'time_bnds'
    # Use encoding to indicate calendar to avoid xarray overwriting attrs
    ds['time'].encoding['calendar'] = 'noleap'
    ds = ds.assign({'time_bnds': xr.DataArray(time_bnds, dims=('time', 'nv'))})

    out_time_units = 'days since 1850-01-01'

    out_file = tmp_path / 'time_units.nc'
    write_netcdf(ds, str(out_file), format='NETCDF4', engine='netcdf4')

    # Validate: prefer time_bnds carrying the same units (and calendar) as
    # time; otherwise accept no units if the lower bound values numerically
    # equal the time values. Always check numeric alignment.
    with netCDF4.Dataset(out_file, mode='r') as nc:
        time_var = nc.variables['time']
        tb_var = nc.variables['time_bnds']

        assert 'units' in time_var.ncattrs()
        t_units = time_var.getncattr('units')
        assert t_units == out_time_units
        tb_has_units = 'units' in tb_var.ncattrs()
        if tb_has_units:
            assert tb_var.getncattr('units') == t_units
            # If calendar is present on time, require it on time_bnds as well
            if 'calendar' in time_var.ncattrs():
                t_cal = time_var.getncattr('calendar')
                assert 'calendar' in tb_var.ncattrs()
                assert tb_var.getncattr('calendar') == t_cal

        # Compare numeric content: time[:] vs time_bnds[:, 0]
        t_vals = np.array(time_var[:])
        tb_lower = np.array(tb_var[:, 0])
        # Use allclose to allow for float representation
        assert np.allclose(t_vals, tb_lower)


def test_no_microseconds_units_when_numeric_input(tmp_path):
    """Numeric time arrays with 'microseconds since ...' units should error
    out during encoding rather than silently converting to a different unit.
    """
    # Construct integer microseconds since 1850-01-01 for 2000, 2001
    base = cftime.DatetimeNoLeap(1850, 1, 1)
    t0 = cftime.DatetimeNoLeap(2000, 1, 1)
    t1 = cftime.DatetimeNoLeap(2001, 1, 1)

    # deltas in days then to microseconds
    def days(a, b):
        # approximate delta in days using cftime date2num with days units
        return cftime.date2num(
            a, 'days since 0001-01-01', calendar='noleap'
        ) - cftime.date2num(b, 'days since 0001-01-01', calendar='noleap')

    us_per_day = 24 * 60 * 60 * 1_000_000
    t_vals = np.array(
        [
            int(days(t0, base) * us_per_day),
            int(days(t1, base) * us_per_day),
        ],
        dtype=np.int64,
    )

    tb_vals = np.array(
        [
            [
                int(days(t0, base) * us_per_day),
                int(days(t1, base) * us_per_day),
            ],
            [
                int(days(t1, base) * us_per_day),
                int(
                    days(cftime.DatetimeNoLeap(2002, 1, 1), base) * us_per_day
                ),
            ],
        ],
        dtype=np.int64,
    )

    ds = xr.Dataset()
    ds = ds.assign_coords(time=xr.DataArray(t_vals, dims=('time',)))
    ds['time'].attrs['bounds'] = 'time_bnds'
    ds['time'].attrs['units'] = 'microseconds since 1850-01-01 00:00:00'
    ds['time'].attrs['calendar'] = 'noleap'
    ds['time_bnds'] = xr.DataArray(tb_vals, dims=('time', 'bnds'))
    ds['time_bnds'].attrs['units'] = 'microseconds since 1850-01-01 00:00:00'
    ds['time_bnds'].attrs['calendar'] = 'noleap'

    out_file = tmp_path / 'no_microseconds.nc'
    with pytest.raises(ValueError):
        write_netcdf(ds, str(out_file), format='NETCDF4', engine='netcdf4')
