import cftime
import netCDF4
import numpy as np
import xarray as xr
from xarray.coding.times import CFDatetimeCoder

from i7aof.io import ensure_cf_time_encoding, write_netcdf


def _decode_like_open_dataset(da: xr.DataArray) -> xr.DataArray:
    coder = CFDatetimeCoder(use_cftime=True)
    return xr.DataArray(
        coder.decode(da.variable), dims=da.dims, coords=da.coords
    )


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
        # add cftime noleap times covering 2000, 2001, 2002,
        # and yearly bounds [start_of_year, start_of_next_year]
        times = [
            cftime.datetime(2000, 1, 1, calendar='noleap'),
            cftime.datetime(2001, 1, 1, calendar='noleap'),
            cftime.datetime(2002, 1, 1, calendar='noleap'),
        ]

        time_bnds = [
            [
                cftime.datetime(2000, 1, 1, calendar='noleap'),
                cftime.datetime(2001, 1, 1, calendar='noleap'),
            ],
            [
                cftime.datetime(2001, 1, 1, calendar='noleap'),
                cftime.datetime(2002, 1, 1, calendar='noleap'),
            ],
            [
                cftime.datetime(2002, 1, 1, calendar='noleap'),
                cftime.datetime(2003, 1, 1, calendar='noleap'),
            ],
        ]

        decoded_times = _decode_like_open_dataset(
            xr.DataArray(data=times, dims=('time',))
        )

        decoded_time_bnds = _decode_like_open_dataset(
            xr.DataArray(data=time_bnds, dims=('time', 'bnds'))
        )

        ds = ds.assign_coords(time=decoded_times)
        ds['time'].attrs['bounds'] = 'time_bnds'
        # Use encoding to indicate calendar to avoid xarray overwriting attrs
        ds['time'].encoding['calendar'] = 'noleap'
        ds = ds.assign(time_bnds=decoded_time_bnds)
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
        has_fill_values=['vari'],
    )

    with netCDF4.Dataset(out_file, mode='r') as nc:
        # varf should not contain _FillValue
        assert '_FillValue' not in nc.variables['varf'].ncattrs()
        # vari should contain _FillValue even though it has no NaNs
        assert '_FillValue' in nc.variables['vari'].ncattrs()


def test_time_and_time_bnds_units_alignment(tmp_path):
    out_time_units = 'days since 1850-01-01'

    for has_fill_values in [None, ['varf']]:
        ds = _make_ds(with_time=True)

        if has_fill_values is None:
            out_base = 'time_units_default_fill.nc'
        else:
            out_base = 'time_units_varf_fill.nc'

        out_file = tmp_path / out_base
        write_netcdf(
            ds,
            str(out_file),
            format='NETCDF4',
            engine='netcdf4',
            has_fill_values=has_fill_values,
        )

        # Validate: prefer time_bnds carrying the same units (and calendar) as
        # time; otherwise accept no units if the lower bound values numerically
        # equal the time values. Always check numeric alignment.
        with netCDF4.Dataset(out_file, mode='r') as nc:
            time_var = nc.variables['time']
            tb_var = nc.variables['time_bnds']

            assert 'units' in time_var.ncattrs()
            t_units = time_var.getncattr('units')
            assert t_units == out_time_units
            assert 'units' not in tb_var.ncattrs()

            # Compare numeric content: time[:] vs time_bnds[:, 0]
            t_vals = np.array(time_var[:])
            tb_lower = np.array(tb_var[:, 0])
            # Use allclose to allow for float representation
            assert np.allclose(t_vals, tb_lower)


def test_ensure_cf_time_encoding_copies_time():
    ds_src = _make_ds(with_time=True)

    # target with matching time length but different (placeholder) coord
    ds_tgt = xr.Dataset({'a': ('time', np.arange(3))})
    ensure_cf_time_encoding(ds=ds_tgt, time_source=ds_src)
    assert 'time' in ds_tgt.coords
    assert 'time_bnds' in ds_tgt
