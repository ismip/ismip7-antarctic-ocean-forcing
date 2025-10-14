import netCDF4
import numpy as np
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
