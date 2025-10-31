import numpy as np
import xarray as xr
from mpas_tools.config import MpasConfigParser

from i7aof.coords import (
    attach_grid_coords,
    dataset_with_var_and_bounds,
    propagate_time_from,
)
from i7aof.grid.ismip import ensure_ismip_grid


def _make_config_and_grid(tmp_path):
    config = MpasConfigParser()
    # Tiny grid: 2x2 horizontal, single z layer
    dx = dy = (6081 - 1) * 1000
    config.set('ismip_grid', 'dx', str(dx))
    config.set('ismip_grid', 'dy', str(dy))
    config.set('ismip_grid', 'dz', '1800')
    config.set('ismip_grid', 'dz_extrap', '1800')
    config.set('workdir', 'base_dir', str(tmp_path))
    path = ensure_ismip_grid(config)
    return config, xr.open_dataset(path)


def test_attach_grid_coords_injects_lat_lon_bounds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config, ds_grid = _make_config_and_grid(tmp_path)

    # Make a minimal 3D variable on (z, y, x)
    z = ds_grid['z']
    y = ds_grid['y']
    x = ds_grid['x']
    data = xr.DataArray(
        np.zeros((z.sizes['z'], y.sizes['y'], x.sizes['x']), dtype=np.float32),
        dims=('z', 'y', 'x'),
    )
    ds = xr.Dataset({'var': data})

    out = attach_grid_coords(ds, config)

    # lat/lon present with bounds
    assert 'lat' in out.coords
    assert 'lon' in out.coords
    assert 'lat_bnds' in out
    assert 'lon_bnds' in out
    # x/y present with bounds
    assert 'x' in out.coords and 'x_bnds' in out
    assert 'y' in out.coords and 'y_bnds' in out
    # vertical present with bounds
    assert 'z' in out.coords and 'z_bnds' in out


def test_dataset_with_var_and_bounds_selects_bounds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config, ds_grid = _make_config_and_grid(tmp_path)
    # Build a thetao variable defined on (z, y, x) to exercise bounds logic
    z = ds_grid['z']
    y = ds_grid['y']
    x = ds_grid['x']
    thetao = xr.DataArray(
        np.zeros((z.sizes['z'], y.sizes['y'], x.sizes['x']), dtype=np.float32),
        dims=('z', 'y', 'x'),
        name='thetao',
    )
    ds = xr.Dataset({'thetao': thetao})
    ds = attach_grid_coords(ds, config)
    out = dataset_with_var_and_bounds(ds, 'thetao')
    # bounds retained
    for name in ('x_bnds', 'y_bnds', 'lat_bnds', 'lon_bnds', 'z_bnds'):
        assert (name in out) or (name == 'z_bnds' and 'z_extrap_bnds' in out)


def test_propagate_time_from_matches_sizes():
    # source with 3 times and bounds
    t_src = xr.date_range(
        '2000-01-01', periods=3, freq='YS', calendar='noleap', use_cftime=True
    )
    ds_src = xr.Dataset().assign_coords(time=('time', t_src))
    ds_src['time'].attrs['bounds'] = 'time_bnds'
    ds_src['time_bnds'] = xr.DataArray(
        np.empty((3, 2), dtype='datetime64[ns]'), dims=('time', 'bnds')
    )

    # target with matching time length but different (placeholder) coord
    ds_tgt = xr.Dataset({'a': ('time', np.arange(3))})
    out = propagate_time_from(ds_tgt, ds_src)
    assert 'time' in out.coords
    assert 'time_bnds' in out
