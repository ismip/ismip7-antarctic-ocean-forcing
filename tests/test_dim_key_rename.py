import numpy as np
import pytest
import xarray as xr
from mpas_tools.config import MpasConfigParser

from i7aof.io import write_netcdf
from i7aof.remap.clim import _preprocess_climatology_input
from i7aof.remap.shared import _remap_horiz


def _make_climatology_config(*, use_new_dims: bool) -> MpasConfigParser:
    config = MpasConfigParser()
    config.set('climatology', 'lat_var', 'latitude')
    config.set('climatology', 'lon_var', 'longitude')
    config.set('climatology', 'lev_var', 'pressure')
    if use_new_dims:
        config.set('climatology', 'y_dim', 'ny')
        config.set('climatology', 'x_dim', 'nx')
    else:
        config.set('climatology', 'lat_dim', 'ny')
        config.set('climatology', 'lon_dim', 'nx')
    config.set('climatology', 'lev_dim', 'nz')
    config.set('climatology', 'ct_var', 'ct')
    config.set('climatology', 'sa_var', 'sa')
    config.set('climatology', 'ct_mse_var', 'ct_mse')
    config.set('climatology', 'sa_mse_var', 'sa_mse')
    config.set('climatology', 'mse_threshold', '1e9')
    return config


def test_preprocess_climatology_input_uses_y_dim_and_x_dim(tmp_path):
    config = _make_climatology_config(use_new_dims=True)
    ds = xr.Dataset(
        data_vars={
            'ct': (('nz', 'ny', 'nx'), np.ones((2, 3, 4), dtype=np.float32)),
            'sa': (('nz', 'ny', 'nx'), np.ones((2, 3, 4), dtype=np.float32)),
            'ct_mse': (
                ('nz', 'ny', 'nx'),
                np.zeros((2, 3, 4), dtype=np.float32),
            ),
            'sa_mse': (
                ('nz', 'ny', 'nx'),
                np.zeros((2, 3, 4), dtype=np.float32),
            ),
            'latitude': ('ny', np.linspace(-80.0, -78.0, 3)),
            'longitude': ('nx', np.linspace(0.0, 30.0, 4)),
            'pressure': ('nz', np.array([10.0, 20.0])),
        }
    )
    ds['pressure'].attrs['units'] = 'dbar'
    in_filename = tmp_path / 'clim_input.nc'
    write_netcdf(ds, in_filename)

    out_filename = _preprocess_climatology_input(
        config, str(in_filename), str(tmp_path)
    )

    out = xr.open_dataset(out_filename)
    try:
        assert out['ct'].dims == ('lev', 'lat', 'lon')
        assert out['sa'].dims == ('lev', 'lat', 'lon')
        assert 'lat' in out.coords
        assert 'lon' in out.coords
        assert 'lev_bnds' in out
    finally:
        out.close()


def test_preprocess_climatology_input_rejects_old_dim_keys(tmp_path):
    config = _make_climatology_config(use_new_dims=False)
    ds = xr.Dataset(
        data_vars={
            'ct': (('nz', 'ny', 'nx'), np.ones((1, 1, 1), dtype=np.float32)),
            'sa': (('nz', 'ny', 'nx'), np.ones((1, 1, 1), dtype=np.float32)),
            'ct_mse': (('nz', 'ny', 'nx'), np.zeros((1, 1, 1))),
            'sa_mse': (('nz', 'ny', 'nx'), np.zeros((1, 1, 1))),
            'latitude': ('ny', np.array([-80.0])),
            'longitude': ('nx', np.array([0.0])),
            'pressure': ('nz', np.array([10.0])),
        }
    )
    ds['pressure'].attrs['units'] = 'dbar'
    in_filename = tmp_path / 'clim_input_old_keys.nc'
    write_netcdf(ds, in_filename)

    with pytest.raises(ValueError, match=r'\[climatology\] y_dim'):
        _preprocess_climatology_input(config, str(in_filename), str(tmp_path))


def test_remap_horiz_uses_cmip_x_dim(monkeypatch):
    config = MpasConfigParser()
    config.set('remap', 'method', 'bilinear')
    config.set('remap', 'threshold', '1e-3')
    config.set('cmip_dataset', 'lat_var', 'lat')
    config.set('cmip_dataset', 'lon_var', 'lon')
    config.set('cmip_dataset', 'y_dim', 'y')
    config.set('cmip_dataset', 'x_dim', 'x')

    called = {}
    ds = xr.Dataset(
        {'ct': (('time', 'y', 'x'), np.zeros((1, 2, 3), dtype=np.float32))},
        coords={'lon': ('x', np.array([0.0, 1.0, 2.0]))},
    )

    def fake_read_dataset(*args, **kwargs):
        return ds

    def fake_add_periodic_lon(dataset, lon_var, periodic_dim):
        called['lon_var'] = lon_var
        called['periodic_dim'] = periodic_dim
        return dataset

    monkeypatch.setattr('i7aof.remap.shared.read_dataset', fake_read_dataset)
    monkeypatch.setattr(
        'i7aof.remap.shared.add_periodic_lon', fake_add_periodic_lon
    )
    monkeypatch.setattr(
        'i7aof.remap.shared._build_and_remap_mask',
        lambda **kwargs: xr.Dataset(),
    )
    monkeypatch.setattr(
        'i7aof.remap.shared._remap_data_variables',
        lambda **kwargs: [xr.Dataset({'ct': kwargs['ds']['ct']})],
    )
    monkeypatch.setattr(
        'i7aof.remap.shared._validate_z_extrap', lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        'i7aof.remap.shared._concat_chunks', lambda chunks: chunks[0]
    )
    monkeypatch.setattr(
        'i7aof.remap.shared._finalize_and_write',
        lambda **kwargs: None,
    )

    _remap_horiz(
        config=config,
        in_filename='ignored.nc',
        out_filename='ignored_out.nc',
        model_prefix='model',
        tmpdir='.',
        logger=None,
        has_fill_values=['ct'],
    )

    assert called == {'lon_var': 'lon', 'periodic_dim': 'x'}


def test_remap_horiz_rejects_old_dim_keys(monkeypatch):
    config = MpasConfigParser()
    config.set('remap', 'method', 'bilinear')
    config.set('remap', 'threshold', '1e-3')
    config.set('cmip_dataset', 'lon_var', 'lon')
    config.set('cmip_dataset', 'lat_dim', 'y')
    config.set('cmip_dataset', 'lon_dim', 'x')

    monkeypatch.setattr(
        'i7aof.remap.shared.read_dataset',
        lambda *args, **kwargs: xr.Dataset(),
    )

    with pytest.raises(ValueError, match=r'\[cmip_dataset\] y_dim'):
        _remap_horiz(
            config=config,
            in_filename='ignored.nc',
            out_filename='ignored_out.nc',
            model_prefix='model',
            tmpdir='.',
            logger=None,
            has_fill_values=['ct'],
        )
