import numpy as np
import pytest
import xarray as xr
from mpas_tools.config import MpasConfigParser

from i7aof.remap import drop_duplicate_lon
from i7aof.remap.shared import _remap_horiz


def _lon_dataset(lon, units='degrees_east'):
    """A small dataset with a 1D longitude coordinate and a data var."""
    data = np.arange(lon.size, dtype=np.float32)
    ds = xr.Dataset(
        {'field': ('lon', data)},
        coords={'lon': ('lon', lon)},
    )
    ds['lon'].attrs['units'] = units
    return ds


def test_drop_duplicate_lon_removes_seam_column():
    # -180..180 inclusive: the -180 and +180 columns are duplicates
    lon = np.linspace(-180.0, 180.0, 5)  # -180, -90, 0, 90, 180
    ds = _lon_dataset(lon)
    out = drop_duplicate_lon(ds, lon_var='lon')
    assert out.sizes['lon'] == 4
    np.testing.assert_array_equal(out['lon'].values, lon[:-1])
    # data var is trimmed consistently
    np.testing.assert_array_equal(out['field'].values, np.arange(4))


def test_drop_duplicate_lon_keeps_non_duplicate():
    # spans less than a full circle: nothing should be dropped
    lon = np.linspace(-180.0, 90.0, 4)
    ds = _lon_dataset(lon)
    out = drop_duplicate_lon(ds, lon_var='lon')
    assert out.sizes['lon'] == 4
    np.testing.assert_array_equal(out['lon'].values, lon)


def test_drop_duplicate_lon_handles_radians():
    lon = np.linspace(-np.pi, np.pi, 5)
    ds = _lon_dataset(lon, units='radians')
    out = drop_duplicate_lon(ds, lon_var='lon')
    assert out.sizes['lon'] == 4


def test_drop_duplicate_lon_rejects_2d():
    lon2d = np.array([[0.0, 1.0], [0.0, 1.0]])
    ds = xr.Dataset(coords={'lon': (('y', 'x'), lon2d)})
    with pytest.raises(ValueError, match='1 dimension'):
        drop_duplicate_lon(ds, lon_var='lon')


def test_remap_horiz_1d_drops_duplicate_and_forces_global(monkeypatch):
    """A 1D global climatology grid is conditioned and forced global."""
    config = MpasConfigParser()
    config.set('remap', 'method', 'bilinear')
    config.set('remap', 'threshold', '1e-3')

    lon = np.linspace(-180.0, 180.0, 5)
    ds = xr.Dataset(
        {'ct': (('time', 'lat', 'lon'), np.zeros((1, 2, 5), np.float32))},
        coords={'lon': ('lon', lon), 'lat': ('lat', np.array([-80.0, -70.0]))},
    )
    ds['lon'].attrs['units'] = 'degrees_east'

    captured = {}

    monkeypatch.setattr('i7aof.remap.shared.read_dataset', lambda *a, **k: ds)

    def fake_add_periodic_lon(*args, **kwargs):
        captured['add_periodic_called'] = True
        return kwargs.get('ds', args[0] if args else None)

    monkeypatch.setattr(
        'i7aof.remap.shared.add_periodic_lon', fake_add_periodic_lon
    )

    def fake_mask(**kwargs):
        captured['mask_regional'] = kwargs['regional']
        captured['mask_nlon'] = kwargs['ds'].sizes['lon']
        return xr.Dataset()

    def fake_data(**kwargs):
        captured['data_regional'] = kwargs['regional']
        return [xr.Dataset({'ct': kwargs['ds']['ct']})]

    monkeypatch.setattr('i7aof.remap.shared._build_and_remap_mask', fake_mask)
    monkeypatch.setattr('i7aof.remap.shared._remap_data_variables', fake_data)
    monkeypatch.setattr(
        'i7aof.remap.shared._validate_z_extrap', lambda *a, **k: None
    )
    monkeypatch.setattr(
        'i7aof.remap.shared._concat_chunks', lambda chunks: chunks[0]
    )
    monkeypatch.setattr(
        'i7aof.remap.shared._finalize_and_write', lambda **k: None
    )

    _remap_horiz(
        config=config,
        in_filename='ignored.nc',
        out_filename='ignored_out.nc',
        model_prefix='clim',
        tmpdir='.',
        logger=None,
        has_fill_values=['ct'],
        lat_var='lat',
        lon_var='lon',
        x_dim='lon',
    )

    # 1D path: duplicate seam column dropped (5 -> 4), source forced global,
    # and the 2D periodic workaround is not used.
    assert captured['mask_nlon'] == 4
    assert captured['mask_regional'] is False
    assert captured['data_regional'] is False
    assert 'add_periodic_called' not in captured
