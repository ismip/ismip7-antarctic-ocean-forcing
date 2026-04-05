import gsw
import numpy as np
import pytest
import xarray as xr

from i7aof.convert.teos10 import (
    _get_and_validate_positive,
    _pressure_from_z,
)


def test_get_and_validate_positive_uses_vertical_dimension() -> None:
    coord = xr.DataArray(
        np.array(
            [
                [[50.0, 0.0], [50.0, 0.0]],
                [[100.0, 50.0], [100.0, 50.0]],
            ]
        ),
        dims=('lev', 'y', 'x'),
    )

    positive = _get_and_validate_positive(coord, 'depth', validation_dim='lev')

    assert positive == 'down'


def test_pressure_from_z_ignores_stale_positive_down_metadata() -> None:
    z = xr.DataArray(
        np.array([-5.0, -50.0]),
        dims=('lev',),
        attrs={'positive': 'down'},
    )
    lat = xr.DataArray(np.array([-70.0]), dims=('y',))

    pressure = _pressure_from_z(z, lat)
    expected = gsw.p_from_z(z.values[:, None, None], lat.values[:, None])

    np.testing.assert_allclose(pressure, expected)


def test_pressure_from_z_does_not_flip_mixed_sign_teos10_z() -> None:
    z = xr.DataArray(
        np.array([10.0, -10.0]),
        dims=('lev',),
        attrs={'positive': 'down'},
    )
    lat = xr.DataArray(np.array([-70.0]), dims=('y',))

    pressure = _pressure_from_z(z, lat)
    expected = gsw.p_from_z(z.values[:, None, None], lat.values[:, None])

    np.testing.assert_allclose(pressure, expected)


def test_pressure_from_z_rejects_positive_up_with_depth_like_values() -> None:
    z = xr.DataArray(
        np.array([5.0, 50.0]),
        dims=('lev',),
        attrs={'positive': 'up'},
    )
    lat = xr.DataArray(np.array([-70.0]), dims=('y',))

    with pytest.raises(ValueError, match="positive='up'"):
        _pressure_from_z(z, lat)
