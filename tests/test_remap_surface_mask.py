import numpy as np
import xarray as xr

from i7aof.io import write_netcdf
from i7aof.remap.shared import _prepare_vert_coords_and_mask


def _make_preprocessed(tmp_path):
    """A tiny climatology-like preprocessed file.

    ``lev`` is positive-up and descending (surface at index 0), matching
    what ``_preprocess_climatology_input`` produces from pressures.
    Column (lat=0, lon=0) has a fill value at the surface but valid data
    below; column (lat=0, lon=1) is fully valid.
    """
    lev = np.array([-5.0, -50.0, -500.0])  # surface (max z) is index 0
    lev_bnds = np.array([[0.0, -27.5], [-27.5, -275.0], [-275.0, -725.0]])
    ct = np.array(
        [
            [[np.nan, 1.0]],  # surface: fill at lon=0, valid at lon=1
            [[2.0, 2.0]],
            [[3.0, 3.0]],
        ]
    )
    sa = np.full_like(ct, 34.0)
    ds = xr.Dataset(
        data_vars={
            'ct': (('lev', 'lat', 'lon'), ct),
            'sa': (('lev', 'lat', 'lon'), sa),
            'lev_bnds': (('lev', 'd2'), lev_bnds),
        },
        coords={
            'lev': ('lev', lev),
            'lat': ('lat', np.array([0.0])),
            'lon': ('lon', np.array([0.0, 1.0])),
        },
    )
    ds['lev'].attrs = {'units': 'm', 'positive': 'up'}
    path = tmp_path / 'preprocessed.nc'
    write_netcdf(ds, str(path))
    return str(path)


def test_surface_mask_invalidates_whole_column(tmp_path):
    in_filename = _make_preprocessed(tmp_path)
    _, _, src_valid = _prepare_vert_coords_and_mask(
        in_filename, ['ct', 'sa'], mask_from_surface=True
    )
    # Column with a surface fill value is fully invalid at every level.
    assert not bool(src_valid.isel(lat=0, lon=0).any())
    # Fully valid column stays valid everywhere.
    assert bool(src_valid.isel(lat=0, lon=1).all())


def test_without_surface_mask_keeps_subsurface(tmp_path):
    in_filename = _make_preprocessed(tmp_path)
    _, _, src_valid = _prepare_vert_coords_and_mask(
        in_filename, ['ct', 'sa'], mask_from_surface=False
    )
    col = src_valid.isel(lat=0, lon=0).values
    surface_idx = int(np.asarray(src_valid['lev'].values).argmax())
    # Surface fill stays invalid, but the subsurface levels remain valid.
    assert not bool(col[surface_idx])
    subsurface = np.delete(col, surface_idx)
    assert subsurface.all()
