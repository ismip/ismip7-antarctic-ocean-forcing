import configparser
from pathlib import Path

import netCDF4

from i7aof.grid.ismip import (
    get_ismip_grid_filename,
    write_ismip_grid,
)


def _make_config(dx_m: float, dy_m: float, dz_m: float, dzx_m: float):
    cfg = configparser.ConfigParser()
    cfg.add_section('ismip_grid')
    cfg.set('ismip_grid', 'dx', str(dx_m))
    cfg.set('ismip_grid', 'dy', str(dy_m))
    cfg.set('ismip_grid', 'dz', str(dz_m))
    cfg.set('ismip_grid', 'dz_extrap', str(dzx_m))
    return cfg


def test_write_ismip_grid_minimal(tmp_path, monkeypatch):
    # Choose large dx,dy to get a tiny 2x2 grid for speed
    # ((nx_base-1)*1000)/dx + 1 -> 2 when dx = (nx_base-1)*1000
    dx = dy = (6081 - 1) * 1000
    dz = 1800.0
    dzx = 1800.0

    cfg = _make_config(dx, dy, dz, dzx)

    # run in a temp dir so output is isolated
    monkeypatch.chdir(tmp_path)

    # write and open file
    write_ismip_grid(cfg)
    out_file = Path(get_ismip_grid_filename(cfg))
    assert out_file.exists()

    with netCDF4.Dataset(out_file, mode='r') as nc:
        # required variables
        required = [
            'x',
            'y',
            'z',
            'z_extrap',
            'lat',
            'lon',
            'x_bnds',
            'y_bnds',
            'lat_bnds',
            'lon_bnds',
            'z_bnds',
            'z_extrap_bnds',
        ]
        for name in required:
            assert name in nc.variables, f'missing variable: {name}'

        # dims for bounds
        assert 'bnds' in nc.dimensions
        assert nc.dimensions['bnds'].size == 2
        assert 'nv' in nc.dimensions
        assert nc.dimensions['nv'].size == 4

        # no _FillValue on grid variables
        for name in required:
            assert '_FillValue' not in nc.variables[name].ncattrs()

        # attrs for x/y
        x = nc.variables['x']
        y = nc.variables['y']
        assert x.getncattr('units') == 'm'
        assert x.getncattr('standard_name') == 'projection_x_coordinate'
        assert x.getncattr('long_name') == 'x coordinate of projection'
        assert x.getncattr('axis') == 'X'
        assert x.getncattr('bounds') == 'x_bnds'
        assert y.getncattr('units') == 'm'
        assert y.getncattr('standard_name') == 'projection_y_coordinate'
        assert y.getncattr('long_name') == 'y coordinate of projection'
        assert y.getncattr('axis') == 'Y'
        assert y.getncattr('bounds') == 'y_bnds'

        # attrs for lat/lon
        lat = nc.variables['lat']
        lon = nc.variables['lon']
        assert lat.getncattr('units') == 'degrees_north'
        assert lat.getncattr('standard_name') == 'latitude'
        assert lat.getncattr('long_name') == 'latitude coordinate'
        assert lat.getncattr('bounds') == 'lat_bnds'
        assert lon.getncattr('units') == 'degrees_east'
        assert lon.getncattr('standard_name') == 'longitude'
        assert lon.getncattr('long_name') == 'longitude coordinate'
        assert lon.getncattr('bounds') == 'lon_bnds'

        # attrs for z, z_extrap
        z = nc.variables['z']
        zx = nc.variables['z_extrap']
        for var in (z, zx):
            assert var.getncattr('units') == 'm'
            assert var.getncattr('standard_name') == 'height'
            assert (
                var.getncattr('long_name')
                == 'height relative to sea surface (positive up)'
            )
            assert var.getncattr('positive') == 'up'
            assert var.getncattr('axis') == 'Z'
        assert z.getncattr('bounds') == 'z_bnds'
        assert zx.getncattr('bounds') == 'z_extrap_bnds'
