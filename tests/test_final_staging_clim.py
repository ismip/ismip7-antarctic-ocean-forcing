import os

import xarray as xr
from mpas_tools.config import MpasConfigParser

from i7aof.convert.ct_sa_to_thetao_so import _publish_final_climatology


def _config(tmp_path):
    config = MpasConfigParser()
    config.set('workdir', 'base_dir', str(tmp_path))
    config.set('output', 'version', 'v4')
    config.set('climatology', 'climatology_start_year', '1972')
    config.set('climatology', 'climatology_end_year', '2024')
    return config


def test_publish_ct_and_sa_to_final(tmp_path):
    config = _config(tmp_path)
    src_dir = tmp_path / 'extrap'
    src_dir.mkdir()
    published = {}
    for var in ('ct', 'sa'):
        src = src_dir / f'OI_Climatology_ismip8km_60m_{var}_extrap.nc'
        xr.Dataset({var: ('x', [0.0])}).to_netcdf(str(src))
        _publish_final_climatology(
            config=config, clim_name='clim_test', var=var, in_path=str(src)
        )
        expected = os.path.join(
            str(tmp_path),
            'final',
            'AIS',
            'obs',
            'ocean',
            'climatology',
            'clim_test',
            var,
            'v4',
            f'{var}_AIS_obs_ocean_climatology_clim_test_v4_1972-2024.nc',
        )
        published[var] = expected

    for var, path in published.items():
        assert os.path.exists(path), f'{var} not staged at {path}'
