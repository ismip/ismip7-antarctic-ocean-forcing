"""
Path helpers for ISMIP7 workflow outputs.

This module standardizes intermediate and final output locations and
constructs filenames that follow the ISMIP7 naming conventions.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from i7aof.version import __version__

STAGE_DIRS = {
    'split': '01_split',
    'convert_cmip': '02_cmip_to_ct_sa',
    'remap': '03_remap',
    'extrap': '04_extrap',
    'biascorr': '05_biascorr',
    'ct_sa_to_tf': '06_ct_sa_to_tf',
    'annual': '07_annual',
    'ct_sa_to_thetao_so': '08_ct_sa_to_thetao_so',
}


def get_workdir_base(config) -> str:
    return config.get('workdir', 'base_dir')


def get_intermediate_root(config) -> str:
    base = get_workdir_base(config)
    if config.has_option('workdir', 'intermediate_dir'):
        sub = config.get('workdir', 'intermediate_dir')
    else:
        sub = 'intermediate'
    return os.path.join(base, sub)


def get_final_root(config) -> str:
    base = get_workdir_base(config)
    if config.has_option('workdir', 'final_dir'):
        sub = config.get('workdir', 'final_dir')
    else:
        sub = 'final'
    return os.path.join(base, sub)


def get_stage_dir(config, stage: str) -> str:
    if stage not in STAGE_DIRS:
        raise KeyError(f'Unknown stage key: {stage}')
    return os.path.join(get_intermediate_root(config), STAGE_DIRS[stage])


def get_output_version(config) -> str:
    if config.has_option('output', 'version'):
        version = str(config.get('output', 'version')).strip()
        if version:
            return version
    version = __version__.strip()
    if version.startswith('v'):
        return version
    return f'v{version}'


def parse_year_range(name: str) -> Optional[str]:
    """Extract a YYYY-YYYY year range from a filename or return None."""
    # Accept YYYYMM-YYYYMM or YYYY-YYYY, return YYYY-YYYY
    match = re.search(r'(\d{4})\d{0,2}-(\d{4})\d{0,2}', name)
    if not match:
        return None
    return f'{match.group(1)}-{match.group(2)}'


def build_cmip_final_dir(
    config,
    *,
    model: str,
    scenario: str,
    variable: str,
    version: Optional[str] = None,
    extras: Optional[str] = None,
) -> str:
    version = version or get_output_version(config)
    base = get_final_root(config)
    if extras:
        return os.path.join(
            base,
            'AIS',
            model,
            scenario,
            'Ocean',
            'extras',
            extras,
            variable,
            version,
        )
    return os.path.join(
        base, 'AIS', model, scenario, 'Ocean', variable, version
    )


def build_cmip_final_filename(
    *,
    variable: str,
    model: str,
    scenario: str,
    version: str,
    year_range: str,
    extras: Optional[str] = None,
) -> str:
    if extras:
        return (
            f'{variable}_AIS_{model}_{scenario}_Ocean_extras_{extras}_'
            f'{version}_{year_range}.nc'
        )
    return f'{variable}_AIS_{model}_{scenario}_Ocean_{version}_{year_range}.nc'


def build_obs_climatology_dir(
    config,
    *,
    clim_name: str,
    variable: str,
    version: Optional[str] = None,
) -> str:
    version = version or get_output_version(config)
    base = get_final_root(config)
    return os.path.join(
        base,
        'AIS',
        'Obs',
        'Ocean',
        'climatology',
        clim_name,
        variable,
        version,
    )


def build_obs_climatology_filename(
    *,
    variable: str,
    clim_name: str,
    version: str,
    year_range: str,
) -> str:
    return (
        f'{variable}_AIS_Obs_Ocean_climatology_{clim_name}_'
        f'{version}_{year_range}.nc'
    )


def build_imbie_basins_dir(config, *, version: Optional[str] = None) -> str:
    version = version or get_output_version(config)
    base = get_final_root(config)
    return os.path.join(base, 'AIS', 'Obs', 'Ocean', 'IMBIE-basins', version)


def build_imbie_basins_filename(*, version: str) -> str:
    return f'IMBIE-basins_AIS_Obs_Ocean_{version}.nc'


def build_topography_dir(
    config, *, dataset: str, version: Optional[str] = None
) -> str:
    version = version or get_output_version(config)
    base = get_final_root(config)
    return os.path.join(
        base, 'AIS', 'Obs', 'Ocean', 'topography', dataset, version
    )


def build_topography_filename(*, dataset: str, version: str) -> str:
    return f'{dataset}_AIS_Obs_Ocean_topography_{version}.nc'


def build_grid_dir(
    config, *, hres: str, vres: str, version: Optional[str] = None
) -> str:
    version = version or get_output_version(config)
    base = get_final_root(config)
    res = f'{hres}-{vres}'
    return os.path.join(base, 'AIS', 'Grid', 'Ocean', 'ISMIP7', res, version)


def build_grid_filename(*, hres: str, vres: str, version: str) -> str:
    res = f'{hres}-{vres}'
    return f'ISMIP7_{res}_AIS_Grid_Ocean_{version}.nc'
