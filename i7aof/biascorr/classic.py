"""
Bias correct the extrapolated CMIP data towards climatology.

This is the classical method with bias correction in geographic space.

Workflow
========

"""

import os
from typing import List

from mpas_tools.config import MpasConfigParser

from i7aof.cmip import get_model_prefix
from i7aof.grid.ismip import get_res_string


def biascorr_cmip(
    model: str,
    scenario: str,
    clim_name: str,
    workdir: str | None = None,
    user_config_filename: str | None = None,
    variables: tuple[str, ...] = ('ct', 'sa'),
):
    """
    Bias correct CMIP ct/sa in two stages:

    1) extract the bias in ct and sa
    2) apply the bias correction in ct and sa

    Parameters
    ----------
    model: str
        Name of the CMIP model to bias correct
    scenario: str
        The name of the scenario ('historical', 'ssp585', etc.)
    clim_name: str
        The name of the reference climatology
    workdir : str, optional
        The base work directory within which the bias corrected files will be
        placed
    user_config_filename : str, optional
        The path to a file with user config options that override the
        defaults
    variables : tuple of str, optional
        Variables to extrapolate (default: ``('ct', 'sa')``).
    """

    (
        config,
        workdir,
        extrap_dir,
        outdir,
        ismip_res_str,
        model_prefix,
    ) = _load_config_and_paths(
        model=model,
        workdir=workdir,
        user_config_filename=user_config_filename,
        scenario=scenario,
        clim_name=clim_name,
    )

    in_files = _collect_extrap_outputs(extrap_dir, ismip_res_str)
    if not in_files:
        raise FileNotFoundError(
            'No extrapolated files found. Run: ismip7-antarctic-extrap-cmip'
        )

    print(in_files)


# helper functions


def _load_config_and_paths(
    model,
    workdir,
    user_config_filename,
    scenario,
    clim_name,
):
    model_prefix = get_model_prefix(model)

    config = MpasConfigParser()
    config.add_from_package('i7aof', 'default.cfg')
    config.add_from_package('i7aof.cmip', f'{model_prefix}.cfg')
    config.add_from_package('i7aof.clim', f'{clim_name}.cfg')
    if user_config_filename is not None:
        config.add_user_config(user_config_filename)

    if workdir is None:
        if config.has_option('workdir', 'base_dir'):
            workdir = config.get('workdir', 'base_dir')
        else:
            raise ValueError(
                'Missing configuration option: [workdir] base_dir. '
                'Please supply a user config file that defines this option.'
            )
    assert workdir is not None, (
        'Internal error: workdir should be resolved to a string'
    )

    extrap_dir = os.path.join(
        workdir, 'extrap', model, scenario, 'Omon', 'ct_sa'
    )

    outdir = os.path.join(
        workdir, 'biascorr', model, scenario, clim_name, 'Omon', 'ct_sa'
    )
    os.makedirs(outdir, exist_ok=True)
    os.chdir(workdir)

    ismip_res_str = get_res_string(config, extrap=True)
    return config, workdir, extrap_dir, outdir, ismip_res_str, model_prefix


def _collect_extrap_outputs(extrap_dir: str, ismip_res_str: str) -> List[str]:
    if not os.path.isdir(extrap_dir):
        return []
    result: List[str] = []
    for name in sorted(os.listdir(extrap_dir)):
        if f'ismip{ismip_res_str}' in name and ('ct' in name or 'sa' in name):
            result.append(os.path.join(extrap_dir, name))
    return result
