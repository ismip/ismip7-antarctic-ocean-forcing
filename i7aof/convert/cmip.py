#!/usr/bin/env python
import argparse
import os

import numpy as np
import xarray as xr
from mpas_tools.config import MpasConfigParser

from i7aof.cmip import get_model_prefix
from i7aof.convert.paths import get_ct_sa_output_paths
from i7aof.convert.teos10 import convert_dataset_to_ct_sa
from i7aof.io import write_netcdf


def convert_cmip(
    model,
    scenario,
    inputdir=None,
    workdir=None,
    user_config_filename=None,
):
    """
    Convert CMIP thetao/so monthly files to ct/sa on the native grid.

    Each thetao/so input pair is aligned and converted to a single output
    file containing variables 'ct' and 'sa'. Existing outputs are skipped to
    allow resuming. Files are written under:

        {workdir}/convert/{model}/{scenario}/Omon/ct_sa/

    Inputs are discovered from the config section ``[{scenario}_files]``
    entries ``thetao`` and ``so``. Output filenames are derived from the
    thetao basenames by replacing the variable token with ``ct_sa``.

    Parameters
    ----------
    model : str
        Name of the CMIP model (used to select the model config and to
        construct output paths).
    scenario : str
        Scenario key (e.g., 'historical', 'ssp585') used to pick input file
        lists from the config.
    inputdir : str, optional
        Base directory where the relative input file paths are resolved. If
        not provided, uses ``[inputdir] base_dir`` from the config.
    workdir : str, optional
        Base working directory where outputs will be written. If not
        provided, uses ``[workdir] base_dir`` from the config.
    user_config_filename : str, optional
        Optional user config that overrides defaults (paths, variable names,
        chunk sizes, etc.).

    Returns
    -------
    None
        Writes converted datasets to disk; does not return a value.
    """

    model_prefix = get_model_prefix(model)

    config = MpasConfigParser()
    config.add_from_package('i7aof', 'default.cfg')
    config.add_from_package('i7aof.cmip', f'{model_prefix}.cfg')
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

    if inputdir is None:
        if config.has_option('inputdir', 'base_dir'):
            inputdir = config.get('inputdir', 'base_dir')
        else:
            raise ValueError(
                'Missing configuration option: [inputdir] base_dir. '
                'Please supply a user config file that defines this option.'
            )

    outdir = os.path.join(workdir, 'convert', model, scenario, 'Omon', 'ct_sa')
    os.makedirs(outdir, exist_ok=True)

    lat_var = config.get('cmip_dataset', 'lat_var')
    lon_var = config.get('cmip_dataset', 'lon_var')
    # allow override from [convert_cmip]
    time_chunk_str = (
        config.get('convert_cmip', 'time_chunk')
        if config.has_option('convert_cmip', 'time_chunk')
        else 'None'
    )
    if time_chunk_str in ('', 'None', 'none'):
        time_chunk = None
    else:
        time_chunk = int(time_chunk_str)
    depth_var = (
        config.get('convert_cmip', 'depth_var')
        if config.has_option('convert_cmip', 'depth_var')
        else 'lev'
    )

    thetao_paths = list(config.getexpression(f'{scenario}_files', 'thetao'))
    so_paths = list(config.getexpression(f'{scenario}_files', 'so'))

    if len(thetao_paths) != len(so_paths):
        raise ValueError(
            'Mismatched number of thetao and so files for scenario '
            f"'{scenario}'."
        )

    # derive output paths once from config; ensures consistent naming
    out_paths = get_ct_sa_output_paths(
        config=config,
        model=model,
        scenario=scenario,
        workdir=workdir,
    )

    for th_rel, so_rel, out_abs in zip(
        thetao_paths, so_paths, out_paths, strict=True
    ):
        th_abs = os.path.join(inputdir, th_rel)
        so_abs = os.path.join(inputdir, so_rel)

        if os.path.exists(out_abs):
            print(f'Converted file exists, skipping: {out_abs}')
            continue

        print(f'Converting to CT/SA: {os.path.basename(out_abs)}')

        ds_thetao = xr.open_dataset(
            th_abs, decode_times=False, chunks={'time': time_chunk}
        )
        ds_so = xr.open_dataset(
            so_abs, decode_times=False, chunks={'time': time_chunk}
        )

        # strict alignment to catch mismatched time/lev
        ds_thetao, ds_so = xr.align(ds_thetao, ds_so, join='exact')

        ds_ctsa = convert_dataset_to_ct_sa(
            ds_thetao,
            ds_so,
            thetao_var='thetao',
            so_var='so',
            lat_var=lat_var,
            lon_var=lon_var,
            depth_var=depth_var,
        )

        # ensure float32 types for compact storage
        ds_ctsa['ct'] = ds_ctsa['ct'].astype(np.float32)
        ds_ctsa['sa'] = ds_ctsa['sa'].astype(np.float32)

        write_netcdf(ds_ctsa, out_abs, progress_bar=True)


def main():
    parser = argparse.ArgumentParser(
        description='Convert CMIP thetao/so to ct/sa on native grid.'
    )
    parser.add_argument(
        '-m',
        '--model',
        dest='model',
        type=str,
        required=True,
        help='Name of the CMIP model to convert (required).',
    )
    parser.add_argument(
        '-s',
        '--scenario',
        dest='scenario',
        type=str,
        required=True,
        help='Scenario to convert (historical, ssp585, ...: required).',
    )
    parser.add_argument(
        '-i',
        '--inputdir',
        dest='inputdir',
        type=str,
        required=False,
        help='Path to base input directory (optional).',
    )
    parser.add_argument(
        '-w',
        '--workdir',
        dest='workdir',
        type=str,
        required=False,
        help='Path to base working directory (optional).',
    )
    parser.add_argument(
        '-c',
        '--config',
        dest='config',
        type=str,
        default=None,
        help='Path to user config file (optional).',
    )
    args = parser.parse_args()

    convert_cmip(
        model=args.model,
        scenario=args.scenario,
        inputdir=args.inputdir,
        workdir=args.workdir,
        user_config_filename=args.config,
    )
