import argparse
import os
import re
from typing import Sequence, Tuple

import numpy as np
import xarray as xr
from mpas_tools.config import MpasConfigParser

from i7aof.config import load_config
from i7aof.io import read_dataset, write_netcdf


def split_cmip(
    model: str,
    scenario: str,
    inputdir: str | None = None,
    workdir: str | None = None,
    user_config_filename: str | None = None,
) -> None:
    """
    Split CMIP monthly datasets (thetao/so) into files of N months.

    Outputs are written under:

        {workdir}/split/{model}/{scenario}/Omon/{variable}/

    Each output filename is based on the input basename with any trailing
    "_YYYY[MM]-YYYY[MM]" suffix removed and replaced by
    "_{startyear}-{endyear}.nc".

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
        Optional user config that overrides defaults.
    """

    print(f'Splitting CMIP model "{model}" scenario "{scenario}" datasets.')

    config = load_config(
        model=model,
        inputdir=inputdir,
        workdir=workdir,
        user_config_filename=user_config_filename,
    )

    if not config.has_option('inputdir', 'base_dir'):
        raise ValueError(
            'Missing configuration option: [inputdir] base_dir. '
            'Please supply a user config file or command-line option that '
            'defines this option.'
        )
    workdir_base: str = config.get('workdir', 'base_dir')
    inputdir_base: str = config.get('inputdir', 'base_dir')

    out_base_dir = os.path.join(workdir_base, 'split', model, scenario, 'Omon')
    os.makedirs(out_base_dir, exist_ok=True)

    months_per_file = _parse_months_per_file(config)
    if months_per_file is None or months_per_file <= 0:
        raise ValueError(
            'Invalid or missing [split_cmip] months_per_file. '
            'Please set a positive integer.'
        )

    # Variables to split. Extendable if needed.
    var_list = ['thetao', 'so']

    for var in var_list:
        if not config.has_option(f'{scenario}_files', var):
            # Skip missing variable list in config
            continue
        rel_paths = list(config.getexpression(f'{scenario}_files', var))
        if not rel_paths:
            continue

        outdir = os.path.join(out_base_dir, var)
        os.makedirs(outdir, exist_ok=True)
        print(f'Writing split {var} files to: {outdir}')

        for rel in rel_paths:
            in_abs = os.path.join(inputdir_base, rel)
            print(f'Splitting {var}: {in_abs}')
            _split_one_file(in_abs, outdir, months_per_file, var_name=var)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Split CMIP thetao/so monthly files into N-month chunks.'
    )
    parser.add_argument(
        '-m', '--model', dest='model', required=True, help='CMIP model name.'
    )
    parser.add_argument(
        '-s', '--scenario', dest='scenario', required=True, help='Scenario.'
    )
    parser.add_argument(
        '-i',
        '--inputdir',
        dest='inputdir',
        required=False,
        help='Input base dir.',
    )
    parser.add_argument(
        '-w',
        '--workdir',
        dest='workdir',
        required=False,
        help='Work base dir.',
    )
    parser.add_argument(
        '-c', '--config', dest='config', default=None, help='User config file.'
    )
    args = parser.parse_args()
    split_cmip(
        model=args.model,
        scenario=args.scenario,
        inputdir=args.inputdir,
        workdir=args.workdir,
        user_config_filename=args.config,
    )


def _parse_months_per_file(config: MpasConfigParser) -> int | None:
    if not config.has_option('split_cmip', 'months_per_file'):
        return None
    raw = config.get('split_cmip', 'months_per_file')
    if raw in ('', 'None', 'none'):
        return None
    return int(raw)


def _split_one_file(
    in_abs: str, outdir: str, months_per_file: int, var_name: str
) -> None:
    ds = read_dataset(in_abs)
    fill_and_compress = [var_name]
    try:
        if 'time' not in ds.dims:
            # No time: just copy once with cleaned name
            prefix = _derive_prefix_from_input(in_abs)
            out_abs = os.path.join(outdir, f'{prefix}.nc')
            write_netcdf(
                ds,
                out_abs,
                has_fill_values=fill_and_compress,
                compression=fill_and_compress,
            )
            return

        nt = int(ds.sizes['time'])
        ranges = _compute_file_ranges(nt, months_per_file)
        prefix = _derive_prefix_from_input(in_abs)
        for t0, t1 in ranges:
            start_year = _extract_year(ds['time'].isel(time=t0))
            end_year = _extract_year(ds['time'].isel(time=t1 - 1))
            out_abs = os.path.join(
                outdir, f'{prefix}_{start_year}-{end_year}.nc'
            )
            ds_chunk = ds.isel(time=slice(t0, t1))
            print(f'  -> {out_abs}')
            write_netcdf(
                ds_chunk,
                out_abs,
                has_fill_values=fill_and_compress,
                compression=fill_and_compress,
            )
    finally:
        ds.close()


def _compute_file_ranges(
    nt: int, months_per_file: int
) -> Sequence[Tuple[int, int]]:
    return [
        (s, min(s + months_per_file, nt))
        for s in range(0, nt, months_per_file)
    ]


def _derive_prefix_from_input(in_abs: str) -> str:
    base = os.path.basename(in_abs)
    name, _ext = os.path.splitext(base)
    # strip trailing date-range pattern if present: _YYYY[MM]-YYYY[MM]
    m = re.search(r'_(\d{4})(?:\d{2})?-(\d{4})(?:\d{2})?$', name)
    if m:
        name = name[: m.start()]
    return name


def _extract_year(t_da: xr.DataArray) -> int:
    val = t_da.values.item() if hasattr(t_da.values, 'item') else t_da.values
    year = getattr(val, 'year', None)
    if year is not None:
        return int(year)
    try:
        s = np.datetime_as_string(val, unit='Y')
        return int(str(s)[:4])
    except (TypeError, ValueError, AttributeError):  # pragma: no cover
        # Fall back to string parsing if not a numpy datetime64
        return int(str(val)[:4])
