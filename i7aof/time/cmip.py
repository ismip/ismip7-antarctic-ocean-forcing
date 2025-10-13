"""
CMIP annual averaging driver for bias-corrected outputs.

Discovers monthly CT/SA files under:

    workdir/biascorr/<model>/<scenario>/<clim_name>/Omon/ct_sa

and monthly TF files under:

    workdir/biascorr/<model>/<scenario>/<clim_name>/Omon/tf

Computes weighted annual means using i7aof.time.average.annual_average and
writes results into a common directory:

    workdir/biascorr/<model>/<scenario>/<clim_name>/Oyr/ct_sa_tf

Output filenames insert the suffix "_ann" before the extension.
"""

from __future__ import annotations

import argparse
import os
from typing import List

from mpas_tools.config import MpasConfigParser

from i7aof.cmip import get_model_prefix
from i7aof.time.average import annual_average

__all__ = [
    'compute_cmip_annual_averages',
    'main',
]


def compute_cmip_annual_averages(
    *,
    model: str,
    scenario: str,
    clim_name: str,
    workdir: str | None = None,
    user_config_filename: str | None = None,
    overwrite: bool = False,
) -> List[str]:
    """
    Compute annual means for bias-corrected monthly CT, SA, and TF.

    Parameters
    ----------
    model : str
        CMIP model name.
    scenario : str
        Scenario key (e.g., 'historical', 'ssp585').
    clim_name : str
        Name of the reference climatology used in bias correction.
    workdir : str, optional
        Base working directory; if omitted, uses config
        "[workdir] base_dir".
    user_config_filename : str, optional
        Optional user config overriding defaults.
    overwrite : bool, optional
        Overwrite annual outputs if they already exist.

    Returns
    -------
    list of str
        Paths of output annual files created or found.
    """

    # Load configuration to resolve default workdir
    config = MpasConfigParser()
    config.add_from_package('i7aof', 'default.cfg')
    config.add_from_package('i7aof.cmip', f'{get_model_prefix(model)}.cfg')
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
    assert workdir is not None

    # Monthly input directories (bias-corrected)
    ct_sa_dir = os.path.join(
        workdir, 'biascorr', model, scenario, clim_name, 'Omon', 'ct_sa'
    )
    tf_dir = os.path.join(
        workdir, 'biascorr', model, scenario, clim_name, 'Omon', 'tf'
    )

    # Annual output directory (combined variables)
    out_dir = os.path.join(
        workdir, 'biascorr', model, scenario, clim_name, 'Oyr', 'ct_sa_tf'
    )
    os.makedirs(out_dir, exist_ok=True)

    # Collect monthly files to average
    in_files: list[str] = []
    for d in (ct_sa_dir, tf_dir):
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                if name.endswith('.nc'):
                    in_files.append(os.path.join(d, name))

    if not in_files:
        raise FileNotFoundError(
            'No monthly bias-corrected CT/SA/TF files found under:\n'
            f'  {ct_sa_dir}\n  {tf_dir}\n'
            'Run bias correction and TF computation first.'
        )

    # Compute annual averages into the requested directory.
    # Call per-file to avoid potential multi-file issues in annual_average.
    outputs: list[str] = []
    for path in in_files:
        outs = annual_average([path], out_dir=out_dir, overwrite=overwrite)
        outputs.extend(outs)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Average monthly bias-corrected CT/SA and TF to annual means. '
            'Outputs go to Oyr/ct_sa_tf with "_ann" suffix.'
        )
    )
    parser.add_argument('-m', '--model', required=True, help='CMIP model.')
    parser.add_argument(
        '-s', '--scenario', required=True, help='Scenario (historical, ...).'
    )
    parser.add_argument(
        '-c', '--clim', dest='clim_name', required=True, help='Climatology.'
    )
    parser.add_argument(
        '-w', '--workdir', required=False, help='Base working directory.'
    )
    parser.add_argument(
        '-C',
        '--config',
        dest='config',
        default=None,
        help='Path to user config (optional).',
    )
    parser.add_argument(
        '--overwrite', action='store_true', help='Overwrite existing outputs.'
    )
    args = parser.parse_args()

    outputs = compute_cmip_annual_averages(
        model=args.model,
        scenario=args.scenario,
        clim_name=args.clim_name,
        workdir=args.workdir,
        user_config_filename=args.config,
        overwrite=args.overwrite,
    )
    for out_path in outputs:
        print(out_path)
