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

To avoid read/write conflicts and enable safe resumption after interruptions,
outputs are written to a temporary file in the same directory and then
atomically renamed to the final filename once the write completes.
"""

from __future__ import annotations

import argparse
import os
import uuid
from typing import List

from i7aof.config import load_config
from i7aof.coords import attach_grid_coords, strip_fill_on_non_data
from i7aof.io import read_dataset, write_netcdf
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
    progress: bool = True,
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
    config = load_config(
        model=model,
        clim_name=clim_name,
        workdir=workdir,
        user_config_filename=user_config_filename,
    )
    workdir_base: str = config.get('workdir', 'base_dir')

    # Monthly input directories (bias-corrected)
    ct_sa_dir = os.path.join(
        workdir_base, 'biascorr', model, scenario, clim_name, 'Omon', 'ct_sa'
    )
    tf_dir = os.path.join(
        workdir_base, 'biascorr', model, scenario, clim_name, 'Omon', 'tf'
    )

    # Annual output directory (combined variables)
    out_dir = os.path.join(
        workdir_base, 'biascorr', model, scenario, clim_name, 'Oyr', 'ct_sa_tf'
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
        outs = annual_average(
            [path], out_dir=out_dir, overwrite=overwrite, progress=progress
        )
        # Attach ISMIP grid coordinates/bounds and strip fills on non-data
        for out_path in outs:
            tmp_path = f'{out_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}'
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            ds_ann = None
            wrote_tmp = False
            try:
                ds_ann = read_dataset(out_path)
                ds_ann = attach_grid_coords(ds_ann, config)
                data_vars = [
                    str(v) for v in ds_ann.data_vars if v not in ds_ann.coords
                ]
                ds_ann = strip_fill_on_non_data(ds_ann, data_vars=data_vars)
                # Only 'ct', 'sa', and 'tf' should carry _FillValue; all
                # others (including coords and bounds) should not.
                fill_and_compress = {'ct': True, 'sa': True, 'tf': True}
                write_netcdf(
                    ds_ann,
                    tmp_path,
                    progress_bar=progress,
                    has_fill_values=fill_and_compress,
                    compression=fill_and_compress,
                )
                wrote_tmp = True
            finally:
                try:
                    if ds_ann is not None:
                        ds_ann.close()
                finally:
                    if wrote_tmp:
                        # After successful write, atomically replace the target
                        os.replace(tmp_path, out_path)
                    elif os.path.exists(tmp_path):
                        # Clean up a partial temp file on failure
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
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
    parser.add_argument(
        '--no-progress',
        dest='progress',
        action='store_false',
        help='Disable progress bars while writing NetCDF files.',
    )
    parser.set_defaults(progress=True)
    args = parser.parse_args()

    outputs = compute_cmip_annual_averages(
        model=args.model,
        scenario=args.scenario,
        clim_name=args.clim_name,
        workdir=args.workdir,
        user_config_filename=args.config,
        overwrite=args.overwrite,
        progress=args.progress,
    )
    for out_path in outputs:
        print(out_path)
