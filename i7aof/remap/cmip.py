#!/usr/bin/env python
import argparse
import os
import shutil

from mpas_tools.logging import LoggingContext

from i7aof.cmip import get_model_prefix
from i7aof.config import load_config
from i7aof.convert.paths import get_ct_sa_output_paths
from i7aof.grid.ismip import get_res_string, write_ismip_grid
from i7aof.remap.shared import (
    _remap_horiz,
    _vert_mask_interp_norm_multi,
)


def remap_cmip(
    model,
    scenario,
    workdir=None,
    user_config_filename=None,
):
    """
    Remap pre-converted CMIP ct/sa to the ISMIP grid in two stages:

    1) vertical interpolation to ISMIP 'z_extrap' levels, then
    2) horizontal remapping to the ISMIP lat/lon grid.

    Prerequisite
    - Run the conversion step first so inputs contain variables 'ct' and
      'sa' on the native grid. Use either:
        * Python: i7aof.convert.cmip.convert_cmip
        * CLI: ismip7-antarctic-convert-cmip
      Then run the remap CLI: ismip7-antarctic-remap-cmip.

    This function orchestrates the basic flow per input file:
    - Prepare output dirs and ensure the ISMIP grid exists.
    - For each monthly file:
      * Vertical pipeline (see _vert_mask_interp_norm): mask invalid source
        points -> interpolate in z -> normalize.
      * Horizontal remap of the vertically processed data to ISMIP grid.

    Parameters
    ----------
    model : str
        Name of the CMIP model to remap
    scenario : str
        The name of the scenario ('historical', 'ssp585', etc.)
    workdir : str, optional
        The base work directory within which the remapped files will be
        placed
    user_config_filename : str, optional
        The path to a file with user config options that override the
        defaults
    """

    (
        config,
        workdir,
        outdir,
        ismip_res_str,
        model_prefix,
    ) = _load_config_and_paths(
        model=model,
        workdir=workdir,
        user_config_filename=user_config_filename,
        scenario=scenario,
    )

    # Build input/output lists for ct/sa
    in_files, out_files = _build_io_lists(
        config=config,
        scenario=scenario,
        outdir=outdir,
        ismip_res_str=ismip_res_str,
        model=model,
        workdir=workdir,
    )

    # Ensure the destination ISMIP grid files exist (used by both steps)
    write_ismip_grid(config)

    for index, pair_or_file in enumerate(in_files):
        _process_one(
            index=index,
            pair_or_file=pair_or_file,
            out_filename=out_files[index],
            outdir=outdir,
            config=config,
            model_prefix=model_prefix,
        )


def main():
    parser = argparse.ArgumentParser(
        description='Remap CT and SA fields from CMIP to ISMIP grid.'
    )
    parser.add_argument(
        '-m',
        '--model',
        dest='model',
        type=str,
        required=True,
        help='Name of the CMIP model to remap (required).',
    )
    parser.add_argument(
        '-s',
        '--scenario',
        dest='scenario',
        type=str,
        required=True,
        help=(
            'Name of the scenario to remap ("historical", "ssp585", etc.: '
            'required).'
        ),
    )
    parser.add_argument(
        '-w',
        '--workdir',
        dest='workdir',
        type=str,
        required=False,
        help='Path to the base working directory (optional).',
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

    remap_cmip(
        model=args.model,
        scenario=args.scenario,
        workdir=args.workdir,
        user_config_filename=args.config,
    )


# helper functions


def _load_config_and_paths(
    model,
    workdir,
    user_config_filename,
    scenario,
):
    model_prefix = get_model_prefix(model)

    config = load_config(
        model=model,
        workdir=workdir,
        user_config_filename=user_config_filename,
    )

    workdir_base: str = config.get('workdir', 'base_dir')
    outdir = os.path.join(
        workdir_base, 'remap', model, scenario, 'Omon', 'ct_sa'
    )
    os.makedirs(outdir, exist_ok=True)
    os.chdir(workdir_base)

    ismip_res_str = get_res_string(config, extrap=True)
    return config, workdir_base, outdir, ismip_res_str, model_prefix


def _build_io_lists(
    config,
    scenario,
    outdir,
    ismip_res_str,
    model,
    workdir,
):
    """Build lists of input and output files for ct/sa remapping.

    Inputs are the pre-converted ct_sa files on the native grid, whose paths
    are derived from the thetao/so config using a shared helper to ensure
    consistent naming across convert and remap stages.
    """
    in_files = []
    out_files = []

    # Derive absolute paths to ct_sa native-grid files under workdir/convert
    ct_sa_abs_paths = get_ct_sa_output_paths(
        config=config,
        model=model,
        scenario=scenario,
        workdir=workdir,
    )

    for abs_filename in ct_sa_abs_paths:
        base_filename = os.path.basename(abs_filename)
        if 'gn' not in base_filename:
            raise ValueError(
                f'Expected input to be on native grid (gn): {base_filename}'
            )
        out_filename = base_filename.replace('gn', f'ismip{ismip_res_str}')
        out_filename = os.path.join(outdir, out_filename)
        in_files.append(abs_filename)
        out_files.append(out_filename)

    return in_files, out_files


def _process_one(
    index,
    pair_or_file,
    out_filename,
    outdir,
    config,
    model_prefix,
):
    if os.path.exists(out_filename):
        print(f'Remapped file exists, skipping: {out_filename}')
        return

    # Per-file tmp dirs for clarity and clean-up
    vert_tmpdir = os.path.join(outdir, f'tmp_vert_interp_ct_sa_{index}')
    os.makedirs(vert_tmpdir, exist_ok=True)

    horiz_tmpdir = os.path.join(outdir, f'tmp_horiz_remap_ct_sa_{index}')
    os.makedirs(horiz_tmpdir, exist_ok=True)

    in_filename = pair_or_file
    vert_interp_filenames = _vert_mask_interp_norm_multi(
        config, in_filename, outdir, ['ct', 'sa'], vert_tmpdir
    )

    with LoggingContext(__name__) as logger:
        _remap_horiz(
            config,
            vert_interp_filenames,
            out_filename,
            model_prefix,
            horiz_tmpdir,
            logger,
        )

    # Always clean up tmp dirs for this input file
    shutil.rmtree(vert_tmpdir)
    shutil.rmtree(horiz_tmpdir)
