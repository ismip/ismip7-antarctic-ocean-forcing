"""Extrapolate a remapped climatology (no time dimension) to depth levels.

This mirrors the CMIP extrapolation workflow but simplifies it to a single
pass per variable because the climatology file has no time dimension. It
reuses shared helpers in ``i7aof.extrap.shared`` for prerequisite data,
namelist rendering, executable invocation, and finalization.

Expected input (produced by ``ismip7-antarctic-remap-clim``):

    workdir/<intermediate>/03_remap/climatology/<name>/<file>_ismip<res>.nc

Outputs (per variable) are written to:

    workdir/<intermediate>/04_extrap/climatology/<name>/<file>_ismip<res>_<var>_extrap.nc

Two external Fortran executables are invoked sequentially:

    * i7aof_extrap_horizontal
    * i7aof_extrap_vertical

The combined namelist is rendered from the existing template.
"""

import argparse
import logging
import os
import shutil

import xarray as xr
from mpas_tools.config import MpasConfigParser
from mpas_tools.logging import LoggingContext

from i7aof.config import load_config
from i7aof.extrap.shared import (
    _apply_under_ice_mask_to_file,
    _ensure_imbie_masks,
    _ensure_topography,
    _finalize_output_with_grid,
    _prepare_input_single,
    _render_namelist,
    _run_exe_capture,
    _vertically_resample_to_coarse_ismip_grid,
)
from i7aof.grid.ismip import ensure_ismip_grid, get_res_string
from i7aof.io import read_dataset
from i7aof.paths import (
    get_stage_dir,
)

__all__ = ['extrap_climatology', 'main']


def extrap_climatology(
    clim_name: str,
    *,
    workdir: str | None = None,
    user_config_filename: str | None = None,
    variables: tuple[str, ...] = ('ct', 'sa'),
    keep_intermediate: bool = False,
):
    """Extrapolate a remapped climatology file for the given variables.

    Parameters
    ----------
    clim_name : str
        Name of the climatology (must match the file used in remap step).
    workdir : str, optional
        Base working directory (defaults to ``[workdir] base_dir``).
    user_config_filename : str, optional
        Optional user config file path overriding defaults.
    variables : tuple of str, optional
        Variables to extrapolate (default: ``('ct', 'sa')``).
    keep_intermediate : bool, optional
        Keep temporary directory if True.
    """
    config = load_config(
        model=None,
        clim_name=clim_name,
        workdir=workdir,
        user_config_filename=user_config_filename,
    )
    workdir_base: str = config.get('workdir', 'base_dir')

    # Locate remapped input file
    remap_dir = os.path.join(
        get_stage_dir(config, 'remap'), 'climatology', clim_name
    )
    if not os.path.isdir(remap_dir):
        raise FileNotFoundError(
            f'Remapped climatology directory not found: {remap_dir}. '
            'Run ismip7-antarctic-remap-clim first.'
        )
    # Choose the single NetCDF file containing ct/sa remapped fields
    candidates = [
        f for f in sorted(os.listdir(remap_dir)) if f.endswith('.nc')
    ]
    if not candidates:
        raise FileNotFoundError(
            f'No NetCDF files found in {remap_dir}; cannot extrapolate.'
        )
    if len(candidates) > 1:
        # Heuristic: prefer one that contains 'ct_sa' if present
        ct_sa = [c for c in candidates if 'ct_sa' in c]
        base_in = ct_sa[0] if ct_sa else candidates[0]
    else:
        base_in = candidates[0]
    in_path = os.path.join(remap_dir, base_in)

    out_dir = os.path.join(
        get_stage_dir(config, 'extrap'), 'climatology', clim_name
    )
    os.makedirs(out_dir, exist_ok=True)

    basin_file = _ensure_imbie_masks(config, workdir_base)
    grid_file = ensure_ismip_grid(config)
    topo_file = _ensure_topography(config, workdir_base)

    with LoggingContext(__name__):
        logger = logging.getLogger(__name__)
        logger.info(
            f'Starting climatology extrapolation for {clim_name}: {in_path}'
        )

        for var in variables:
            if var not in read_dataset(in_path):
                logger.warning(
                    f"Variable '{var}' missing in input file; skipping."
                )
                continue
            # Output naming: insert variable + _extrap before .nc
            stem, ext = os.path.splitext(base_in)
            # ensure variable distinguishes outputs if original file has ct_sa
            if 'ct_sa' in stem:
                stem_var = stem.replace('ct_sa', f'{var}')
            else:
                stem_var = f'{stem}_{var}'
            out_file = os.path.join(out_dir, f'{stem_var}_extrap.nc')
            _ensure_extrapolated_file(
                in_path=in_path,
                out_file=out_file,
                grid_file=grid_file,
                topo_file=topo_file,
                basin_file=basin_file,
                variable=var,
                config=config,
                logger=logger,
                keep_intermediate=keep_intermediate,
            )
            # After extrapolation (or if it already existed), resample.
            # Use shared Zarr-based helper for consistent performance.
            res_extrap = get_res_string(config, extrap=True)
            res_final = get_res_string(config, extrap=False)
            out_nc = out_file.replace(
                f'ismip{res_extrap}', f'ismip{res_final}'
            )
            if os.path.abspath(out_nc) == os.path.abspath(out_file):
                stem_nc, ext_nc = os.path.splitext(out_file)
                out_nc = f'{stem_nc}_z{ext_nc}'
            base_nc = os.path.splitext(os.path.basename(out_nc))[0]
            zarr_store = os.path.join(
                os.path.dirname(out_nc), f'{base_nc}.zarr'
            )
            _vertically_resample_to_coarse_ismip_grid(
                in_path=out_file,
                grid_file=grid_file,
                variable=var,
                config=config,
                out_nc=out_nc,
                time_chunk=None,
                zarr_store=zarr_store,
                logger=logger,
            )


def main():
    parser = argparse.ArgumentParser(
        description='Extrapolate remapped climatology ct/sa (no time).'
    )
    parser.add_argument(
        '-n',
        '--clim',
        dest='clim_name',
        required=True,
        help='Climatology name used in remap step (required).',
    )
    parser.add_argument(
        '-w',
        '--workdir',
        dest='workdir',
        required=False,
        help='Base working directory (optional; else from config).',
    )
    parser.add_argument(
        '-c',
        '--config',
        dest='config',
        default=None,
        help='Path to user config file (optional).',
    )
    parser.add_argument(
        '-V',
        '--variables',
        nargs='+',
        default=['ct', 'sa'],
        help='Variables to extrapolate (default: ct sa).',
    )
    parser.add_argument(
        '--keep-intermediate',
        action='store_true',
        help='Keep temporary NetCDF / namelist files.',
    )
    args = parser.parse_args()

    extrap_climatology(
        clim_name=args.clim_name,
        workdir=args.workdir,
        user_config_filename=args.config,
        variables=tuple(args.variables),
        keep_intermediate=args.keep_intermediate,
    )


def _ensure_extrapolated_file(
    *,
    in_path: str,
    out_file: str,
    grid_file: str,
    topo_file: str,
    basin_file: str,
    variable: str,
    config: MpasConfigParser,
    logger: logging.Logger,
    keep_intermediate: bool,
) -> None:
    """Run Fortran extrapolation and finalize output if missing.

    If ``out_file`` already exists, this is a no-op.
    """
    if os.path.exists(out_file):
        logger.info(f'Extrapolated output exists: {out_file}; skipping.')
        return

    stem = os.path.splitext(os.path.basename(out_file))[0]
    tmp_dir = os.path.join(os.path.dirname(out_file), f'{stem}_tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    prepared = os.path.join(tmp_dir, f'input_{variable}.nc')
    horiz_tmp = os.path.join(tmp_dir, f'horizontal_{variable}.nc')
    vert_tmp = os.path.join(tmp_dir, f'vertical_{variable}.nc')
    namelist_path = os.path.join(tmp_dir, f'{variable}.nml')
    log_path = os.path.join(tmp_dir, 'logs', f'{variable}.log')

    # Prepare single input
    _prepare_input_single(
        in_path=in_path,
        grid_path=grid_file,
        out_prepared_path=prepared,
        variable=variable,
        logger=logger,
        # add dummy singleton time so Fortran expectation is met
        add_dummy_time=True,
    )

    # Optional pre-extrap masking under grounded/floating ice using ice_frac
    # from topography. Controlled by config knobs.
    mask_enabled = config.has_option('extrap', 'mask_under_ice')
    if mask_enabled:
        thr = config.getfloat('extrap', 'under_ice_threshold')
        _apply_under_ice_mask_to_file(
            prepared_path=prepared,
            topo_file=topo_file,
            variable=variable,
            threshold=thr,
            logger=logger,
        )

    # Render namelist
    namelist_txt = _render_namelist(
        file_in=prepared,
        horizontal_out=horiz_tmp,
        vertical_out=vert_tmp,
        basin_file=basin_file,
        topo_file=topo_file,
        variable=variable,
    )
    with open(namelist_path, 'w', encoding='utf-8') as nf:
        nf.write(namelist_txt)

    # Run executables sequentially
    for exe, phase in (
        ('i7aof_extrap_horizontal', 'horizontal'),
        ('i7aof_extrap_vertical', 'vertical'),
    ):
        _run_exe_capture(
            exe, namelist_path, log_path, phase=phase, logger=logger
        )

    if not os.path.exists(vert_tmp):
        raise FileNotFoundError(
            f'Expected vertical output missing: {vert_tmp}'
        )
    # can't use read_dataset because the Fortran output doesn't have
    # the required time_bnds variable
    ds_vert = xr.open_dataset(vert_tmp, decode_times=False)
    # Drop the dummy singleton time dimension added for Fortran, but only
    # in the climatology workflow.

    if 'time' in ds_vert.dims:
        # Remove the singleton dim and clean up time variables/bounds
        ds_vert = ds_vert.isel(time=0, drop=True)
        # Drop any leftover scalar coord/variable and common bounds names
        for var in ['time', 'time_bnds']:
            if var in ds_vert.coords or var in ds_vert.data_vars:
                ds_vert = ds_vert.drop_vars([var], errors='ignore')
    _finalize_output_with_grid(
        ds_in=ds_vert,
        config=config,
        final_out_path=out_file,
        variable=variable,
        logger=logger,
        src_attr_path=in_path,
    )
    ds_vert.close()

    if not keep_intermediate:
        try:
            shutil.rmtree(tmp_dir)
        except OSError:
            logger.warning(f'Failed to remove temporary directory {tmp_dir}')
    else:
        logger.info(f'Keeping intermediates in {tmp_dir}')
