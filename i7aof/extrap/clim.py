"""Extrapolate a remapped climatology (no time dimension) to depth levels.

This mirrors the CMIP extrapolation workflow but simplifies it to a single
pass per variable because the climatology file has no time dimension. It
reuses shared helpers in ``i7aof.extrap.shared`` for prerequisite data,
namelist rendering, executable invocation, and finalization.

Expected input (produced by ``ismip7-antarctic-remap-clim``):

    workdir/remap/climatology/<name>/<file>_ismip<res>.nc

Outputs (per variable) are written to:

    workdir/extrap/climatology/<name>/<file>_ismip<res>_<var>_extrap.nc

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

from i7aof.extrap.shared import (
    _ensure_imbie_masks,
    _ensure_ismip_grid,
    _ensure_topography,
    _finalize_output_with_grid,
    _prepare_input_single,
    _render_namelist,
    _run_exe_capture,
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
    config = MpasConfigParser()
    config.add_from_package('i7aof', 'default.cfg')
    config.add_from_package('i7aof.clim', f'{clim_name}.cfg')
    if user_config_filename is not None:
        config.add_user_config(user_config_filename)

    if workdir is None:
        if config.has_option('workdir', 'base_dir'):
            workdir = config.get('workdir', 'base_dir')
        else:  # pragma: no cover
            raise ValueError(
                'Missing configuration option: [workdir] base_dir.'
            )
    assert workdir is not None

    # Locate remapped input file
    remap_dir = os.path.join(workdir, 'remap', 'climatology', clim_name)
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

    out_dir = os.path.join(workdir, 'extrap', 'climatology', clim_name)
    os.makedirs(out_dir, exist_ok=True)

    basin_file = _ensure_imbie_masks(config, workdir)
    grid_file = _ensure_ismip_grid(config, workdir)
    topo_file = _ensure_topography(config, workdir)

    with LoggingContext(__name__):
        logger = logging.getLogger(__name__)
        logger.info(
            f'Starting climatology extrapolation for {clim_name}: {in_path}'
        )

        for var in variables:
            if var not in xr.open_dataset(in_path, decode_times=False):
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
            if os.path.exists(out_file):
                logger.info(f'Output exists, skipping: {out_file}')
                continue
            tmp_dir = os.path.join(out_dir, f'{stem_var}_tmp')
            os.makedirs(tmp_dir, exist_ok=True)
            prepared = os.path.join(tmp_dir, f'input_{var}.nc')
            horiz_tmp = os.path.join(tmp_dir, f'horizontal_{var}.nc')
            vert_tmp = os.path.join(tmp_dir, f'vertical_{var}.nc')
            namelist_path = os.path.join(tmp_dir, f'{var}.nml')
            log_path = os.path.join(tmp_dir, 'logs', f'{var}.log')

            # Prepare single input
            _prepare_input_single(
                in_path=in_path,
                grid_path=grid_file,
                out_prepared_path=prepared,
                variable=var,
                logger=logger,
                # add dummy singleton time so Fortran expectation is met
                add_dummy_time=True,
            )

            # Render namelist
            namelist_txt = _render_namelist(
                file_in=prepared,
                horizontal_out=horiz_tmp,
                vertical_out=vert_tmp,
                basin_file=basin_file,
                topo_file=topo_file,
                variable=var,
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
            ds_vert = xr.open_dataset(vert_tmp, decode_times=False)
            _finalize_output_with_grid(
                ds_in=ds_vert,
                grid_path=grid_file,
                final_out_path=out_file,
                variable=var,
                logger=logger,
                # drop the dummy time dimension from final output
                drop_singleton_time=True,
            )
            ds_vert.close()

            if not keep_intermediate:
                try:
                    shutil.rmtree(tmp_dir)
                except OSError:
                    logger.warning(
                        f'Failed to remove temporary directory {tmp_dir}'
                    )
            else:
                logger.info(f'Keeping intermediates in {tmp_dir}')


def main():
    parser = argparse.ArgumentParser(
        description='Extrapolate remapped climatology ct/sa (no time).'
    )
    parser.add_argument(
        '-n',
        '--clim_name',
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
