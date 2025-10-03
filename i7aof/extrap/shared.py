"""Shared helpers for CMIP and climatology extrapolation workflows.

This module factors out logic from the CMIP extrapolation workflow so a
climatology (no time dimension) workflow can reuse the same building
blocks:

    * Ensuring prerequisite datasets (ISMIP grid, IMBIE basins, topography)
    * Rendering the combined Fortran namelist (horizontal + vertical)
    * Preparing an input file with required coordinates/variables
    * Invoking the external Fortran executables while capturing logs
    * Finalizing outputs by injecting grid variables

The CMIP workflow adds time chunking / parallelism on top of these core
operations; the climatology workflow just runs a single (no-time) pass.
"""

import logging
import os
import shutil
import subprocess

import xarray as xr
from dask import config as dask_config
from jinja2 import BaseLoader, Environment
from mpas_tools.config import MpasConfigParser

from i7aof.extrap import load_template_text
from i7aof.grid.ismip import (
    get_horiz_res_string,
    get_ismip_grid_filename,
    write_ismip_grid,
)
from i7aof.imbie.masks import make_imbie_masks
from i7aof.io import write_netcdf
from i7aof.topo import get_topo

__all__ = [
    '_ensure_ismip_grid',
    '_ensure_imbie_masks',
    '_ensure_topography',
    '_render_namelist',
    '_prepare_input_single',
    '_run_exe_capture',
    '_finalize_output_with_grid',
]


def _ensure_imbie_masks(config: MpasConfigParser, workdir: str) -> str:
    """Ensure IMBIE basin mask file exists under ``workdir`` and return it."""
    res = get_horiz_res_string(config)
    basin_file = os.path.join(workdir, 'imbie', f'basinNumbers_{res}.nc')
    if not os.path.exists(basin_file):
        cwd = os.getcwd()
        try:
            os.makedirs(os.path.join(workdir, 'imbie'), exist_ok=True)
            os.chdir(workdir)
            make_imbie_masks(config)
        finally:
            os.chdir(cwd)
        if not os.path.exists(basin_file):
            raise FileNotFoundError(
                f'Failed to generate IMBIE basin file: {basin_file}'
            )
    return basin_file


def _ensure_topography(config: MpasConfigParser, workdir: str) -> str:
    """Ensure topography on ISMIP grid exists; build if missing."""
    logger = logging.getLogger(__name__)
    cwd = os.getcwd()
    try:
        os.makedirs(os.path.join(workdir, 'topo'), exist_ok=True)
        os.chdir(workdir)
        topo_obj = get_topo(config, logger)
        topo_path = topo_obj.get_topo_on_ismip_path()
        if not os.path.exists(topo_path):
            try:
                topo_obj.download_and_preprocess_topo()
            except FileNotFoundError as e:  # pragma: no cover
                raise FileNotFoundError(
                    f'Topography prerequisite missing: {e}. '
                    'Please fetch required source data.'
                ) from e
            topo_obj.remap_topo_to_ismip()
        if not os.path.exists(topo_path):  # pragma: no cover
            raise FileNotFoundError(
                f'Failed to build topography file: {topo_path}'
            )
    finally:
        os.chdir(cwd)
    return os.path.join(workdir, topo_path)


def _ensure_ismip_grid(config: MpasConfigParser, workdir: str) -> str:
    grid_rel = get_ismip_grid_filename(config)
    grid_abs = os.path.join(workdir, grid_rel)
    if not os.path.exists(grid_abs):
        cwd = os.getcwd()
        try:
            os.makedirs(os.path.dirname(grid_abs), exist_ok=True)
            os.chdir(workdir)
            write_ismip_grid(config)
        finally:
            os.chdir(cwd)
        if not os.path.exists(grid_abs):  # pragma: no cover
            raise FileNotFoundError(
                f'Failed to generate ISMIP grid file: {grid_abs}'
            )
    return grid_abs


def _render_namelist(
    *,
    file_in: str,
    horizontal_out: str,
    vertical_out: str,
    basin_file: str,
    topo_file: str,
    variable: str,
    z_name: str = 'z_extrap',
) -> str:
    """Render the combined Fortran namelist for one (no-time) run."""
    template_txt = load_template_text()
    env = Environment(
        loader=BaseLoader(),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.from_string(template_txt)
    rendered = template.render(
        file_in=file_in,
        horizontal_out=horizontal_out,
        vertical_out=vertical_out,
        file_basin=basin_file,
        file_topo=topo_file,
        var_name=variable,
        z_name=z_name,
    )
    if not rendered.endswith('\n'):
        rendered += '\n'
    return rendered


def _prepare_input_single(
    *,
    in_path: str,
    grid_path: str,
    out_prepared_path: str,
    variable: str,
    logger: logging.Logger,
    add_dummy_time: bool = False,
) -> None:
    """Prepare a single input file for Fortran executables.

    If ``add_dummy_time`` is True and no ``time`` dimension exists, a
    size-1 ``time`` dimension and coordinate variable are added with
    simple CF-compatible attributes so the legacy Fortran codes (which
    currently assume the presence of a time dimension) can operate.
    The climatology driver later drops this singleton dimension from
    the final product so user-facing files remain time-less.
    """
    ds_in = xr.open_dataset(in_path, decode_times=False)
    ds_grid = xr.open_dataset(grid_path, decode_times=False)

    for dim in ('x', 'y'):
        if dim not in ds_in.dims:
            raise KeyError(
                f"Input file {in_path} missing required dimension '{dim}'."
            )

    # inject x/y coordinates if absent
    inject = {}
    for coord in ('x', 'y'):
        if coord not in ds_in.variables:
            inject[coord] = ds_grid[coord]
    if inject:
        ds_in = ds_in.assign_coords(inject)

    if variable not in ds_in:
        raise KeyError(
            f"Variable '{variable}' not found in {in_path}. Available: "
            f'{", ".join(sorted(ds_in.data_vars))}'
        )

    keep = {variable, 'x', 'y'}
    if 'z_extrap' in ds_in:
        keep.add('z_extrap')
    elif 'z' in ds_in:
        keep.add('z')
    drop_vars = [v for v in ds_in.variables if v not in keep]
    if drop_vars:
        ds_in = ds_in.drop_vars(drop_vars, errors='ignore')

    # Optionally add a dummy time dimension (length 1) with minimal
    # metadata so the Fortran executables (which expect time) succeed.
    # We place 'time' as the most slowly varying dimension (first in
    # NetCDF / xarray order) to match the ordering used by CMIP inputs:
    # (time, z_extrap, y, x). Fortran then indexes in reverse memory
    # order with its (x, y, z, time) expectations.
    if add_dummy_time and 'time' not in ds_in.dims:
        ds_in = ds_in.expand_dims({'time': [0]})
        # Provide required attributes accessed by the Fortran codes
        ds_in['time'].attrs.setdefault('units', 'days since 0001-01-01')
        ds_in['time'].attrs.setdefault('calendar', 'gregorian')
        # Ensure a history attribute exists (Fortran reads it)
        ds_in.attrs.setdefault(
            'history',
            'Added dummy time dimension (length 1) for extrapolation.',
        )
        # Reorder the target variable to (time, z, y, x) in file order so
        # that time is the slowest-varying dimension, consistent with
        # CMIP workflow outputs.
        vert_name = None
        for cand in ('z_extrap', 'z'):
            if cand in ds_in[variable].dims:
                vert_name = cand
                break
        if vert_name is None:
            raise ValueError(
                f"Could not identify vertical dimension for '{variable}' "
                'to order dimensions before extrapolation.'
            )
        # Build target order ensuring all required dims exist
        target_order = ['time', vert_name, 'y', 'x']
        # Some datasets may already have a different ordering; transpose safely
        present = [d for d in target_order if d in ds_in[variable].dims]
        # Only transpose if full target set is present
        if set(present) == set(ds_in[variable].dims):
            ds_in[variable] = ds_in[variable].transpose(*target_order)

    logger.info(
        'Prepared single input for variable '
        f"'{variable}' -> "
        f'{out_prepared_path}'
    )
    tmp_out = f'{out_prepared_path}.tmp'
    try:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
    except Exception:  # pragma: no cover - best effort
        pass
    with dask_config.set(scheduler='synchronous'):
        write_netcdf(
            ds_in,
            tmp_out,
            has_fill_values=lambda name, var: name == variable,
            format='NETCDF4',
            engine='netcdf4',
            progress_bar=False,
        )
    os.replace(tmp_out, out_prepared_path)


def _run_exe_capture(
    exe: str, namelist: str, log_path: str, phase: str, logger: logging.Logger
) -> None:
    """Run a Fortran executable capturing output."""
    if shutil.which(exe) is None:
        raise FileNotFoundError(f"Executable '{exe}' not found on PATH.")
    cmd = [exe, namelist]
    if shutil.which('stdbuf') is not None:
        cmd = ['stdbuf', '-o0', '-e0'] + cmd
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('OPENBLAS_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')
    with open(log_path, 'a', encoding='utf-8') as lf:
        lf.write(f'== Phase: {phase} ==\n')
        lf.write(f'Command: {" ".join(cmd)}\n')
        lf.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
        proc.stdout.close()
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


def _finalize_output_with_grid(
    *,
    ds_in: xr.Dataset,
    grid_path: str,
    final_out_path: str,
    variable: str,
    logger: logging.Logger,
    drop_singleton_time: bool = False,
) -> None:
    """Finalize a (single) vertical output by injecting grid variables."""
    ds_grid = xr.open_dataset(grid_path, decode_times=False)
    coord_names = ['x', 'y']
    var_names = [
        'x_bnds',
        'y_bnds',
        'lat',
        'lon',
        'lat_bnds',
        'lon_bnds',
        'crs',
    ]
    ds_out = ds_in
    drop_list = [v for v in (coord_names + var_names) if v in ds_out]
    if drop_list:
        ds_out = ds_out.drop_vars(drop_list, errors='ignore')
    to_add_coords = {v: ds_grid[v] for v in coord_names if v in ds_grid}
    to_add_vars = {v: ds_grid[v] for v in var_names if v in ds_grid}
    if to_add_coords:
        ds_out = ds_out.assign_coords(to_add_coords)
    if to_add_vars:
        ds_out = ds_out.assign(to_add_vars)

    # Optionally remove a dummy singleton time dimension that may have
    # been added during preparation for legacy Fortran executables.
    if (
        drop_singleton_time
        and 'time' in ds_out.dims
        and ds_out.sizes.get('time', 0) == 1
    ):
        ds_out = ds_out.isel(time=0, drop=True)
        for tb in ('time_bnds', 'time_bounds'):
            if tb in ds_out:
                ds_out = ds_out.drop_vars(tb, errors='ignore')
    logger.info(f'Writing extrapolated output: {final_out_path}')
    write_netcdf(
        ds_out,
        final_out_path,
        has_fill_values=lambda name, var: name == variable,
        progress_bar=True,
    )
