"""
Extrapolate remapped CMIP ``ct``/``sa`` fields horizontally and vertically.

Workflow
========
Consumes remapped monthly ``ct``/``sa`` files produced by the remap step:
  workdir/remap/<model>/<scenario>/Omon/ct_sa/*ismip<res>.nc

Produces per input file and per variable vertically extrapolated outputs:
  workdir/extrap/<model>/<scenario>/Omon/ct_sa/*ismip<res>_extrap.nc
with ``ct_sa`` in the filename replaced by the variable name (``ct`` or
``sa``).

Two external Fortran executables are invoked sequentially for each variable:
  * i7aof_extrap_horizontal  (&horizontal_extrapolation namelist)
  * i7aof_extrap_vertical    (&vertical_extrapolation namelist)

A single combined namelist (containing both groups) is rendered from the
Jinja2 template ``namelist_template.nml.j2`` via :func:`load_template_text`.

Supporting data (auto-generated if missing)
------------------------------------------
If required inputs for IMBIE basin masks or topography are absent, the
workflow attempts to build them on the fly inside ``workdir``:

    * IMBIE basins: uses :func:`i7aof.imbie.masks.make_imbie_masks` to produce
        ``imbie/basinNumbers_<res>.nc`` (downloading shapefiles as needed).
    * Topography: constructs the configured dataset (e.g. BedMachine) via
        :func:`i7aof.topo.get_topo`; if the remapped ISMIP file is missing it
        runs ``download_and_preprocess_topo()`` (which may require a manually
        downloaded source file for licensed data) followed by
        ``remap_topo_to_ismip()``.

If a required manual download (e.g. BedMachine original file) is not present
an informative ``FileNotFoundError`` is raised.

Execution is strictly serial for memory safety (no parallelism).

Logging uses :class:`mpas_tools.logging.LoggingContext` so future redirecting
to log files requires no code changes.
"""

import argparse
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Sequence

import xarray as xr
from jinja2 import BaseLoader, Environment
from mpas_tools.config import MpasConfigParser
from mpas_tools.logging import LoggingContext

from i7aof.cmip import get_model_prefix
from i7aof.extrap import load_template_text
from i7aof.grid.ismip import (
    get_horiz_res_string,
    get_ismip_grid_filename,
    get_res_string,
    write_ismip_grid,
)
from i7aof.imbie.masks import make_imbie_masks
from i7aof.io import write_netcdf
from i7aof.topo import get_topo

__all__ = ['extrap_cmip', 'main']


@dataclass
class FileTask:
    """Per-variable extrapolation work for one remapped input file."""

    in_path: str
    out_path: str  # final vertical output
    namelist_path: str  # combined rendered namelist
    variable: str  # e.g. 'ct' or 'sa'
    tmp_dir: str  # directory for all intermediates for this file


def extrap_cmip(
    model: str,
    scenario: str,
    workdir: str | None = None,
    user_config_filename: str | None = None,
    variables: Sequence[str] = ('ct', 'sa'),
    keep_intermediate: bool = False,
) -> None:
    """
    Extrapolate remapped CMIP ``ct``/``sa`` fields horizontally and vertically.

    Workflow
    ========
    Consumes remapped monthly ``ct``/``sa`` files produced by the remap step:
    workdir/remap/<model>/<scenario>/Omon/ct_sa/*ismip<res>.nc

    Produces per input file and per variable vertically extrapolated outputs:
    workdir/extrap/<model>/<scenario>/Omon/ct_sa/*ismip<res>_extrap.nc
    with ``ct_sa`` in the filename replaced by the variable name (``ct`` or
    ``sa``).

    Two external Fortran executables are invoked sequentially for each
    variable:

        * i7aof_extrap_horizontal  (&horizontal_extrapolation namelist)

        * i7aof_extrap_vertical    (&vertical_extrapolation namelist)

    A single combined namelist (containing both groups) is rendered from the
    Jinja2 template ``namelist_template.nml.j2`` via
    :func:`load_template_text`.

    Supporting data (auto-generated if missing)
    ------------------------------------------
    If the IMBIE basin mask (``imbie/basinNumbers_<res>.nc``) or the ISMIP
    remapped topography file is absent, they are generated on demand. IMBIE
    shapefiles are downloaded automatically; topography may require a
    manually obtained source file (e.g. BedMachine) before preprocessing.

    Execution is strictly serial for memory safety (no parallelism).

    Logging uses :class:`mpas_tools.logging.LoggingContext` so future
    redirecting to log files requires no code changes.

    Parameters
    ----------
    model : str
        CMIP model name.
    scenario : str
        Scenario key (e.g., ``historical`` or ``ssp585``).
    workdir : str, optional
        Base working directory; if omitted, uses config ``[workdir] base_dir``.
    user_config_filename : str, optional
        Optional user config overriding defaults.
    variables : sequence of str, optional
        Variable names to extrapolate (default: ``ct`` and ``sa``).
    keep_intermediate : bool, optional
        Keep horizontal temp and namelist files if True.
    """

    (
        config,
        workdir,
        remap_dir,
        out_dir,
        ismip_res_str,
        _model_prefix,
    ) = _prepare_paths_and_config(
        model=model,
        scenario=scenario,
        workdir=workdir,
        user_config_filename=user_config_filename,
    )

    in_files = _collect_remap_outputs(remap_dir, ismip_res_str)
    if not in_files:
        raise FileNotFoundError(
            'No remapped ct/sa files found. Run: ismip7-antarctic-remap-cmip'
        )

    basin_file = _ensure_imbie_masks(config, workdir)
    grid_file = _ensure_ismip_grid(config, workdir)
    topo_file = _ensure_topography(config, workdir)

    for in_file in in_files:
        base = os.path.basename(in_file)
        for var in variables:
            out_base = base.replace('ct_sa', var)
            out_path = os.path.join(out_dir, out_base)
            stem = os.path.splitext(os.path.basename(out_path))[0]
            tmp_dir = os.path.join(os.path.dirname(out_path), f'{stem}_tmp')
            namelist_path = os.path.join(tmp_dir, f'{var}.nml')
            task = FileTask(
                in_path=in_file,
                out_path=out_path,
                namelist_path=namelist_path,
                variable=var,
                tmp_dir=tmp_dir,
            )
            _process_task(
                task,
                config=config,
                basin_file=basin_file,
                grid_file=grid_file,
                topo_file=topo_file,
                keep_intermediate=keep_intermediate,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Extrapolate remapped CMIP ct/sa (horizontal + vertical) using '
            'Fortran executables.'
        )
    )
    parser.add_argument(
        '-m',
        '--model',
        dest='model',
        required=True,
        help='CMIP model name (required).',
    )
    parser.add_argument(
        '-s',
        '--scenario',
        dest='scenario',
        required=True,
        help='Scenario key (historical, ssp585, ...: required).',
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
        help='Keep horizontal intermediate NetCDF and namelist files.',
    )
    args = parser.parse_args()

    extrap_cmip(
        model=args.model,
        scenario=args.scenario,
        workdir=args.workdir,
        user_config_filename=args.config,
        variables=args.variables,
        keep_intermediate=args.keep_intermediate,
    )


def _prepare_paths_and_config(
    model: str,
    scenario: str,
    workdir: str | None,
    user_config_filename: str | None,
):
    model_prefix = get_model_prefix(model)
    config = MpasConfigParser()
    config.add_from_package('i7aof', 'default.cfg')
    config.add_from_package('i7aof.cmip', f'{model_prefix}.cfg')
    if user_config_filename is not None:
        config.add_user_config(user_config_filename)
    if workdir is None:
        if config.has_option('workdir', 'base_dir'):
            workdir = config.get('workdir', 'base_dir')
        else:  # pragma: no cover
            raise ValueError(
                'Missing configuration option: [workdir] base_dir.'
            )
    # At this point workdir must be a concrete string for path joins
    assert workdir is not None, (
        'Internal error: workdir should be resolved to a string'
    )
    remap_dir = os.path.join(
        workdir, 'remap', model, scenario, 'Omon', 'ct_sa'
    )
    out_dir = os.path.join(workdir, 'extrap', model, scenario, 'Omon', 'ct_sa')
    os.makedirs(out_dir, exist_ok=True)
    ismip_res_str = get_res_string(config)
    return config, workdir, remap_dir, out_dir, ismip_res_str, model_prefix


def _collect_remap_outputs(remap_dir: str, ismip_res_str: str) -> List[str]:
    if not os.path.isdir(remap_dir):
        return []
    result: List[str] = []
    for name in sorted(os.listdir(remap_dir)):
        if f'ismip{ismip_res_str}' in name and 'ct_sa' in name:
            result.append(os.path.join(remap_dir, name))
    return result


def _ensure_imbie_masks(config, workdir: str) -> str:
    """Ensure IMBIE basin mask NetCDF exists; build it if missing.

    Returns
    -------
    str
        Path to basin numbers NetCDF file.
    """
    # The output naming in make_imbie_masks is basinNumbers_<res>.nc

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
        if not os.path.exists(basin_file):  # safety check
            raise FileNotFoundError(
                f'Failed to generate IMBIE basin file: {basin_file}'
            )
    return basin_file


def _ensure_topography(config, workdir: str) -> str:
    """Ensure topography file on ISMIP grid exists; build it if missing.

    Returns
    -------
    str
        Path to the ISMIP-grid topography NetCDF file.
    """
    logger = logging.getLogger(__name__)
    cwd = os.getcwd()
    try:
        os.makedirs(os.path.join(workdir, 'topo'), exist_ok=True)
        os.chdir(workdir)
        topo_obj = get_topo(config, logger)
        topo_path = topo_obj.get_topo_on_ismip_path()
        if not os.path.exists(topo_path):
            # Need intermediate preprocessed file first
            try:
                topo_obj.download_and_preprocess_topo()
            except FileNotFoundError as e:
                # Provide actionable message then re-raise
                raise FileNotFoundError(
                    f'Topography prerequisite missing: {e}. '
                    'Please fetch required source data (see docs).'
                ) from e
            topo_obj.remap_topo_to_ismip()
        if not os.path.exists(topo_path):
            raise FileNotFoundError(
                f'Failed to build topography file: {topo_path}'
            )
    finally:
        os.chdir(cwd)
    return os.path.join(workdir, topo_path)


def _ensure_ismip_grid(config, workdir: str) -> str:
    """Ensure ISMIP grid exists under workdir and return absolute path."""
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
        if not os.path.exists(grid_abs):
            raise FileNotFoundError(
                f'Failed to generate ISMIP grid file: {grid_abs}'
            )
    return grid_abs


def _render_namelist(
    file_in: str,
    horizontal_out: str,
    vertical_out: str,
    basin_file: str,
    topo_file: str,
    variable: str,
) -> str:
    template_txt = load_template_text()
    env = Environment(
        loader=BaseLoader(),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.from_string(template_txt)
    return template.render(
        file_in=file_in,
        horizontal_out=horizontal_out,
        vertical_out=vertical_out,
        file_basin=basin_file,
        file_topo=topo_file,
        var_name=variable,
        z_name='z_extrap',
    )


def _process_task(
    task: FileTask,
    config: MpasConfigParser,
    basin_file: str,
    grid_file: str,
    topo_file: str,
    keep_intermediate: bool,
) -> None:
    if os.path.exists(task.out_path):
        print(f'Extrapolated file exists, skipping: {task.out_path}')
        return
    os.makedirs(os.path.dirname(task.out_path), exist_ok=True)
    os.makedirs(task.tmp_dir, exist_ok=True)

    # We'll preprocess per time chunk to keep memory and file sizes in check
    # Determine chunking in time (default 120 months)
    try:
        time_chunk = config.getint('extrap_cmip', 'time_chunk')
    except Exception:
        # Fallback safety: default to 120 if section/option is missing
        time_chunk = 120

    with LoggingContext(__name__):
        logger = logging.getLogger(__name__)
        logger.info(
            f'Starting extrapolation (chunked): {task.in_path} '
            f'(variable={task.variable}, chunk={time_chunk})'
        )

        horiz_exec = 'i7aof_extrap_horizontal'
        vert_exec = 'i7aof_extrap_vertical'
        # Verify required executables exist once
        for exe in (horiz_exec, vert_exec):
            if shutil.which(exe) is None:
                raise FileNotFoundError(
                    f"Required executable '{exe}' not found on PATH."
                )

        # Open source input lazily to compute chunk indices
        ds_meta = xr.open_dataset(task.in_path, decode_times=False)
        if 'time' not in ds_meta.dims:
            # No time dimension; process as a single synthetic chunk
            time_indices = [(0, 1)]
        else:
            n_time = ds_meta.sizes['time']
            time_indices = [
                (i0, min(i0 + time_chunk, n_time))
                for i0 in range(0, n_time, time_chunk)
            ]

        vertical_chunks: List[str] = []

        for i0, i1 in time_indices:
            # Prepare per-chunk input file
            if i1 == 0:
                # Empty time dimension; nothing to do
                continue
            input_chunk = os.path.join(task.tmp_dir, f'input_{i0}_{i1}.nc')
            if not os.path.exists(input_chunk):
                # Preprocess only this time slice with grid coordinates
                _prepare_input_with_coords(
                    task.in_path,
                    grid_file,
                    input_chunk,
                    task.variable,
                    time_slice=(i0, i1) if 'time' in ds_meta.dims else None,
                )

            horizontal_tmp = os.path.join(
                task.tmp_dir, f'horizontal_{i0}_{i1}.nc'
            )
            vertical_tmp = os.path.join(task.tmp_dir, f'vertical_{i0}_{i1}.nc')

            # Render a per-chunk namelist
            namelist_contents = _render_namelist(
                file_in=input_chunk,
                horizontal_out=horizontal_tmp,
                vertical_out=vertical_tmp,
                basin_file=basin_file,
                topo_file=topo_file,
                variable=task.variable,
            )
            namelist_path = os.path.join(
                task.tmp_dir, f'{task.variable}_{i0}_{i1}.nml'
            )
            with open(namelist_path, 'w', encoding='utf-8') as f:
                f.write(namelist_contents)

            logger.info(
                '  Chunk %d:%d -> horiz %s, vert %s',
                i0,
                i1,
                os.path.basename(horizontal_tmp),
                os.path.basename(vertical_tmp),
            )

            # Run horizontal and vertical phases if outputs are missing
            if not os.path.exists(horizontal_tmp):
                _run_exe(horiz_exec, namelist_path, logger, 'horizontal')
            if not os.path.exists(vertical_tmp):
                _run_exe(vert_exec, namelist_path, logger, 'vertical')

            if not os.path.exists(vertical_tmp):
                raise FileNotFoundError(
                    f'Expected vertical output missing: {vertical_tmp}'
                )
            vertical_chunks.append(vertical_tmp)

        if not vertical_chunks:
            raise RuntimeError(
                'No vertical output chunks were produced; aborting.'
            )

        # Concatenate chunks along time in-memory (lazy) and finalize once
        if len(vertical_chunks) == 1:
            ds_final_in = xr.open_dataset(
                vertical_chunks[0], decode_times=False
            )
        else:
            ds_list = [
                xr.open_dataset(path, decode_times=False)
                for path in vertical_chunks
            ]
            ds_final_in = xr.concat(ds_list, dim='time', join='exact')

        _finalize_output_with_grid_ds(
            ds_final_in,
            grid_file,
            task.out_path,
            task.variable,
            logger,
        )

        if not keep_intermediate:
            _cleanup_intermediate(task, logger)
        else:
            logger.info('Keeping intermediate files and namelists.')


def _run_exe(exe: str, namelist: str, logger, phase: str) -> None:
    # Fortran executables expect the namelist file path as argv(1)
    logger.info(f'Running {phase} extrapolation executable: {exe} {namelist}')
    try:
        proc = subprocess.run(
            [exe, namelist],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )
        logger.info(f'{exe} output:\n{proc.stdout.strip()}')
    except subprocess.CalledProcessError as e:  # pragma: no cover
        logger.error(
            f'Execution failed for {exe} (phase={phase}) rc={e.returncode}'
        )
        logger.error(f'Combined stdout/stderr:\n{e.stdout}')
        raise


def _cleanup_intermediate(task: FileTask, logger) -> None:
    # Remove the entire temp directory safely
    tmp_dir = task.tmp_dir
    out_parent = os.path.dirname(task.out_path)
    safe_parent = os.path.dirname(tmp_dir) == out_parent
    safe_suffix = os.path.basename(tmp_dir).endswith('_tmp')
    if os.path.isdir(tmp_dir) and safe_parent and safe_suffix:
        try:
            shutil.rmtree(tmp_dir)
            logger.info(f'Removed temporary directory: {tmp_dir}')
        except OSError as e:  # pragma: no cover
            logger.warning(
                f'Failed to remove temporary directory {tmp_dir}: {e}'
            )


def _prepare_input_with_coords(
    in_path: str,
    grid_path: str,
    out_prepared_path: str,
    var_name: str,
    time_slice: tuple[int, int] | None = None,
) -> None:
    """Ensure input has x/y coordinates; add from ISMIP grid if missing.

    Always writes a prepared copy to out_prepared_path.
    """
    if os.path.exists(out_prepared_path):
        print(
            f'Prepared input file exists, not rewriting: {out_prepared_path}'
        )
        return

    ds_in = xr.open_dataset(in_path, chunks={'time': 1}, decode_times=False)
    if time_slice is not None and 'time' in ds_in.dims:
        i0, i1 = time_slice
        ds_in = ds_in.isel(time=slice(i0, i1))
    ds_grid = xr.open_dataset(grid_path, decode_times=False)

    to_add = {}
    # Ensure dims exist and match sizes
    for dim in ('x', 'y'):
        if dim not in ds_in.dims:
            raise KeyError(
                f"Input file {in_path} missing required dimension '{dim}'."
            )
        if dim in ds_in.variables:
            continue
        # Use grid coordinates; basic size check
        if ds_in.sizes[dim] != ds_grid.sizes[dim]:
            raise ValueError(
                f"Dim size mismatch for '{dim}': input={ds_in.sizes[dim]} "
                f'grid={ds_grid.sizes[dim]}'
            )
        to_add[dim] = ds_grid[dim]

    if to_add:
        ds_in = ds_in.assign(to_add)

    # Keep only variables needed by the Fortran tools: the target variable
    # and essential coordinate variables (time, x, y, z or z_extrap)
    keep_vars = {var_name, 'time', 'x', 'y'}
    if 'z_extrap' in ds_in.variables:
        keep_vars.add('z_extrap')
    elif 'z' in ds_in.variables:
        keep_vars.add('z')

    drop_list = [v for v in ds_in.variables if v not in keep_vars]
    if drop_list:
        ds_in = ds_in.drop_vars(drop_list, errors='ignore')

    # Ensure the target variable gets a fill value; others do not
    write_netcdf(
        ds_in,
        out_prepared_path,
        has_fill_values=lambda name, var: name == var_name,
        progress_bar=True,
    )


def _finalize_output_with_grid_ds(
    ds_in: xr.Dataset,
    grid_path: str,
    final_out_path: str,
    variable: str,
    logger,
) -> None:
    """Finalize from an xarray Dataset by injecting grid vars and writing.

    This variant avoids creating a large concatenated temporary file by
    operating directly on a (possibly lazily concatenated) Dataset.
    """
    ds_out = ds_in
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

    drop_list = [v for v in (coord_names + var_names) if v in ds_out.variables]
    if drop_list:
        ds_out = ds_out.drop_vars(drop_list, errors='ignore')

    to_add_coords = {v: ds_grid[v] for v in coord_names if v in ds_grid}
    to_add_vars = {v: ds_grid[v] for v in var_names if v in ds_grid}

    if to_add_coords:
        names = ', '.join(sorted(to_add_coords.keys()))
        logger.info(f'Overwriting grid coordinates in output: {names}')
        ds_out = ds_out.assign_coords(to_add_coords)

    if to_add_vars:
        names = ', '.join(sorted(to_add_vars.keys()))
        logger.info(f'Overwriting grid variables in output: {names}')
        ds_out = ds_out.assign(to_add_vars)

    write_netcdf(
        ds_out,
        final_out_path,
        has_fill_values=lambda name, var: name == variable,
        progress_bar=True,
    )
