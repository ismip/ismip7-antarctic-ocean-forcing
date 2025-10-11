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

Parallel, chunked execution
---------------------------
Time is processed in chunks to limit memory and file sizes. Chunks run
serially or in parallel (process-based) depending on configuration. Each
chunk writes a per-chunk input, runs the Fortran executables, and records
stdout/stderr and any Python traceback in a chunk log under
``<out>_tmp/logs``. Abrupt worker failures are detected and reported with
the set of completed vs. pending chunks for quick triage.

Supporting data (auto-generated if missing)
------------------------------------------
If required inputs for IMBIE basin masks or topography are absent, the
workflow attempts to build them on the fly inside ``workdir``:

    * IMBIE basins: uses :func:`i7aof.imbie.masks.make_imbie_masks` to
      produce ``imbie/basinNumbers_<res>.nc`` (downloading shapefiles as
      needed).

    * Topography: constructs the configured dataset (e.g. BedMachine)
        via :func:`i7aof.topo.get_topo`; if the remapped ISMIP file is
        missing it runs ``download_and_preprocess_topo()`` (which may
        require a manually downloaded source file for licensed data)
        followed by``remap_topo_to_ismip()``.

If a required manual download (e.g., BedMachine source) is not present
an informative ``FileNotFoundError`` is raised.
"""

import argparse
import faulthandler
import json
import logging
import multiprocessing as mp
import os
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import xarray as xr
from dask import config as dask_config
from mpas_tools.config import MpasConfigParser
from mpas_tools.logging import LoggingContext

from i7aof.cmip import get_model_prefix
from i7aof.extrap.shared import (
    _apply_under_ice_mask_to_file,
    _ensure_imbie_masks,
    _ensure_ismip_grid,
    _ensure_topography,
    _finalize_output_with_grid,
    _render_namelist,
    _run_exe_capture,
    _vertically_resample_to_coarse_ismip_grid,
)
from i7aof.grid.ismip import get_res_string
from i7aof.io import write_netcdf

__all__ = ['extrap_cmip', 'main']

# Relax HDF5 file locking on parallel filesystems to reduce I/O errors when
# writing from worker processes (safe on GPFS/Lustre; ignored otherwise).
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')


@dataclass
class FileTask:
    """Per-variable extrapolation work for one remapped input file."""

    in_path: str
    out_path: str  # final vertical output
    namelist_path: str  # combined rendered namelist
    variable: str  # e.g. 'ct' or 'sa'
    tmp_dir: str  # directory for all intermediates for this file


@dataclass
class ChunkFailed(Exception):
    """Represents a failure in a specific time chunk, with context."""

    i0: int
    i1: int
    log_path: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - formatting helper
        return (
            f'Chunk {self.i0}:{self.i1} failed: {self.message} '
            f'(see log: {self.log_path})'
        )


def extrap_cmip(
    model: str,
    scenario: str,
    workdir: str | None = None,
    user_config_filename: str | None = None,
    variables: Sequence[str] = ('ct', 'sa'),
    keep_intermediate: bool = False,
    num_workers: int | str | None = None,
) -> None:
    """
    Extrapolate remapped CMIP ``ct``/``sa`` fields horizontally and
    vertically in time chunks, optionally in parallel.

    Overview
    --------
    For each input file under
    ``workdir/remap/<model>/<scenario>/Omon/ct_sa/*ismip<res>.nc``, this
    function:

      1. Splits the time dimension into chunks (``extrap_cmip.time_chunk``).

      2. For each chunk, writes a per-chunk input on the ISMIP grid with
         coordinates ensured and only required variables kept.

      3. Invokes the Fortran executables sequentially
         (horizontal then vertical) using a rendered namelist.

      4. Captures Fortran stdout/stderr in a per-chunk log and appends a
         Python traceback on any error for that chunk.

      5. Concatenates vertical outputs along time and injects grid
         coordinates/variables into the final output file.

    Execution is process-parallel with a configurable number of workers.
    When a worker fails, the failing chunk indices and log path are
    reported; if the pool crashes, the set of completed vs. pending chunks
    is logged to direct debugging. Dask is required and used with a
    synchronous scheduler during per-chunk input writes to avoid
    multi-threaded HDF5 access. BLAS/OMP threads are pinned to 1 for each
    worker. HDF5 file locking is relaxed by default.

    Parameters
    ----------
    model : str
        CMIP model name.
    scenario : str
        Scenario key (e.g., ``historical`` or ``ssp585``).
    workdir : str, optional
        Base working directory; if omitted, uses config
        ``[workdir] base_dir``.
    user_config_filename : str, optional
        Optional user config overriding defaults.
    variables : sequence of str, optional
        Variable names to extrapolate (default: ``ct`` and ``sa``).
    keep_intermediate : bool, optional
        Keep horizontal temps and namelists if True; otherwise delete the
        ``*_tmp`` directory after finalization.
    num_workers : int | str | None, optional
        Number of parallel workers. Pass an integer, or ``"auto"``/``0``
        to use ``cpu_count - 1``. If omitted, read from config
        ``[extrap_cmip] num_workers``.
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
                num_workers_override=num_workers,
                workdir=workdir,
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
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        default=None,
        help=(
            'Number of parallel workers (processes). Use "auto" or 0 to '
            'leave one core free. Default: from config (extrap_cmip).'
        ),
    )
    args = parser.parse_args()

    extrap_cmip(
        model=args.model,
        scenario=args.scenario,
        workdir=args.workdir,
        user_config_filename=args.config,
        variables=args.variables,
        keep_intermediate=args.keep_intermediate,
        num_workers=args.num_workers,
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
    # Persist workdir into config for downstream consumers (path resolution)
    config.set('workdir', 'base_dir', workdir)
    remap_dir = os.path.join(
        workdir, 'remap', model, scenario, 'Omon', 'ct_sa'
    )
    out_dir = os.path.join(workdir, 'extrap', model, scenario, 'Omon', 'ct_sa')
    os.makedirs(out_dir, exist_ok=True)
    ismip_res_str = get_res_string(config, extrap=True)
    return config, workdir, remap_dir, out_dir, ismip_res_str, model_prefix


def _collect_remap_outputs(remap_dir: str, ismip_res_str: str) -> List[str]:
    if not os.path.isdir(remap_dir):
        return []
    result: List[str] = []
    for name in sorted(os.listdir(remap_dir)):
        if f'ismip{ismip_res_str}' in name and 'ct_sa' in name:
            result.append(os.path.join(remap_dir, name))
    return result


def _process_task(
    task: FileTask,
    config: MpasConfigParser,
    basin_file: str,
    grid_file: str,
    topo_file: str,
    keep_intermediate: bool,
    num_workers_override: int | str | None = None,
    workdir: str | None = None,
) -> None:
    # Ensure the extrapolated file exists (run if missing)
    _ensure_extrapolated_file(
        task=task,
        config=config,
        basin_file=basin_file,
        grid_file=grid_file,
        topo_file=topo_file,
        keep_intermediate=keep_intermediate,
        num_workers_override=num_workers_override,
    )

    # After extrapolation (or if it already existed), conservatively resample
    # Append results to a Zarr store per time chunk, then convert to NetCDF.
    vert_time_chunk = config.getint('extrap_cmip', 'time_chunk_resample')
    # Determine final NetCDF and Zarr store paths
    res_extrap = get_res_string(config, extrap=True)
    res_final = get_res_string(config, extrap=False)
    out_nc = task.out_path.replace(f'ismip{res_extrap}', f'ismip{res_final}')
    if os.path.abspath(out_nc) == os.path.abspath(task.out_path):
        stem, ext = os.path.splitext(task.out_path)
        out_nc = f'{stem}_z{ext}'
    out_dir_nc = os.path.dirname(out_nc)
    base = os.path.splitext(os.path.basename(out_nc))[0]
    zarr_store = os.path.join(out_dir_nc, f'{base}.zarr')
    _vertically_resample_to_coarse_ismip_grid(
        in_path=task.out_path,
        grid_file=grid_file,
        variable=task.variable,
        config=config,
        out_nc=out_nc,
        time_chunk=vert_time_chunk,
        zarr_store=zarr_store,
        logger=logging.getLogger(__name__),
    )

    # ds_meta closed automatically by context manager above


def _execute_time_chunks(
    *,
    task: FileTask,
    grid_file: str,
    basin_file: str,
    topo_file: str,
    time_indices: List[Tuple[int, int]],
    num_workers: int,
    has_time: bool,
    logger,
    mask_enabled: bool,
    mask_threshold: float,
) -> List[Tuple[int, int, str]]:
    """Execute time chunks serially or in parallel and return outputs."""
    vertical_chunks: List[Tuple[int, int, str]] = []

    if num_workers <= 1:
        for i0, i1 in time_indices:
            if i1 == 0:
                continue
            logger.info(f'  Submitting chunk {i0}:{i1} (serial)')
            tup = _run_chunk_worker(
                i0=i0,
                i1=i1,
                in_path=task.in_path,
                grid_file=grid_file,
                basin_file=basin_file,
                topo_file=topo_file,
                variable=task.variable,
                tmp_dir=task.tmp_dir,
                has_time=has_time,
                mask_enabled=mask_enabled,
                mask_threshold=mask_threshold,
            )
            vertical_chunks.append(tup)
            logger.info(
                f'  Completed chunk {i0}:{i1} -> {os.path.basename(tup[2])}'
            )
        return vertical_chunks

    # Parallel execution
    logger.info(
        f'Launching {min(num_workers, len(time_indices))} workers for '
        f'{len(time_indices)} chunks'
    )
    futures = []
    fut_to_idx: dict = {}
    with ProcessPoolExecutor(
        max_workers=num_workers, mp_context=mp.get_context('spawn')
    ) as ex:
        for i0, i1 in time_indices:
            if i1 == 0:
                continue
            logger.info(f'  Submitting chunk {i0}:{i1}')
            fut = ex.submit(
                _run_chunk_worker,
                i0=i0,
                i1=i1,
                in_path=task.in_path,
                grid_file=grid_file,
                basin_file=basin_file,
                topo_file=topo_file,
                variable=task.variable,
                tmp_dir=task.tmp_dir,
                has_time=has_time,
                mask_enabled=mask_enabled,
                mask_threshold=mask_threshold,
            )
            futures.append(fut)
            fut_to_idx[fut] = (i0, i1)
        try:
            for fut in as_completed(futures):
                try:
                    tup = fut.result()
                except ChunkFailed as ce:
                    for f in futures:
                        f.cancel()
                    logger.error(str(ce))
                    raise
                except Exception as e:
                    # Unexpected worker exception; include chunk indices
                    # if known
                    i0i1 = fut_to_idx.get(fut, ('?', '?'))
                    for f in futures:
                        f.cancel()
                    logger.error(
                        f'Chunk {i0i1[0]}:{i0i1[1]} raised '
                        f'{type(e).__name__}: {e}'
                    )
                    raise
                else:
                    vertical_chunks.append(tup)
                    logger.info(
                        f'  Completed chunk {tup[0]}:{tup[1]} -> '
                        f'{os.path.basename(tup[2])}'
                    )
        except BrokenProcessPool as e:
            # Infer completed chunks from presence of output files
            done_set = set()
            for i0, i1 in time_indices:
                status_path = _status_path(task.tmp_dir, task.variable, i0, i1)
                status = _read_status(status_path)
                if status.get('vertical', False):
                    done_set.add((i0, i1))
            pending_set = set(time_indices) - done_set
            logger.error(
                'Worker pool crashed: '
                f'{e}. Completed: {sorted(done_set)}; '
                f'pending/unknown: {sorted(pending_set)}. '
                'Check per-chunk logs under '
                f'{os.path.join(task.tmp_dir, "logs")}.'
            )
            raise
    return vertical_chunks


def _run_chunk_worker(
    *,
    i0: int,
    i1: int,
    in_path: str,
    grid_file: str,
    basin_file: str,
    topo_file: str,
    variable: str,
    tmp_dir: str,
    has_time: bool,
    mask_enabled: bool,
    mask_threshold: float,
) -> Tuple[int, int, str]:
    """Process a single time chunk and return (i0, i1, vertical_tmp_path)."""
    log_dir = os.path.join(tmp_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{variable}_t{i0}-{i1}.log')
    status_path = _status_path(tmp_dir, variable, i0, i1)

    # Prepare per-chunk input file and write logs into the chunk log
    input_chunk = os.path.join(tmp_dir, f'input_{i0}_{i1}.nc')
    horizontal_tmp = os.path.join(tmp_dir, f'horizontal_{i0}_{i1}.nc')
    vertical_tmp = os.path.join(tmp_dir, f'vertical_{i0}_{i1}.nc')

    # Read and reconcile status with existing files to avoid false positives
    status = _read_status(status_path)
    status = _reconcile_status_with_files(
        status,
        input_chunk=input_chunk,
        horizontal_tmp=horizontal_tmp,
        vertical_tmp=vertical_tmp,
    )
    # Persist reconciliation if it changed anything
    _write_status_atomic(status_path, status)

    # Enable faulthandler to capture fatal errors (e.g., segfault) to the log
    fh_fault = open(log_path, 'a', encoding='utf-8')
    try:
        fh_fault.write('== faulthandler enabled ==\n')
        fh_fault.flush()
        faulthandler.enable(file=fh_fault, all_threads=True)
    except Exception:
        # Don't fail if faulthandler can't be enabled; continue
        pass

    try:
        # Phase: prepare
        chunk_logger = logging.getLogger(f'{__name__}.chunk.{i0}_{i1}')
        chunk_logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        fh.setFormatter(
            logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        )
        chunk_logger.addHandler(fh)
        try:
            if not status.get('prepare', False):
                chunk_logger.info('== Phase: prepare_input ==')
                _prepare_input_with_coords(
                    in_path,
                    grid_file,
                    input_chunk,
                    variable,
                    time_slice=(i0, i1) if has_time else None,
                    logger=chunk_logger,
                )
                # Apply under-ice mask before marking prepare done.
                if mask_enabled:
                    _apply_under_ice_mask_to_file(
                        prepared_path=input_chunk,
                        topo_file=topo_file,
                        variable=variable,
                        threshold=mask_threshold,
                        logger=chunk_logger,
                    )
                _mark_stage_done(status_path, 'prepare')
            # Render a per-chunk namelist after paths are defined
            namelist_contents = _render_namelist(
                file_in=input_chunk,
                horizontal_out=horizontal_tmp,
                vertical_out=vertical_tmp,
                basin_file=basin_file,
                topo_file=topo_file,
                variable=variable,
            )
            namelist_path = os.path.join(tmp_dir, f'{variable}_{i0}_{i1}.nml')
            with open(namelist_path, 'w', encoding='utf-8') as f:
                f.write(namelist_contents)
        finally:
            chunk_logger.removeHandler(fh)
            fh.close()

        # Phase: executables
        if not status.get('horizontal', False):
            # Remove existing output to avoid stale data if present
            try:
                if os.path.exists(horizontal_tmp):
                    os.remove(horizontal_tmp)
            except Exception:
                pass
            _run_exe_capture(
                'i7aof_extrap_horizontal',
                namelist_path,
                log_path,
                'horizontal',
                logger=chunk_logger,
            )
            if not os.path.exists(horizontal_tmp):
                raise FileNotFoundError(
                    f'Expected horizontal output missing: {horizontal_tmp}'
                )
            _mark_stage_done(status_path, 'horizontal')
            status = _read_status(status_path)
        if not status.get('vertical', False):
            try:
                if os.path.exists(vertical_tmp):
                    os.remove(vertical_tmp)
            except Exception:
                pass
            _run_exe_capture(
                'i7aof_extrap_vertical',
                namelist_path,
                log_path,
                'vertical',
                logger=chunk_logger,
            )
            if not os.path.exists(vertical_tmp):
                raise FileNotFoundError(
                    f'Expected vertical output missing: {vertical_tmp}'
                )
            _mark_stage_done(status_path, 'vertical')

        if not os.path.exists(vertical_tmp):
            raise FileNotFoundError(
                f'Expected vertical output missing: {vertical_tmp}'
            )
        return (i0, i1, vertical_tmp)
    except BaseException as err:
        # Append traceback to per-chunk log for easier debugging
        try:
            with open(log_path, 'a', encoding='utf-8') as lf:
                lf.write('== Python exception ==\n')
                lf.write(traceback.format_exc())
        finally:
            pass
        raise ChunkFailed(
            i0=i0, i1=i1, log_path=log_path, message=str(err)
        ) from err
    finally:
        try:
            faulthandler.disable()
        except Exception:
            pass
        try:
            fh_fault.close()
        except Exception:
            pass


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
    logger: logging.Logger | None = None,
) -> None:
    """Ensure input has x/y coordinates; add from ISMIP grid if missing.

    Always writes a prepared copy to out_prepared_path.
    """
    log = logger or logging.getLogger(__name__)

    ds_in = xr.open_dataset(in_path, chunks={'time': 1}, decode_times=False)
    if time_slice is not None and 'time' in ds_in.dims:
        i0, i1 = time_slice
        ds_in = ds_in.isel(time=slice(i0, i1))
    ds_grid = xr.open_dataset(grid_path, decode_times=False)

    # Log dimensions early for debugging
    dims_repr = ', '.join(f'{k}={v}' for k, v in ds_in.sizes.items())
    log.info(
        f'Prepared input initial dims after slice: {dims_repr} '
        f"(var='{var_name}')"
    )

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

    # Validate target variable exists before dropping others
    if var_name not in ds_in.variables:
        available = ', '.join(sorted(ds_in.data_vars.keys()))
        raise KeyError(
            f"Variable '{var_name}' not found in input {in_path}. "
            f'Available: {available}'
        )

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

    # More logging: variable dims and dtype
    v = ds_in[var_name]
    v_dims = 'x'.join(str(s) for s in v.shape)
    log.info(
        f"Variable '{var_name}' shape after prep: {v_dims}, dtype={v.dtype}"
    )

    # Ensure the target variable gets a fill value; others do not
    if time_slice is not None:
        i0, i1 = time_slice
        log.info(
            f'Writing prepared input slice {i0}:{i1} to {out_prepared_path}'
        )
    else:
        log.info(f'Writing prepared input to {out_prepared_path}')

    # Use synchronous dask scheduler to avoid multi-threaded HDF5 I/O
    log.info('Using dask scheduler: synchronous for to_netcdf')
    tmp_out = f'{out_prepared_path}.tmp'
    # Clean any stale tmp
    try:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
    except Exception:
        pass

    with dask_config.set(scheduler='synchronous'):
        write_netcdf(
            ds_in,
            tmp_out,
            has_fill_values=lambda name, var: name == var_name,
            format='NETCDF4',
            engine='netcdf4',
            progress_bar=False,
        )

    # Atomically move into place
    os.replace(tmp_out, out_prepared_path)

    # Close datasets to release file handles and log completion
    ds_in.close()
    ds_grid.close()

    size_bytes = os.path.getsize(out_prepared_path)
    size_gb = size_bytes / (1024**3)
    log.info(
        f'Finished writing prepared input to {out_prepared_path} '
        f'({size_gb:.2f} GiB)'
    )


def _ensure_extrapolated_file(
    *,
    task: FileTask,
    config: MpasConfigParser,
    basin_file: str,
    grid_file: str,
    topo_file: str,
    keep_intermediate: bool,
    num_workers_override: int | str | None = None,
) -> None:
    """Run chunked extrapolation and finalize output if missing.

    If the final extrapolated file already exists, this is a no-op.
    """
    if os.path.exists(task.out_path):
        # Nothing to do
        return

    os.makedirs(os.path.dirname(task.out_path), exist_ok=True)
    os.makedirs(task.tmp_dir, exist_ok=True)

    # Determine chunking in time
    time_chunk = config.getint('extrap_cmip', 'time_chunk')

    # Number of parallel workers and per-chunk logging
    if num_workers_override is not None:
        raw_workers = str(num_workers_override)
    else:
        raw_workers = (
            config.get('extrap_cmip', 'num_workers')
            if config.has_option('extrap_cmip', 'num_workers')
            else '1'
        )

    rw = raw_workers.strip().lower()
    if rw in ('auto', '0'):
        cpu_cnt = os.cpu_count() or 1
        num_workers = max(cpu_cnt - 1, 1)
    else:
        try:
            num_workers = int(raw_workers)
        except ValueError:
            num_workers = 1

    with LoggingContext(__name__):
        logger = logging.getLogger(__name__)
        logger.info(
            f'Starting extrapolation (chunked): {task.in_path} '
            f'(variable={task.variable}, chunk={time_chunk})'
        )

        # Open source input lazily to compute chunk indices
        with xr.open_dataset(task.in_path, decode_times=False) as ds_meta:
            has_time = 'time' in ds_meta.dims
            if not has_time:
                # No time dimension; process as a single synthetic chunk
                time_indices = [(0, 1)]
            else:
                n_time = ds_meta.sizes['time']
                time_indices = [
                    (i0, min(i0 + time_chunk, n_time))
                    for i0 in range(0, n_time, time_chunk)
                ]

        # Determine masking options from config
        mask_enabled = config.getboolean('extrap', 'mask_under_ice')
        mask_threshold = config.getfloat('extrap', 'under_ice_threshold')

        # Execute all chunks (serial or parallel)
        vertical_chunks = _execute_time_chunks(
            task=task,
            grid_file=grid_file,
            basin_file=basin_file,
            topo_file=topo_file,
            time_indices=time_indices,
            num_workers=num_workers,
            has_time=has_time,
            logger=logger,
            mask_enabled=mask_enabled,
            mask_threshold=mask_threshold,
        )

        # Sort by start index and concatenate along time
        vertical_chunks.sort(key=lambda t: t[0])
        if len(vertical_chunks) == 1:
            ds_final_in = xr.open_dataset(
                vertical_chunks[0][2], decode_times=False
            )
        else:
            ds_list = [
                xr.open_dataset(path, decode_times=False, chunks={'time': 1})
                for (_i0, _i1, path) in vertical_chunks
            ]
            logger.info(
                f'Concatenating {len(ds_list)} vertical chunks along time...'
            )
            ds_final_in = xr.concat(ds_list, dim='time', join='exact')
            logger.info('Concatenation complete.')

        _finalize_output_with_grid(
            ds_in=ds_final_in,
            grid_path=grid_file,
            final_out_path=task.out_path,
            variable=task.variable,
            logger=logger,
            drop_singleton_time=False,
        )

        if not keep_intermediate:
            _cleanup_intermediate(task, logger)
        else:
            logger.info('Keeping intermediate files and namelists.')


# -----------------------------
# Per-chunk status management
# -----------------------------


def _status_path(tmp_dir: str, variable: str, i0: int, i1: int) -> str:
    """Return the JSON status file path for a chunk."""
    status_dir = os.path.join(tmp_dir, 'status')
    os.makedirs(status_dir, exist_ok=True)
    return os.path.join(status_dir, f'{variable}_{i0}_{i1}.json')


def _read_status(path: str) -> dict:
    """Read a status JSON; return default structure if missing/corrupt."""
    default = {'prepare': False, 'horizontal': False, 'vertical': False}
    if not os.path.exists(path):
        return default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Ensure required keys exist
        for k in default:
            data.setdefault(k, False)
        return data
    except Exception:
        # If unreadable, start fresh (safer than assuming done)
        return default


def _write_status_atomic(path: str, status: dict) -> None:
    """Write status JSON atomically (write temp then rename)."""
    tmp = f'{path}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=2, sort_keys=True)
        f.write('\n')
    os.replace(tmp, path)


def _mark_stage_done(path: str, stage: str) -> dict:
    """Mark a stage as done in the status file and write it; return status."""
    status = _read_status(path)
    status[stage] = True
    _write_status_atomic(path, status)
    return status


def _reconcile_status_with_files(
    status: dict, *, input_chunk: str, horizontal_tmp: str, vertical_tmp: str
) -> dict:
    """If status says done but file is missing, reset that stage to False."""
    if status.get('prepare', False) and not os.path.exists(input_chunk):
        status['prepare'] = False
    if status.get('horizontal', False) and not os.path.exists(horizontal_tmp):
        status['horizontal'] = False
    if status.get('vertical', False) and not os.path.exists(vertical_tmp):
        status['vertical'] = False
    return status
