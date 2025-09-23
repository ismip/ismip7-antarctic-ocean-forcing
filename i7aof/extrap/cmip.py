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
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Sequence

from jinja2 import BaseLoader, Environment
from mpas_tools.config import MpasConfigParser
from mpas_tools.logging import LoggingContext

from i7aof.cmip import get_model_prefix
from i7aof.extrap import load_template_text
from i7aof.grid.ismip import get_res_string
from i7aof.imbie.masks import make_imbie_masks
from i7aof.topo import get_topo

__all__ = ['extrap_cmip', 'main']


@dataclass
class FileTask:
    """Per-variable extrapolation work for one remapped input file."""

    in_path: str
    out_path: str  # final vertical output
    horizontal_tmp: str  # horizontal intermediate
    namelist_path: str  # combined rendered namelist
    variable: str  # e.g. 'ct' or 'sa'


def extrap_cmip(
    model: str,
    scenario: str,
    workdir: str | None = None,
    user_config_filename: str | None = None,
    variables: Sequence[str] = ('ct', 'sa'),
    keep_intermediate: bool = False,
    dry_run: bool = False,
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
    dry_run : bool, optional
        If True, list planned actions only.
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
    topo_file = _ensure_topography(config, workdir)

    for in_file in in_files:
        base = os.path.basename(in_file)
        for var in variables:
            out_base = base.replace('ct_sa', var)
            out_path = os.path.join(out_dir, out_base)
            horizontal_tmp = out_path.replace('.nc', '_horizontal_tmp.nc')
            namelist_path = out_path.replace('.nc', f'_{var}.nml')
            task = FileTask(
                in_path=in_file,
                out_path=out_path,
                horizontal_tmp=horizontal_tmp,
                namelist_path=namelist_path,
                variable=var,
            )
            _process_task(
                task,
                basin_file=basin_file,
                topo_file=topo_file,
                keep_intermediate=keep_intermediate,
                dry_run=dry_run,
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
        '--dry-run',
        action='store_true',
        help='Show planned actions without running executables.',
    )
    args = parser.parse_args()

    extrap_cmip(
        model=args.model,
        scenario=args.scenario,
        workdir=args.workdir,
        user_config_filename=args.config,
        variables=args.variables,
        keep_intermediate=args.keep_intermediate,
        dry_run=args.dry_run,
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
    from i7aof.grid.ismip import get_horiz_res_string

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
    import logging

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
    )


def _process_task(
    task: FileTask,
    basin_file: str,
    topo_file: str,
    keep_intermediate: bool,
    dry_run: bool,
) -> None:
    if os.path.exists(task.out_path):
        print(f'Extrapolated file exists, skipping: {task.out_path}')
        return
    os.makedirs(os.path.dirname(task.out_path), exist_ok=True)

    namelist_contents = _render_namelist(
        file_in=task.in_path,
        horizontal_out=task.horizontal_tmp,
        vertical_out=task.out_path,
        basin_file=basin_file,
        topo_file=topo_file,
        variable=task.variable,
    )
    with open(task.namelist_path, 'w', encoding='utf-8') as f:
        f.write(namelist_contents)

    import logging

    with LoggingContext(__name__):
        logger = logging.getLogger(__name__)
        logger.info(
            f'Starting extrapolation: {task.in_path} '
            f'(variable={task.variable})'
        )
        logger.info(f'  Namelist: {task.namelist_path}')
        logger.info(f'  Horizontal tmp: {task.horizontal_tmp}')
        logger.info(f'  Final output: {task.out_path}')

        horiz_exec = 'i7aof_extrap_horizontal'
        vert_exec = 'i7aof_extrap_vertical'
        for exe in (horiz_exec, vert_exec):
            if shutil.which(exe) is None:
                raise FileNotFoundError(
                    f"Required executable '{exe}' not found on PATH."
                )

        if dry_run:
            logger.info('Dry-run mode: skipping Fortran execution.')
        else:
            _run_exe(horiz_exec, task.namelist_path, logger, 'horizontal')
            _run_exe(vert_exec, task.namelist_path, logger, 'vertical')

        if not keep_intermediate:
            _cleanup_intermediate(task, logger)
        else:
            logger.info('Keeping intermediate horizontal file and namelist.')


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
    for path in (task.horizontal_tmp, task.namelist_path):
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f'Removed intermediate: {path}')
        except OSError as e:  # pragma: no cover
            logger.warning(f'Failed to remove intermediate {path}: {e}')
