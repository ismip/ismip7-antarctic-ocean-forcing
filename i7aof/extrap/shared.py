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

import numpy as np
import xarray as xr
from dask import config as dask_config
from jinja2 import BaseLoader, Environment
from mpas_tools.config import MpasConfigParser
from xarray.coders import CFDatetimeCoder

from i7aof.coords import (
    attach_grid_coords,
    ensure_cf_time_encoding,
    propagate_time_from,
    strip_fill_on_non_data,
)
from i7aof.extrap import load_template_text
from i7aof.grid.ismip import get_horiz_res_string
from i7aof.imbie.masks import make_imbie_masks
from i7aof.io import read_dataset, write_netcdf
from i7aof.io_zarr import append_to_zarr, finalize_zarr_to_netcdf
from i7aof.time.bounds import inject_time_bounds
from i7aof.topo import get_topo
from i7aof.vert.resamp import VerticalResampler

__all__ = [
    '_ensure_imbie_masks',
    '_ensure_topography',
    '_render_namelist',
    '_prepare_input_single',
    '_run_exe_capture',
    '_finalize_output_with_grid',
    '_apply_under_ice_mask_to_file',
    '_vertically_resample_to_coarse_ismip_grid',
]


##
# Fill-value handling is centralized in i7aof.coords.strip_fill_on_non_data.
# Use that helper instead of the local variant that used to live here.


def _ensure_imbie_masks(config: MpasConfigParser, workdir: str) -> str:
    """Ensure IMBIE basin mask file exists under ``workdir`` and return it."""
    res = get_horiz_res_string(config)
    basin_file = os.path.join(
        workdir, 'imbie2', f'basin_numbers_ismip{res}.nc'
    )
    if not os.path.exists(basin_file):
        cwd = os.getcwd()
        try:
            os.makedirs(os.path.join(workdir, 'imbie2'), exist_ok=True)
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
    ds_in = read_dataset(in_path)
    ds_grid = read_dataset(grid_path)

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
    except OSError:  # pragma: no cover - best effort
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
    config: MpasConfigParser,
    final_out_path: str,
    variable: str,
    logger: logging.Logger,
    src_attr_path: str,
    time_bounds: tuple[str, xr.DataArray] | None = None,
    time_prefer_source: xr.Dataset | None = None,
) -> None:
    """Finalize a vertical output by injecting grid + restoring time bounds.

    Parameters
    ----------
    ds_in : xr.Dataset
        Dataset to finalize (output of concatenated vertical chunks).
    config : MpasConfigParser
        Configuration object for locating ISMIP grid.
    final_out_path : str
        Path to final extrapolated NetCDF file.
    variable : str
        Name of the data variable being extrapolated (e.g., 'ct' or 'sa').
    logger : logging.Logger
        Logger for status messages.
    src_attr_path : str
        Path to original remapped input used to copy variable attrs + time.
    time_bounds : (str, DataArray) | None, optional
        Captured time bounds tuple (name, DataArray) from the original
        remapped input file. Injected if present.
    time_prefer_source : xr.Dataset | None, optional
        Dataset whose time coordinate metadata should be preferred when
        applying CF encodings (calendar inference).
    """
    # Attach ISMIP grid coordinates and geodetic metadata
    ds_out = attach_grid_coords(ds_in, config)

    # Copy original variable attributes and capture time metadata
    with read_dataset(src_attr_path) as src:
        ds_out[variable].attrs = dict(src[variable].attrs)
        if 'time' in ds_out.dims and 'time' in src.dims:
            # Propagate time coordinate values (and existing bounds if any)
            ds_out = propagate_time_from(ds_out, src, apply_cf_encoding=False)
            # Inject captured time bounds explicitly (source may have lost
            # them)
            if time_bounds is not None:
                inject_time_bounds(ds_out, time_bounds)
            # Apply CF-compliant shared encoding for time/time_bnds
            ensure_cf_time_encoding(
                ds_out,
                units='days since 1850-01-01 00:00:00',
                calendar=None,
                prefer_source=time_prefer_source or src,
            )

    # Remove _FillValue from non-data variables
    ds_out = strip_fill_on_non_data(ds_out, data_vars=(variable,))

    logger.info(f'Writing extrapolated output: {final_out_path}')
    write_netcdf(
        ds_out,
        final_out_path,
        has_fill_values=lambda name, var: name == variable,
        progress_bar=True,
    )


def _apply_under_ice_mask_to_file(
    *,
    prepared_path: str,
    topo_file: str,
    variable: str,
    threshold: float,
    logger: logging.Logger | None = None,
) -> None:
    """Mask values under ice using topography ``ice_frac`` and rewrite file.

    This applies a 2D (y, x) mask where ``ice_frac > threshold`` to the
    target variable in ``prepared_path``. The masked dataset is written back
    atomically to ``prepared_path`` using a ``.tmp`` file and replace.

    Parameters
    ----------
    prepared_path : str
        Path to the prepared NetCDF file that the Fortran tools will read.
    topo_file : str
        Path to the topography file on the ISMIP grid (must contain
        ``ice_frac`` with dims (y, x)).
    variable : str
        Name of the variable to mask (e.g., 'ct' or 'sa').
    threshold : float
        Mask where ``ice_frac > threshold``.
    logger : logging.Logger, optional
        Logger for info messages.
    """
    log = logger or logging.getLogger(__name__)
    ds_prep = read_dataset(prepared_path)
    ds_topo = read_dataset(topo_file)

    if 'ice_frac' not in ds_topo:
        raise KeyError(
            f"Topography file '{topo_file}' is missing required 'ice_frac'."
        )
    if variable not in ds_prep:
        raise KeyError(
            f"Variable '{variable}' not found in prepared file "
            f"'{prepared_path}'. Available: {', '.join(ds_prep.data_vars)}"
        )

    mask = ds_topo['ice_frac'] > threshold  # dims: (y, x)
    before = ds_prep[variable]
    after = before.where(~mask, other=np.nan)
    # preserve attrs
    after.attrs = before.attrs
    ds_prep[variable] = after

    tmp_out = f'{prepared_path}.tmp'
    try:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
    except OSError:  # best effort
        pass

    log.info(
        f"Applying under-ice mask to '{variable}' with threshold {threshold} "
        f'-> {prepared_path}'
    )
    # Write atomically; ensure only target var has fill value
    with dask_config.set(scheduler='synchronous'):
        write_netcdf(
            ds_prep,
            tmp_out,
            has_fill_values=lambda name, var: name == variable,
            format='NETCDF4',
            engine='netcdf4',
            progress_bar=False,
        )
    os.replace(tmp_out, prepared_path)

    # Close datasets
    ds_prep.close()
    ds_topo.close()


def _vertically_resample_to_coarse_ismip_grid(
    *,
    in_path: str,
    grid_file: str,
    variable: str,
    config: MpasConfigParser,
    out_nc: str,
    time_chunk: int | None,
    zarr_store: str,
    logger: logging.Logger | None = None,
) -> str:
    """
    Resample CT and SA vertically from the finer interpolation grid to the
    coarser ISMIP grid.  Process the data in time chunks to Zarr, convert to
    NetCDF, and return out_nc.

    - Removes any pre-existing Zarr store at the given path.
    - Appends along time if present; otherwise writes a single slice.
    - Preserves chunk encodings for the target variable to speed NetCDF write.
    """
    log = logger or logging.getLogger(__name__)
    if os.path.exists(out_nc):
        log.info(f'Resampled output exists, skipping: {out_nc}')
        return out_nc

    # Prepare resampler and optional z_bnds once
    with read_dataset(grid_file) as ds_grid:
        z_src = ds_grid['z_extrap']
        src_valid = xr.DataArray(
            np.ones(z_src.shape, dtype=np.float32),
            dims=('z_extrap',),
            coords={'z_extrap': z_src},
        )
        resampler = VerticalResampler(
            src_valid=src_valid,
            src_coord='z_extrap',
            dst_coord='z',
            config=config,
        )
        z_bnds = ds_grid['z_bnds'] if 'z_bnds' in ds_grid else None

    # Zarr store will be recreated on first append if it exists

    # Build chunk indices and capture original time coordinate (values + attrs)
    # so we can reattach metadata after the Zarr round-trip.
    # Build time chunk indices directly; full span when no chunking
    with read_dataset(in_path) as _ds_meta:
        if 'time' in _ds_meta.dims:
            n_time = int(_ds_meta.sizes['time'])
            if time_chunk and time_chunk > 0:
                indices = [
                    (i0, min(i0 + time_chunk, n_time))
                    for i0 in range(0, n_time, time_chunk)
                ]
            else:
                indices = [(0, n_time)]
        else:
            indices = [(0, 1)]

    first = True
    in_chunks = {'time': time_chunk} if time_chunk else None
    with xr.open_dataset(
        in_path,
        decode_times=CFDatetimeCoder(use_cftime=True),
        chunks=in_chunks,
    ) as ds_in:
        for i0, i1 in indices:
            ds_slice = (
                ds_in.isel(time=slice(i0, i1))
                if 'time' in ds_in.dims
                else ds_in
            )
            if variable not in ds_slice:
                raise KeyError(
                    f"Variable '{variable}' not found in {in_path}."
                )
            da_out = resampler.resample(ds_slice[variable])
            # Preserve original variable attributes (units, long_name, etc.)
            da_out.attrs = dict(ds_slice[variable].attrs)
            ds_res = ds_slice.copy()
            ds_res[variable] = da_out.astype(ds_slice[variable].dtype)
            if 'z_extrap' in ds_res:
                ds_res = ds_res.drop_vars(['z_extrap'])
            if z_bnds is not None:
                ds_res['z_bnds'] = z_bnds
            # Chunk this slice fully in time for contiguous appends
            if 'time' in ds_res.dims:
                ds_res = ds_res.chunk({'time': max(i1 - i0, 1)})

            # Write/append to Zarr using shared helper
            if first:
                log.info(f'Creating Zarr store: {zarr_store}')
            first = append_to_zarr(
                ds=ds_res,
                zarr_store=zarr_store,
                first=first,
                append_dim='time' if 'time' in ds_res.dims else None,
            )

    # Convert Zarr to NetCDF once using shared helper; preserve chunk encoding
    log.info('Converting Zarr to NetCDF...')

    # Intentionally nested: captures in_path and config from outer scope
    # to avoid threading multiple parameters through the finalize call.
    def _post(ds_z: xr.Dataset) -> xr.Dataset:
        # Reopen the original source and propagate time/time_bnds
        try:
            with read_dataset(
                in_path,
                decode_times=CFDatetimeCoder(use_cftime=True),
            ) as _src:
                ds_z = propagate_time_from(
                    ds_z,
                    _src,
                    apply_cf_encoding=True,
                    units='days since 1850-01-01 00:00:00',
                )
        except Exception as e:  # pragma: no cover - non-fatal
            log.warning('Failed to propagate time metadata: %s', e)

        # Attach ISMIP grid coordinates and geodetic coords; validate dims
        ds_z = attach_grid_coords(ds_z, config)

        # Drop fine vertical coord and its bounds if still present after
        # resampling to the coarse ISMIP vertical grid.
        drop_names = [n for n in ('z_extrap', 'z_extrap_bnds') if n in ds_z]
        if drop_names:
            ds_z = ds_z.drop_vars(drop_names, errors='ignore')

        # Ensure coordinates and non-target variables do not carry _FillValue
        ds_z = strip_fill_on_non_data(ds_z, data_vars=(variable,))

        var_da = ds_z[variable]
        chunks = getattr(var_da, 'chunks', None)
        if chunks and all(c is not None for c in chunks):
            var_da.encoding['chunksizes'] = tuple(chunks)
        return ds_z

    finalize_zarr_to_netcdf(
        zarr_store=zarr_store,
        out_nc=out_nc,
        has_fill_values=lambda name, v, _v=variable: name == _v,
        progress_bar=True,
        postprocess=_post,
    )
    return out_nc
