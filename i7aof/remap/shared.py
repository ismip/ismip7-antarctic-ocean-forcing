import os
import shutil
import tempfile

import numpy as np
import xarray as xr

from i7aof.coords import (
    attach_grid_coords,
    ensure_cf_time_encoding,
    propagate_time_from,
    strip_fill_on_non_data,
)
from i7aof.grid.ismip import get_ismip_grid_filename
from i7aof.io import read_dataset, write_netcdf
from i7aof.remap import add_periodic_lon, remap_lat_lon_to_ismip
from i7aof.time.bounds import inject_time_bounds
from i7aof.vert.interp import VerticalInterpolator, fix_src_z_coord


def _vert_mask_interp_norm_multi(
    config, in_filename, outdir, variables, tmpdir
):
    """
    Mask, vertically interpolate, and normalize variables to the ISMIP
    vertical coordinate.

    What it does
    - Reads variables in ``variables`` (e.g., ``['ct', 'sa']``) using the
        source vertical coordinate ``lev`` with bounds ``lev_bnds``.
    - Fixes the vertical coordinate via ``fix_src_z_coord`` (meters,
        positive up) and builds a combined validity mask from the first
        time step of each variable. If no ``time`` dimension exists, the
        full array is used.
    - Runs a 3-stage pipeline, writing temporary files in ``tmpdir``:
        1) mask: zero-out invalid cells and standardize vertical metadata
             (``lev``, ``lev_bnds``); writes ``src_valid``.
        2) interp: interpolate each variable to ``z_extrap`` from the
             ISMIP grid; writes ``src_frac_interp`` (valid fraction per
             column) and ``z_extrap_bnds``; drops ``lev``.
        3) normalize: renormalize by ``src_frac_interp`` above a threshold
             to avoid low bias where only part of a column is valid.

    Time handling
    - Works with or without a ``time`` dimension. If time is present,
        stages are chunked by ``[remap_cmip] vert_time_chunk`` from config.

    Returns
    - Absolute path to the final normalized file to feed the horizontal
        remapping step.
    """

    # Developer notes (vertical overview):
    # - Stage files in tmpdir allow resuming when the normalized file
    #   exists.
    # - The validity mask is the logical AND across all variables; when
    #   time exists, the first time slice is used to keep masks
    #   time-invariant.
    # - Normalization uses the vertically interpolated fraction to avoid
    #   damping intensive fields near partial coverage.

    mask_filename = os.path.join(tmpdir, 'vars_masked.nc')
    interp_filename = os.path.join(tmpdir, 'vars_interp.nc')
    normalized_filename = os.path.join(tmpdir, 'vars_normalized.nc')

    if os.path.exists(normalized_filename):
        print(
            'Vertically interpolated file exists, skipping: '
            f'{normalized_filename}'
        )
        return normalized_filename

    ds_ismip = read_dataset(get_ismip_grid_filename(config))

    lev, lev_bnds, src_valid = _prepare_vert_coords_and_mask(
        in_filename, variables
    )

    time_chunk = config.getint('remap_cmip', 'vert_time_chunk')
    interpolator = VerticalInterpolator(
        src_valid=src_valid,
        src_coord='lev',
        dst_coord='z_extrap',
        config=config,
    )

    _write_mask_stage(
        in_filename,
        variables,
        interpolator,
        lev,
        lev_bnds,
        mask_filename,
        time_chunk,
    )

    _write_interp_stage(
        mask_filename,
        variables,
        interpolator,
        ds_ismip,
        interp_filename,
        time_chunk,
    )

    _write_normalize_stage(
        interp_filename,
        variables,
        interpolator,
        ds_ismip,
        normalized_filename,
        time_chunk,
    )

    print(
        'Vertical interpolation completed and saved to '
        f"'{normalized_filename}'."
    )

    return normalized_filename


def _prepare_vert_coords_and_mask(in_filename, variables):
    with read_dataset(in_filename) as ds:
        lev, lev_bnds = fix_src_z_coord(ds, 'lev', 'lev_bnds')
        ds = ds.assign_coords({'lev': ('lev', lev.data)})
        ds['lev'].attrs = lev.attrs
        ds['lev_bnds'] = lev_bnds
        src_valid = None
        for var in variables:
            da = ds[var]
            if 'time' in da.dims:
                valid = da.isel(time=0).notnull().drop_vars(['time'])
            else:
                valid = da.notnull()
            src_valid = valid if src_valid is None else (src_valid & valid)
    return lev, lev_bnds, src_valid


def _write_mask_stage(
    in_filename,
    variables,
    interpolator,
    lev,
    lev_bnds,
    mask_filename,
    time_chunk,
):
    ds = read_dataset(in_filename)
    if 'time' in ds.dims:
        ds = ds.chunk({'time': time_chunk})
    ds_out = xr.Dataset()
    ds_out = ds_out.assign_coords(
        {
            'lev': ('lev', lev.data),
            'lev_bnds': (('lev', 'd2'), lev_bnds.data),
        }
    )
    ds_out['lev'].attrs = lev.attrs
    for var in variables:
        da_masked = interpolator.mask_and_sort(ds[var])
        ds_out[var] = da_masked.astype(np.float32)
    ds_out['src_valid'] = interpolator.src_valid.astype(np.float32)

    has_fill_values = variables + ['src_valid']
    write_netcdf(
        ds_out,
        mask_filename,
        progress_bar=True,
        has_fill_values=has_fill_values,
    )


def _write_interp_stage(
    mask_filename,
    variables,
    interpolator,
    ds_ismip,
    interp_filename,
    time_chunk,
):
    ds = read_dataset(mask_filename)
    if 'time' in ds.dims:
        ds = ds.chunk({'time': time_chunk})
    ds_out = xr.Dataset()
    for var in variables:
        da_interp = interpolator.interp(ds[var])
        ds_out[var] = da_interp.astype(np.float32)
    ds_out['src_frac_interp'] = interpolator.src_frac_interp.astype(np.float32)
    ds_out['z_extrap_bnds'] = ds_ismip['z_extrap_bnds']
    if 'lev' in ds_out:
        ds_out = ds_out.drop_vars(['lev'])

    has_fill_values = variables + ['src_frac_interp']
    write_netcdf(
        ds_out,
        interp_filename,
        progress_bar=True,
        has_fill_values=has_fill_values,
    )


def _write_normalize_stage(
    interp_filename,
    variables,
    interpolator,
    ds_ismip,
    normalized_filename,
    time_chunk,
):
    ds = read_dataset(interp_filename)
    if 'time' in ds.dims:
        ds = ds.chunk({'time': time_chunk})
    ds_out = xr.Dataset()
    for var in variables:
        da_norm = interpolator.normalize(ds[var])
        ds_out[var] = da_norm.astype(np.float32)
    ds_out['src_frac_interp'] = interpolator.src_frac_interp.astype(np.float32)
    ds_out['z_extrap_bnds'] = ds_ismip['z_extrap_bnds']

    has_fill_values = variables + ['src_frac_interp']
    write_netcdf(
        ds_out,
        normalized_filename,
        progress_bar=True,
        has_fill_values=has_fill_values,
    )


def _run_remap_with_temp_cwd(
    *,
    in_filename,
    in_grid_name,
    out_filename,
    map_dir,
    method,
    config,
    logger,
    lon_var,
    lat_var,
    renormalize=None,
):
    """Run remap_lat_lon_to_ismip inside a temporary working directory.

    This isolates ESMF PET* log files per invocation to avoid collisions and
    clutter. The temporary directory is created under the output file's
    directory and removed on success; on failure, it is preserved.
    """
    # Create a temp dir within the destination directory for logs
    out_parent = os.path.dirname(out_filename)
    os.makedirs(out_parent, exist_ok=True)
    tmp_cwd = tempfile.mkdtemp(prefix='esmf_logs_', dir=out_parent)
    orig_cwd = os.getcwd()
    success = False
    try:
        os.chdir(tmp_cwd)
        kwargs = dict(
            in_filename=in_filename,
            in_grid_name=in_grid_name,
            out_filename=out_filename,
            map_dir=map_dir,
            method=method,
            config=config,
            logger=logger,
            lon_var=lon_var,
            lat_var=lat_var,
        )
        if renormalize is not None:
            kwargs['renormalize'] = renormalize
        remap_lat_lon_to_ismip(**kwargs)
        success = True
    finally:
        try:
            os.chdir(orig_cwd)
        finally:
            if success:
                shutil.rmtree(tmp_cwd, ignore_errors=True)
            else:
                logger.warning(
                    'Preserving ESMF log directory for debugging: %s',
                    tmp_cwd,
                )


def _remap_horiz(
    config,
    in_filename,
    out_filename,
    model_prefix,
    tmpdir,
    logger,
    has_fill_values,
    lat_var=None,
    lon_var=None,
    lon_dim=None,
    *,
    time_bounds: tuple[str, xr.DataArray] | None = None,
    time_prefer_source: xr.Dataset | None = None,
):
    """High-level orchestration for horizontal remapping."""
    method = config.get('remap', 'method')
    renorm_threshold = config.getfloat('remap', 'threshold')
    # Resolve coordinate variable names if not provided
    lat_var = lat_var or (
        config.get('cmip_dataset', 'lat_var')
        if config.has_option('cmip_dataset', 'lat_var')
        else 'lat'
    )
    lon_var = lon_var or (
        config.get('cmip_dataset', 'lon_var')
        if config.has_option('cmip_dataset', 'lon_var')
        else 'lon'
    )
    lon_dim = lon_dim or (
        config.get('cmip_dataset', 'lon_dim')
        if config.has_option('cmip_dataset', 'lon_dim')
        else 'lon'
    )
    in_grid_name = model_prefix
    ds = read_dataset(in_filename, chunks={'time': 1})
    if method == 'bilinear':
        ds = add_periodic_lon(ds, lon_var=lon_var, periodic_dim=lon_dim)
    ds_mask = _build_and_remap_mask(
        ds=ds,
        tmpdir=tmpdir,
        in_grid_name=in_grid_name,
        method=method,
        config=config,
        logger=logger,
        lon_var=lon_var,
        lat_var=lat_var,
    )
    remapped_chunks = _remap_data_variables(
        ds=ds,
        tmpdir=tmpdir,
        in_grid_name=in_grid_name,
        method=method,
        config=config,
        logger=logger,
        lon_var=lon_var,
        lat_var=lat_var,
        renorm_threshold=renorm_threshold,
        has_fill_values=has_fill_values,
    )
    _validate_z_extrap(remapped_chunks)
    ds_final = _concat_chunks(remapped_chunks)
    _finalize_and_write(
        ds_final=ds_final,
        ds_mask=ds_mask,
        ds_source=ds,
        out_filename=out_filename,
        config=config,
        time_bounds=time_bounds,
        time_prefer_source=time_prefer_source,
        has_fill_values=has_fill_values,
    )


def _build_and_remap_mask(
    *,
    ds: xr.Dataset,
    tmpdir: str,
    in_grid_name: str,
    method: str,
    config,
    logger,
    lon_var: str,
    lat_var: str,
) -> xr.Dataset:
    """Create and horizontally remap the src_frac_interp mask dataset."""
    input_mask_path = os.path.join(tmpdir, 'input_mask.nc')
    output_mask_path = os.path.join(tmpdir, 'output_mask.nc')
    output_mask_tmp = f'{output_mask_path}.tmp'
    if os.path.exists(output_mask_path):
        return read_dataset(output_mask_path)
    ds_mask = ds[['src_frac_interp']].copy()
    write_netcdf(
        ds_mask,
        input_mask_path,
        progress_bar=True,
        has_fill_values=['src_frac_interp'],
        compression=['src_frac_interp'],
    )
    try:
        if os.path.exists(output_mask_tmp):
            os.remove(output_mask_tmp)
    except OSError:
        pass
    _run_remap_with_temp_cwd(
        in_filename=input_mask_path,
        in_grid_name=in_grid_name,
        out_filename=output_mask_tmp,
        map_dir=tmpdir,
        method=method,
        config=config,
        logger=logger,
        lon_var=lon_var,
        lat_var=lat_var,
        renormalize=None,
    )
    if not os.path.exists(output_mask_tmp):
        raise FileNotFoundError(
            f'Mask remap failed to produce: {output_mask_tmp}'
        )
    os.replace(output_mask_tmp, output_mask_path)
    return read_dataset(output_mask_path)


def _remap_data_variables(
    *,
    ds: xr.Dataset,
    tmpdir: str,
    in_grid_name: str,
    method: str,
    config,
    logger,
    lon_var: str,
    lat_var: str,
    renorm_threshold: float,
    has_fill_values: list[str],
) -> list[xr.Dataset]:
    """Remap data variables in ds either single-pass or chunked in time."""
    if 'time' not in ds.dims:
        return _remap_no_time(
            ds,
            tmpdir,
            in_grid_name,
            method,
            config,
            logger,
            lon_var,
            lat_var,
            renorm_threshold,
            has_fill_values,
        )
    return _remap_with_time(
        ds,
        tmpdir,
        in_grid_name,
        method,
        config,
        logger,
        lon_var,
        lat_var,
        renorm_threshold,
        has_fill_values,
    )


def _validate_z_extrap(remapped_chunks: list[xr.Dataset]) -> None:
    """Ensure all remapped chunks have identical unique z_extrap coordinate."""
    if not remapped_chunks:
        raise ValueError('No remapped chunks produced.')
    z0 = remapped_chunks[0]['z_extrap'].values
    if len(np.unique(z0)) != len(z0):
        raise ValueError('First chunk has duplicate z_extrap values.')
    for i, ds_chk in enumerate(remapped_chunks[1:], start=1):
        z = ds_chk['z_extrap'].values
        if z.shape != z0.shape or not np.allclose(z, z0):
            raise ValueError(
                f'Inconsistent z_extrap in chunk {i}: shape/values mismatch.'
            )


def _concat_chunks(remapped_chunks: list[xr.Dataset]) -> xr.Dataset:
    """Concatenate chunks along time dimension when present."""
    if len(remapped_chunks) == 1 and 'time' not in remapped_chunks[0].dims:
        return remapped_chunks[0]
    return xr.concat(remapped_chunks, dim='time', join='exact')


def _finalize_and_write(
    *,
    ds_final: xr.Dataset,
    ds_mask: xr.Dataset,
    ds_source: xr.Dataset,
    out_filename: str,
    config,
    time_bounds: tuple[str, xr.DataArray] | None,
    time_prefer_source: xr.Dataset | None,
    has_fill_values: list[str] | None,
) -> None:
    """Attach metadata, ensure encodings, and atomically write final file."""
    # Attach horizontally remapped src_frac_interp (time-invariant)
    ds_final['src_frac_interp'] = ds_mask['src_frac_interp']
    ds_final = attach_grid_coords(ds_final, config)
    if 'time' in ds_source.dims:
        ds_final = propagate_time_from(
            ds_final,
            ds_source,
            apply_cf_encoding=False,
        )
        if time_bounds is not None:
            inject_time_bounds(ds_final, time_bounds)
        ensure_cf_time_encoding(
            ds_final,
            units='days since 1850-01-01 00:00:00',
            calendar=None,
            prefer_source=time_prefer_source or ds_source,
        )
    data_vars = [
        str(v) for v in ds_final.data_vars if v not in ds_final.coords
    ]
    ds_final = strip_fill_on_non_data(ds_final, data_vars=data_vars)
    final_tmp = f'{out_filename}.tmp'
    try:
        if os.path.exists(final_tmp):
            os.remove(final_tmp)
    except OSError:
        pass
    write_netcdf(
        ds_final,
        final_tmp,
        progress_bar=True,
        has_fill_values=has_fill_values,
    )
    if not os.path.exists(final_tmp):
        raise FileNotFoundError(
            f'Expected final remap output missing: {final_tmp}'
        )
    os.replace(final_tmp, out_filename)


def _remap_no_time(
    ds,
    tmpdir,
    in_grid_name,
    method,
    config,
    logger,
    lon_var,
    lat_var,
    renorm_threshold,
    has_fill_values,
):
    input_chunk_path = os.path.join(tmpdir, 'input_single.nc')
    output_chunk_path = os.path.join(tmpdir, 'output_single.nc')
    output_chunk_tmp = f'{output_chunk_path}.tmp'

    subset = ds
    if 'src_frac_interp' in subset:
        subset = subset.drop_vars(['src_frac_interp'])
    write_netcdf(
        subset,
        input_chunk_path,
        progress_bar=True,
        has_fill_values=has_fill_values,
    )

    # Write to a temp file to support safe resume on interruption
    try:
        if os.path.exists(output_chunk_tmp):
            os.remove(output_chunk_tmp)
    except OSError:
        pass
    _run_remap_with_temp_cwd(
        in_filename=input_chunk_path,
        in_grid_name=in_grid_name,
        out_filename=output_chunk_tmp,
        map_dir=tmpdir,
        method=method,
        config=config,
        logger=logger,
        lon_var=lon_var,
        lat_var=lat_var,
        renormalize=renorm_threshold,
    )
    if not os.path.exists(output_chunk_tmp):
        raise FileNotFoundError(
            f'Expected remap output missing: {output_chunk_tmp}'
        )
    os.replace(output_chunk_tmp, output_chunk_path)
    remapped_chunk = read_dataset(output_chunk_path)
    return [remapped_chunk]


def _remap_with_time(
    ds,
    tmpdir,
    in_grid_name,
    method,
    config,
    logger,
    lon_var,
    lat_var,
    renorm_threshold,
    has_fill_values,
):
    chunk_size = config.getint('remap_cmip', 'horiz_time_chunk')
    n_time = ds.sizes['time']
    time_indices = np.arange(0, n_time, chunk_size)

    remapped_chunks = []

    for i_start in time_indices:
        i_end = min(i_start + chunk_size, n_time)

        input_chunk_path = os.path.join(tmpdir, f'input_{i_start}_{i_end}.nc')
        output_chunk_path = os.path.join(
            tmpdir, f'output_{i_start}_{i_end}.nc'
        )
        output_chunk_tmp = f'{output_chunk_path}.tmp'
        if os.path.exists(output_chunk_path):
            print(
                f'Skipping remapping for chunk {i_start}-{i_end} '
                f'(already exists).'
            )
            remapped_chunk = read_dataset(
                output_chunk_path,
                chunks={'time': 1},
            )
            remapped_chunks.append(remapped_chunk)
            continue

        subset = ds.isel(time=slice(i_start, i_end))
        if 'src_frac_interp' in subset:
            subset = subset.drop_vars(['src_frac_interp'])

        # Load the subset into memory to reduce dask scheduler overhead and
        # many small I/O operations when writing temporary chunk files.
        subset = subset.load()

        write_netcdf(
            subset,
            input_chunk_path,
            progress_bar=False,
            has_fill_values=has_fill_values,
            engine='h5netcdf',
        )

        # Write to a temp file to support safe resume on interruption
        try:
            if os.path.exists(output_chunk_tmp):
                os.remove(output_chunk_tmp)
        except OSError:
            pass
        _run_remap_with_temp_cwd(
            in_filename=input_chunk_path,
            in_grid_name=in_grid_name,
            out_filename=output_chunk_tmp,
            map_dir=tmpdir,
            method=method,
            config=config,
            logger=logger,
            lon_var=lon_var,
            lat_var=lat_var,
            renormalize=renorm_threshold,
        )
        if not os.path.exists(output_chunk_tmp):
            raise FileNotFoundError(
                f'Expected remap output missing: {output_chunk_tmp}'
            )
        os.replace(output_chunk_tmp, output_chunk_path)

        remapped_chunk = read_dataset(
            output_chunk_path,
            chunks={'time': 1},
        )
        remapped_chunks.append(remapped_chunk)

    return remapped_chunks
