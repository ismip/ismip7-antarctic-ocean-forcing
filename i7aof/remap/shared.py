import os

import numpy as np
import xarray as xr

from i7aof.grid.ismip import get_ismip_grid_filename
from i7aof.io import write_netcdf
from i7aof.remap import add_periodic_lon, remap_lat_lon_to_ismip
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

    ds_ismip = xr.open_dataset(get_ismip_grid_filename(config))

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


def _assign_coords_and_bounds(
    ds_src: xr.Dataset,
    ds_dest: xr.Dataset,
    coord_names: tuple[str, ...] = ('lat', 'lon', 'time'),
) -> xr.Dataset:
    """Copy coordinates and their bounds variables from ds_src to ds_dest.

    - Adds coord variables in coord_names if present in ds_src.
    - Adds conventional ``<coord>_bnds`` variables when present.
    - Honors CF ``bounds`` attribute to include non-conventional names like
      ``time_bounds``.
    - Does not overwrite variables already present in ds_dest.
    """
    ds_out = ds_dest
    for cname in coord_names:
        if cname in ds_src and cname not in ds_out:
            ds_out[cname] = ds_src[cname]
        # Conventional bounds variable (<coord>_bnds)
        bname_conv = f'{cname}_bnds'
        if bname_conv in ds_src and bname_conv not in ds_out:
            ds_out[bname_conv] = ds_src[bname_conv]
        # CF bounds attribute (supports time_bounds, etc.)
        if cname in ds_src:
            bname_attr = ds_src[cname].attrs.get('bounds')
            if (
                isinstance(bname_attr, str)
                and bname_attr in ds_src
                and bname_attr not in ds_out
            ):
                ds_out[bname_attr] = ds_src[bname_attr]
    return ds_out


def _prepare_vert_coords_and_mask(in_filename, variables):
    with xr.open_dataset(in_filename, decode_times=False) as ds:
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
    ds = xr.open_dataset(in_filename, decode_times=False)
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
    ds_out = _assign_coords_and_bounds(ds, ds_out)
    write_netcdf(
        ds_out,
        mask_filename,
        progress_bar=True,
        has_fill_values=lambda name, var: name
        in set(variables + ['src_valid']),
    )


def _write_interp_stage(
    mask_filename,
    variables,
    interpolator,
    ds_ismip,
    interp_filename,
    time_chunk,
):
    ds = xr.open_dataset(mask_filename, decode_times=False)
    if 'time' in ds.dims:
        ds = ds.chunk({'time': time_chunk})
    ds_out = xr.Dataset()
    for var in variables:
        da_interp = interpolator.interp(ds[var])
        ds_out[var] = da_interp.astype(np.float32)
    ds_out['src_frac_interp'] = interpolator.src_frac_interp.astype(np.float32)
    ds_out = _assign_coords_and_bounds(ds, ds_out)
    ds_out['z_extrap_bnds'] = ds_ismip['z_extrap_bnds']
    if 'lev' in ds_out:
        ds_out = ds_out.drop_vars(['lev'])
    write_netcdf(
        ds_out,
        interp_filename,
        progress_bar=True,
        has_fill_values=lambda name, var: name
        in set(variables + ['src_frac_interp']),
    )


def _write_normalize_stage(
    interp_filename,
    variables,
    interpolator,
    ds_ismip,
    normalized_filename,
    time_chunk,
):
    ds = xr.open_dataset(interp_filename, decode_times=False)
    if 'time' in ds.dims:
        ds = ds.chunk({'time': time_chunk})
    ds_out = xr.Dataset()
    for var in variables:
        da_norm = interpolator.normalize(ds[var])
        ds_out[var] = da_norm.astype(np.float32)
    ds_out['src_frac_interp'] = interpolator.src_frac_interp.astype(np.float32)
    ds_out = _assign_coords_and_bounds(ds, ds_out)
    ds_out['z_extrap_bnds'] = ds_ismip['z_extrap_bnds']
    write_netcdf(
        ds_out,
        normalized_filename,
        progress_bar=True,
        has_fill_values=lambda name, var: name
        in set(variables + ['src_frac_interp']),
    )


def _remap_horiz(
    config,
    in_filename,
    out_filename,
    model_prefix,
    tmpdir,
    logger,
    lat_var=None,
    lon_var=None,
    lon_dim=None,
):
    """
     Horizontally remap a vertically processed dataset to the ISMIP grid.

     Input/Output
     - Input is the output from the vertical pipeline (already masked,
        interpolated to ``z_extrap``, and normalized) stored in
        ``in_filename``. Output is written to ``out_filename`` on the ISMIP
        grid.

     Steps
     1) Open the input lazily. If ``method == 'bilinear'``, append a
         periodic longitude column to prevent a dateline seam.
     2) Build a lightweight mask dataset that contains only
         ``src_frac_interp`` and remap it once without renormalization.
         This carries the vertically interpolated valid fraction to the
         ISMIP grid and is reused across all time chunks (or the single
         no-time remap).
     3) Remap data variables in time chunks of
         ``[remap_cmip] horiz_time_chunk`` (if a time dimension exists);
         if no time dimension, do a single remap. Before remapping, drop
         ``src_frac_interp`` so renormalization does not act on it.
     4) Concatenate remapped chunks along time (if any), attach the
         horizontally remapped ``src_frac_interp`` from step 2, and write
         the final file. If the output has the expected (y, x) shape,
         attach ISMIP x/y coordinates and bounds.

     Notes
     - Horizontal renormalization is applied during remapping with the
        threshold ``[remap] threshold`` to handle partial source-cell
        coverage.
     - ``lat_var``, ``lon_var``, and ``lon_dim`` are provided or read from
        ``[cmip_dataset]`` by default so this works for CMIP or climatology.
    - ``model_prefix`` is used as the source grid identifier in map names.
    """

    method = config.get('remap', 'method')
    renorm_threshold = config.getfloat('remap', 'threshold')

    # Determine coordinate var/dim names
    if lat_var is None:
        lat_var = (
            config.get('cmip_dataset', 'lat_var')
            if config.has_option('cmip_dataset', 'lat_var')
            else 'lat'
        )
    if lon_var is None:
        lon_var = (
            config.get('cmip_dataset', 'lon_var')
            if config.has_option('cmip_dataset', 'lon_var')
            else 'lon'
        )
    if lon_dim is None:
        lon_dim = (
            config.get('cmip_dataset', 'lon_dim')
            if config.has_option('cmip_dataset', 'lon_dim')
            else 'lon'
        )

    in_grid_name = model_prefix

    # Open dataset lazily
    ds = xr.open_dataset(in_filename, chunks={'time': 1}, decode_times=False)
    # Capture original time bounds metadata to re-attach if remapping drops it
    tbname, tbda = _capture_time_bounds_dataset(ds)

    if method == 'bilinear':
        # ensure periodic longitude to avoid seam
        ds = add_periodic_lon(ds, lon_var=lon_var, periodic_dim=lon_dim)

    input_mask_path = os.path.join(tmpdir, 'input_mask.nc')
    output_mask_path = os.path.join(tmpdir, 'output_mask.nc')
    if os.path.exists(output_mask_path):
        ds_mask = xr.open_dataset(output_mask_path, decode_times=False)
    else:
        ds_mask = ds.copy()
        # Keep only src_frac_interp in the mask file
        keep_vars = ['src_frac_interp']
        ds_mask = ds_mask[keep_vars]
        write_netcdf(
            ds_mask,
            input_mask_path,
            progress_bar=True,
            has_fill_values=lambda name, var: name == 'src_frac_interp',
        )

        # remap the mask without renormalizing
        remap_lat_lon_to_ismip(
            in_filename=input_mask_path,
            in_grid_name=in_grid_name,
            out_filename=output_mask_path,
            map_dir=tmpdir,
            method=method,
            config=config,
            logger=logger,
            lon_var=lon_var,
            lat_var=lat_var,
        )
        ds_mask = xr.open_dataset(output_mask_path, decode_times=False)

    # If no time axis, do a single remap; otherwise chunk in time
    if 'time' not in ds.dims:
        remapped_chunks = _remap_no_time(
            ds,
            tmpdir,
            in_grid_name,
            method,
            config,
            logger,
            lon_var,
            lat_var,
            renorm_threshold,
        )
    else:
        remapped_chunks = _remap_with_time(
            ds,
            tmpdir,
            in_grid_name,
            method,
            config,
            logger,
            lon_var,
            lat_var,
            renorm_threshold,
        )

    # Sanity: ensure all chunks carry identical, unique z_extrap
    z0 = remapped_chunks[0]['z_extrap'].values
    if len(np.unique(z0)) != len(z0):
        raise ValueError(
            'First chunk has duplicate z_extrap values — aborting.'
        )
    for i, ds_chk in enumerate(remapped_chunks[1:], start=1):
        z = ds_chk['z_extrap'].values
        if z.shape != z0.shape or not np.allclose(z, z0):
            raise ValueError(
                f'Inconsistent z_extrap in chunk {i}: shape/values mismatch; '
                'possible corruption.'
            )

    # Concatenate along time if needed (or just take the single chunk)
    if len(remapped_chunks) == 1 and 'time' not in remapped_chunks[0].dims:
        ds_final = remapped_chunks[0]
    else:
        ds_final = xr.concat(remapped_chunks, dim='time', join='exact')

    # attach horizontally remapped src_frac_interp (time-invariant)
    ds_final['src_frac_interp'] = ds_mask['src_frac_interp']

    # Ensure x/y projection coordinates are present from the ISMIP grid
    ds_final = _attach_ismip_xy_if_match(ds_final, config)

    # Ensure time bounds variable and attribute are present on final output
    ds_final = _restore_time_bounds_dataset(ds_final, tbname, tbda)

    write_netcdf(
        ds_final,
        out_filename,
        progress_bar=True,
        has_fill_values=lambda name, var: name
        in set([v for v in ds_final.data_vars if v not in ds_final.coords]),
    )


def _capture_time_bounds_dataset(ds):
    tbname = None
    tbda = None
    if 'time' in ds and 'time' in ds.coords:
        tbname_attr = ds['time'].attrs.get('bounds')
        if isinstance(tbname_attr, str) and tbname_attr in ds:
            tbname = tbname_attr
            tbda = ds[tbname]
    return tbname, tbda


def _restore_time_bounds_dataset(ds, tbname, tbda):
    if 'time' in ds and tbname is not None:
        ds['time'].attrs['bounds'] = tbname
        if tbname not in ds and tbda is not None:
            if int(ds.sizes.get('time', -1)) == int(
                tbda.sizes.get('time', -2)
            ):
                ds[tbname] = tbda
    return ds


def _attach_ismip_xy_if_match(ds_final, config):
    try:
        ds_ismip = xr.open_dataset(get_ismip_grid_filename(config))
        if (
            'y' in ds_final.dims
            and 'x' in ds_final.dims
            and ds_final.sizes.get('y') == ds_ismip.sizes.get('y')
            and ds_final.sizes.get('x') == ds_ismip.sizes.get('x')
        ):
            ds_final = ds_final.assign_coords(
                {
                    'x': ('x', ds_ismip['x'].values),
                    'y': ('y', ds_ismip['y'].values),
                }
            )
            ds_final['x'].attrs = ds_ismip['x'].attrs
            ds_final['y'].attrs = ds_ismip['y'].attrs
            if 'x_bnds' in ds_ismip and 'x_bnds' not in ds_final:
                ds_final['x_bnds'] = ds_ismip['x_bnds']
            if 'y_bnds' in ds_ismip and 'y_bnds' not in ds_final:
                ds_final['y_bnds'] = ds_ismip['y_bnds']
        else:
            print(
                'Warning: Could not attach x/y from ISMIP grid because '
                'dimensions do not match expected (y, x).'
            )
    except Exception as exc:
        print(f'Warning: failed to attach x/y from ISMIP grid: {exc}')
    return ds_final


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
):
    input_chunk_path = os.path.join(tmpdir, 'input_single.nc')
    output_chunk_path = os.path.join(tmpdir, 'output_single.nc')

    subset = ds
    if 'src_frac_interp' in subset:
        subset = subset.drop_vars(['src_frac_interp'])
    write_netcdf(
        subset,
        input_chunk_path,
        progress_bar=True,
        has_fill_values=lambda name, var, _subset=subset: name
        in set([v for v in _subset.data_vars if v not in _subset.coords]),
    )

    remap_lat_lon_to_ismip(
        in_filename=input_chunk_path,
        in_grid_name=in_grid_name,
        out_filename=output_chunk_path,
        map_dir=tmpdir,
        method=method,
        config=config,
        logger=logger,
        lon_var=lon_var,
        lat_var=lat_var,
        renormalize=renorm_threshold,
    )
    remapped_chunk = xr.open_dataset(output_chunk_path, decode_times=False)
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
        if os.path.exists(output_chunk_path):
            print(
                f'Skipping remapping for chunk {i_start}-{i_end} '
                f'(already exists).'
            )
            remapped_chunk = xr.open_dataset(
                output_chunk_path,
                chunks={'time': 1},
                decode_times=False,
            )
            remapped_chunks.append(remapped_chunk)
            continue

        subset = ds.isel(time=slice(i_start, i_end))
        if 'src_frac_interp' in subset:
            subset = subset.drop_vars(['src_frac_interp'])
        write_netcdf(
            subset,
            input_chunk_path,
            progress_bar=True,
            has_fill_values=lambda name, var, _subset=subset: name
            in set([v for v in _subset.data_vars if v not in _subset.coords]),
        )

        remap_lat_lon_to_ismip(
            in_filename=input_chunk_path,
            in_grid_name=in_grid_name,
            out_filename=output_chunk_path,
            map_dir=tmpdir,
            method=method,
            config=config,
            logger=logger,
            lon_var=lon_var,
            lat_var=lat_var,
            renormalize=renorm_threshold,
        )

        remapped_chunk = xr.open_dataset(
            output_chunk_path, chunks={'time': 1}, decode_times=False
        )
        remapped_chunks.append(remapped_chunk)

    return remapped_chunks
