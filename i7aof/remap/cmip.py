#!/usr/bin/env python
import argparse
import os
import shutil

import numpy as np
import xarray as xr
from mpas_tools.config import MpasConfigParser
from mpas_tools.logging import LoggingContext

from i7aof.cmip import get_model_prefix
from i7aof.convert.paths import get_ct_sa_output_paths
from i7aof.grid.ismip import (
    get_ismip_grid_filename,
    get_res_string,
    write_ismip_grid,
)
from i7aof.io import write_netcdf
from i7aof.remap import add_periodic_lon, remap_lat_lon_to_ismip
from i7aof.vert.interp import VerticalInterpolator, fix_src_z_coord


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

    config = MpasConfigParser()
    config.add_from_package('i7aof', 'default.cfg')
    config.add_from_package('i7aof.cmip', f'{model_prefix}.cfg')
    if user_config_filename is not None:
        config.add_user_config(user_config_filename)

    if workdir is None:
        if config.has_option('workdir', 'base_dir'):
            workdir = config.get('workdir', 'base_dir')
        else:
            raise ValueError(
                'Missing configuration option: [workdir] base_dir. '
                'Please supply a user config file that defines this option.'
            )

    outdir = os.path.join(workdir, 'remap', model, scenario, 'Omon', 'ct_sa')
    os.makedirs(outdir, exist_ok=True)
    os.chdir(workdir)

    ismip_res_str = get_res_string(config)
    return config, workdir, outdir, ismip_res_str, model_prefix


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

    try:
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
    finally:
        # Always clean up tmp dirs for this input file
        shutil.rmtree(vert_tmpdir)
        shutil.rmtree(horiz_tmpdir)


def _vert_mask_interp_norm_multi(
    config, in_filename, outdir, variables, tmpdir
):
    """Mask, vertically interpolate and normalize a multi-var dataset."""

    # Overview of the vertical pipeline (per ct_sa native file):
    # 1) Define stage filenames under tmpdir (masked -> interp -> normalized).
    # 2) If the final normalized file exists, return it (resumable).
    # 3) Open the ISMIP grid to get 'z_extrap' and 'z_extrap_bnds' (targets).
    # 4) Open the source dataset and standardize the vertical coordinate:
    #    fix_src_z_coord() yields a monotonic 'lev' and valid 'lev_bnds',
    #    which are reattached as coordinates.
    # 5) Build a combined source-validity mask at the first time step from all
    #    variables (ct and sa): src_valid = AND over variables. The time
    #    dimension is dropped so the mask can be reused for all times.
    # 6) Read vertical time-chunk settings and construct VerticalInterpolator
    #    with src_valid, src_coord='lev' and dst_coord='z_extrap'.
    # 7) Run three stages:
    #    a) mask: apply mask and vertical sorting on 'lev'; write 'src_valid'.
    #    b) interp: interpolate masked profiles to 'z_extrap'; write
    #       'src_frac_interp' (fraction of valid source levels per column),
    #       attach 'z_extrap_bnds', and drop 'lev'.
    #    c) normalize: renormalize interpolated profiles using src_frac_interp.
    # 8) Return the path to the normalized file for downstream horizontal
    #    remapping.

    mask_filename = os.path.join(tmpdir, 'ctsa_masked.nc')
    interp_filename = os.path.join(tmpdir, 'ctsa_interp.nc')
    normalized_filename = os.path.join(tmpdir, 'ctsa_normalized.nc')

    if os.path.exists(normalized_filename):
        print(
            'Vertically interpolated file exists, skipping: '
            f'{normalized_filename}'
        )
        return normalized_filename

    ds_ismip = xr.open_dataset(get_ismip_grid_filename(config))

    with xr.open_dataset(in_filename, decode_times=False) as ds:
        lev, lev_bnds = fix_src_z_coord(ds, 'lev', 'lev_bnds')
        ds = ds.assign_coords({'lev': ('lev', lev.data)})
        ds['lev'].attrs = lev.attrs
        ds['lev_bnds'] = lev_bnds

        # combine validity from all variables at first time slice
        src_valid = None
        for var in variables:
            valid = ds[var].isel(time=0).notnull().drop_vars(['time'])
            src_valid = valid if src_valid is None else (src_valid & valid)

    time_chunk = config.getint('remap_cmip', 'vert_time_chunk')

    interpolator = VerticalInterpolator(
        src_valid=src_valid,
        src_coord='lev',
        dst_coord='z_extrap',
        config=config,
    )

    # mask stage
    ds = xr.open_dataset(in_filename, decode_times=False)
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
    for var in ['lat', 'lon', 'time']:
        ds_out[var] = ds[var]
        var_bnds = f'{var}_bnds'
        if var_bnds in ds:
            ds_out[var_bnds] = ds[var_bnds]
    write_netcdf(ds_out, mask_filename, progress_bar=True)

    # interp stage
    ds = xr.open_dataset(mask_filename, decode_times=False)
    ds = ds.chunk({'time': time_chunk})
    ds_out = xr.Dataset()
    for var in variables:
        da_interp = interpolator.interp(ds[var])
        ds_out[var] = da_interp.astype(np.float32)
    ds_out['src_frac_interp'] = interpolator.src_frac_interp.astype(np.float32)
    for var in ['lat', 'lon', 'time']:
        ds_out[var] = ds[var]
        var_bnds = f'{var}_bnds'
        if var_bnds in ds:
            ds_out[var_bnds] = ds[var_bnds]
    ds_out['z_extrap_bnds'] = ds_ismip['z_extrap_bnds']
    ds_out = ds_out.drop_vars(['lev'])
    write_netcdf(ds_out, interp_filename, progress_bar=True)

    # normalize stage
    ds = xr.open_dataset(interp_filename, decode_times=False)
    ds = ds.chunk({'time': time_chunk})
    ds_out = xr.Dataset()
    for var in variables:
        da_norm = interpolator.normalize(ds[var])
        ds_out[var] = da_norm.astype(np.float32)
    ds_out['src_frac_interp'] = interpolator.src_frac_interp.astype(np.float32)
    for var in ['lat', 'lon', 'time']:
        ds_out[var] = ds[var]
        var_bnds = f'{var}_bnds'
        if var_bnds in ds:
            ds_out[var_bnds] = ds[var_bnds]
    ds_out['z_extrap_bnds'] = ds_ismip['z_extrap_bnds']
    write_netcdf(ds_out, normalized_filename, progress_bar=True)

    print(
        'Vertical interpolation completed and saved to '
        f"'{normalized_filename}'."
    )

    return normalized_filename


def _remap_horiz(
    config,
    in_filename,
    out_filename,
    model_prefix,
    tmpdir,
    logger,
):
    """
    Horizontally remap the vertically processed dataset to the ISMIP grid.

    Workflow
    - Input is the single-file output of the vertical pipeline (already masked,
      vertically interpolated and normalized). We map it to the ISMIP lat/lon
      grid and write the final product.
    - Steps:
      1) Open the source dataset lazily; if ``method == 'bilinear'``, add a
         periodic longitude column to avoid a seam at the dateline.
      2) Build a lightweight, time-invariant "mask" dataset by dropping the
         main variable and time coordinates, and remap it once without
         renormalization. This carries ancillary variables (notably
         ``src_frac_interp`` from the vertical stage) onto the ISMIP grid. The
         remapped mask is reused for all time chunks and broadcast in time.
      3) Remap the data in time chunks of size
         ``[remap_cmip] horiz_time_chunk`` for memory efficiency. Each chunk is
         written to a temporary file and remapped with
         ``renormalize = [remap] threshold`` so horizontal cell-coverage
         fractions are handled consistently.
      4) Concatenate remapped chunks along time, attach the horizontally
         remapped ``src_frac_interp`` from the mask (time-invariant), and write
         the final output file.

    Horizontal renormalization vs. vertical normalization
    - Vertical normalization (already applied) corrects for partial coverage in
      the water column using the vertically interpolated validity fraction
      (``src_frac_interp``) per column.
    - Horizontal renormalization (this step) corrects for partial horizontal
      coverage of source cells contributing to a destination ISMIP cell. If the
      covered fraction exceeds ``[remap] threshold``, values are scaled by that
      fraction; otherwise the destination cell is masked to avoid inflating
      noise.

    Parameters
    ----------
    config : MpasConfigParser
        Configuration with remapping options (method, thresholds, grid
        metadata, chunk sizes).
    in_filename : str
        Path to the per-file output from the vertical pipeline.
    out_filename : str
        Path where the horizontally remapped file will be written.
    variable : str
        Name of the data variable to remap.
    model_prefix : str
        Short identifier for the source grid used in map-file names/metadata.
    tmpdir : str
        Directory for temporary mask and chunk files.
    logger : logging.Logger
        Logger used to capture remapping tool output.
    """

    method = config.get('remap', 'method')
    renorm_threshold = config.getfloat('remap', 'threshold')
    lat_var = config.get('remap_cmip', 'lat_var')
    lon_var = config.get('remap_cmip', 'lon_var')
    lon_dim = config.get('remap_cmip', 'lon_dim')
    in_grid_name = model_prefix

    # Open dataset (but do not load into memory)
    ds = xr.open_dataset(in_filename, chunks={'time': 1}, decode_times=False)

    if method == 'bilinear':
        # we need to add a periodic longitude value or remapping will have a
        # seam
        ds = add_periodic_lon(ds, lon_var=lon_var, periodic_dim=lon_dim)

    input_mask_path = os.path.join(tmpdir, 'input_mask.nc')
    output_mask_path = os.path.join(tmpdir, 'output_mask.nc')
    if os.path.exists(output_mask_path):
        ds_mask = xr.open_dataset(output_mask_path, decode_times=False)
    else:
        ds_mask = ds.copy()
        # Keep only src_frac_interp and coords in the mask file
        keep_vars = ['src_frac_interp']
        ds_mask = ds_mask[keep_vars]
        write_netcdf(ds_mask, input_mask_path, progress_bar=True)

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

    # remap in 10-year chunks (by default)
    chunk_size = config.getint('remap_cmip', 'horiz_time_chunk')
    n_time = ds.sizes['time']
    time_indices = np.arange(0, n_time, chunk_size)

    remapped_chunks = []

    for i_start in time_indices:
        i_end = min(i_start + chunk_size, n_time)

        input_chunk_path = os.path.join(tmpdir, f'input_{i_start}_{i_end}.nc')

        # Remapped output path
        output_chunk_path = os.path.join(
            tmpdir, f'output_{i_start}_{i_end}.nc'
        )
        if os.path.exists(output_chunk_path):
            print(
                f'Skipping remapping for chunk {i_start}-{i_end} '
                f'(already exists).'
            )
            # Load remapped chunk
            remapped_chunk = xr.open_dataset(
                output_chunk_path, chunks={'time': 1}, decode_times=False
            )
            remapped_chunks.append(remapped_chunk)
            continue

        # Slice dataset
        subset = ds.isel(time=slice(i_start, i_end))
        # drop fraction before remap so renormalize doesn't touch it
        if 'src_frac_interp' in subset:
            subset = subset.drop_vars(['src_frac_interp'])

        # Write temporary input chunk
        write_netcdf(subset, input_chunk_path, progress_bar=True)

        # Run remapping
        remap_lat_lon_to_ismip(
            in_filename=input_chunk_path,
            in_grid_name=in_grid_name,
            out_filename=output_chunk_path,
            map_dir='.',
            method=method,
            config=config,
            logger=logger,
            lon_var=lon_var,
            lat_var=lat_var,
            renormalize=renorm_threshold,
        )

        # Load remapped chunk
        remapped_chunk = xr.open_dataset(
            output_chunk_path, chunks={'time': 1}, decode_times=False
        )
        remapped_chunks.append(remapped_chunk)

    # Concatenate all remapped chunks along time
    ds_final = xr.concat(remapped_chunks, dim='time')
    ds_final['src_frac_interp'] = ds_mask['src_frac_interp']

    # Save final output
    write_netcdf(ds_final, out_filename, progress_bar=True)
