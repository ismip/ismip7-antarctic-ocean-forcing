#!/usr/bin/env python
import argparse
import os
import shutil

import numpy as np
import xarray as xr
from mpas_tools.config import MpasConfigParser
from mpas_tools.logging import LoggingContext

from i7aof.cmip import get_model_prefix
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
    variable,
    scenario,
    inputdir=None,
    workdir=None,
    user_config_filename=None,
):
    """
    Remap CMIP data to the ISMIP grid with two stages:
    1) vertical interpolation to ISMIP z_extrap levels, then
    2) horizontal remapping to the ISMIP lat/lon grid.

    This function orchestrates the basic flow per input file:
    - Prepare output dirs and ensure the ISMIP grid exists.
    - For each monthly file:
      * Vertical pipeline (see _vert_mask_interp_norm):
        mask invalid source points -> interpolate in z -> normalize.
      * Horizontal remap of the vertically processed data to ISMIP grid.

    Parameters
    ----------
    model : str
        Name of the CMIP model to remap

    variable : {'thetao', 'so'}
        The name of the variable to remap

    scenario : str
        The name of the scenario ('historical', 'ssp585', etc.)

    inputdir : str, optional
        The base input directory where the CMIP monthly input files are
        located

    workdir : str, optional
        The base work directory within which the remapped files will be placed

    user_config_filename : str, optional
        The path to a file with user config options that override the defaults
    """

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
                'Please supply a user config file that '
                'defines this option.'
            )

    if inputdir is None:
        if config.has_option('inputdir', 'base_dir'):
            inputdir = config.get('inputdir', 'base_dir')
        else:
            raise ValueError(
                'Missing configuration option: [inputdir] base_dir. '
                'Please supply a user config file that '
                'defines this option.'
            )

    outdir = os.path.join(workdir, 'remap', model, scenario, 'Omon', variable)
    os.makedirs(outdir, exist_ok=True)

    os.chdir(workdir)

    ismip_res_str = get_res_string(config)

    in_rel_paths = config.getexpression(f'{scenario}_files', variable)

    in_files = []
    out_files = []

    for rel_filename in in_rel_paths:
        base_filename = os.path.basename(rel_filename)
        if 'gn' not in base_filename:
            raise ValueError(
                f'Expected input to be on native grid (gn): {base_filename}'
            )

        out_filename = base_filename.replace('gn', f'ismip{ismip_res_str}')
        out_filename = os.path.join(outdir, out_filename)

        in_filename = os.path.join(inputdir, rel_filename)
        in_files.append(in_filename)
        out_files.append(out_filename)

    # Ensure the destination ISMIP grid files exist (used by both steps)
    write_ismip_grid(config)

    for index, (in_filename, out_filename) in enumerate(
        zip(in_files, out_files, strict=True)
    ):
        if os.path.exists(out_filename):
            print(f'Remapped file exists, skipping: {out_filename}')
            continue

        # Per-file tmp dirs for clarity and clean-up
        # Vertical stage tmp directory (mask -> interp -> normalize)
        vert_tmpdir = os.path.join(
            outdir, f'tmp_vert_interp_{variable}_{index}'
        )
        os.makedirs(vert_tmpdir, exist_ok=True)

        # Horizontal stage tmp directory (time-chunked remap + masks)
        horiz_tmpdir = os.path.join(
            outdir, f'tmp_horiz_remap_{variable}_{index}'
        )
        os.makedirs(horiz_tmpdir, exist_ok=True)

        # 1) Vertical pipeline: masking -> vertical interpolation -> normalize
        vert_interp_filenames = _vert_mask_interp_norm(
            config, in_filename, outdir, variable, vert_tmpdir
        )

        with LoggingContext(__name__) as logger:
            # 2) Horizontal remap to ISMIP lat/lon grid
            # Requires a logger to capture output from ncremap calls (we use
            # stdout and stderr, rather than a log file here)
            _remap_horiz(
                config,
                vert_interp_filenames,
                out_filename,
                variable,
                model_prefix,
                horiz_tmpdir,
                logger,
            )

        # Always clean up tmp dirs for this input file
        shutil.rmtree(vert_tmpdir)
        shutil.rmtree(horiz_tmpdir)


def main():
    parser = argparse.ArgumentParser(
        description='Remap CMIP model data to ISMIP grid.'
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
        '-v',
        '--variable',
        dest='variable',
        type=str,
        required=True,
        help='Name of the variable to remap ("thetao" or "so": required).',
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
        '-i',
        '--inputdir',
        dest='inputdir',
        type=str,
        required=False,
        help='Path to the base input directory (optional).',
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
        variable=args.variable,
        scenario=args.scenario,
        inputdir=args.inputdir,
        workdir=args.workdir,
        user_config_filename=args.config,
    )


## helper functions for vertical interpolation


def _vert_mask(
    interpolator,
    src_filename,
    mask_filename,
    time_chunk,
    variable,
    lev,
    lev_bnds,
):
    """
    Mask the dataset before vertical interpolation.

    Why a mask?
    - Construct a boolean mask of valid source values on the native 3D grid
      (lev × lat × lon). It is used to:
      1) zero-out invalid data in all 3D variables prior to interpolation;
      2) be vertically interpolated to ISMIP z_extrap levels along with
         the data, yielding a valid fraction on the ISMIP grid (written
         later as 'src_frac_interp');
      3) normalize the vertically interpolated fields using that fraction.

    This step writes an intermediate file containing the masked variable,
    standardized vertical coordinates ('lev', 'lev_bnds'), and the
    'src_valid' mask needed by later stages.
    """

    ds = xr.open_dataset(src_filename, decode_times=False)
    ds = ds.chunk({'time': time_chunk})

    ds_out = xr.Dataset()

    ds_out = ds_out.assign_coords(
        {
            'lev': ('lev', lev.data),
            'lev_bnds': (('lev', 'd2'), lev_bnds.data),
        }
    )
    ds_out['lev'].attrs = lev.attrs

    da = ds[variable]

    da_masked = interpolator.mask_and_sort(da)
    ds_out[variable] = da_masked.astype(np.float32)

    ds_out['src_valid'] = interpolator.src_valid.astype(np.float32)
    for var in ['lat', 'lon', 'time']:
        ds_out[var] = ds[var]
        var_bnds = f'{var}_bnds'
        ds_out[var_bnds] = ds[var_bnds]

    write_netcdf(ds_out, mask_filename, progress_bar=True)


def _vert_interp(
    interpolator,
    ds_ismip,
    time_chunk,
    mask_filename,
    interp_filename,
    variable,
):
    """Vertically interpolate the dataset"""
    # open it again to get a clean dataset
    ds = xr.open_dataset(mask_filename, decode_times=False)
    ds = ds.chunk({'time': time_chunk})
    da_masked = ds[variable]

    da_interp = interpolator.interp(da_masked)

    ds_out = xr.Dataset()
    ds_out[variable] = da_interp.astype(np.float32)
    ds_out['src_frac_interp'] = interpolator.src_frac_interp.astype(np.float32)
    for var in ['lat', 'lon', 'time']:
        ds_out[var] = ds[var]
        var_bnds = f'{var}_bnds'
        ds_out[var_bnds] = ds[var_bnds]
    ds_out['z_extrap_bnds'] = ds_ismip['z_extrap_bnds']
    ds_out = ds_out.drop_vars(['lev'])

    write_netcdf(ds_out, interp_filename, progress_bar=True)


def _vert_normalize(
    interpolator,
    ds_ismip,
    time_chunk,
    interp_filename,
    normalize_filename,
    variable,
):
    """
    Normalize the dataset following vertical interpolation.

    Why normalization?
    - Intensive fields like temperature and salinity should not be reduced
      just because only a fraction of the destination cell is valid on the
      source grid. Without renormalization, values near bathymetry or
      partially sampled columns would be biased low.
    - We use the vertically interpolated validity (written as
      'src_frac_interp') to renormalize the field after interpolation:
        * where the valid fraction is sufficiently large, divide by that
          fraction to restore the correct magnitude;
        * where the fraction is too small to be reliable, set the result to
          a mask/missing value rather than amplify noise.

    Inputs/Outputs
    - Reads the vertically interpolated file from the previous step.
    - Writes the normalized variable and carries through 'src_frac_interp'
      and 'z_extrap_bnds'.
    """
    # open it again to get a clean dataset
    ds = xr.open_dataset(interp_filename, decode_times=False)
    ds = ds.chunk({'time': time_chunk})
    da_interp = ds[variable]

    da_normalized = interpolator.normalize(da_interp)

    ds_out = xr.Dataset()
    ds_out[variable] = da_normalized.astype(np.float32)
    ds_out['src_frac_interp'] = interpolator.src_frac_interp.astype(np.float32)
    for var in ['lat', 'lon', 'time']:
        ds_out[var] = ds[var]
        var_bnds = f'{var}_bnds'
        ds_out[var_bnds] = ds[var_bnds]
    ds_out['z_extrap_bnds'] = ds_ismip['z_extrap_bnds']

    write_netcdf(ds_out, normalize_filename, progress_bar=True)


def _vert_mask_interp_norm(config, in_filename, outdir, variable, tmpdir):
    """Mask, vertically interpolate and normalize the dataset"""

    # Overview of the vertical pipeline for a single input file:
    # 1) Prepare per-stage temporary filenames under tmpdir.
    # 2) Fast-path return if the final normalized file already exists.
    # 3) Open the ISMIP grid (needed for z_extrap and its bounds).
    # 4) Open the source dataset and standardize the vertical coordinate:
    #    - fix_src_z_coord() enforces a monotonic 'lev' and valid 'lev_bnds',
    #      returning sanitized arrays we re-attach to the dataset.
    # 5) Build a source-validity mask from the first time slice to mark ocean
    #    points with valid data (shape: lev x lat x lon). The time dimension is
    #    dropped so the mask can be reused across time.
    # 6) Read configuration for time chunking during vertical steps (memory
    #    management) and construct a VerticalInterpolator that carries the
    #    validity mask and target vertical coordinate ('z_extrap').
    # 7) Run the three stages:
    #    a) _vert_mask: apply mask/sorting on the source vertical coordinate.
    #    b) _vert_interp: interpolate masked profiles to z_extrap; writes
    #       'src_frac_interp' (fraction of valid source levels used).
    #    c) _vert_normalize: normalize interpolated profiles for consistency.
    # 8) Return the path to the normalized per-file output for downstream
    #    horizontal remapping.

    # 1) Temporary file paths for each vertical stage output
    mask_filename = os.path.join(tmpdir, f'{variable}_masked.nc')
    interp_filename = os.path.join(tmpdir, f'{variable}_interp.nc')
    normalized_filename = os.path.join(tmpdir, f'{variable}_normalized.nc')

    # 2) Skip computation if the final vertical product already exists
    if os.path.exists(normalized_filename):
        print(
            f'Vertically interpolated file exists, skipping: '
            f'{normalized_filename}'
        )
        return normalized_filename

    # 3) ISMIP grid contains 'z_extrap' and 'z_extrap_bnds' needed later
    ds_ismip = xr.open_dataset(get_ismip_grid_filename(config))

    # 4) Open the source dataset and ensure a clean, monotonic vertical axis
    with xr.open_dataset(in_filename, decode_times=False) as ds:
        # Standardize the source vertical coordinate and bounds
        lev, lev_bnds = fix_src_z_coord(ds, 'lev', 'lev_bnds')
        ds = ds.assign_coords({'lev': ('lev', lev.data)})
        ds['lev'].attrs = lev.attrs
        ds['lev_bnds'] = lev_bnds

        # 5) Compute validity mask from the first time sample; drop time dim
        src_valid = ds[variable].isel(time=0).notnull()
        src_valid = src_valid.drop_vars(['time'])

    # 6) Chunk size for vertical operations (memory/performance control)
    time_chunk = config.getint('remap_cmip', 'vert_time_chunk')

    # Create the interpolator that will perform masking/interp/normalization
    interpolator = VerticalInterpolator(
        src_valid=src_valid,
        src_coord='lev',
        dst_coord='z_extrap',
        config=config,
    )

    # 7a) Mask/sort the source data on the vertical axis and write output
    _vert_mask(
        interpolator,
        in_filename,
        mask_filename,
        time_chunk,
        variable,
        lev,
        lev_bnds,
    )

    # 7b) Interpolate masked data to z_extrap levels; writes src_frac_interp
    _vert_interp(
        interpolator,
        ds_ismip,
        time_chunk,
        mask_filename,
        interp_filename,
        variable,
    )

    # 7c) Normalize the interpolated profiles and finalize vertical product
    _vert_normalize(
        interpolator,
        ds_ismip,
        time_chunk,
        interp_filename,
        normalized_filename,
        variable,
    )

    print(
        f'Vertical interpolation completed and saved to '
        f"'{normalized_filename}'."
    )

    # 8) Hand back the normalized file for horizontal remapping
    return normalized_filename


## helper functions for horizontal interpolation


def _remap_horiz(
    config,
    in_filename,
    out_filename,
    variable,
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
        ds_mask = ds_mask.drop_vars([variable, 'time', 'time_bnds'])
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
