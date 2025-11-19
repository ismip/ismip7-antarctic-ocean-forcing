import argparse
import glob
import os
import re
import shutil
from typing import Sequence, Tuple

import numpy as np
import xarray as xr
from mpas_tools.config import MpasConfigParser
from tqdm import tqdm

from i7aof.config import load_config
from i7aof.convert.teos10 import convert_dataset_to_ct_sa
from i7aof.coords import ensure_cf_time_encoding
from i7aof.io import read_dataset
from i7aof.io_zarr import append_to_zarr, finalize_zarr_to_netcdf
from i7aof.time.bounds import capture_time_bounds, inject_time_bounds


def convert_cmip_to_ct_sa(
    model: str,
    scenario: str,
    workdir: str | None = None,
    user_config_filename: str | None = None,
) -> None:
    """
    Convert thetao/so monthly files to ct/sa for a CMIP model.

    Conversion is performed per thetao/so pair. Outputs store only ct & sa
    plus coordinate variables; bounds variables for depth/lat/lon are
    injected only in the final merged file to avoid acquiring a spurious
    time dimension.

    Parameters
    ----------
    model : str
        Name of the CMIP model (used to select the model config and to
        construct output paths).
    scenario : str
        Scenario key (e.g., 'historical', 'ssp585') used to pick input file
        lists from the config.
    inputdir : str, optional
        Base directory where the relative input file paths are resolved. If
        not provided, uses ``[inputdir] base_dir`` from the config.
    workdir : str, optional
        Base working directory where outputs will be written. If not
        provided, uses ``[workdir] base_dir`` from the config.
    user_config_filename : str, optional
        Optional user config that overrides defaults (paths, variable names,
        chunk sizes, etc.).
    """

    print(f'Converting CMIP model "{model}" scenario "{scenario}" to CT/SA.')

    config = load_config(
        model=model,
        inputdir=None,
        workdir=workdir,
        user_config_filename=user_config_filename,
    )

    workdir_base: str = config.get('workdir', 'base_dir')

    print(f'Using working directory: {workdir_base}')

    outdir = os.path.join(
        workdir_base, 'convert', model, scenario, 'Omon', 'ct_sa'
    )
    os.makedirs(outdir, exist_ok=True)

    print(f'Using output directory: {outdir}')

    lat_var = config.get('cmip_dataset', 'lat_var')
    lon_var = config.get('cmip_dataset', 'lon_var')
    depth_var = (
        config.get('convert_cmip', 'depth_var')
        if config.has_option('convert_cmip', 'depth_var')
        else 'lev'
    )
    time_chunk = _parse_time_chunk(config)

    # Inputs now come from the split workflow under workdir/split
    split_base = os.path.join(workdir_base, 'split', model, scenario, 'Omon')
    th_dir = os.path.join(split_base, 'thetao')
    so_dir = os.path.join(split_base, 'so')
    if not os.path.isdir(th_dir) or not os.path.isdir(so_dir):
        raise ValueError(
            'Expected split inputs not found. Please run the split workflow: '
            f"missing directories '{th_dir}' or '{so_dir}'."
        )

    th_files = sorted(glob.glob(os.path.join(th_dir, '*.nc')))
    so_files = sorted(glob.glob(os.path.join(so_dir, '*.nc')))

    # Pair files by their trailing year range _YYYY-YYYY
    def _key(path: str) -> tuple[int, int]:
        base = os.path.basename(path)
        m = re.search(r'_(\d{4})-(\d{4})\.nc$', base)
        if not m:
            raise ValueError(
                'Split filename missing year range suffix: ' + base
            )
        y0, y1 = int(m.group(1)), int(m.group(2))
        return y0, y1

    th_map = {_key(p): p for p in th_files}
    so_map = {_key(p): p for p in so_files}
    missing = [k for k in th_map.keys() if k not in so_map]
    if missing:
        raise ValueError(
            'Missing matching so files for thetao ranges: '
            + ', '.join([f'{a}-{b}' for a, b in sorted(missing)])
        )

    for y0y1 in sorted(th_map.keys()):
        th_abs = th_map[y0y1]
        so_abs = so_map[y0y1]
        # output path based on thetao basename with variable token replaced
        th_base = os.path.basename(th_abs)
        ct_base = (
            th_base.replace('thetao', 'ct_sa')
            if 'thetao' in th_base
            else f'ct_sa_{th_base}'
        )
        out_abs = os.path.join(outdir, ct_base)
        if os.path.exists(out_abs):
            print(f'Converted file exists, skipping: {out_abs}')
            continue
        print(f'Converting to CT/SA:\n{th_abs}\n{so_abs}\n{out_abs}')
        _process_file_pair(
            th_abs,
            so_abs,
            out_abs,
            depth_var,
            lat_var,
            lon_var,
            time_chunk,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert CMIP thetao/so to ct/sa on native grid.'
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
        help='Scenario (historical, ssp585, ...: required).',
    )
    parser.add_argument(
        '-w',
        '--workdir',
        dest='workdir',
        required=False,
        help='Base working directory (optional).',
    )
    parser.add_argument(
        '-c',
        '--config',
        dest='config',
        default=None,
        help='Path to user config file (optional).',
    )
    args = parser.parse_args()
    convert_cmip_to_ct_sa(
        model=args.model,
        scenario=args.scenario,
        workdir=args.workdir,
        user_config_filename=args.config,
    )


def _parse_time_chunk(config: MpasConfigParser) -> int | None:
    if not config.has_option('convert_cmip', 'time_chunk'):
        return None
    raw = config.get('convert_cmip', 'time_chunk')
    if raw in ('', 'None', 'none'):
        return None
    return int(raw)


def _process_file_pair(
    th_abs: str,
    so_abs: str,
    out_abs: str,
    depth_var: str,
    lat_var: str,
    lon_var: str,
    time_chunk: int | None,
) -> None:
    ds_thetao = read_dataset(th_abs)
    ds_so = read_dataset(so_abs)
    # Standardize time bounds name to 'time_bnds' (from e.g., 'time_bounds')
    ds_thetao = _standardize_time_bounds_to_time_bnds(ds_thetao)
    ds_so = _standardize_time_bounds_to_time_bnds(ds_so)
    # Enforce presence of time/time_bnds after normalization; there is no
    # point in proceeding without well-defined time bounds.
    if 'time' not in ds_thetao.dims:
        raise ValueError(
            'Expected a time dimension in thetao input but none were found. '
            f'File: {th_abs}'
        )
    if 'time_bnds' not in ds_thetao.variables:
        raise ValueError(
            'Expected time_bnds variable in thetao input after '
            'standardization but none were found. Ensure original CMIP '
            'files provide time bounds. File: '
            f'{th_abs}'
        )
    ds_thetao, ds_so = xr.align(ds_thetao, ds_so, join='exact')

    time_indices = _compute_time_indices(ds_thetao, time_chunk)
    out_base = os.path.splitext(os.path.basename(out_abs))[0]
    zarr_store = os.path.join(os.path.dirname(out_abs), f'{out_base}.zarr')
    if os.path.isdir(zarr_store):
        shutil.rmtree(zarr_store, ignore_errors=True)

    bounds_records = _capture_bounds(ds_thetao, depth_var, lat_var, lon_var)
    time_bounds = capture_time_bounds(ds_thetao)
    first = True
    for t0, t1 in tqdm(time_indices, desc='time chunks', unit='chunk'):
        ds_th_chunk = ds_thetao.isel(time=slice(t0, t1))
        ds_so_chunk = ds_so.isel(time=slice(t0, t1))
        ds_ctsa = _convert_chunk_and_strip_bounds(
            ds_th_chunk,
            ds_so_chunk,
            lat_var,
            lon_var,
            depth_var,
            bounds_records,
            time_bounds,
        )
        first = append_to_zarr(
            ds=ds_ctsa,
            zarr_store=zarr_store,
            first=first,
            append_dim='time',
        )

    # Intentionally nested: captures ds_thetao and ensures CF encoding
    # without widening the helper API.
    def _post(ds_z: xr.Dataset) -> xr.Dataset:
        # Restore spatial bounds/coords for depth/lat/lon
        _inject_bounds(ds_z, ds_thetao, bounds_records)

        # Ensure time coordinate (values + attrs) and time_bnds come
        # directly from the standardized thetao dataset so any attribute
        # manipulations in intermediate chunks don't erase the bounds
        # relationship.
        ds_z['time'] = ds_thetao['time']
        inject_time_bounds(ds_z, time_bounds)

        # Ensure CF-consistent encodings for time/time_bnds so units "stick".
        ensure_cf_time_encoding(
            ds_z,
            units='days since 1850-01-01 00:00:00',
            calendar=None,
            prefer_source=ds_thetao,
        )
        return ds_z

    finalize_zarr_to_netcdf(
        zarr_store=zarr_store,
        out_nc=out_abs,
        has_fill_values=['ct', 'sa'],
        compression=['ct', 'sa'],
        progress_bar=True,
        postprocess=_post,
    )
    ds_thetao.close()
    ds_so.close()


def _standardize_time_bounds_to_time_bnds(ds: xr.Dataset) -> xr.Dataset:
    """Rename any time bounds variable to 'time_bnds' and update attrs.

    If the dataset has a time coordinate with a bounds attribute (commonly
    'time_bounds'), this function renames that variable to 'time_bnds' and sets
    ds['time'].attrs['bounds'] = 'time_bnds'. If no bounds are present, the
    dataset is returned unchanged.
    """
    if 'time' not in ds:
        raise ValueError(
            'Expected time coordinate but none was found. '
            'Ensure the dataset provides time.'
        )
    tbname: str | None = None
    tcoord = ds['time']
    battr = tcoord.attrs.get('bounds') if hasattr(tcoord, 'attrs') else None
    if isinstance(battr, str) and battr in ds:
        tbname = battr
    else:
        # Fallback: common conventions
        if 'time_bnds' in ds:
            tbname = 'time_bnds'
        elif 'time_bounds' in ds:
            tbname = 'time_bounds'

    if tbname is None:
        raise ValueError(
            'Expected time bounds for time coordinate but none were found. '
            'Ensure the dataset provides time_bnds/time_bounds.'
        )

    if tbname != 'time_bnds':
        ds = ds.rename({tbname: 'time_bnds'})

    bnds_var = ds['time_bnds']
    dims = list(bnds_var.dims)
    if len(dims) == 2:
        _time_dim, second_dim = dims
        if second_dim != 'bnds':
            second_size = ds.sizes.get(second_dim)
            # Rename only if no conflicting 'bnds' dimension exists
            if 'bnds' not in ds.dims or ds.sizes.get('bnds') == second_size:
                ds = ds.rename({second_dim: 'bnds'})
    # Ensure the time coord points to the standardized name
    t_attrs = dict(getattr(ds['time'], 'attrs', {}))
    t_attrs['bounds'] = 'time_bnds'
    ds['time'].attrs = t_attrs
    return ds


def _compute_time_indices(
    ds_thetao: xr.Dataset, time_chunk: int | None
) -> Sequence[Tuple[int, int]]:
    nt = ds_thetao.sizes['time']
    if time_chunk is None or time_chunk <= 0:
        return [(0, nt)]
    chunk = min(time_chunk, nt)
    return [(start, min(start + chunk, nt)) for start in range(0, nt, chunk)]


def _capture_bounds(
    ds_thetao: xr.Dataset,
    depth_var: str,
    lat_var: str,
    lon_var: str,
) -> list[tuple[str, str, xr.DataArray]]:
    records: list[tuple[str, str, xr.DataArray]] = []
    for coord_name in (depth_var, lat_var, lon_var):
        if coord_name not in ds_thetao:
            continue
        coord_da = ds_thetao[coord_name]
        bname = coord_da.attrs.get('bounds')
        if isinstance(bname, str) and bname in ds_thetao:
            records.append((coord_name, bname, ds_thetao[bname]))
    missing = [
        c
        for c in (depth_var, lat_var, lon_var)
        if c not in {r[0] for r in records}
    ]
    if missing:
        raise ValueError(
            'Missing expected bounds for coordinates: ' + ', '.join(missing)
        )
    return records


def _convert_chunk_and_strip_bounds(
    ds_th_chunk: xr.Dataset,
    ds_so_chunk: xr.Dataset,
    lat_var: str,
    lon_var: str,
    depth_var: str,
    bounds_records: list[tuple[str, str, xr.DataArray]],
    time_bounds: tuple[str, xr.DataArray] | None,
) -> xr.Dataset:
    ds_ctsa = convert_dataset_to_ct_sa(
        ds_th_chunk,
        ds_so_chunk,
        thetao_var='thetao',
        so_var='so',
        lat_var=lat_var,
        lon_var=lon_var,
        depth_var=depth_var,
    )
    ds_ctsa['ct'] = ds_ctsa['ct'].astype(np.float32)
    ds_ctsa['sa'] = ds_ctsa['sa'].astype(np.float32)
    for coord_name, bname, _bda in bounds_records:
        if (
            coord_name in ds_ctsa.coords
            and 'bounds' in ds_ctsa[coord_name].attrs
        ):
            ds_ctsa[coord_name].attrs.pop('bounds', None)
        if bname in ds_ctsa.variables:
            ds_ctsa = ds_ctsa.drop_vars(bname)
    # Drop time bounds and its attribute (time is mandatory here)
    if time_bounds is not None:
        tbname, _tbda = time_bounds
        ds_ctsa['time'].attrs.pop('bounds', None)
        if tbname in ds_ctsa.variables:
            ds_ctsa = ds_ctsa.drop_vars(tbname)
    return ds_ctsa


def _inject_bounds(
    ds_final: xr.Dataset,
    ds_thetao: xr.Dataset,
    bounds_records: list[tuple[str, str, xr.DataArray]],
) -> None:
    for coord_name, bname, bda in bounds_records:
        ds_final[bname] = bda.load()
        if coord_name not in ds_final.coords:
            ds_final = ds_final.assign_coords(
                {coord_name: ds_thetao[coord_name]}
            )
        ds_final[coord_name].attrs['bounds'] = bname
