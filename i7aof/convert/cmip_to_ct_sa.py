import argparse
import os
import shutil
import uuid
from typing import List, Sequence, Tuple

import numpy as np
import xarray as xr
from mpas_tools.config import MpasConfigParser
from tqdm import tqdm

from i7aof.cmip import get_model_prefix
from i7aof.convert.paths import get_ct_sa_output_paths
from i7aof.convert.teos10 import convert_dataset_to_ct_sa
from i7aof.io import write_netcdf


def convert_cmip_to_ct_sa(
    model: str,
    scenario: str,
    inputdir: str | None = None,
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

    config = _load_config(model, user_config_filename)

    workdir = _get_or_config_path(config, workdir, 'workdir')
    inputdir = _get_or_config_path(config, inputdir, 'inputdir')
    _ensure_config_base_dirs(config, workdir=workdir, inputdir=inputdir)

    outdir = os.path.join(workdir, 'convert', model, scenario, 'Omon', 'ct_sa')
    os.makedirs(outdir, exist_ok=True)

    lat_var = config.get('cmip_dataset', 'lat_var')
    lon_var = config.get('cmip_dataset', 'lon_var')
    depth_var = (
        config.get('convert_cmip', 'depth_var')
        if config.has_option('convert_cmip', 'depth_var')
        else 'lev'
    )
    time_chunk = _parse_time_chunk(config)

    thetao_paths = list(config.getexpression(f'{scenario}_files', 'thetao'))
    so_paths = list(config.getexpression(f'{scenario}_files', 'so'))
    if len(thetao_paths) != len(so_paths):  # pragma: no cover - sanity guard
        raise ValueError(
            'Mismatched number of thetao and so files for scenario '
            f"'{scenario}'."
        )

    out_paths = get_ct_sa_output_paths(
        config=config,
        model=model,
        scenario=scenario,
        workdir=workdir,
    )

    for th_rel, so_rel, out_abs in zip(
        thetao_paths, so_paths, out_paths, strict=True
    ):
        if os.path.exists(out_abs):
            print(f'Converted file exists, skipping: {out_abs}')
            continue
        print(f'Converting to CT/SA: {os.path.basename(out_abs)}')
        th_abs = os.path.join(inputdir, th_rel)
        so_abs = os.path.join(inputdir, so_rel)
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
        '-i',
        '--inputdir',
        dest='inputdir',
        required=False,
        help='Base input directory (optional).',
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
        inputdir=args.inputdir,
        workdir=args.workdir,
        user_config_filename=args.config,
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _load_config(model: str, user_cfg: str | None) -> MpasConfigParser:
    config = MpasConfigParser()
    config.add_from_package('i7aof', 'default.cfg')
    config.add_from_package('i7aof.cmip', f'{get_model_prefix(model)}.cfg')
    if user_cfg is not None:
        config.add_user_config(user_cfg)
    return config


def _get_or_config_path(
    config: MpasConfigParser, supplied: str | None, section: str
) -> str:
    if supplied is not None:
        return supplied
    if config.has_option(section, 'base_dir'):
        return config.get(section, 'base_dir')
    raise ValueError(
        f'Missing configuration option: [{section}] base_dir. '
        'Please supply a user config file that defines this option.'
    )


def _parse_time_chunk(config: MpasConfigParser) -> int | None:
    if not config.has_option('convert_cmip', 'time_chunk'):
        return None
    raw = config.get('convert_cmip', 'time_chunk')
    if raw in ('', 'None', 'none'):
        return None
    return int(raw)


def _ensure_config_base_dirs(
    config: MpasConfigParser,
    *,
    workdir: str | None = None,
    inputdir: str | None = None,
) -> None:
    """Persist provided workdir/inputdir into config base_dir options."""
    if workdir is not None:
        config.set('workdir', 'base_dir', workdir)
    if inputdir is not None:
        config.set('inputdir', 'base_dir', inputdir)


def _process_file_pair(
    th_abs: str,
    so_abs: str,
    out_abs: str,
    depth_var: str,
    lat_var: str,
    lon_var: str,
    time_chunk: int | None,
) -> None:
    ds_thetao = xr.open_dataset(th_abs, decode_times=False)
    ds_so = xr.open_dataset(so_abs, decode_times=False)
    ds_thetao, ds_so = xr.align(ds_thetao, ds_so, join='exact')

    time_indices = _compute_time_indices(ds_thetao, time_chunk)
    out_base = os.path.splitext(os.path.basename(out_abs))[0]
    tmp_dir = os.path.join(
        os.path.dirname(out_abs), f'.tmp_{out_base}_{uuid.uuid4().hex}'
    )
    os.makedirs(tmp_dir, exist_ok=True)

    bounds_records = _capture_bounds(ds_thetao, depth_var, lat_var, lon_var)
    chunk_files: List[str] = []
    try:
        for t0, t1 in tqdm(time_indices, desc='time chunks', unit='chunk'):
            if 'time' in ds_thetao.dims:
                ds_th_chunk = ds_thetao.isel(time=slice(t0, t1))
                ds_so_chunk = ds_so.isel(time=slice(t0, t1))
            else:
                ds_th_chunk = ds_thetao
                ds_so_chunk = ds_so
            ds_ctsa = _convert_chunk_and_strip_bounds(
                ds_th_chunk,
                ds_so_chunk,
                lat_var,
                lon_var,
                depth_var,
                bounds_records,
            )
            if 'time' in ds_thetao.dims:
                chunk_name = f'{out_base}_part-{t0:06d}-{t1:06d}.nc'
            else:
                chunk_name = f'{out_base}_part-static.nc'
            chunk_path = os.path.join(tmp_dir, chunk_name)
            write_netcdf(ds_ctsa, chunk_path, progress_bar=False)
            chunk_files.append(chunk_path)

        if len(chunk_files) == 1:
            ds_final = xr.open_dataset(chunk_files[0], decode_times=False)
        else:
            ds_final = xr.open_mfdataset(
                chunk_files,
                combine='nested',
                concat_dim='time',
                decode_times=False,
            )
        _inject_bounds(ds_final, ds_thetao, bounds_records)
        write_netcdf(ds_final, out_abs, progress_bar=True)
        ds_final.close()
    finally:
        ds_thetao.close()
        ds_so.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _compute_time_indices(
    ds_thetao: xr.Dataset, time_chunk: int | None
) -> Sequence[Tuple[int, int]]:
    if 'time' not in ds_thetao.dims:
        return [(0, 0)]
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
