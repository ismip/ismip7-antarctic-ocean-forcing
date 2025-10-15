import argparse
import os
import shutil
import warnings
from typing import List, Sequence, Tuple

import gsw
import numpy as np
import xarray as xr
from mpas_tools.config import MpasConfigParser
from tqdm import tqdm

from i7aof.cmip import get_model_prefix
from i7aof.convert.teos10 import _pressure_from_z
from i7aof.extrap.shared import _ensure_ismip_grid
from i7aof.grid.ismip import get_res_string
from i7aof.io import write_netcdf
from i7aof.io_zarr import append_to_zarr

__all__ = [
    'cmip_ct_sa_ann_to_thetao_so_tf',
    'main_cmip',
    'clim_ct_sa_to_thetao_so',
    'main_clim',
]

# Standard variable attributes
THETAO_ATTRS = {
    'units': 'degC',
    'long_name': 'Sea Water Potential Temperature',
}
SO_ATTRS = {
    'units': '1e-3',
    'long_name': 'Sea Water Practical Salinity',
}


# --------------------------- CMIP WORKFLOW ---------------------------


def cmip_ct_sa_ann_to_thetao_so_tf(
    *,
    model: str,
    scenario: str,
    clim_name: str,
    workdir: str | None = None,
    user_config_filename: str | None = None,
    time_chunk_years: int | None = None,
    progress: bool = True,
) -> List[str]:
    """
    Convert annual-mean CT/SA back to thetao/so and copy TF to final layout.

    Inputs are discovered under:

        workdir/biascorr/<model>/<scenario>/<clim_name>/Oyr/ct_sa_tf

    Annual outputs are written into:

        workdir/biascorr/<model>/<scenario>/<clim_name>/Oyr/thetao_so_tf

    - thetao/so are computed from ct/sa per time chunk and written via a
      temporary Zarr store for robustness and memory efficiency.
    - tf files are simply copied/read-through from the input annual folder.

    Parameters
    ----------
    model, scenario, clim_name : str
        Identify the CMIP dataset and climatology context.
    workdir : str, optional
        Base working directory; if omitted, uses config "[workdir] base_dir".
    user_config_filename : str, optional
        Optional user config overriding defaults.
    time_chunk_years : int, optional
        Number of annual time steps per chunk. If None, read from config
        section [ct_sa_to_thetao_so] time_chunk_years; if missing, process
        all time at once.
    progress : bool, optional
        If True (default), show a tqdm progress bar over time chunks.

    Returns
    -------
    list of str
        Paths of output annual thetao/so/tf files created or found.
    """
    config = _load_config(model, clim_name, user_config_filename)
    workdir = _get_or_config_path(config, workdir, 'workdir')
    config.set('workdir', 'base_dir', workdir)

    # Directories
    in_dir = os.path.join(
        workdir, 'biascorr', model, scenario, clim_name, 'Oyr', 'ct_sa_tf'
    )
    out_dir = os.path.join(
        workdir, 'biascorr', model, scenario, clim_name, 'Oyr', 'thetao_so_tf'
    )
    os.makedirs(out_dir, exist_ok=True)

    # Chunk size (years) from config unless provided
    if time_chunk_years is None:
        time_chunk_years = _parse_time_chunk_years(config)

    # Ensure ISMIP grid for coordinates and for pressure/lat in SP_from_SA
    grid_path = _ensure_ismip_grid(config, workdir)
    ismip_res_str = get_res_string(config, extrap=False)

    # Collect annual CT/SA pairs and TF files from input directory
    ct_sa_pairs = _collect_annual_ct_sa_files(in_dir, ismip_res_str)
    tf_files = _collect_annual_tf_files(in_dir, ismip_res_str)

    outputs: List[str] = []

    if not ct_sa_pairs and not tf_files:
        raise FileNotFoundError(
            'No annual CT/SA or TF files found. Expected under: ' + in_dir
        )

    # First copy/convert TF files (simple pass-through)
    for tf_in in tf_files:
        tf_out = os.path.join(out_dir, os.path.basename(tf_in))
        if os.path.exists(tf_out):
            print(f'TF annual exists, skipping: {tf_out}')
            outputs.append(tf_out)
            continue
        print(f'Copying TF annual: {os.path.basename(tf_out)}')
        _copy_netcdf(tf_in, tf_out)
        outputs.append(tf_out)

    # Now convert CT/SA -> thetao and so per pair of files
    for ct_path, sa_path in ct_sa_pairs:
        out_thetao = _output_path_for_thetao_annual(ct_path, out_dir)
        out_so = _output_path_for_so_annual(sa_path, out_dir)
        thetao_exists = os.path.exists(out_thetao)
        so_exists = os.path.exists(out_so)
        if thetao_exists and so_exists:
            print(
                'thetao/so annual exist, skipping: '
                f'{os.path.basename(out_thetao)}, {os.path.basename(out_so)}'
            )
            outputs.extend([out_thetao, out_so])
            continue
        print(
            'Converting CT/SA->thetao/so (annual): '
            f'{os.path.basename(out_thetao)}, {os.path.basename(out_so)}'
        )
        _process_ct_sa_annual_pair(
            ct_path=ct_path,
            sa_path=sa_path,
            grid_path=grid_path,
            out_thetao=out_thetao,
            out_so=out_so,
            time_chunk_years=time_chunk_years,
            progress=progress,
        )
        if not thetao_exists:
            outputs.append(out_thetao)
        if not so_exists:
            outputs.append(out_so)

    return outputs


def main_cmip() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Convert annual-mean CT/SA to thetao/so and copy TF into '
            'Oyr/thetao_so_tf.'
        )
    )
    parser.add_argument('-m', '--model', required=True, help='CMIP model.')
    parser.add_argument(
        '-s', '--scenario', required=True, help='Scenario (historical, ...).'
    )
    parser.add_argument(
        '-c', '--clim', dest='clim_name', required=True, help='Climatology.'
    )
    parser.add_argument(
        '-w', '--workdir', required=False, help='Base working directory.'
    )
    parser.add_argument(
        '-C',
        '--config',
        dest='config',
        default=None,
        help='Path to user config (optional).',
    )
    parser.add_argument(
        '--time-chunk-years',
        dest='time_chunk_years',
        default=None,
        help='Years per chunk for processing (optional).',
    )
    parser.add_argument(
        '--no-progress',
        dest='no_progress',
        action='store_true',
        help='Disable tqdm progress bar.',
    )
    args = parser.parse_args()
    tcy = (
        int(args.time_chunk_years)
        if args.time_chunk_years not in (None, 'None', '')
        else None
    )
    outs = cmip_ct_sa_ann_to_thetao_so_tf(
        model=args.model,
        scenario=args.scenario,
        clim_name=args.clim_name,
        workdir=args.workdir,
        user_config_filename=args.config,
        time_chunk_years=tcy,
        progress=not args.no_progress,
    )
    for p in outs:
        print(p)


# --------------------------- CLIMATOLOGY WORKFLOW ---------------------------


def clim_ct_sa_to_thetao_so(
    clim_name: str,
    *,
    workdir: str | None = None,
    user_config_filename: str | None = None,
    progress: bool = True,
) -> List[str]:
    """
    Convert extrapolated climatology CT/SA (no time) to thetao/so.

    Discovers extrapolated CT/SA files under:

        workdir/extrap/climatology/<clim_name>

    and writes separate thetao and so files next to them by replacing
    `_ct_extrap` with `_thetao_extrap` and `_so_extrap` in the filenames
    (preserving any `_z` suffix).

    Returns list of output file paths created or found.
    """
    config = _load_clim_config(clim_name, user_config_filename)
    workdir = _get_or_config_path(config, workdir, 'workdir')
    config.set('workdir', 'base_dir', workdir)

    in_dir = os.path.join(workdir, 'extrap', 'climatology', clim_name)
    ismip_res_str = get_res_string(config, extrap=False)
    pairs = _collect_extrap_clim_ct_sa_pairs(in_dir, ismip_res_str)
    if not pairs:
        raise FileNotFoundError(
            'No extrapolated climatology ct/sa files found. Expected under: '
            + in_dir
        )

    grid_path = _ensure_ismip_grid(config, workdir)
    outputs: List[str] = []
    for ct_path, sa_path in pairs:
        out_thetao = _output_path_for_extrap_thetao(ct_path)
        out_so = _output_path_for_extrap_so(ct_path)
        thetao_exists = os.path.exists(out_thetao)
        so_exists = os.path.exists(out_so)
        if thetao_exists and so_exists:
            print(
                'thetao/so exist, skipping: '
                f'{os.path.basename(out_thetao)}, {os.path.basename(out_so)}'
            )
            outputs.extend([out_thetao, out_so])
            continue
        print(
            'Computing thetao/so (clim): '
            f'{os.path.basename(out_thetao)}, {os.path.basename(out_so)}'
        )
        ds_out = _process_ct_sa_clim_pair(
            ct_path=ct_path,
            sa_path=sa_path,
            grid_path=grid_path,
            progress=progress,
        )
        if not thetao_exists:
            ds_write = _dataset_for_output(ds_out, 'thetao')
            write_netcdf(
                ds_write,
                out_thetao,
                progress_bar=progress,
                has_fill_values=True,
            )
            outputs.append(out_thetao)
        if not so_exists:
            ds_write = _dataset_for_output(ds_out, 'so')
            write_netcdf(
                ds_write,
                out_so,
                progress_bar=progress,
                has_fill_values=True,
            )
            outputs.append(out_so)

    return outputs


def main_clim() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Convert extrapolated climatology CT/SA (no time) to thetao/so.'
        )
    )
    parser.add_argument(
        '-c', '--clim', dest='clim_name', required=True, help='Climatology.'
    )
    parser.add_argument(
        '-w', '--workdir', required=False, help='Base working directory.'
    )
    parser.add_argument(
        '-C',
        '--config',
        dest='config',
        default=None,
        help='Path to user config (optional).',
    )
    parser.add_argument(
        '--no-progress',
        dest='no_progress',
        action='store_true',
        help='Disable tqdm progress bar.',
    )
    args = parser.parse_args()

    outs = clim_ct_sa_to_thetao_so(
        clim_name=args.clim_name,
        workdir=args.workdir,
        user_config_filename=args.config,
        progress=not args.no_progress,
    )
    for p in outs:
        print(p)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(
    model: str, clim_name: str, user_cfg: str | None
) -> MpasConfigParser:
    config = MpasConfigParser()
    config.add_from_package('i7aof', 'default.cfg')
    config.add_from_package('i7aof.cmip', f'{get_model_prefix(model)}.cfg')
    config.add_from_package('i7aof.clim', f'{clim_name}.cfg')
    if user_cfg is not None:
        config.add_user_config(user_cfg)
    return config


def _load_clim_config(
    clim_name: str, user_cfg: str | None
) -> MpasConfigParser:
    config = MpasConfigParser()
    config.add_from_package('i7aof', 'default.cfg')
    config.add_from_package('i7aof.clim', f'{clim_name}.cfg')
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


def _parse_time_chunk_years(config: MpasConfigParser) -> int | None:
    if not config.has_option('ct_sa_to_thetao_so', 'time_chunk_years'):
        return None
    raw = config.get('ct_sa_to_thetao_so', 'time_chunk_years')
    if raw in ('', 'None', 'none'):
        return None
    return int(raw)


def _collect_annual_ct_sa_files(
    in_dir: str, ismip_res_str: str
) -> List[Tuple[str, str]]:
    """Collect pairs of annual CT and SA files in the given directory.

    Pairs files named with 'ct' and 'sa' tokens that share the same pattern,
    filtered by the ISMIP resolution string.
    """
    if not os.path.isdir(in_dir):
        return []
    ct_files: List[str] = []
    sa_set: set[str] = set()
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith('.nc'):
            continue
        if f'ismip{ismip_res_str}' not in name:
            continue
        path = os.path.join(in_dir, name)
        if 'ct' in name:
            ct_files.append(path)
        elif 'sa' in name:
            sa_set.add(path)
    pairs: List[Tuple[str, str]] = []
    for ct in ct_files:
        base = os.path.basename(ct)
        sa_candidate = os.path.join(
            os.path.dirname(ct), base.replace('ct', 'sa')
        )
        if sa_candidate in sa_set:
            pairs.append((ct, sa_candidate))
    return pairs


def _collect_annual_tf_files(in_dir: str, ismip_res_str: str) -> List[str]:
    if not os.path.isdir(in_dir):
        return []
    files: List[str] = []
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith('.nc'):
            continue
        if f'ismip{ismip_res_str}' not in name:
            continue
        if 'tf' in name:
            files.append(os.path.join(in_dir, name))
    return files


def _output_path_for_thetao_so(ct_or_sa_annual_path: str, out_dir: str) -> str:
    base = os.path.basename(ct_or_sa_annual_path)
    # Prefer replacing a 'ct_sa' token if present (legacy combined files)
    if 'ct_sa' in base:
        out_base = base.replace('ct_sa', 'thetao_so')
    elif 'ct' in base:
        out_base = base.replace('ct', 'thetao_so')
    elif 'sa' in base:
        out_base = base.replace('sa', 'thetao_so')
    else:
        out_base = f'thetao_so_{base}'
    return os.path.join(out_dir, out_base)


def _output_path_for_thetao_annual(ct_path: str, out_dir: str) -> str:
    base = os.path.basename(ct_path)
    out_base = base.replace('ct', 'thetao')
    return os.path.join(out_dir, out_base)


def _output_path_for_so_annual(sa_path: str, out_dir: str) -> str:
    base = os.path.basename(sa_path)
    out_base = base.replace('sa', 'so')
    return os.path.join(out_dir, out_base)


def _collect_extrap_clim_ct_sa_pairs(
    in_dir: str, ismip_res_str: str
) -> List[Tuple[str, str]]:
    """Collect pairs of extrapolated climatology ct/sa files in a folder.

    Looks for files matching patterns like:
      *_ct_extrap.nc, *_ct_extrap_z.nc and pairs them with corresponding
      *_sa_extrap*.nc files.
    """
    if not os.path.isdir(in_dir):
        return []
    ct_files: List[str] = []
    sa_set: set[str] = set()
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith('.nc'):
            continue
        # Only keep files for the post-vertical-resampling ISMIP resolution
        if ismip_res_str not in name:
            continue
        path = os.path.join(in_dir, name)
        if '_ct_extrap' in name:
            ct_files.append(path)
        if '_sa_extrap' in name:
            sa_set.add(path)
    pairs: List[Tuple[str, str]] = []
    for ct in ct_files:
        base = os.path.basename(ct)
        sa_base = base.replace('_ct_extrap', '_sa_extrap')
        sa_candidate = os.path.join(in_dir, sa_base)
        if sa_candidate in sa_set:
            pairs.append((ct, sa_candidate))
    return pairs


def _output_path_for_extrap_thetao(ct_path: str) -> str:
    """Return thetao output path alongside extrapolated CT file.

    Replaces `_ct_extrap` with `_thetao_extrap` (preserves optional `_z`).
    """
    base = os.path.basename(ct_path)
    out_base = base.replace('_ct_extrap', '_thetao_extrap')
    return os.path.join(os.path.dirname(ct_path), out_base)


def _output_path_for_extrap_so(ct_path: str) -> str:
    """Return so output path alongside extrapolated CT file.

    Replaces `_ct_extrap` with `_so_extrap` (preserves optional `_z`).
    """
    base = os.path.basename(ct_path)
    out_base = base.replace('_ct_extrap', '_so_extrap')
    return os.path.join(os.path.dirname(ct_path), out_base)


def _compute_time_indices(
    ds: xr.Dataset, time_chunk_years: int | None
) -> Sequence[Tuple[int, int]]:
    if 'time' not in ds.dims:
        return [(0, 1)]
    nt = ds.sizes['time']
    if time_chunk_years is None or time_chunk_years <= 0:
        return [(0, nt)]
    chunk = min(time_chunk_years, nt)
    return [(i0, min(i0 + chunk, nt)) for i0 in range(0, nt, chunk)]


def _copy_netcdf(src: str, dst: str) -> None:
    # Simple pass-through copy using project helper
    with xr.open_dataset(src, decode_times=True, use_cftime=True) as ds:
        write_netcdf(ds, dst, progress_bar=True, has_fill_values=True)


def _process_ct_sa_clim_pair(
    *,
    ct_path: str,
    sa_path: str,
    grid_path: str,
    progress: bool,
) -> xr.Dataset:
    """Process a CT/SA pair (no time) into combined thetao/so NetCDF.

    Reads ct and sa datasets, computes thetao and so using TEOS-10,
    attaches ISMIP coordinates, and writes a single NetCDF file.
    """
    ds_ct = xr.open_dataset(ct_path, decode_times=True, use_cftime=True)
    ds_sa = xr.open_dataset(sa_path, decode_times=True, use_cftime=True)
    ds_ct, ds_sa = xr.align(ds_ct, ds_sa, join='exact')

    ds_grid, lat, lon, p = _open_grid_and_pressure(grid_path)

    ct = ds_ct['ct']
    sa = ds_sa['sa']
    ds_out = _build_thetao_so_dataset(
        ct=ct,
        sa=sa,
        lat=lat,
        lon=lon,
        p=p,
        ds_grid=ds_grid,
        time_bnds=None,
    )

    # Enforce float32 dtype (mirrors annual safeguard)
    for v in ['thetao', 'so']:
        if v in ds_out and ds_out[v].dtype != 'float32':
            ds_out[v] = ds_out[v].astype('float32')

    return ds_out


def _dataset_for_output(ds: xr.Dataset, var_name: str) -> xr.Dataset:
    """Return a dataset with the variable plus any bounds variables.

    Coordinates are kept automatically; this ensures bounds variables are
    also included when saving a single variable dataset.
    """
    bound_names = [
        'time_bnds',
        'x_bnds',
        'y_bnds',
        'z_bnds',
        'lat_bnds',
        'lon_bnds',
    ]
    present = [b for b in bound_names if b in ds]
    return ds[[var_name] + present]


def _process_ct_sa_annual_pair(
    *,
    ct_path: str,
    sa_path: str,
    grid_path: str,
    out_thetao: str,
    out_so: str,
    time_chunk_years: int | None,
    progress: bool,
) -> None:
    ds_ct = xr.open_dataset(ct_path, decode_times=True, use_cftime=True)
    ds_sa = xr.open_dataset(sa_path, decode_times=True, use_cftime=True)
    ds_ct, ds_sa = xr.align(ds_ct, ds_sa, join='exact')

    ds_grid, lat, lon, p = _open_grid_and_pressure(grid_path)

    # Determine Zarr store for this output (use thetao basename)
    out_dir = os.path.dirname(out_thetao)
    base = os.path.splitext(os.path.basename(out_thetao))[0]
    zarr_store = os.path.join(out_dir, f'{base}.zarr')

    # Compute time indices based on CT dataset (aligned with SA)
    time_indices = _compute_time_indices(ds_ct, time_chunk_years)
    iterator = time_indices
    if progress:
        iterator = tqdm(
            time_indices,
            total=len(time_indices),
            desc='time chunks',
            leave=False,
        )

    first = True
    for i0, i1 in iterator:
        if 'time' in ds_ct.dims:
            ds_ct_chunk = ds_ct.isel(time=slice(i0, i1))
            ds_sa_chunk = ds_sa.isel(time=slice(i0, i1))
        else:
            ds_ct_chunk = ds_ct
            ds_sa_chunk = ds_sa

        ct = ds_ct_chunk['ct']
        sa = ds_sa_chunk['sa']

        ds_out = _build_thetao_so_dataset(
            ct=ct,
            sa=sa,
            lat=lat,
            lon=lon,
            p=p,
            ds_grid=ds_grid,
            time_bnds=(
                ds_ct_chunk['time_bnds']
                if 'time_bnds' in ds_ct_chunk
                else None
            ),
        )

        # Append chunk to Zarr
        first = append_to_zarr(
            ds=ds_out,
            zarr_store=zarr_store,
            first=first,
            append_dim='time' if 'time' in ds_out.dims else None,
        )

    # Prepare postprocess to enforce dtype/attrs
    def _post(ds_final: xr.Dataset) -> xr.Dataset:
        if 'thetao' in ds_final:
            if ds_final['thetao'].dtype != 'float32':
                ds_final['thetao'] = ds_final['thetao'].astype('float32')
            ds_final['thetao'].attrs = THETAO_ATTRS
        if 'so' in ds_final:
            if ds_final['so'].dtype != 'float32':
                ds_final['so'] = ds_final['so'].astype('float32')
            ds_final['so'].attrs = SO_ATTRS
        return ds_final

    # Open Zarr, apply post, and write two separate NetCDF outputs
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=(
                'Consolidated metadata is currently not part in the Zarr '
                'format 3 specification.'
            ),
            category=UserWarning,
        )
        ds_final = xr.open_zarr(zarr_store, consolidated=False)
    try:
        ds_final = _post(ds_final)

        # Replace Zarr-derived time/time_bnds with original cftime-coordinates
        # from the source NetCDF to avoid epoch-based units and dtype drift.
        if 'time' in ds_final.dims and 'time' in ds_ct.dims:
            if ds_final.sizes.get('time') == ds_ct.sizes.get('time'):
                # Assign cftime time coord from source
                ds_final = ds_final.assign_coords(time=ds_ct['time'])
                # Carry bounds variable from source if available
                if 'time_bnds' in ds_ct:
                    ds_final['time_bnds'] = ds_ct['time_bnds']
                    ds_final['time'].attrs['bounds'] = 'time_bnds'

        # Write thetao dataset (with bounds)
        if not os.path.exists(out_thetao):
            ds_write = _dataset_for_output(ds_final, 'thetao')
            write_netcdf(
                ds_write,
                out_thetao,
                has_fill_values=lambda name, _v: name == 'thetao',
                progress_bar=progress,
            )
        # Write so dataset (with bounds)
        if not os.path.exists(out_so):
            ds_write = _dataset_for_output(ds_final, 'so')
            write_netcdf(
                ds_write,
                out_so,
                has_fill_values=lambda name, _v: name == 'so',
                progress_bar=progress,
            )
    finally:
        ds_final.close()
        shutil.rmtree(zarr_store, ignore_errors=True)
        ds_ct.close()
        ds_sa.close()
        ds_grid.close()


def _assign_coord_with_bounds(
    ds_out: xr.Dataset, ds_grid: xr.Dataset, coord: str
) -> None:
    if coord not in ds_grid:
        return
    # set as a coordinate so it is kept when selecting variables
    ds_out.coords[coord] = ds_grid[coord]
    bname = ds_grid[coord].attrs.get('bounds', f'{coord}_bnds')
    if bname in ds_grid:
        ds_out[bname] = ds_grid[bname]
        ds_out[bname].attrs = ds_grid[bname].attrs.copy()
    ds_out[coord].attrs['bounds'] = bname


def _open_grid_and_pressure(
    grid_path: str,
) -> Tuple[xr.Dataset, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Open ISMIP grid once and compute pressure from z and lat.

    Returns: (ds_grid, lat, lon, p)
    """
    ds_grid = xr.open_dataset(grid_path, decode_times=True, use_cftime=True)
    for name in ('lat', 'lon', 'z'):
        if name not in ds_grid:
            raise KeyError(
                f'ISMIP grid file missing required variable: {name}'
            )
    lat = ds_grid['lat']
    lon = ds_grid['lon']
    z = ds_grid['z']
    # Compute pressure as a NumPy array (typically shape (Z, Y, X))
    p_np = _pressure_from_z(z, lat)
    # Build dims/coords to match the computed pressure shape
    z_dim = z.dims[0] if isinstance(z.dims, tuple) else z.dims
    p_dims = (z_dim,) + tuple(lat.dims)
    p_coords = {z_dim: z}
    for d in lat.dims:
        if d in ds_grid:
            p_coords[d] = ds_grid[d]
        elif d in lat.coords:
            p_coords[d] = lat[d]
    p = xr.DataArray(p_np, dims=p_dims, coords=p_coords)
    return ds_grid, lat, lon, p


def _build_thetao_so_dataset(
    *,
    ct: xr.DataArray,
    sa: xr.DataArray,
    lat: xr.DataArray,
    lon: xr.DataArray,
    p: xr.DataArray,
    ds_grid: xr.Dataset,
    time_bnds: xr.DataArray | None,
) -> xr.Dataset:
    """Compute thetao/so and build an output dataset with grid coords.

    - Computes SP from SA with pressure, lon, lat using gsw.SP_from_SA
    - Computes potential temperature from CT and SA using gsw.pt_from_CT
    - Attaches x/y/z coords + lat/lon and their bounds from ds_grid
    - Carries time_bnds if provided
    """
    sp_np = gsw.SP_from_SA(sa.values, np.asarray(p), lon.values, lat.values)
    thetao_np = gsw.pt_from_CT(sa.values, ct.values)

    thetao = xr.DataArray(
        np.asarray(thetao_np).astype('float32', copy=False),
        dims=ct.dims,
        coords=ct.coords,
    ).assign_attrs(THETAO_ATTRS)
    so = xr.DataArray(
        np.asarray(sp_np).astype('float32', copy=False),
        dims=sa.dims,
        coords=sa.coords,
    ).assign_attrs(SO_ATTRS)

    ds_out = xr.Dataset({'thetao': thetao, 'so': so})
    if time_bnds is not None:
        ds_out['time_bnds'] = time_bnds
    _assign_coord_with_bounds(ds_out, ds_grid, 'x')
    _assign_coord_with_bounds(ds_out, ds_grid, 'y')
    _assign_coord_with_bounds(ds_out, ds_grid, 'z')
    # Include geodetic coordinates and their bounds from the grid
    if 'lat' in ds_grid:
        ds_out.coords['lat'] = ds_grid['lat']
        if 'lat_bnds' in ds_grid:
            ds_out['lat_bnds'] = ds_grid['lat_bnds']
    if 'lon' in ds_grid:
        ds_out.coords['lon'] = ds_grid['lon']
        if 'lon_bnds' in ds_grid:
            ds_out['lon_bnds'] = ds_grid['lon_bnds']
    return ds_out
