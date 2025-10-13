import argparse
import os
from typing import List, Sequence, Tuple

import xarray as xr
from mpas_tools.config import MpasConfigParser
from tqdm import tqdm

from i7aof.cmip import get_model_prefix
from i7aof.convert.teos10 import _pressure_from_z, compute_ct_freezing
from i7aof.extrap.shared import _ensure_ismip_grid
from i7aof.grid.ismip import get_res_string
from i7aof.io_zarr import append_to_zarr, finalize_zarr_to_netcdf

__all__ = ['cmip_ct_sa_to_tf', 'main_cmip', 'clim_ct_sa_to_tf', 'main_clim']


def cmip_ct_sa_to_tf(
    model: str,
    scenario: str,
    clim_name: str,
    workdir: str | None = None,
    user_config_filename: str | None = None,
    time_chunk: int | None = None,
    progress: bool = True,
) -> None:
    """
    Compute monthly thermal forcing from bias-corrected CT/SA.

    Discovers bias-corrected monthly ct/sa files under:

        workdir/biascorr/<model>/<scenario>/<clim_name>/Omon/ct_sa

    and writes monthly thermal forcing into:

        workdir/biascorr/<model>/<scenario>/<clim_name>/Omon/tf

    Output filenames replace "ct" with "tf" (e.g., *_ct.nc -> *_tf.nc).

    Parameters
    ----------
    model : str
        CMIP model name.
    scenario : str
        Scenario key (e.g., 'historical', 'ssp585').
    clim_name : str
        Name of the reference climatology used in bias correction.
    workdir : str, optional
        Base working directory; if omitted, uses config
        "[workdir] base_dir".
    user_config_filename : str, optional
        Optional user config overriding defaults.
    time_chunk : int | None, optional
        Number of time steps per processing chunk. If None, read from
        config section [ct_sa_to_tf] time_chunk; if missing, process all
        time at once.
    progress : bool, optional
        If True (default), show a tqdm progress bar over time chunks.
    """
    config = _load_config(model, clim_name, user_config_filename)
    workdir = _get_or_config_path(config, workdir, 'workdir')
    config.set('workdir', 'base_dir', workdir)

    # Input/output directories
    in_dir = os.path.join(
        workdir, 'biascorr', model, scenario, clim_name, 'Omon', 'ct_sa'
    )
    out_dir = os.path.join(
        workdir, 'biascorr', model, scenario, clim_name, 'Omon', 'tf'
    )
    os.makedirs(out_dir, exist_ok=True)

    # Ensure ISMIP grid exists and get grid path; used for coords and lat/z
    grid_path = _ensure_ismip_grid(config, workdir)
    ismip_res_str = get_res_string(config, extrap=False)

    # time chunk from config unless provided
    if time_chunk is None:
        time_chunk = _parse_time_chunk(config)

    pairs = _collect_biascorr_ct_sa_pairs(in_dir, ismip_res_str)
    if not pairs:
        raise FileNotFoundError(
            'No bias-corrected ct/sa files found. Expected under: ' + in_dir
        )

    for ct_path, sa_path in pairs:
        out_nc = _output_path_for_tf(ct_path, out_dir)
        if os.path.exists(out_nc):
            print(f'TF exists, skipping: {out_nc}')
            continue
        print(f'Computing TF: {os.path.basename(out_nc)}')
        _process_ct_sa_pair(
            ct_path=ct_path,
            sa_path=sa_path,
            grid_path=grid_path,
            out_nc=out_nc,
            time_chunk=time_chunk,
            progress=progress,
        )


def main_cmip() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Compute monthly thermal forcing from ct/sa using '
            'TEOS-10 CT_freezing.'
        )
    )
    parser.add_argument('-m', '--model', required=True, help='CMIP model.')
    parser.add_argument(
        '-s',
        '--scenario',
        required=True,
        help='Scenario (historical, ...).',
    )
    parser.add_argument(
        '-c',
        '--clim',
        dest='clim_name',
        required=True,
        help='Climatology.',
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
        '--time-chunk',
        dest='time_chunk',
        default=None,
        help='Time chunk size for processing (optional).',
    )
    parser.add_argument(
        '--no-progress',
        dest='no_progress',
        action='store_true',
        help='Disable tqdm progress bar.',
    )
    args = parser.parse_args()
    tc = (
        int(args.time_chunk)
        if args.time_chunk not in (None, 'None', '')
        else None
    )
    cmip_ct_sa_to_tf(
        model=args.model,
        scenario=args.scenario,
        clim_name=args.clim_name,
        workdir=args.workdir,
        user_config_filename=args.config,
        time_chunk=tc,
        progress=not args.no_progress,
    )


def clim_ct_sa_to_tf(
    clim_name: str,
    *,
    workdir: str | None = None,
    user_config_filename: str | None = None,
    progress: bool = True,
) -> None:
    """
    Compute TF from extrapolated climatology CT/SA (no time dimension).

    Discovers extrapolated CT/SA files under:

        workdir/extrap/climatology/<clim_name>

    and writes TF files next to them by replacing `_ct_extrap` with
    `_tf_extrap` in the filenames (preserving any `_z` suffix).
    """
    config = _load_clim_config(clim_name, user_config_filename)
    workdir = _get_or_config_path(config, workdir, 'workdir')
    config.set('workdir', 'base_dir', workdir)

    in_dir = os.path.join(workdir, 'extrap', 'climatology', clim_name)
    pairs = _collect_extrap_clim_ct_sa_pairs(in_dir)
    if not pairs:
        raise FileNotFoundError(
            'No extrapolated climatology ct/sa files found. Expected under: '
            + in_dir
        )

    grid_path = _ensure_ismip_grid(config, workdir)
    for ct_path, sa_path in pairs:
        out_nc = _output_path_for_extrap_tf(ct_path)
        if os.path.exists(out_nc):
            print(f'TF exists, skipping: {out_nc}')
            continue
        print(f'Computing TF (clim): {os.path.basename(out_nc)}')
        _process_ct_sa_pair(
            ct_path=ct_path,
            sa_path=sa_path,
            grid_path=grid_path,
            out_nc=out_nc,
            time_chunk=None,
            progress=progress,
        )


def main_clim() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Compute thermal forcing from extrapolated climatology ct/sa '
            '(no time dimension).'
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

    clim_ct_sa_to_tf(
        clim_name=args.clim_name,
        workdir=args.workdir,
        user_config_filename=args.config,
        progress=not args.no_progress,
    )


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
    if not config.has_option('ct_sa_to_tf', 'time_chunk'):
        return None
    raw = config.get('ct_sa_to_tf', 'time_chunk')
    if raw in ('', 'None', 'none'):
        return None
    return int(raw)


def _collect_biascorr_ct_sa_pairs(
    in_dir: str, ismip_res_str: str
) -> List[Tuple[str, str]]:
    if not os.path.isdir(in_dir):
        return []
    ct_files: List[str] = []
    sa_set: set[str] = set()
    for name in sorted(os.listdir(in_dir)):
        if f'ismip{ismip_res_str}' not in name:
            continue
        path = os.path.join(in_dir, name)
        if 'ct' in name and name.endswith('.nc'):
            ct_files.append(path)
        elif 'sa' in name and name.endswith('.nc'):
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


def _output_path_for_tf(ct_path: str, out_dir: str) -> str:
    base = os.path.basename(ct_path)
    tf_base = base.replace('ct', 'tf')
    return os.path.join(out_dir, tf_base)


def _collect_extrap_clim_ct_sa_pairs(in_dir: str) -> List[Tuple[str, str]]:
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


def _output_path_for_extrap_tf(ct_path: str) -> str:
    """Return TF output path alongside extrapolated CT file.

    Replaces `_ct_extrap` with `_tf_extrap` (and preserves optional `_z`).
    """
    base = os.path.basename(ct_path)
    tf_base = base.replace('_ct_extrap', '_tf_extrap')
    return os.path.join(os.path.dirname(ct_path), tf_base)


def _load_clim_config(
    clim_name: str, user_cfg: str | None
) -> MpasConfigParser:
    config = MpasConfigParser()
    config.add_from_package('i7aof', 'default.cfg')
    config.add_from_package('i7aof.clim', f'{clim_name}.cfg')
    if user_cfg is not None:
        config.add_user_config(user_cfg)
    return config


def _compute_time_indices(
    ds: xr.Dataset, time_chunk: int | None
) -> Sequence[Tuple[int, int]]:
    if 'time' not in ds.dims:
        return [(0, 1)]
    nt = ds.sizes['time']
    if time_chunk is None or time_chunk <= 0:
        return [(0, nt)]
    chunk = min(time_chunk, nt)
    return [(i0, min(i0 + chunk, nt)) for i0 in range(0, nt, chunk)]


def _assign_coord_with_bounds(
    ds_out: xr.Dataset, ds_grid: xr.Dataset, coord: str
) -> None:
    """Assign a 1D coordinate and its bounds to ds_out from ISMIP grid."""
    if coord not in ds_grid:
        return
    ds_out[coord] = ds_grid[coord]
    bname = ds_grid[coord].attrs.get('bounds', f'{coord}_bnds')
    if bname in ds_grid:
        ds_out[bname] = ds_grid[bname]
        ds_out[bname].attrs = ds_grid[bname].attrs.copy()
    ds_out[coord].attrs['bounds'] = bname


def _process_ct_sa_pair(
    *,
    ct_path: str,
    sa_path: str,
    grid_path: str,
    out_nc: str,
    time_chunk: int | None,
    progress: bool,
) -> None:
    # Open inputs
    ds_ct = xr.open_dataset(ct_path, decode_times=False)
    ds_sa = xr.open_dataset(sa_path, decode_times=False)
    ds_ct, ds_sa = xr.align(ds_ct, ds_sa, join='exact')

    # Load ISMIP grid for coords and lat/z (for pressure calc)
    ds_grid = xr.open_dataset(grid_path, decode_times=False)
    lat = ds_grid['lat'] if 'lat' in ds_grid else None
    if lat is None:
        raise KeyError('ISMIP grid file missing required variable: lat')
    if 'z' not in ds_grid:
        raise KeyError('ISMIP grid file missing required variable: z')
    z = ds_grid['z']
    # Pre-compute pressure (dbar) once for the whole grid; reused for all
    # time chunks to avoid repeated gsw.p_from_z calls inside the loop.
    p = _pressure_from_z(z, lat)
    p_da = xr.DataArray(p)

    # Determine per-file temporary Zarr store
    out_dir = os.path.dirname(out_nc)
    base = os.path.splitext(os.path.basename(out_nc))[0]
    zarr_store = os.path.join(out_dir, f'{base}.zarr')
    # No need to pre-clean; append_to_zarr will remove existing store on first
    # creation to ensure a fresh write.

    time_indices = _compute_time_indices(ds_ct, time_chunk)
    first = True
    iterator = time_indices
    if progress:
        iterator = tqdm(
            time_indices,
            total=len(time_indices),
            desc='time chunks',
            leave=False,
        )
    # Define TF attributes once and apply to each chunk and final dataset
    tf_attrs = {
        'units': 'degC',
        'long_name': 'Thermal Forcing',
        'comment': (
            'Computed as Conservative Temperature minus TEOS-10 '
            'freezing CT (gsw.CT_freezing) with saturation_fraction=0.'
        ),
    }
    for i0, i1 in iterator:
        if 'time' in ds_ct.dims:
            ds_ct_chunk = ds_ct.isel(time=slice(i0, i1))
            ds_sa_chunk = ds_sa.isel(time=slice(i0, i1))
        else:
            ds_ct_chunk = ds_ct
            ds_sa_chunk = ds_sa

        ct = ds_ct_chunk['ct']
        sa = ds_sa_chunk['sa']
        # Compute CT_freezing at saturation_fraction=0.0 using precomputed
        # pressure to avoid recomputing p_from_z every chunk.
        ct_freeze = compute_ct_freezing(
            sa=sa, z_or_p=p_da, lat=None, is_pressure=True
        )
        tf = (ct - ct_freeze).astype('float32')

        # Assemble output dataset for this chunk
        ds_out = xr.Dataset({'tf': tf})
        # Copy monthly time bounds if present on inputs
        if 'time_bnds' in ds_ct_chunk:
            ds_out['time_bnds'] = ds_ct_chunk['time_bnds']
        # Attach ISMIP coordinates and bounds
        _assign_coord_with_bounds(ds_out, ds_grid, 'x')
        _assign_coord_with_bounds(ds_out, ds_grid, 'y')
        _assign_coord_with_bounds(ds_out, ds_grid, 'z')

        # Variable attributes
        ds_out['tf'].attrs = tf_attrs

        # Append to Zarr store (create on first write)
        first = append_to_zarr(
            ds=ds_out,
            zarr_store=zarr_store,
            first=first,
            append_dim='time' if 'time' in ds_out.dims else None,
        )

    # Finalize: convert Zarr to NetCDF (reapplying TF attrs) and remove store
    def _post(ds_final: xr.Dataset) -> xr.Dataset:
        if 'tf' in ds_final:
            ds_final['tf'].attrs = tf_attrs
        return ds_final

    finalize_zarr_to_netcdf(
        zarr_store=zarr_store,
        out_nc=out_nc,
        has_fill_values=lambda name, _v: name == 'tf',
        progress_bar=True,
        postprocess=_post,
    )

    ds_ct.close()
    ds_sa.close()
    ds_grid.close()
