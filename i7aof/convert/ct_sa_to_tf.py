import argparse
import os
import shutil
from typing import Callable, List, Sequence, Tuple

import xarray as xr
from mpas_tools.config import MpasConfigParser
from tqdm import tqdm

from i7aof.config import load_config
from i7aof.convert.teos10 import _pressure_from_z, compute_ct_freezing
from i7aof.coords import (
    attach_grid_coords,
)
from i7aof.grid.ismip import ensure_ismip_grid, get_res_string
from i7aof.io import ensure_cf_time_encoding, read_dataset
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

    and writes corrected monthly CT/SA/TF into:

        workdir/biascorr/<model>/<scenario>/<clim_name>/Omon/ct_sa_tf0

    where TF is clipped to be nonnegative and CT is adjusted to be equal to
    CT_freezing wherever TF would otherwise be negative.

    Output filenames replace "ct" with "tf" (e.g., ``*_ct.nc -> *_tf.nc``).

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
    config = load_config(
        model=model,
        clim_name=clim_name,
        workdir=workdir,
        user_config_filename=user_config_filename,
    )
    workdir_base: str = config.get('workdir', 'base_dir')

    # Input/output directories
    in_dir = os.path.join(
        workdir_base, 'biascorr', model, scenario, clim_name, 'Omon', 'ct_sa'
    )
    out_dir = os.path.join(
        workdir_base,
        'biascorr',
        model,
        scenario,
        clim_name,
        'Omon',
        'ct_sa_tf0',
    )
    os.makedirs(out_dir, exist_ok=True)

    # Ensure ISMIP grid exists and get grid path; used for coords and lat/z
    ismip_res_str = get_res_string(config, extrap=False)

    # time chunk from config unless provided
    if time_chunk is None:
        time_chunk = _parse_time_chunk(config)
    # use_poly from config (default True)
    use_poly = _parse_use_poly(config)

    pairs = _collect_biascorr_ct_sa_pairs(in_dir, ismip_res_str)
    if not pairs:
        raise FileNotFoundError(
            'No bias-corrected ct/sa files found. Expected under: ' + in_dir
        )

    for ct_path, sa_path in pairs:
        out_tf = _output_path_for_tf(ct_path, out_dir)
        out_ct = _output_path_for_ct(ct_path, out_dir)
        out_sa = _output_path_for_sa(sa_path, out_dir)

        tf_exists = os.path.exists(out_tf)
        ct_exists = os.path.exists(out_ct)
        sa_exists = os.path.exists(out_sa)

        if tf_exists and ct_exists and sa_exists:
            print(f'CT/SA/TF exist, skipping: {os.path.basename(out_tf)}')
            continue

        # Copy SA for a self-contained monthly directory
        if not sa_exists:
            shutil.copyfile(sa_path, out_sa)

        print(
            f'Computing TF (tf>=0) + corrected CT: {os.path.basename(out_tf)}'
        )
        _process_ct_sa_pair(
            ct_path=ct_path,
            sa_path=sa_path,
            config=config,
            out_tf=out_tf,
            out_ct=out_ct,
            time_chunk=time_chunk,
            use_poly=use_poly,
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
    config = load_config(
        model=None,
        clim_name=clim_name,
        workdir=workdir,
        user_config_filename=user_config_filename,
    )
    workdir_base: str = config.get('workdir', 'base_dir')

    in_dir = os.path.join(workdir_base, 'extrap', 'climatology', clim_name)
    ismip_res_str = get_res_string(config, extrap=False)
    pairs = _collect_extrap_clim_ct_sa_pairs(in_dir, ismip_res_str)
    if not pairs:
        raise FileNotFoundError(
            'No extrapolated climatology ct/sa files found. Expected under: '
            + in_dir
        )

    for ct_path, sa_path in pairs:
        out_nc = _output_path_for_extrap_tf(ct_path)
        if os.path.exists(out_nc):
            print(f'TF exists, skipping: {out_nc}')
            continue
        print(f'Computing TF (clim): {os.path.basename(out_nc)}')
        _process_ct_sa_pair(
            ct_path=ct_path,
            sa_path=sa_path,
            config=config,
            out_tf=out_nc,
            out_ct=None,
            time_chunk=None,
            use_poly=_parse_use_poly(config),
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


def _parse_time_chunk(config: MpasConfigParser) -> int | None:
    if not config.has_option('ct_sa_to_tf', 'time_chunk'):
        return None
    raw = config.get('ct_sa_to_tf', 'time_chunk')
    if raw in ('', 'None', 'none'):
        return None
    return int(raw)


def _parse_use_poly(config: MpasConfigParser) -> bool:
    if not config.has_option('ct_sa_to_tf', 'use_poly'):
        return True
    raw = str(config.get('ct_sa_to_tf', 'use_poly')).strip().lower()
    return raw not in ('0', 'false', 'no', '')


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


def _output_path_for_ct(ct_path: str, out_dir: str) -> str:
    return os.path.join(out_dir, os.path.basename(ct_path))


def _output_path_for_sa(sa_path: str, out_dir: str) -> str:
    return os.path.join(out_dir, os.path.basename(sa_path))


def _collect_extrap_clim_ct_sa_pairs(
    in_dir: str, ismip_res_str: str
) -> List[Tuple[str, str]]:
    """Collect pairs of extrapolated climatology ct/sa files in a folder.

    Looks for files matching patterns like:
        ``*_ct_extrap.nc``, ``*_ct_extrap_z.nc`` and pairs them with
        corresponding ``*_sa_extrap*.nc`` files.
    """
    if not os.path.isdir(in_dir):
        return []
    ct_files: List[str] = []
    sa_set: set[str] = set()
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith('.nc'):
            continue
        # Only keep files corresponding to the post-vertical-resampling
        # resolution (e.g., containing the ISMIP res string)
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


def _output_path_for_extrap_tf(ct_path: str) -> str:
    """Return TF output path alongside extrapolated CT file.

    Replaces `_ct_extrap` with `_tf_extrap` (and preserves optional `_z`).
    """
    base = os.path.basename(ct_path)
    tf_base = base.replace('_ct_extrap', '_tf_extrap')
    return os.path.join(os.path.dirname(ct_path), tf_base)


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


def _open_and_align_ct_sa(
    ct_path: str, sa_path: str
) -> Tuple[xr.Dataset, xr.Dataset]:
    ds_ct = read_dataset(ct_path)
    ds_sa = read_dataset(sa_path)
    return xr.align(ds_ct, ds_sa, join='exact')


def _load_grid_and_pressure(config) -> Tuple[xr.Dataset, xr.DataArray]:
    """Load ISMIP grid and precompute pressure (dbar) as a DataArray."""

    grid_path = ensure_ismip_grid(config)
    ds_grid = read_dataset(grid_path)
    lat = ds_grid['lat'] if 'lat' in ds_grid else None
    if lat is None:
        raise KeyError('ISMIP grid file missing required variable: lat')
    if 'z' not in ds_grid:
        raise KeyError('ISMIP grid file missing required variable: z')
    z = ds_grid['z']
    p = _pressure_from_z(z, lat)
    return ds_grid, xr.DataArray(p)


def _zarr_store_from_nc(out_nc: str) -> str:
    out_dir = os.path.dirname(out_nc)
    base = os.path.splitext(os.path.basename(out_nc))[0]
    return os.path.join(out_dir, f'{base}.zarr')


def _time_indices_iterator(
    ds: xr.Dataset, time_chunk: int | None, progress: bool
) -> Sequence[Tuple[int, int]] | tqdm:
    time_indices = _compute_time_indices(ds, time_chunk)
    if not progress:
        return time_indices
    return tqdm(
        time_indices,
        total=len(time_indices),
        desc='time chunks',
        leave=False,
    )


def _slice_time_if_present(
    ds_ct: xr.Dataset, ds_sa: xr.Dataset, i0: int, i1: int
) -> Tuple[xr.Dataset, xr.Dataset]:
    if 'time' not in ds_ct.dims:
        return ds_ct, ds_sa
    return ds_ct.isel(time=slice(i0, i1)), ds_sa.isel(time=slice(i0, i1))


def _tf_variable_attrs() -> dict:
    return {
        'units': 'degC',
        'long_name': 'Thermal Forcing',
        'comment': (
            'Computed as Conservative Temperature minus TEOS-10 '
            'freezing CT (gsw.CT_freezing) with saturation_fraction=0.'
        ),
    }


def _corrected_ct_variable_attrs(ct_var: xr.DataArray) -> dict:
    ct_attrs = dict(ct_var.attrs)
    ct_attrs.setdefault('units', 'degC')
    ct_attrs.setdefault('long_name', 'Conservative Temperature')
    ct_attrs['comment'] = (
        'Bias-corrected CT, adjusted to be consistent with nonnegative TF: '
        'where TF would be negative, CT is set to TEOS-10 CT_freezing '
        '(gsw.CT_freezing) with saturation_fraction=0.'
    )
    return ct_attrs


def _build_tf_and_ct_outputs(
    *,
    ds_ct_chunk: xr.Dataset,
    ds_sa_chunk: xr.Dataset,
    p_da: xr.DataArray,
    use_poly: bool,
    tf_attrs: dict,
    ct_attrs: dict,
    write_ct: bool,
) -> Tuple[xr.Dataset, xr.Dataset | None]:
    ct = ds_ct_chunk['ct']
    sa = ds_sa_chunk['sa']
    ct_freeze = compute_ct_freezing(
        sa=sa, z_or_p=p_da, lat=None, is_pressure=True, use_poly=use_poly
    )

    tf_raw = ct - ct_freeze
    neg = tf_raw < 0.0
    tf = xr.where(neg, 0.0, tf_raw).astype('float32')

    ds_out_tf = xr.Dataset({'tf': tf})
    if 'time_bnds' in ds_ct_chunk:
        ds_out_tf['time_bnds'] = ds_ct_chunk['time_bnds']
    ds_out_tf['tf'].attrs = tf_attrs

    if not write_ct:
        return ds_out_tf, None

    ct_out = xr.where(neg, ct_freeze, ct).astype('float32')
    ds_out_ct = xr.Dataset({'ct': ct_out})
    if 'time_bnds' in ds_ct_chunk:
        ds_out_ct['time_bnds'] = ds_ct_chunk['time_bnds']
    ds_out_ct['ct'].attrs = ct_attrs
    return ds_out_tf, ds_out_ct


def _postprocess_attach_coords(
    *,
    var_name: str,
    var_attrs: dict,
    config,
    time_source: xr.Dataset,
    has_time_dim: bool,
) -> Callable[[xr.Dataset], xr.Dataset]:
    def _post(ds_final: xr.Dataset) -> xr.Dataset:
        if var_name in ds_final:
            if ds_final[var_name].dtype != 'float32':
                ds_final[var_name] = ds_final[var_name].astype('float32')
            ds_final[var_name].attrs = var_attrs
        ds_final = attach_grid_coords(ds_final, config)
        if has_time_dim:
            ensure_cf_time_encoding(ds=ds_final, time_source=time_source)
        return ds_final

    return _post


def _process_ct_sa_pair(
    *,
    ct_path: str,
    sa_path: str,
    config,
    out_tf: str,
    out_ct: str | None,
    time_chunk: int | None,
    use_poly: bool,
    progress: bool,
) -> None:
    ds_ct, ds_sa = _open_and_align_ct_sa(ct_path, sa_path)
    ds_grid, p_da = _load_grid_and_pressure(config)

    zarr_store_tf = _zarr_store_from_nc(out_tf)
    zarr_store_ct = _zarr_store_from_nc(out_ct) if out_ct is not None else None

    tf_attrs = _tf_variable_attrs()
    ct_attrs = _corrected_ct_variable_attrs(ds_ct['ct'])

    iterator = _time_indices_iterator(ds_ct, time_chunk, progress)
    has_time_dim = 'time' in ds_ct.dims
    first_tf = True
    first_ct = True

    for i0, i1 in iterator:
        ds_ct_chunk, ds_sa_chunk = _slice_time_if_present(ds_ct, ds_sa, i0, i1)
        ds_out_tf, ds_out_ct = _build_tf_and_ct_outputs(
            ds_ct_chunk=ds_ct_chunk,
            ds_sa_chunk=ds_sa_chunk,
            p_da=p_da,
            use_poly=use_poly,
            tf_attrs=tf_attrs,
            ct_attrs=ct_attrs,
            write_ct=(out_ct is not None),
        )

        first_tf = append_to_zarr(
            ds=ds_out_tf,
            zarr_store=zarr_store_tf,
            first=first_tf,
            append_dim='time' if 'time' in ds_out_tf.dims else None,
        )

        if zarr_store_ct is not None and ds_out_ct is not None:
            first_ct = append_to_zarr(
                ds=ds_out_ct,
                zarr_store=zarr_store_ct,
                first=first_ct,
                append_dim='time' if 'time' in ds_out_ct.dims else None,
            )

    compression_opts = {'zlib': True, 'complevel': 9, 'shuffle': True}

    post_tf = _postprocess_attach_coords(
        var_name='tf',
        var_attrs=tf_attrs,
        config=config,
        time_source=ds_ct,
        has_time_dim=has_time_dim,
    )
    finalize_zarr_to_netcdf(
        zarr_store=zarr_store_tf,
        out_nc=out_tf,
        postprocess=post_tf,
        has_fill_values=['tf'],
        compression=['tf'],
        progress_bar=True,
        compression_opts=compression_opts,
    )

    if zarr_store_ct is not None and out_ct is not None:
        post_ct = _postprocess_attach_coords(
            var_name='ct',
            var_attrs=ct_attrs,
            config=config,
            time_source=ds_ct,
            has_time_dim=has_time_dim,
        )
        finalize_zarr_to_netcdf(
            zarr_store=zarr_store_ct,
            out_nc=out_ct,
            postprocess=post_ct,
            has_fill_values=['ct'],
            compression=['ct'],
            progress_bar=True,
            compression_opts=compression_opts,
        )

    ds_ct.close()
    ds_sa.close()
    ds_grid.close()
