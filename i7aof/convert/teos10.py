"""
TEOS-10 conversion utilities to compute Absolute Salinity (SA) and
Conservative Temperature (CT) from Practical Salinity (SP) and potential
temperature (PT), with xarray/dask support.

This module is I/O-free and operates on xarray DataArray/Dataset, keeping
variable and dimension names flexible.
"""

import os
import time
from typing import Tuple

import gsw
import numpy as np
import xarray as xr


def _debug_enabled() -> bool:
    val = os.environ.get('I7AOF_DEBUG_TEOS10', '0')
    return str(val).lower() not in ('0', 'false', 'no', '')


def _dbg(*args):
    if _debug_enabled():
        print('[TEOS10]', *args, flush=True)


def compute_sa(
    sp: xr.DataArray,
    z_or_p: xr.DataArray,
    lon: xr.DataArray,
    lat: xr.DataArray,
    is_pressure: bool = False,
    normalize_lon: bool = True,
) -> xr.DataArray:
    """Compute Absolute Salinity (SA) from Practical Salinity (SP).

    Parameters
    ----------
    sp : xr.DataArray
        Practical Salinity (PSS-78, unitless).
    z_or_p : xr.DataArray
        Depth (m, positive down) or TEOS-10 z (m, negative down) or pressure
        (dbar) if ``is_pressure=True``. Will be broadcast to SP.
    lon, lat : xr.DataArray
        Geographic coordinates in degrees. 1D or 2D accepted.
    is_pressure : bool
        If True, interpret ``z_or_p`` as pressure (dbar). Otherwise, compute
        pressure from depth and latitude.
    normalize_lon : bool
        Normalize longitudes to [0, 360).

    Returns
    -------
    xr.DataArray
        Absolute Salinity (g kg-1) with the same grid as ``sp`` and
        attributes ``units`` and ``long_name`` set.
    """
    if normalize_lon:
        lon = _normalize_lon(lon)

    # Prepare lon/lat arrays for broadcasting: ensure 2D (Y,X)
    lon_arr = lon.values
    lat_arr = lat.values
    if lon_arr.ndim == 1 and lat_arr.ndim == 1:
        # Create 2D grids from 1D coords
        lon_arr, lat_arr = np.meshgrid(lon_arr, lat_arr, indexing='xy')

    # Log shapes; rely on NumPy broadcasting inside gsw
    _dbg(
        'SA inputs shapes:',
        'sp',
        tuple(sp.shape),
        'lon',
        tuple(lon_arr.shape),
        'lat',
        tuple(lat_arr.shape),
    )

    if is_pressure:
        # Use provided pressure (dbar)
        p = z_or_p.values
        # If pressure provided as 1D (Z,), reshape to (Z,1,1) for broadcasting
        if p.ndim == 1:
            p = p[:, None, None]
    else:
        # Ensure z is TEOS-10 z (negative downward)
        z = z_or_p
        # If z looks positive-down, flip sign
        if (z.min() >= 0).item() or (
            z.attrs.get('positive', '').lower() == 'down'
        ):
            z = -z
        # Broadcast z and lat to 3D then compute pressure
        t1 = time.perf_counter()
        # Compute pressure using NumPy broadcasting; z as (Z,1,1),
        # lat as (Y,X) -> p becomes (Z,Y,X)
        z_arr = z.values
        if z_arr.ndim == 1:
            z_arr = z_arr[:, None, None]
        p = gsw.p_from_z(z_arr, lat_arr)
        _dbg(
            'p_from_z: z',
            tuple(z_arr.shape),
            'lat',
            tuple(lat_arr.shape),
            '->',
            tuple(np.asarray(p).shape),
        )
        _dbg(f'p_from_z time: {time.perf_counter() - t1:.3f}s')

    t2 = time.perf_counter()
    sa_np = gsw.SA_from_SP(sp.values, np.asarray(p), lon_arr, lat_arr)
    _dbg(
        'SA_from_SP: shapes',
        'SP',
        tuple(sp.shape),
        'p',
        tuple(np.asarray(p).shape),
        'lon',
        tuple(lon_arr.shape),
        'lat',
        tuple(lat_arr.shape),
        '->',
        tuple(np.asarray(sa_np).shape),
    )
    _dbg(f'SA_from_SP time: {time.perf_counter() - t2:.3f}s')
    sa_da = xr.DataArray(
        np.asarray(sa_np).astype(sp.dtype, copy=False),
        dims=sp.dims,
        coords=sp.coords,
    ).assign_attrs(units='g kg-1', long_name='Absolute Salinity')
    # Ensure missing SP propagates to SA explicitly
    sa_da = sa_da.where(np.isfinite(sp))
    return sa_da


def compute_ct(pt: xr.DataArray, sa: xr.DataArray) -> xr.DataArray:
    """Compute Conservative Temperature (CT).

    Parameters
    ----------
    pt : xr.DataArray
        Potential temperature (degC), aligned with ``sa``.
    sa : xr.DataArray
        Absolute Salinity (g kg-1), aligned with ``pt``.

    Returns
    -------
    xr.DataArray
        Conservative Temperature (degC) broadcast to the common grid and
        with attributes ``units`` and ``long_name`` set.
    """
    t0 = time.perf_counter()
    pt_b, sa_b = xr.broadcast(pt, sa)
    _dbg(
        'broadcast CT: pt',
        tuple(pt.shape),
        'sa',
        tuple(sa.shape),
        '->',
        tuple(pt_b.shape),
    )
    _dbg(f'broadcast CT time: {time.perf_counter() - t0:.3f}s')
    t1 = time.perf_counter()
    ct_np = gsw.CT_from_pt(sa.values, pt.values)
    _dbg(
        'CT_from_pt: shapes',
        'SA',
        tuple(sa.shape),
        'PT',
        tuple(pt.shape),
        '->',
        tuple(np.asarray(ct_np).shape),
    )
    _dbg(f'CT_from_pt time: {time.perf_counter() - t1:.3f}s')
    ct_da = xr.DataArray(
        np.asarray(ct_np).astype(pt.dtype, copy=False),
        dims=pt.dims,
        coords=pt.coords,
    ).assign_attrs(units='degC', long_name='Conservative Temperature')
    # Ensure missing PT or SA propagates to CT explicitly
    ct_da = ct_da.where(np.isfinite(pt) & np.isfinite(sa))
    return ct_da


def compute_ct_sa(
    sp: xr.DataArray,
    pt: xr.DataArray,
    z_or_p: xr.DataArray,
    lon: xr.DataArray,
    lat: xr.DataArray,
    is_pressure: bool = False,
    normalize_lon: bool = True,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Compute SA and CT together.

    Parameters
    ----------
    sp : xr.DataArray
        Practical Salinity (PSS-78, unitless).
    pt : xr.DataArray
        Potential temperature (degC), aligned or broadcastable to ``sp``.
    z_or_p : xr.DataArray
        Depth (m, positive down), TEOS-10 z (m, negative down), or pressure
        (dbar) if ``is_pressure=True``.
    lon, lat : xr.DataArray
        Geographic coordinates in degrees. 1D or 2D accepted.
    is_pressure : bool, optional
        If True, interpret ``z_or_p`` as pressure (dbar). Otherwise, compute
        pressure from depth and latitude.
    normalize_lon : bool, optional
        Normalize longitudes to [0, 360).

    Returns
    -------
    (xr.DataArray, xr.DataArray)
        Tuple of (sa, ct), where ``sa`` is Absolute Salinity (g kg-1) and
        ``ct`` is Conservative Temperature (degC), both broadcast to the
        common grid and with standard attributes set.
    """
    t0 = time.perf_counter()
    sa = compute_sa(
        sp,
        z_or_p,
        lon,
        lat,
        is_pressure=is_pressure,
        normalize_lon=normalize_lon,
    )
    _dbg(f'compute_sa total: {time.perf_counter() - t0:.3f}s')
    t1 = time.perf_counter()
    ct = compute_ct(pt, sa)
    _dbg(f'compute_ct total: {time.perf_counter() - t1:.3f}s')
    return sa, ct


def convert_dataset_to_ct_sa(
    ds_thetao: xr.Dataset | xr.DataArray,
    ds_so: xr.Dataset | xr.DataArray,
    thetao_var: str = 'thetao',
    so_var: str = 'so',
    lat_var: str = 'lat',
    lon_var: str = 'lon',
    depth_var: str = 'lev',
    depth_positive: str | None = None,
) -> xr.Dataset:
    """Convert a thetao/so pair to a dataset with ct and sa.

    Parameters
    ----------
    ds_thetao : xr.Dataset or xr.DataArray
        Dataset containing potential temperature variable ``thetao_var``
        (or a DataArray of potential temperature).
    ds_so : xr.Dataset or xr.DataArray
        Dataset containing Practical Salinity variable ``so_var`` (or a
        DataArray of Practical Salinity).
    thetao_var : str, optional
        Name of the potential temperature variable in ``ds_thetao``.
    so_var : str, optional
        Name of the Practical Salinity variable in ``ds_so``.
    lat_var, lon_var : str, optional
        Names of the latitude and longitude coordinate variables.
    depth_var : str, optional
        Name of the depth coordinate (commonly 'lev').
    depth_positive : {"down", "up", None}, optional
        Explicit direction of ``depth_var``. If None, inferred from CF
        attributes when available.

    Returns
    -------
    xr.Dataset
        Dataset with variables:
        - ``ct`` (degC): Conservative Temperature
        - ``sa`` (g kg-1): Absolute Salinity
        Common coordinates (time, depth_var, lat_var, lon_var) are attached
        when present in inputs. Basic provenance comments are set.
    """
    _dbg('convert_dataset_to_ct_sa: start')
    # Pull arrays
    if isinstance(ds_thetao, xr.Dataset):
        pt = ds_thetao[thetao_var]
        lat = ds_thetao[lat_var]
        lon = ds_thetao[lon_var]
        z = ds_thetao[depth_var]
    else:
        pt = ds_thetao
        ds = ds_so.to_dataset(name=so_var)
        lat = ds[lat_var]
        lon = ds[lon_var]
        z = ds[depth_var]

    if isinstance(ds_so, xr.Dataset):
        sp = ds_so[so_var]
    else:
        sp = ds_so

    t_all = time.perf_counter()
    # Align (exact) and broadcast
    pt, sp = xr.align(pt, sp, join='exact')
    _dbg('aligned shapes:', 'pt', tuple(pt.shape), 'sp', tuple(sp.shape))
    # Ensure depth is TEOS-10 z (negative downward)
    z = _depth_to_z(z, positive=depth_positive)
    _dbg('z converted')

    # Performance: eagerly load inputs for this chunk to avoid large dask
    # graphs and scheduler overhead during TEOS-10 transformations.
    # (Safe because caller processes small per-time chunks.)
    t_load = time.perf_counter()
    pt = pt.load()
    sp = sp.load()
    z = z.load()
    lon = lon.load()
    lat = lat.load()
    _dbg(
        'loaded inputs:',
        'pt',
        tuple(pt.shape),
        'sp',
        tuple(sp.shape),
        'z',
        tuple(z.shape),
        'lon',
        tuple(lon.shape),
        'lat',
        tuple(lat.shape),
    )
    _dbg(f'load time: {time.perf_counter() - t_load:.3f}s')

    sa, ct = compute_ct_sa(sp=sp, pt=pt, z_or_p=z, lon=lon, lat=lat)

    ds_out = xr.Dataset({'ct': ct, 'sa': sa})
    # Copy common coords (time, lev, lat, lon). Use what's present in pt.
    for coord in (
        c for c in [depth_var, lat_var, lon_var, 'time'] if c in pt.coords
    ):
        ds_out = ds_out.assign_coords({coord: pt[coord]})

    # Basic provenance
    ds_out['ct'].attrs['comment'] = (
        'Computed with TEOS-10 (gsw) from thetao and so.'
    )
    ds_out['sa'].attrs['comment'] = (
        'Computed with TEOS-10 (gsw) from so, depth, lon, lat.'
    )
    _dbg(
        'convert_dataset_to_ct_sa total:',
        f'{time.perf_counter() - t_all:.3f}s',
    )
    return ds_out


# helper functions


def _normalize_lon(lon: xr.DataArray) -> xr.DataArray:
    """Normalize longitude to the [0, 360) interval.

    Parameters
    ----------
    lon : xr.DataArray
        Longitude values in degrees, any wrap convention.

    Returns
    -------
    xr.DataArray
        Longitudes wrapped into [0, 360), same shape as input.
    """
    lon_vals = lon % 360.0
    lon_vals = xr.where(lon_vals < 0, lon_vals + 360.0, lon_vals)
    lon_vals = xr.where(lon_vals >= 360.0, lon_vals - 360.0, lon_vals)
    lon_out = lon.copy()
    lon_out.data = lon_vals.data
    return lon_out


def _depth_to_z(
    depth: xr.DataArray, positive: str | None = None
) -> xr.DataArray:
    """
    Convert a depth coordinate to TEOS-10 z (m, negative downward).

    Parameters
    ----------
    depth : xr.DataArray
        Depth coordinate (commonly positive downward in meters).
    positive : {"down", "up", None}
        Optional explicit direction. If None, infer from CF attrs if present.

    Returns
    -------
    xr.DataArray
        TEOS-10 z coordinate (m, negative downward) with units and
        long_name attributes set.
    """
    if positive is None:
        positive = depth.attrs.get('positive')
        if positive is not None:
            positive = positive.lower()
    units = depth.attrs.get('units', '').lower()
    # Handle units and convert to meters if necessary (assume meters if
    # unit-less)
    if units in ('cm', 'centimeter', 'centimeters'):
        scale_to_m = 1.0e-2
    elif units in ('m', 'meter', 'meters', ''):
        scale_to_m = 1.0
    else:
        raise ValueError(
            f"Unsupported depth units '{units}'. Expected meters or "
            f'centimeters.'
        )

    # convert to meters first
    depth_m = depth * scale_to_m

    if positive == 'up':
        # depth increases upward -> uncommon, treat as altitude
        z = depth_m
    else:
        # Common case: depth positive downward -> z is negative
        z = -xr.apply_ufunc(np.asarray, depth_m)

    z = z.assign_attrs(
        units='m',
        long_name='height above mean sea level',
    )
    return z
