"""
TEOS-10 conversion utilities to compute Absolute Salinity (SA) and
Conservative Temperature (CT) from Practical Salinity (SP) and potential
temperature (PT), with xarray/dask support.

This module is I/O-free and operates on xarray DataArray/Dataset, keeping
variable and dimension names flexible.
"""

from typing import Tuple

import gsw
import numpy as np
import xarray as xr


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

    # Broadcast lon/lat to SP grid
    lon_b, lat_b = xr.broadcast(lon, lat)
    sp_b, lon_b, lat_b = xr.broadcast(sp, lon_b, lat_b)

    if is_pressure:
        p = z_or_p
        # Broadcast pressure to SP grid
        p = xr.broadcast(sp_b, p)[1]
    else:
        # Ensure z is TEOS-10 z (negative downward)
        z = z_or_p
        # If z looks positive-down, flip sign
        if (z.min() >= 0).item() or (
            z.attrs.get('positive', '').lower() == 'down'
        ):
            z = -z
        # Broadcast z and lat to 3D then compute pressure
        z_b = xr.broadcast(sp_b, z)[1]
        lat_b2 = xr.broadcast(sp_b, lat)[1]
        p = xr.apply_ufunc(
            gsw.p_from_z,
            z_b,
            lat_b2,
            dask='parallelized',
            input_core_dims=[[], []],
            output_core_dims=[[]],
            vectorize=True,
            output_dtypes=[sp.dtype],
        )

    sa = xr.apply_ufunc(
        gsw.SA_from_SP,
        sp_b,
        p,
        lon_b,
        lat_b,
        dask='parallelized',
        input_core_dims=[[], [], [], []],
        output_core_dims=[[]],
        vectorize=True,
        output_dtypes=[sp.dtype],
    )
    sa = sa.assign_attrs(units='g kg-1', long_name='Absolute Salinity')
    return sa


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
    pt_b, sa_b = xr.broadcast(pt, sa)
    ct = xr.apply_ufunc(
        gsw.CT_from_pt,
        sa_b,
        pt_b,
        dask='parallelized',
        input_core_dims=[[], []],
        output_core_dims=[[]],
        vectorize=True,
        output_dtypes=[pt.dtype],
    )
    ct = ct.assign_attrs(units='degC', long_name='Conservative Temperature')
    return ct


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
    sa = compute_sa(
        sp,
        z_or_p,
        lon,
        lat,
        is_pressure=is_pressure,
        normalize_lon=normalize_lon,
    )
    ct = compute_ct(pt, sa)
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

    # Align (exact) and broadcast
    pt, sp = xr.align(pt, sp, join='exact')
    # Ensure depth is TEOS-10 z (negative downward)
    z = _depth_to_z(z, positive=depth_positive)

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
    # Basic units sanity check (assume meters if unit-less)
    if units not in ('m', 'meter', 'meters', ''):
        raise ValueError(
            f"Unsupported depth units '{units}'. Expected meters."
        )

    if positive == 'up':
        # depth increases upward -> uncommon, treat as altitude
        z = depth
    else:
        # Common case: depth positive downward -> z is negative
        z = -xr.apply_ufunc(np.asarray, depth)

    z = z.assign_attrs(
        units='m',
        long_name='height above mean sea level',
    )
    return z
