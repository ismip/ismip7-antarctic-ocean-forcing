"""
Shared utilities for propagating ISMIP grid coordinates and metadata.

This module centralizes logic for:

- Attaching standard ISMIP spatial coordinates (x, y, z/z_extrap) and
  their bounds, as well as geodetic coordinates (lat/lon + bounds) and
  CRS metadata, to output datasets.
- Propagating time/time_bnds coordinates and their key attrs from a
  source dataset to a target dataset. This is useful when intermediate
  storage (e.g., Zarr) may alter encodings or units.
- Selecting a dataset comprised of a target variable plus any bounds
  variables, while preserving coordinates.
- Removing _FillValue encodings/attrs from coordinate variables and
  other auxiliary variables so that only data variables receive fill
  values in outputs.

These helpers are intended to be used across remapping, extrapolation,
time-averaging, bias correction, TF computation, and CT/SA->thetao/so
conversion workflows to ensure consistent coordinate handling.
"""

from typing import Iterable

import xarray as xr
from mpas_tools.config import MpasConfigParser

from i7aof.grid.ismip import ensure_ismip_grid
from i7aof.io import ensure_cftime_time, read_dataset

__all__ = [
    'attach_grid_coords',
    'dataset_with_var_and_bounds',
    'strip_fill_on_non_data',
    'propagate_time_from',
    'vertical_name_for',
    'ensure_cf_time_encoding',
]


def attach_grid_coords(
    ds: xr.Dataset,
    config: MpasConfigParser,
    *,
    z_name: str | None = None,
    include_lat_lon: bool = True,
    validate_xy: bool = True,
) -> xr.Dataset:
    """
    Attach ISMIP grid coordinates, bounds, and CRS to a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Target dataset to receive ISMIP spatial and geodetic coordinates.
    config : mpas_tools.config.MpasConfigParser
        Configuration used to locate or generate the ISMIP grid via
        :func:`i7aof.grid.ismip.ensure_ismip_grid`.
    z_name : str, optional
        Explicit vertical coordinate to attach, usually ``"z"`` or
        ``"z_extrap"``. If not provided, the vertical coordinate name is
        inferred with :func:`vertical_name_for`.
    include_lat_lon : bool, default: True
        When True, also attach ``lat``/``lon`` and their bounds, plus
        ``crs`` if present in the grid file.
    validate_xy : bool, default: True
        Validate that ``(y, x)`` dimensions in ``ds`` (if present) match
        the grid. A mismatch raises ``ValueError``.

    Returns
    -------
    xarray.Dataset
        A new dataset with ISMIP ``x``/``y`` (and bounds), the chosen
        vertical coordinate and bounds, geodetic coordinates (optional),
        and ``crs`` (when available) attached.
    """
    grid_path = ensure_ismip_grid(config)
    ds_grid = read_dataset(grid_path)

    out = ds

    # Optional validation of (y, x) sizes when present
    if validate_xy and 'y' in out.dims and 'x' in out.dims:
        if out.sizes.get('y') != ds_grid.sizes.get('y') or out.sizes.get(
            'x'
        ) != ds_grid.sizes.get('x'):
            raise ValueError(
                'Output (y, x) dimensions do not match ISMIP grid: '
                f'got (y={out.sizes.get("y")}, x={out.sizes.get("x")}), '
                f'expected (y={ds_grid.sizes.get("y")}, '
                f'x={ds_grid.sizes.get("x")}).'
            )

    # x/y as coordinates
    for coord in ('x', 'y'):
        if coord in ds_grid:
            out = out.assign_coords({coord: ds_grid[coord]})
            bname = ds_grid[coord].attrs.get('bounds', f'{coord}_bnds')
            if bname in ds_grid:
                out[bname] = ds_grid[bname]
                # copy attrs to preserve metadata on bounds
                out[bname].attrs = ds_grid[bname].attrs.copy()
            out[coord].attrs['bounds'] = bname

    # determine vertical coordinate name
    vname = z_name or vertical_name_for(out)
    if vname is not None and vname in ds_grid:
        out = out.assign_coords({vname: ds_grid[vname]})
        vbounds = ds_grid[vname].attrs.get('bounds', f'{vname}_bnds')
        if vbounds in ds_grid:
            out[vbounds] = ds_grid[vbounds]
            out[vbounds].attrs = ds_grid[vbounds].attrs.copy()
        out[vname].attrs['bounds'] = vbounds

    if include_lat_lon:
        if 'lat' in ds_grid:
            out = out.assign_coords({'lat': ds_grid['lat']})
            if 'lat_bnds' in ds_grid:
                out['lat_bnds'] = ds_grid['lat_bnds']
                out['lat_bnds'].attrs = ds_grid['lat_bnds'].attrs.copy()
            out['lat'].attrs['bounds'] = 'lat_bnds'
        if 'lon' in ds_grid:
            out = out.assign_coords({'lon': ds_grid['lon']})
            if 'lon_bnds' in ds_grid:
                out['lon_bnds'] = ds_grid['lon_bnds']
                out['lon_bnds'].attrs = ds_grid['lon_bnds'].attrs.copy()
            out['lon'].attrs['bounds'] = 'lon_bnds'
        if 'crs' in ds_grid and 'crs' not in out:
            out['crs'] = ds_grid['crs']

    return out


def dataset_with_var_and_bounds(ds: xr.Dataset, var_name: str) -> xr.Dataset:
    """
    Return a dataset with a variable plus any present bounds variables.

    Parameters
    ----------
    ds : xarray.Dataset
        Source dataset from which to select variables.
    var_name : str
        Name of the data variable to include.

    Returns
    -------
    xarray.Dataset
        Subset dataset containing ``var_name`` and any of these bounds
        variables that exist in ``ds``: ``time_bnds``, ``x_bnds``,
        ``y_bnds``, ``z_bnds``, ``z_extrap_bnds``, ``lat_bnds``,
        ``lon_bnds``. Coordinates are preserved by xarray.
    """
    bound_names = (
        'time_bnds',
        'x_bnds',
        'y_bnds',
        'z_bnds',
        'z_extrap_bnds',
        'lat_bnds',
        'lon_bnds',
    )
    present = [b for b in bound_names if b in ds]
    # ensure unique without changing order
    names: list[str] = [var_name] + [b for b in present if b != var_name]
    return ds[names]


def strip_fill_on_non_data(
    ds: xr.Dataset, data_vars: Iterable[str]
) -> xr.Dataset:
    """
    Remove ``_FillValue`` from all non-data variables.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to modify in-place.
    data_vars : Iterable[str]
        Names of data variables that should retain their ``_FillValue``.
        All other variables (coordinates and auxiliaries) will have the
        key removed from both ``.encoding`` and ``.attrs`` if present.

    Returns
    -------
    xarray.Dataset
        The same dataset instance with non-data variables cleared of
        ``_FillValue``.
    """
    data_set = set(data_vars)
    for name in list(ds.data_vars) + list(ds.coords):
        if name in data_set:
            continue
        var = ds[name]
        enc = getattr(var, 'encoding', None)
        if isinstance(enc, dict) and '_FillValue' in enc:
            try:
                del enc['_FillValue']
            except KeyError:
                pass
        attrs = getattr(var, 'attrs', None)
        if isinstance(attrs, dict) and '_FillValue' in attrs:
            attrs.pop('_FillValue', None)
    return ds


def propagate_time_from(
    target: xr.Dataset,
    source: xr.Dataset,
    *,
    apply_cf_encoding: bool = False,
    units: str | None = None,
    calendar: str | None = None,
    prefer_source: xr.Dataset | None = None,
) -> xr.Dataset:
    """
    Copy time coordinate and bounds from ``source`` into ``target``.

    Parameters
    ----------
    target : xarray.Dataset
        Dataset to receive the time coordinate and bounds.
    source : xarray.Dataset
        Dataset providing ``time`` and, optionally, ``time_bnds``.
    apply_cf_encoding : bool, optional
        When True, also ensure CF-compliant shared encoding (units,
        calendar, dtype) for ``time`` and ``time_bnds`` on the returned
        dataset. Default is False to keep this function side-effect free
        for mid-pipeline use.
    units : str, optional
        CF time units string to apply if ``apply_cf_encoding`` is True.
        Defaults to ``'days since 1850-01-01 00:00:00'`` when omitted.
    calendar : str, optional
        Calendar to apply if ``apply_cf_encoding`` is True. When None,
        inferred from source or target, otherwise falls back to
        ``'proleptic_gregorian'``.
    prefer_source : xarray.Dataset, optional
        If provided, use this dataset as the preferred source for
        calendar inference when applying CF encoding. Defaults to the
        ``source`` dataset.

    Returns
    -------
    xarray.Dataset
        Updated dataset where, if sizes match, ``time`` values and
        attributes come from ``source`` and ``time_bnds`` is attached when
        present. Also sets ``target['time'].attrs['bounds'] = 'time_bnds'``.
        If ``apply_cf_encoding`` is True, shared encodings are applied to
        ``time`` and ``time_bnds`` to ensure CF-compliant serialization.
    """
    if 'time' in target.dims and 'time' in source.dims:
        if target.sizes.get('time') == source.sizes.get('time'):
            target = target.assign_coords(time=source['time'])
            if 'time_bnds' in source:
                target['time_bnds'] = source['time_bnds']
                target['time'].attrs['bounds'] = 'time_bnds'
    if apply_cf_encoding:
        ensure_cf_time_encoding(
            target,
            units=units or 'days since 1850-01-01 00:00:00',
            calendar=calendar,
            prefer_source=prefer_source or source,
        )
    return target


def vertical_name_for(ds: xr.Dataset) -> str | None:
    """
    Infer the preferred vertical coordinate name for a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset whose variables/coordinates will be inspected.

    Returns
    -------
    str or None
        One of ``"z"`` or ``"z_extrap"`` if present, preferring names
        that appear in any data variable dimensions. Returns ``None`` if
        neither vertical coordinate is found.
    """
    candidates = ('z', 'z_extrap')
    # check data variables' dims first
    for v in ds.data_vars:
        dims = getattr(ds[v], 'dims', ())
        for c in candidates:
            if c in dims:
                return c
    # check coords in ds
    for c in candidates:
        if c in ds.coords or c in ds.variables:
            return c
    return None


def ensure_cf_time_encoding(
    ds: xr.Dataset,
    *,
    units: str = 'days since 1850-01-01',
    calendar: str | None = None,
    prefer_source: xr.Dataset | None = None,
) -> xr.Dataset:
    """
    Ensure CF-compliant, shared encoding for ``time`` and ``time_bnds``.

    This sets identical ``units`` and (when appropriate) ``calendar``
    encodings on both variables so xarray encodes them consistently and
    avoids warnings about divergent encodings.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset whose encodings will be updated in-place.
    units : str, optional
        CF time units string to apply. Default is
        ``'days since 1850-01-01 00:00:00'``.
    calendar : str, optional
        Calendar to apply. If ``None``, inferred from ``ds['time'].attrs``
        (or from ``prefer_source['time'].attrs`` when provided). Falls back
        to ``'proleptic_gregorian'``.
    prefer_source : xarray.Dataset, optional
        If provided, use its time attributes as a preferred source for
        calendar inference.

    Returns
    -------
    xarray.Dataset
        The same dataset instance with encodings updated.
    """
    if 'time' not in ds:
        return ds

    # Determine calendar preference: prefer explicit arg, then source, then ds
    cal = calendar

    def _extract_calendar(obj: xr.Dataset) -> str | None:
        if 'time' not in obj:
            return None
        t = obj['time']
        # Prefer encoding (where xarray stores decoded meta), then attrs
        if hasattr(t, 'encoding') and isinstance(t.encoding, dict):
            cal_enc = t.encoding.get('calendar')
            if isinstance(cal_enc, str) and cal_enc:
                return cal_enc
        tattrs = getattr(t, 'attrs', {}) or {}
        return tattrs.get('calendar') or tattrs.get('calendar_type')

    if cal is None and prefer_source is not None:
        cal = _extract_calendar(prefer_source)
    if cal is None:
        cal = _extract_calendar(ds)

    if cal is None:
        raise ValueError(
            "Cannot determine calendar for 'time' variable; "
            "please ensure 'calendar' is set in attributes or encoding."
        )

    # Ensure values are cftime objects for predictable CF encoding
    ensure_cftime_time(ds, cal)

    # Place units/calendar in encoding for CF encoding; remove from attrs
    # to avoid safe_setitem conflicts during serialization.
    t = ds['time']
    if isinstance(getattr(t, 'attrs', None), dict):
        t.attrs.pop('units', None)
        t.attrs.pop('calendar', None)
    if isinstance(getattr(t, 'encoding', None), dict):
        t.encoding['units'] = units
        t.encoding['calendar'] = cal
        t.encoding['dtype'] = 'float64'

    if 'time_bnds' in ds:
        tb = ds['time_bnds']
        if isinstance(getattr(tb, 'attrs', None), dict):
            tb.attrs.pop('units', None)
            tb.attrs.pop('calendar', None)
        if isinstance(getattr(tb, 'encoding', None), dict):
            tb.encoding['units'] = units
            tb.encoding['calendar'] = cal
            tb.encoding['dtype'] = 'float64'

    return ds
