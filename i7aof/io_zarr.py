"""Utilities for writing chunked datasets to Zarr and finalizing to NetCDF.

This module centralizes the common pattern used across workflows:

1) Append one or more chunked xarray Datasets to a temporary Zarr store
2) Open the consolidated Zarr dataset, optionally apply a postprocess
3) Write the final NetCDF once and remove the Zarr store

The helpers wrap xarray's to_zarr/open_zarr and reuse i7aof.io.write_netcdf
for consistent NetCDF writing options (fill values, progress bar, engine).
"""

from __future__ import annotations

import os
import shutil
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from typing import List

import cftime
import numpy as np
import xarray as xr
from xarray.coding.common import SerializationWarning

from i7aof.io import _ensure_cftime_time, write_netcdf

__all__ = ['append_to_zarr', 'finalize_zarr_to_netcdf']


def append_to_zarr(
    *, ds: xr.Dataset, zarr_store: str, first: bool, append_dim: str | None
) -> bool:
    """Append a dataset to a Zarr store, creating it if ``first`` is True.

    Returns the updated value for ``first`` (False after first write).
    """

    # If a previous run marked the Zarr store as ready (or legacy complete
    # marker exists), skip re-writing/append to keep idempotent behavior.
    if _zarr_is_ready(zarr_store) or _zarr_is_complete(zarr_store):
        return False

    ds_to_write = _sanitize_time_attrs(ds)

    if first:
        if os.path.isdir(zarr_store):
            shutil.rmtree(zarr_store, ignore_errors=True)
        with _suppress_zarr_warnings():
            ds_to_write.to_zarr(zarr_store, mode='w')
        return False
    with _suppress_zarr_warnings():
        if append_dim is None:
            # Idempotency: if the store already exists and there's no
            # append dimension, assume the segment is present and skip
            if os.path.isdir(zarr_store):
                return False
            ds_to_write.to_zarr(zarr_store, mode='w')
        else:
            # If the segment's coord values along append_dim are already
            # fully present in the store, skip appending to keep idempotent
            if os.path.isdir(zarr_store) and _segment_already_present(
                zarr_store, append_dim, ds_to_write[append_dim].values
            ):
                return False
            ds_to_write.to_zarr(zarr_store, mode='a', append_dim=append_dim)
    return False


def finalize_zarr_to_netcdf(
    *,
    zarr_store: str,
    out_nc: str,
    has_fill_values: List[str] | None = None,
    compression: List[str] | None = None,
    progress_bar: bool = True,
    postprocess: Callable[[xr.Dataset], xr.Dataset] | None = None,
) -> None:
    """Open a Zarr store, optionally postprocess, then write NetCDF and clean.

    Parameters
    ----------
    zarr_store : str
        Path to the Zarr store directory to consolidate.
    out_nc : str
        Target NetCDF output path.
    has_fill_values : list, optional
        A list of variable names to which to apply fill values.
    compression : list, optional
        A list of variable names to compress.
    progress_bar : bool, optional
        Whether to show a write progress bar (forwarded to write_netcdf).
    postprocess : callable, optional
        Function mapping ``xr.Dataset -> xr.Dataset`` applied before writing
        NetCDF to allow attribute fixes, coordinate/bounds injection, etc.
    """
    # Avoid format-3 consolidated metadata warning and disable consolidated
    # metadata usage since we immediately convert to NetCDF
    with _suppress_zarr_warnings():
        ds = xr.open_zarr(zarr_store, consolidated=False)
    try:
        # Mark Zarr as ready as soon as we can successfully open it. If the
        # subsequent NetCDF write fails, this allows reruns to skip the Zarr
        # append phase and retry conversion only.
        _mark_zarr_ready(zarr_store)
        if postprocess is not None:
            ds = postprocess(ds)
        # Ensure deterministic CF-time encoding: cftime values and matching
        # units/calendar for time and time_bnds right before NetCDF write.
        if 'time' in ds:
            t = ds['time']
            cal = (
                (
                    t.encoding.get('calendar')
                    if isinstance(t.encoding, dict)
                    else None
                )
                or t.attrs.get('calendar')
                or 'proleptic_gregorian'
            )
            _ensure_cftime_time(ds, cal)
            # Force-convert to numeric days since 1850 for both time and
            # time_bnds to avoid backend-specific unit choices.
            tunits = 'days since 1850-01-01 00:00:00'

            # time
            tvals = np.array(
                cftime.date2num(list(ds['time'].values), tunits, calendar=cal),
                dtype=np.float64,
            )
            ds['time'] = xr.DataArray(tvals, dims=ds['time'].dims)
            ds['time'].attrs['units'] = tunits
            ds['time'].attrs['calendar'] = cal
            if isinstance(ds['time'].encoding, dict):
                ds['time'].encoding.clear()

            # time_bnds
            if 'time_bnds' in ds:
                tb_vals = ds['time_bnds'].values
                # vectorize over both columns
                tb0 = np.array(
                    cftime.date2num(list(tb_vals[:, 0]), tunits, calendar=cal),
                    dtype=np.float64,
                )
                tb1 = np.array(
                    cftime.date2num(list(tb_vals[:, 1]), tunits, calendar=cal),
                    dtype=np.float64,
                )
                tb_num = np.stack([tb0, tb1], axis=1)
                ds['time_bnds'] = xr.DataArray(
                    tb_num, dims=ds['time_bnds'].dims
                )
                ds['time_bnds'].attrs['units'] = tunits
                ds['time_bnds'].attrs['calendar'] = cal
                if isinstance(ds['time_bnds'].encoding, dict):
                    ds['time_bnds'].encoding.clear()
        # Write NetCDF to a temporary path first, then atomically move to the
        # final destination on success. This avoids leaving a partially-
        # written NetCDF when failures occur and removes the need for a
        # ".complete" file.
        out_tmp = f'{out_nc}.tmp'
        write_netcdf(
            ds,
            out_tmp,
            has_fill_values=has_fill_values,
            progress_bar=progress_bar,
            compression=compression,
        )
    finally:
        ds.close()
    # Finalize atomically: move tmp -> final, then remove Zarr store and any
    # legacy external marker if present.
    if os.path.isfile(out_tmp):
        os.replace(out_tmp, out_nc)
        shutil.rmtree(zarr_store, ignore_errors=True)
        _cleanup_legacy_complete_marker(zarr_store)


@contextmanager
def _suppress_zarr_warnings():
    """Context manager to silence common Zarr/time serialization warnings.

    Suppresses:
    - Consolidated metadata warning for Zarr format v3
    - Time decoding SerializationWarning about datetime64 range
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=(
                'Consolidated metadata is currently not part in the Zarr '
                'format 3 specification.'
            ),
        )
        warnings.filterwarnings(
            'ignore',
            message=('Unable to decode time axis into full numpy.datetime64'),
            category=SerializationWarning,
        )
        yield


def _zarr_complete_marker(zarr_store: str) -> str:
    """Legacy external marker path (created by older versions).

    Still recognized to keep reruns idempotent, but no longer created.
    """
    return f'{zarr_store}.complete'


def _cleanup_legacy_complete_marker(zarr_store: str) -> None:
    """Remove legacy external marker file if it exists."""
    marker = _zarr_complete_marker(zarr_store)
    try:
        if os.path.isfile(marker):
            os.remove(marker)
    except OSError:
        # Non-fatal cleanup failure
        pass


def _zarr_is_complete(zarr_store: str) -> bool:
    marker = _zarr_complete_marker(zarr_store)
    return os.path.isfile(marker)


def _zarr_ready_marker(zarr_store: str) -> str:
    """Return path to the in-store marker indicating Zarr is ready.

    Using a hidden file inside the Zarr store avoids cluttering the parent
    directory and ensures automatic cleanup when the store is removed.
    """
    return os.path.join(zarr_store, '.i7aof_zarr_ready')


def _mark_zarr_ready(zarr_store: str) -> None:
    """Create/refresh the internal ready marker inside the Zarr store."""
    marker = _zarr_ready_marker(zarr_store)
    try:
        # Ensure the directory exists; if it doesn't, just skip quietly
        if os.path.isdir(zarr_store):
            with open(marker, 'w', encoding='utf-8') as f:
                f.write('ready\n')
    except OSError:
        # Non-fatal: inability to write marker shouldn't fail the pipeline
        pass


def _zarr_is_ready(zarr_store: str) -> bool:
    """Return True if the internal ready marker exists inside the store."""
    marker = _zarr_ready_marker(zarr_store)
    return os.path.isfile(marker)


def _segment_already_present(
    zarr_store: str, append_dim: str, new_coords: np.ndarray
) -> bool:
    """Return True if all values in new_coords already exist in the store."""
    with _suppress_zarr_warnings():
        ds = xr.open_zarr(zarr_store, consolidated=False)
    try:
        if append_dim not in ds:
            return False
        existing = ds[append_dim].values
        # Fast path: if last new coord is less/equal to last existing and
        # first new coord is >= first existing, it's likely contained;
        # confirm via set inclusion to be safe.
        if existing.size == 0 or new_coords.size == 0:
            return False
        if new_coords[0] < existing[0] or new_coords[-1] > existing[-1]:
            return False
        return bool(np.all(np.isin(new_coords, existing)))
    finally:
        ds.close()


def _sanitize_time_attrs(dsin: xr.Dataset) -> xr.Dataset:
    """
    Sanitize attrs on time-like coordinates to avoid conflicts where
    Xarray encoding tries to set 'units'/'calendar' and they already
    exist in attrs. Keep encoding fields only.
    """
    ds_work = dsin
    for name in ['time', 'time_bnds']:
        if name in ds_work:
            da = ds_work[name]
            # Remove reserved fields from attrs if present
            for key in ('units', 'calendar'):
                if key in da.attrs:
                    # make a shallow copy first to avoid mutating input
                    da = da.copy()
                    da.attrs.pop(key, None)
            ds_work[name] = da
    return ds_work
