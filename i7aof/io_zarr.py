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

import xarray as xr
from xarray.coding.common import SerializationWarning

from i7aof.io import write_netcdf

__all__ = ['append_to_zarr', 'finalize_zarr_to_netcdf']


def append_to_zarr(
    *, ds: xr.Dataset, zarr_store: str, first: bool, append_dim: str | None
) -> bool:
    """Append a dataset to a Zarr store, creating it if ``first`` is True.

    Returns the updated value for ``first`` (False after first write).
    """

    ds_to_write = _sanitize_time_attrs(ds)

    if first:
        if os.path.isdir(zarr_store):
            shutil.rmtree(zarr_store, ignore_errors=True)
        with _suppress_zarr_warnings():
            ds_to_write.to_zarr(zarr_store, mode='w')
        return False
    with _suppress_zarr_warnings():
        ds_to_write.to_zarr(zarr_store, mode='a', append_dim=append_dim)
    return False


def finalize_zarr_to_netcdf(
    *,
    zarr_store: str,
    out_nc: str,
    has_fill_values: Callable[[str, xr.DataArray], bool] | None = None,
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
    has_fill_values : callable, optional
        Predicate called as ``has_fill_values(name, var)`` to decide whether
        a variable should use a `_FillValue` in the NetCDF output.
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
        if postprocess is not None:
            ds = postprocess(ds)
        write_netcdf(
            ds,
            out_nc,
            has_fill_values=(
                has_fill_values if has_fill_values else (lambda *_: False)
            ),
            progress_bar=progress_bar,
        )
    finally:
        ds.close()
        shutil.rmtree(zarr_store, ignore_errors=True)


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
