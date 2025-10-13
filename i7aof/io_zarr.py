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
from collections.abc import Callable

import xarray as xr

from i7aof.io import write_netcdf

__all__ = ['append_to_zarr', 'finalize_zarr_to_netcdf']


def append_to_zarr(
    *, ds: xr.Dataset, zarr_store: str, first: bool, append_dim: str | None
) -> bool:
    """Append a dataset to a Zarr store, creating it if ``first`` is True.

    Returns the updated value for ``first`` (False after first write).
    """
    if first:
        if os.path.isdir(zarr_store):
            shutil.rmtree(zarr_store, ignore_errors=True)
        ds.to_zarr(zarr_store, mode='w')
        return False
    ds.to_zarr(zarr_store, mode='a', append_dim=append_dim)
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
    ds = xr.open_zarr(zarr_store)
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
