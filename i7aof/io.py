import os
import subprocess
from pathlib import Path

import netCDF4
import numpy
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from xarray.coders import CFDatetimeCoder


def read_dataset(path, **kwargs) -> xr.Dataset:
    """Open a dataset with package defaults and fix time/time_bnds encodings.

    - Ensures cftime decoding for robust non-standard calendars.
    - If both ``time`` and ``time_bnds`` are present, propagate the same
      units/calendar into ``time_bnds.encoding`` to satisfy CF expectations
      and silence xarray warnings when writing.

    Any extra keyword arguments are passed through to ``xarray.open_dataset``,
    but ``decode_times`` will default to ``CFDatetimeCoder(use_cftime=True)``
    unless explicitly overridden.
    """
    if 'decode_times' not in kwargs:
        kwargs['decode_times'] = CFDatetimeCoder(use_cftime=True)

    ds = xr.open_dataset(path, **kwargs)

    try:
        if 'time' in ds.variables and 'time_bnds' in ds.variables:
            t = ds['time']
            tb = ds['time_bnds']

            # Derive units/calendar from time's encoding or attrs
            t_units = None
            t_cal = None
            if hasattr(t, 'encoding'):
                t_units = t.encoding.get('units')
                t_cal = t.encoding.get('calendar')
            if t_units is None:
                t_units = t.attrs.get('units')
            if t_cal is None:
                t_cal = t.attrs.get('calendar')

            # Only set when available; prefer encoding, avoid attrs to
            # prevent conflicts during Zarr/NetCDF encoding.
            if t_units is not None:
                tb.encoding['units'] = t_units
            if t_cal is not None:
                tb.encoding['calendar'] = t_cal
    except (KeyError, AttributeError, TypeError, ValueError):
        # Be forgiving: reading should not fail due to metadata tweaks
        pass

    return ds


def write_netcdf(
    ds,
    filename,
    fillvalues=None,
    format=None,
    engine=None,
    progress_bar=False,
    has_fill_values=None,
):
    """
    Write an xarray.Dataset to a file with NetCDF4 fill values

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to save

    filename : str
        The path for the NetCDF file to write

    fillvalues : dict, optional
        A dictionary of fill values for different NetCDF types.  Default is
        ``netCDF4.default_fillvals``

    format : {'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT', 'NETCDF3_CLASSIC'}, optional
        The NetCDF file format to use, the default is 'NETCDF4'

    engine : {'netcdf4', 'scipy', 'h5netcdf'}, optional
        The library to use for NetCDF output, the default is 'netcdf4'

    has_fill_values : bool | dict | callable, optional
        Controls whether to apply ``_FillValue`` per variable without scanning
        data:
          - bool: apply to all variables (True adds, False omits)
          - dict: mapping of var_name -> bool
          - callable: function (var_name, var: xarray.DataArray) -> bool
        If omitted (None), the function determines necessity by checking for
        NaNs using xarray's lazy operations (``var.isnull().any().compute()``),
        which is safe for chunked datasets. For unchunked datasets, the scan
        may load data into memory; callers can avoid this by opening datasets
        with Dask chunks.
    """  # noqa: E501
    if fillvalues is None:
        fillvalues = netCDF4.default_fillvals

    numpy_fillvals = {}
    for filltype, fillvalue in fillvalues.items():
        # drop string fill values
        if not filltype.startswith('S'):
            numpy_fillvals[numpy.dtype(filltype)] = fillvalue

    encoding_dict = _build_encoding_dict(ds, numpy_fillvals, has_fill_values)

    if 'time' in ds.dims:
        # make sure the time dimension is unlimited
        ds.encoding['unlimited_dims'] = {'time'}
    else:
        # make sure there are no unlimited dimensions
        ds.encoding['unlimited_dims'] = set()

    # Standardize time encodings across the package to avoid warnings and
    # ensure CF-consistent units between time and time_bnds.
    # Use days since 1850-01-01 as the common reference.
    _apply_time_encoding(ds, encoding_dict)

    # for performance, we have to handle this as a special case
    convert = format == 'NETCDF3_64BIT_DATA'

    if convert:
        out_path = Path(filename)
        out_filename = (
            out_path.parent / f'_tmp_{out_path.stem}.netcdf4{out_path.suffix}'
        )
        format = 'NETCDF4'
        if engine == 'scipy':
            # that's not going to work
            engine = 'netcdf4'
    else:
        out_filename = filename

    write_job = ds.to_netcdf(
        out_filename,
        encoding=encoding_dict,
        format=format,
        engine=engine,
        compute=not progress_bar,
    )

    if progress_bar:
        with ProgressBar():
            print(f'Writing to {out_filename}:')
            write_job.compute()

    if convert:
        args = [
            'ncks',
            '-O',
            '-5',
            out_filename,
            filename,
        ]
        # Ensure all args are strings (important for Path objects)
        args = [str(arg) for arg in args]
        subprocess.run(
            args,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # delete the temporary NETCDF4 file
        os.remove(out_filename)


def _apply_time_encoding(
    ds: xr.Dataset,
    encoding_dict: dict,
    default_time_units: str = 'days since 1850-01-01',
) -> None:
    """
    Ensure consistent time/time_bnds encodings and clear conflicting attrs.
    """
    if 'time' not in ds.variables:
        return
    time_var = ds['time']
    # Remove conflicting attrs that would cause xarray to raise when encoding
    for key in ('units', 'calendar'):
        time_var.attrs.pop(key, None)
    # Determine calendar, prefer existing, otherwise default
    calendar = (
        time_var.encoding.get('calendar')
        if hasattr(time_var, 'encoding')
        else None
    )
    if calendar is None:
        calendar = time_var.attrs.get('calendar')
    if calendar is None:
        calendar = 'proleptic_gregorian'

    # Force units and calendar to be consistent for time and time_bnds
    time_units = default_time_units
    time_var.encoding['units'] = time_units
    time_var.encoding['calendar'] = calendar
    # Also pass explicitly via encoding_dict to ensure backend sees it
    encoding_dict.setdefault('time', {})
    encoding_dict['time']['units'] = time_units
    encoding_dict['time']['calendar'] = calendar

    # Ensure time_bnds (if present) uses identical units/calendar
    if 'time_bnds' in ds.variables:
        tb = ds['time_bnds']
        for key in ('units', 'calendar'):
            tb.attrs.pop(key, None)
        tb.encoding['units'] = time_units
        tb.encoding['calendar'] = calendar
        encoding_dict.setdefault('time_bnds', {})
        encoding_dict['time_bnds']['units'] = time_units
        encoding_dict['time_bnds']['calendar'] = calendar


def _build_encoding_dict(
    dataset: xr.Dataset, numpy_fillvals: dict, has_fill_values
) -> dict:
    """Build encoding dict for variables, including _FillValue decisions."""
    enc: dict = {}
    var_names_local = list(dataset.data_vars.keys()) + list(
        dataset.coords.keys()
    )
    for vn in var_names_local:
        var = dataset[vn]
        enc_v = _var_encoding(vn, var, numpy_fillvals, has_fill_values)
        if enc_v:
            enc[vn] = enc_v
    return enc


def _decide_fill_value(var_name, var, numpy_fillvals, has_fill_values):
    """Return an appropriate _FillValue (or None) for a variable.

    - Respects caller directive (bool|dict|callable) when provided.
    - Otherwise detects NaNs via xarray lazy ops (safe for chunked arrays).
    """
    dtype = getattr(var, 'dtype', None)
    candidate = numpy_fillvals.get(dtype)
    if candidate is None:
        return None

    # Caller directive overrides default behavior
    if has_fill_values is not None:
        if isinstance(has_fill_values, bool):
            return candidate if has_fill_values else None
        if isinstance(has_fill_values, dict):
            choice = has_fill_values.get(var_name)
            if choice is not None:
                return candidate if choice else None
        if callable(has_fill_values):
            try:
                choice = bool(has_fill_values(var_name, var))
                return candidate if choice else None
            except (TypeError, ValueError):
                # Fall back to default behavior
                pass

    # Default: detect NaNs using xarray (works well for chunked data)
    try:
        has_nan = bool(var.isnull().any().compute())
    except (RuntimeError, ValueError):
        has_nan = False
    return candidate if has_nan else None


def _var_encoding(var_name, var, numpy_fillvals, has_fill_values):
    """Compute per-variable encoding for _FillValue.

    - If directive is False, set ``_FillValue`` to None to suppress backend
      defaults.
    - If directive is True, set to the type-appropriate candidate, even if
      NaNs are not present.
    - Otherwise, detect NaNs lazily and set only when needed.
    - Preserve explicit values from var.encoding/attrs.
    """
    encoding = {}

    # Determine explicit directive
    directive = None
    if has_fill_values is not None:
        if isinstance(has_fill_values, bool):
            directive = has_fill_values
        elif isinstance(has_fill_values, dict):
            directive = has_fill_values.get(var_name)
        elif callable(has_fill_values):
            try:
                directive = bool(has_fill_values(var_name, var))
            except (TypeError, ValueError):
                directive = None

    # Explicitly disabled: prevent backend default by setting None
    if directive is False:
        encoding['_FillValue'] = None
        return encoding

    # Explicitly enabled or default behavior
    if directive is True:
        fill = numpy_fillvals.get(getattr(var, 'dtype', None))
    else:
        # Default behavior uses detection without given directive
        fill = _decide_fill_value(var_name, var, numpy_fillvals, None)

    present_in_enc = '_FillValue' in var.encoding
    present_in_attrs = '_FillValue' in var.attrs
    if fill is not None or present_in_enc or present_in_attrs:
        # Preserve explicit None to suppress backend auto-fill when present
        encoding['_FillValue'] = var.encoding.get(
            '_FillValue', var.attrs.get('_FillValue', fill)
        )

    return encoding
