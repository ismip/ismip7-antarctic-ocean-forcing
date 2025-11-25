import os
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cftime
import netCDF4
import numpy
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from xarray.coders import CFDatetimeCoder  # noqa: F401

__all__ = [
    'read_dataset',
    'write_netcdf',
    'ensure_cf_time_encoding',
]


# Default compression options when compression is requested as a boolean
# or via a callable returning True. These options are supported by the
# netCDF4/HDF5-based backends ('netcdf4' and 'h5netcdf').
DEFAULT_COMPRESSION = {
    'zlib': True,
    'complevel': 4,
    'shuffle': True,
}

TIME_UNITS = 'days since 1850-01-01'

NetcdfFormat = Literal[
    'NETCDF4',
    'NETCDF4_CLASSIC',
    'NETCDF3_64BIT',
    'NETCDF3_64BIT_DATA',
    'NETCDF3_CLASSIC',
]
NetcdfEngine = Literal['netcdf4', 'scipy', 'h5netcdf']


def read_dataset(path, **kwargs) -> xr.Dataset:
    """Open a dataset with package defaults and normalize time metadata.

    - Ensures cftime decoding for robust non-standard calendars.
    - If both ``time`` and ``time_bnds`` are present, propagate the same
      units/calendar into ``time_bnds`` attributes (not encoding) so that
      both variables serialize consistently. This avoids placing
      ``units``/``calendar`` in encodings, which recent backends reject.

    Any extra keyword arguments are passed through to
    ``xarray.open_dataset``. By default, this function sets
    ``decode_times=CFDatetimeCoder(use_cftime=True)`` unless explicitly
    overridden, ensuring decoded CF-time coordinates backed by cftime
    objects for predictable behavior across calendars.
    """
    if 'decode_times' not in kwargs:
        kwargs['decode_times'] = CFDatetimeCoder(use_cftime=True)

    ds = xr.open_dataset(path, **kwargs)

    if 'time' in ds.variables:
        if 'bounds' not in ds['time'].attrs:
            raise ValueError(
                "The 'time' variable has no 'bounds' attribute but time "
                'bounds are required for all i7aof workflows with time.'
            )

        # rename time bounds variable to standard name if necessary
        time_bnds_name = ds['time'].attrs['bounds']
        if time_bnds_name != 'time_bnds':
            ds = ds.rename_vars({time_bnds_name: 'time_bnds'})
            ds['time'].attrs['bounds'] = 'time_bnds'

        ensure_cf_time_encoding(ds)

    return ds


def write_netcdf(
    ds: xr.Dataset,
    filename: Union[str, Path],
    fillvalues: Optional[Dict[str, Any]] = None,
    format: Optional[NetcdfFormat] = None,
    engine: Optional[NetcdfEngine] = None,
    progress_bar: bool = False,
    has_fill_values: Optional[Union[bool, List[str]]] = None,
    compression: Optional[Union[bool, List[str]]] = None,
    compression_opts: Optional[Dict[str, Any]] = None,
) -> None:
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

    format : {'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT', 'NETCDF3_64BIT_DATA', 'NETCDF3_CLASSIC'}, optional
        The NetCDF file format to use, the default is 'NETCDF4'

    engine : {'netcdf4', 'scipy', 'h5netcdf'}, optional
        The library to use for NetCDF output, the default is 'h5netcdf' if
        ``compression`` is specified, otherwise 'netcdf4'.

    has_fill_values : bool | list, optional
        Controls whether to apply ``_FillValue`` per variable without scanning
        data:

          - bool: apply to all variables (True adds, False omits)

          - list: the list of variable names to which to apply fill values.

        If omitted (None), the function determines necessity by checking for
        NaNs using xarray's lazy operations (``var.isnull().any().compute()``),
        which is safe for chunked datasets. For unchunked datasets, the scan
        may load data into memory; callers can avoid this by opening datasets
        with Dask chunks.

    compression : bool | list, optional
        Controls variable compression. Accepted forms mirror ``has_fill_values``
        semantics:

          - bool: enable/disable default compression for all variables

          - list: the list of variable names to compress.

        If omitted (None), no compression is enabled by default; callers can
        enable it selectively. Note that support and available options depend
        on the chosen ``engine`` (for example, ``scipy`` does not support
        compression; ``netcdf4``/``h5netcdf`` do).

    compression_opts : dict, optional
        Default compression options to apply when compression is requested via
        a boolean directive. Example: ``{'zlib': True, 'complevel': 4}``.
        These defaults are merged with any per-variable dicts specified in
        ``compression``.
    """  # noqa: E501
    if fillvalues is None:
        fillvalues = netCDF4.default_fillvals

    numpy_fillvals = {}
    for filltype, fillvalue in fillvalues.items():
        # drop string fill values
        if not filltype.startswith('S'):
            numpy_fillvals[numpy.dtype(filltype)] = fillvalue

    # If compression requested and engine unspecified, prefer 'h5netcdf'
    if compression is not None and engine is None:
        engine = 'h5netcdf'

    # If the chosen engine does not support compression, warn and ignore
    if compression is not None and engine == 'scipy':
        warnings.warn(
            'The "scipy" NetCDF engine does not support compression; ignoring '
            'compression directives.',
            UserWarning,
            stacklevel=2,
        )
        compression = None

    encoding_dict = _build_encoding_dict(
        ds,
        numpy_fillvals,
        has_fill_values,
        compression,
        compression_opts,
        engine,
    )

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

    if progress_bar:
        write_job = ds.to_netcdf(
            out_filename,
            encoding=encoding_dict,
            format=format,
            engine=engine,
            compute=False,
        )
        with ProgressBar():
            print(f'Writing to {out_filename}:')
            write_job.compute()
    else:
        ds.to_netcdf(
            out_filename,
            encoding=encoding_dict,
            format=format,
            engine=engine,
            compute=True,
        )

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


def ensure_cf_time_encoding(
    ds: xr.Dataset,
    time_source: xr.Dataset | None = None,
) -> None:
    """

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset whose encodings will be updated in-place.
    time_source : xarray.Dataset, optional
        If provided, use its time variables to replace those from ``ds``.

    Returns
    -------
    xarray.Dataset
        The same dataset instance with encodings updated.
    """
    copy_time = True
    if time_source is None:
        time_source = ds
        copy_time = False

    if 'time' not in time_source.variables:
        raise ValueError(
            'Dataset has no time variable; cannot ensure cftime time encoding.'
        )
    if 'time_bnds' not in time_source.variables:
        raise ValueError(
            'Dataset has time but no time bounds variable. Time bounds are '
            'required for all i7aof workflows with time.'
        )

    # Ensure values are cftime objects for predictable CF encoding
    if not _is_cftime_array(time_source['time']):
        raise ValueError(
            'i7aof workflows require that time be a cftime array.'
        )
    if not _is_cftime_array(time_source['time_bnds']):
        raise ValueError(
            'i7aof workflows require that time_bnds be a cftime array.'
        )

    if copy_time:
        ds['time'] = time_source['time']
        ds['time_bnds'] = time_source['time_bnds']

    cal = _extract_calendar(time_source)

    if cal is None:
        raise ValueError(
            "Cannot determine calendar for 'time' variable; "
            "please ensure 'calendar' is set in attributes or encoding."
        )

    # Remove attributes and instead set encoding to ensure that we get the
    # desired encoding.
    t = ds['time']
    if isinstance(getattr(t, 'attrs', None), dict):
        t.attrs.pop('units', None)
        t.attrs.pop('calendar', None)
    if isinstance(getattr(t, 'encoding', None), dict):
        t.encoding['units'] = TIME_UNITS
        t.encoding['calendar'] = cal
        t.encoding['dtype'] = 'float64'

    tb = ds['time_bnds']
    if isinstance(getattr(tb, 'attrs', None), dict):
        tb.attrs.pop('units', None)
        tb.attrs.pop('calendar', None)
    if isinstance(getattr(tb, 'encoding', None), dict):
        tb.encoding['units'] = TIME_UNITS
        tb.encoding['calendar'] = cal
        tb.encoding['dtype'] = 'float64'


def _extract_calendar(obj: xr.Dataset) -> str | None:
    """Extract calendar string from a dataset's time variable."""
    if 'time' not in obj.variables:
        return None
    t = obj['time']
    # Prefer encoding (where xarray stores decoded meta), then attrs
    if hasattr(t, 'encoding') and isinstance(t.encoding, dict):
        cal_enc = t.encoding.get('calendar')
        if isinstance(cal_enc, str) and cal_enc:
            return cal_enc
    tattrs = getattr(t, 'attrs', {}) or {}
    return tattrs.get('calendar') or tattrs.get('calendar_type')


def _apply_time_encoding(
    ds: xr.Dataset,
    encoding_dict: dict,
) -> None:
    """
    Ensure consistent time/time_bnds encodings and clear conflicting attrs.

    Target invariant for all workflows:

    - time has attrs: bounds="time_bnds", units=<default_time_units>,
      calendar=<calendar derived from input>
    - time_bnds has no attrs
    - neither variable has _FillValue in encoding or attrs
    """
    if 'time' not in ds.variables:
        return

    if 'time_bnds' not in ds.variables:
        raise ValueError(
            'Dataset has time but no time bounds variable. Time bounds are '
            'required for all i7aof workflows with time.'
        )
    t = ds['time']
    tb = ds['time_bnds']
    # Determine calendar: prefer encoding, then attrs, else default
    calendar = t.encoding.get('calendar') if hasattr(t, 'encoding') else None
    if calendar is None:
        calendar = t.attrs.get('calendar')
    if calendar is None:
        raise ValueError(
            "Cannot determine calendar for 'time' variable; "
            "please ensure 'calendar' is set in attributes or encoding."
        )

    # Keep time as decoded cftime and let xarray's CF encoder handle the
    # numeric conversion. We provide a single, consistent encoding for
    # units/calendar and ensure no conflicting attrs/encodings remain.

    # time: attrs (no CF encoding keys here; those are set via encoding)
    if isinstance(getattr(t, 'attrs', None), dict):
        t.attrs['bounds'] = 'time_bnds'
        t.attrs.pop('units', None)
        t.attrs.pop('calendar', None)
        t.attrs.pop('_FillValue', None)
    # time_bnds: clear attrs to follow CF conventions
    if isinstance(getattr(tb, 'attrs', None), dict):
        tb.attrs.clear()

    for var_name in ('time', 'time_bnds'):
        if var_name not in ds.variables:
            continue
        var = ds[var_name]

        # Remove any _FillValue from attrs or encoding
        if isinstance(getattr(var, 'encoding', None), dict):
            enc = var.encoding
            enc['units'] = TIME_UNITS
            enc['calendar'] = calendar
            enc['dtype'] = 'float64'
            enc.pop('_FillValue', None)

        # the encoding_dict overrides the encoding attribute so set that as
        # well
        enc = encoding_dict.get(var_name, {})
        enc['units'] = TIME_UNITS
        enc['calendar'] = calendar
        enc['dtype'] = 'float64'
        # Ensure no _FillValue is set
        enc['_FillValue'] = None
        encoding_dict[var_name] = enc


def _build_encoding_dict(
    dataset: xr.Dataset,
    numpy_fillvals: Dict[numpy.dtype, Any],
    has_fill_values: Optional[Union[bool, List[str]]],
    compression: Optional[Union[bool, List[str]]],
    compression_opts: Optional[Dict[str, Any]],
    engine: Optional[NetcdfEngine],
) -> Dict[str, Dict[str, Any]]:
    """Build encoding dict for variables, including _FillValue decisions."""
    enc: dict = {}
    var_names_local = list(dataset.data_vars.keys()) + list(
        dataset.coords.keys()
    )
    for vn in var_names_local:
        var = dataset[vn]
        enc_v = _var_encoding(
            vn,
            var,
            numpy_fillvals,
            has_fill_values,
            compression,
            compression_opts,
            engine,
        )
        if enc_v:
            enc[vn] = enc_v
    return enc


def _is_cftime_array(var: xr.DataArray) -> bool:
    """Return True if var is an object array of cftime objects."""
    dt = getattr(var, 'dtype', None)
    if dt is None or getattr(var, 'size', 0) == 0:
        return False
    try:
        if not numpy.issubdtype(dt, numpy.dtype('O')):
            return False
    except (TypeError, ValueError):
        return False
    try:
        v0 = var.values.flat[0]
    except (AttributeError, IndexError, TypeError, ValueError):
        return False
    # Prefer direct isinstance when available
    if isinstance(v0, cftime.datetime):
        return True
    # Fallback: detect cftime classes by module name
    mod = getattr(getattr(v0, '__class__', None), '__module__', '')
    return isinstance(mod, str) and mod.startswith('cftime')


def _decide_fill_value(
    var_name: str,
    var: xr.DataArray,
    numpy_fillvals: Dict[Any, Any],
    has_fill_values: Optional[Union[bool, List[str]]],
) -> Tuple[bool, Optional[Any]]:
    dtype = getattr(var, 'dtype', None)
    candidate = numpy_fillvals.get(dtype)

    # 1. Global modes: has_fill_values is bool
    if isinstance(has_fill_values, bool):
        if has_fill_values:  # global enable
            # Use candidate when available, otherwise explicitly suppress
            return True, candidate if candidate is not None else None
        # global disable: suppress for all
        return True, None

    # 2. List mode: per-variable control
    if isinstance(has_fill_values, list):
        in_list = var_name in has_fill_values
        if in_list:
            # Listed var must have a candidate
            if candidate is None:
                raise TypeError(
                    f"Variable '{var_name}' (dtype={dtype}) is listed in "
                    'has_fill_values but has no corresponding numpy_fillval.'
                )
            return True, candidate
        # Not listed: always suppress backend default
        return True, None

    # 3. Auto mode (has_fill_values is None)
    if candidate is None:
        return False, None

    present_in_enc = '_FillValue' in getattr(var, 'encoding', {})
    present_in_attrs = '_FillValue' in getattr(var, 'attrs', {})
    if present_in_enc or present_in_attrs:
        existing = var.encoding.get('_FillValue', var.attrs.get('_FillValue'))
        return True, existing

    try:
        has_nan = bool(var.isnull().any().compute())
    except (RuntimeError, ValueError):
        has_nan = False
    if has_nan:
        return True, candidate

    return False, None


def _decide_compression(
    var_name: str,
    var: xr.DataArray,
    compression: Optional[Union[bool, List[str]]],
    default_opts: Optional[Dict[str, Any]],
    engine: Optional[NetcdfEngine],
) -> Optional[Dict[str, Any]]:
    """
    Return a compression dict for a variable or None.

    Supported return values are a dict of encoding keys (e.g. {'zlib': True,
    'complevel': 4}) or None. The ``compression`` argument supports the same
    three forms as ``has_fill_values``: bool or list. If a boolean
    True is provided, ``default_opts`` are used. If a list contains the
    variable name, compression is applied.
    """
    if compression is None:
        return None

    # If engine cannot support compression, signal None
    if engine == 'scipy':
        return None

    # Determine default options (fall back to module default when None)
    if default_opts is None:
        default_opts = DEFAULT_COMPRESSION

    # Caller directive overrides default behavior
    if isinstance(compression, bool):
        return default_opts if compression else None

    if isinstance(compression, list):
        return default_opts if var_name in compression else None

    return None


def _var_encoding(
    var_name: str,
    var: xr.DataArray,
    numpy_fillvals: Dict[Any, Any],
    has_fill_values: Optional[Union[bool, List[str]]],
    compression: Optional[Union[bool, List[str]]],
    compression_opts: Optional[Dict[str, Any]],
    engine: Optional[NetcdfEngine],
) -> Dict[str, Any]:
    """
    Compute per-variable encoding for _FillValue and compression.

    Fill-value logic mirrors existing behavior. Compression decision follows
    a similar directive system and preserves explicit encoding keys when
    present on the variable.
    """
    encoding = {}

    # Consolidated fill-value decision
    set_fill, fill_val = _decide_fill_value(
        var_name, var, numpy_fillvals, has_fill_values
    )
    if set_fill:
        # remove any existing _FillValue to avoid bypassing our logic
        var.encoding.pop('_FillValue', None)
        encoding['_FillValue'] = fill_val

    # Decide compression options and merge in, preserving any explicit
    # encoding values present on the variable.
    comp_opts = _decide_compression(
        var_name,
        var,
        compression,
        compression_opts,
        engine,
    )
    if comp_opts:
        var_enc = getattr(var, 'encoding', {}) or {}
        for k, v in comp_opts.items():
            if k in var_enc:
                # Preserve explicit per-variable encoding keys
                continue
            encoding[k] = v

    return encoding
