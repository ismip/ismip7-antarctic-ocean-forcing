import os
import subprocess
import warnings
from pathlib import Path

import cftime
import netCDF4
import numpy
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from xarray.coders import CFDatetimeCoder  # noqa: F401

# Default compression options when compression is requested as a boolean
# or via a callable returning True. These options are supported by the
# netCDF4/HDF5-based backends ('netcdf4' and 'h5netcdf').
DEFAULT_COMPRESSION = {
    'zlib': True,
    'complevel': 4,
    'shuffle': True,
}


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

    if 'time' in ds.variables and 'time_bnds' in ds.variables:
        t = ds['time']
        tb = ds['time_bnds']

        # Derive units/calendar from time's encoding or attrs
        t_units = None
        t_cal = None
        if hasattr(t, 'encoding') and isinstance(t.encoding, dict):
            t_units = t.encoding.get('units')
            t_cal = t.encoding.get('calendar')
        if t_units is None:
            t_units = t.attrs.get('units')
        if t_cal is None:
            t_cal = t.attrs.get('calendar')

        # Normalize: place in encoding for CF encoding, ensure attrs don't
        # carry conflicting keys that would trigger safe_setitem.
        if isinstance(getattr(tb, 'attrs', None), dict):
            tb.attrs.pop('units', None)
            tb.attrs.pop('calendar', None)
        if isinstance(getattr(tb, 'encoding', None), dict):
            if t_units is not None:
                tb.encoding['units'] = t_units
            if t_cal is not None:
                tb.encoding['calendar'] = t_cal

    # Deterministically decode/normalize time and time_bnds to cftime
    if 'time' in ds.variables:
        t = ds['time']
        cal = None
        if hasattr(t, 'encoding') and isinstance(t.encoding, dict):
            cal = t.encoding.get('calendar')
        if cal is None and hasattr(t, 'attrs') and isinstance(t.attrs, dict):
            cal = t.attrs.get('calendar')
        if cal is None:
            cal = 'proleptic_gregorian'
        _ensure_cftime_time(ds, cal)

    return ds


def write_netcdf(
    ds,
    filename,
    fillvalues=None,
    format=None,
    engine=None,
    progress_bar=False,
    has_fill_values=None,
    compression=None,
    compression_opts=None,
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

    compression : bool | dict | callable, optional
        Controls variable compression. Accepted forms mirror ``has_fill_values``
        semantics:

          - bool: enable/disable default compression for all variables

          - dict: mapping of var_name -> bool or var_name -> dict of explicit
                  encoding compression options (e.g.
                  ``{'zlib': True, 'complevel': 4}``)

          - callable: function (var_name, var: xarray.DataArray) -> bool or
                      dict

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


def _apply_time_encoding(  # noqa: C901
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
    was_num_time = numpy.issubdtype(time_var.dtype, numpy.number)
    # Determine calendar, prefer existing (encoding), then attrs, else default
    calendar = (
        time_var.encoding.get('calendar')
        if hasattr(time_var, 'encoding')
        else None
    )
    if calendar is None:
        calendar = time_var.attrs.get('calendar')
    if calendar is None:
        calendar = 'proleptic_gregorian'

    # CF encoding expects units/calendar declared for consistent serialization.
    time_units = default_time_units

    # Intentionally nested helpers: these are only used within
    # _apply_time_encoding; keeping them local reduces clutter and avoids
    # expanding the module surface area.
    def _units_in(var: xr.DataArray):
        u = None
        if isinstance(getattr(var, 'attrs', None), dict):
            u = var.attrs.get('units') or u
        if isinstance(getattr(var, 'encoding', None), dict) and u is None:
            u = var.encoding.get('units')
        return u

    def _validate_numeric_units(units_str: str, kind: str) -> None:
        u = (units_str or '').lower().strip()
        if not (u.startswith('days since') or u.startswith('seconds since')):
            raise ValueError(
                f"Unsupported numeric {kind} units: '{units_str}'. "
                "Supported prefixes: 'days since', 'seconds since'."
            )

    # Normalize dtype for time
    is_dt64 = numpy.issubdtype(time_var.dtype, numpy.datetime64)
    is_num = numpy.issubdtype(time_var.dtype, numpy.number)
    units_present = _units_in(time_var)
    if is_dt64:
        raise ValueError(
            'time coordinate is numpy.datetime64 at write time; this is not '
            'supported in this workflow. Ensure time is decoded to cftime '
            'earlier (use read_dataset) or provide numeric CF time with '
            "supported units ('days since' or 'seconds since')."
        )
    if is_num and isinstance(units_present, str) and units_present:
        # Validate units without converting to cftime
        _validate_numeric_units(units_present, 'time')
    # Clear conflicting attrs for cftime; preserve attrs for numeric

    def _apply_meta(var: xr.DataArray):
        # Clear attrs to avoid conflicts; set encoding appropriately
        if isinstance(getattr(var, 'attrs', None), dict):
            if _is_cftime_array(var):
                var.attrs.pop('units', None)
                var.attrs.pop('calendar', None)
            else:
                var.attrs['units'] = time_units
                var.attrs['calendar'] = calendar
        if isinstance(getattr(var, 'encoding', None), dict):
            if _is_cftime_array(var):
                var.encoding['units'] = time_units
                var.encoding['calendar'] = calendar
                var.encoding['dtype'] = 'float64'
            else:
                var.encoding.pop('units', None)
                var.encoding.pop('calendar', None)
                var.encoding['dtype'] = 'float64'

    _apply_meta(time_var)
    # Guard: numeric time must remain numeric within this function
    if was_num_time and _is_cftime_array(ds['time']):
        raise ValueError(
            'time was numeric entering _apply_time_encoding but became '
            'cftime (object) during encoding. Refusing to write to avoid '
            'microseconds units. This indicates an unintended conversion.'
        )

    # Ensure time_bnds (if present) uses identical units/calendar
    if 'time_bnds' in ds.variables:
        tb = ds['time_bnds']
        was_num_tb = numpy.issubdtype(tb.dtype, numpy.number)
        # Ensure dtype consistency for bounds; validate when numeric+units
        is_dt64_tb = numpy.issubdtype(tb.dtype, numpy.datetime64)
        is_num_tb = numpy.issubdtype(tb.dtype, numpy.number)
        units_tb = _units_in(tb)
        if is_dt64_tb:
            raise ValueError(
                'time_bnds is numpy.datetime64 at write time; this is not '
                'supported in this workflow. Ensure bounds are cftime '
                'earlier or provide numeric CF time with supported units.'
            )
        if is_num_tb and isinstance(units_tb, str) and units_tb:
            _validate_numeric_units(units_tb, 'time_bnds')
        _apply_meta(tb)
        # Guard: numeric time_bnds must remain numeric
        if was_num_tb and _is_cftime_array(ds['time_bnds']):
            raise ValueError(
                'time_bnds was numeric entering _apply_time_encoding but '
                'became cftime (object) during encoding. Refusing to write '
                'to avoid microseconds units. This indicates an unintended '
                'conversion.'
            )


def _build_encoding_dict(
    dataset: xr.Dataset,
    numpy_fillvals: dict,
    has_fill_values,
    compression,
    compression_opts,
    engine,
) -> dict:
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


def _cftime_class_for(calendar: str):
    cal = (calendar or '').lower()
    mapping = {
        'noleap': cftime.DatetimeNoLeap,
        '365_day': cftime.DatetimeNoLeap,
        'all_leap': cftime.DatetimeAllLeap,
        '366_day': cftime.DatetimeAllLeap,
        'gregorian': cftime.DatetimeGregorian,
        'proleptic_gregorian': cftime.DatetimeProlepticGregorian,
        'julian': cftime.DatetimeJulian,
        'standard': cftime.DatetimeProlepticGregorian,
    }
    return mapping.get(cal, cftime.DatetimeProlepticGregorian)


def _to_cftime_array(datetimes: numpy.ndarray, calendar: str) -> numpy.ndarray:
    """
    Convert numpy.datetime64 array to cftime objects for the given calendar.
    """
    cls = _cftime_class_for(calendar)
    # Ensure nanosecond resolution for consistent extraction
    dtns = datetimes.astype('datetime64[ns]')
    # Convert to Python-like components via ISO strings, then extract parts.
    out = numpy.empty(dtns.size, dtype=object)
    # Use ISO parsing with numpy for speed; fallback to python components
    # Convert to structured y-m-d h:m:s via string slicing
    iso = dtns.astype('datetime64[ns]').astype(str)
    for i, s in enumerate(iso):
        # s like 'YYYY-MM-DDThh:mm:ss.nnnnnnnnn'
        date_time = s.split('T')
        y, m, d = map(int, date_time[0].split('-'))
        if len(date_time) > 1:
            hms = date_time[1]
            h, mi, sec = hms.split(':')
            h = int(h)
            mi = int(mi)
            # sec may contain fractional seconds
            if '.' in sec:
                s_int, frac = sec.split('.')
                s_val = int(s_int)
                # cftime supports microseconds
                micro = int(round(int(frac[:6].ljust(6, '0'))))
            else:
                s_val = int(sec)
                micro = 0
        else:
            h = mi = s_val = 0
            micro = 0
        out[i] = cls(y, m, d, h, mi, s_val, microsecond=micro)
    return out.reshape(datetimes.shape)


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


def _num_to_cftime(
    values: numpy.ndarray, units: str, calendar: str
) -> numpy.ndarray:
    """
    Convert numeric time values with supported CF units into cftime objects.

    Supported units prefixes: 'days since', 'seconds since'.
    Raises ValueError for unsupported units (e.g., 'microseconds since').
    """
    units_l = units.lower().strip()
    if not (
        units_l.startswith('days since') or units_l.startswith('seconds since')
    ):
        raise ValueError(
            f"Unsupported time units for conversion to cftime: '{units}'. "
            "Supported prefixes: 'days since', 'seconds since'."
        )
    # cftime.num2date may raise ValueError for malformed units; let it surface.
    return numpy.array(
        cftime.num2date(values.astype(float), units, calendar=calendar),
        dtype=object,
    )


def _ensure_cftime_time(ds: xr.Dataset, calendar: str) -> None:
    """
    Ensure ds['time'] and ds['time_bnds'] (if present) use cftime objects.

    Converts from numpy.datetime64 or numeric arrays with supported
    CF units ('days since', 'seconds since'). If numeric units are
    not present or unsupported (e.g., 'microseconds since'), no
    implicit conversion is performed here; callers can decide to error
    or skip CF encoding based on the result.
    """
    if 'time' in ds.variables:
        t = ds['time']
        if not _is_cftime_array(t):
            if numpy.issubdtype(t.dtype, numpy.datetime64):
                ds['time'] = xr.DataArray(
                    _to_cftime_array(t.values, calendar), dims=t.dims
                )
            elif numpy.issubdtype(t.dtype, numpy.number):
                # numeric time with possible units in attrs/encoding
                units = None
                if hasattr(t, 'attrs') and isinstance(t.attrs, dict):
                    units = t.attrs.get('units')
                if (
                    units is None
                    and hasattr(t, 'encoding')
                    and isinstance(t.encoding, dict)
                ):
                    units = t.encoding.get('units')
                if isinstance(units, str) and units:
                    ds['time'] = xr.DataArray(
                        _num_to_cftime(t.values, units, calendar), dims=t.dims
                    )
    if 'time_bnds' in ds.variables:
        tb = ds['time_bnds']
        if not _is_cftime_array(tb):
            if numpy.issubdtype(tb.dtype, numpy.datetime64):
                ds['time_bnds'] = xr.DataArray(
                    _to_cftime_array(tb.values, calendar), dims=tb.dims
                )
            elif numpy.issubdtype(tb.dtype, numpy.number):
                units = None
                if hasattr(tb, 'attrs') and isinstance(tb.attrs, dict):
                    units = tb.attrs.get('units')
                if (
                    units is None
                    and hasattr(tb, 'encoding')
                    and isinstance(tb.encoding, dict)
                ):
                    units = tb.encoding.get('units')
                if isinstance(units, str) and units:
                    ds['time_bnds'] = xr.DataArray(
                        _num_to_cftime(tb.values, units, calendar),
                        dims=tb.dims,
                    )


def _decide_fill_value(var_name, var, numpy_fillvals, has_fill_values):
    """
    Return an appropriate _FillValue (or None) for a variable.

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


def _decide_compression(var_name, var, compression, default_opts, engine):
    """
    Return a compression dict for a variable or None.

    Supported return values are a dict of encoding keys (e.g. {'zlib': True,
    'complevel': 4}) or None. The ``compression`` argument supports the same
    three forms as ``has_fill_values``: bool, dict, or callable. If a boolean
    True is provided, ``default_opts`` are used. If a dict maps the variable
    name to either a bool or a dict, that mapping is respected. For callables,
    the callable may return a bool or a dict.
    """
    if compression is None:
        return None

    # If engine cannot support compression, signal None
    if engine == 'scipy':
        return None

    # Determine default options (fall back to module default when None)
    if default_opts is None:
        default_opts = DEFAULT_COMPRESSION
    default_opts = dict(default_opts or {})

    # Caller directive overrides default behavior
    if isinstance(compression, bool):
        return default_opts if compression else None

    if isinstance(compression, dict):
        choice = compression.get(var_name)
        if choice is None:
            return None
        if isinstance(choice, dict):
            # merge defaults under explicit values
            opts = dict(default_opts)
            opts.update(choice)
            return opts
        return default_opts if bool(choice) else None

    if callable(compression):
        try:
            res = compression(var_name, var)
        except (TypeError, ValueError):
            return None
        if res is None:
            return None
        if isinstance(res, bool):
            return default_opts if res else None
        if isinstance(res, dict):
            opts = dict(default_opts)
            opts.update(res)
            return opts
    return None


def _var_encoding(
    var_name,
    var,
    numpy_fillvals,
    has_fill_values,
    compression,
    compression_opts,
    engine,
):
    """
    Compute per-variable encoding for _FillValue and compression.

    Fill-value logic mirrors existing behavior. Compression decision follows
    a similar directive system and preserves explicit encoding keys when
    present on the variable.
    """
    encoding = {}

    # Determine explicit fill-value directive
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

    # Handle fill-value decisions without returning early so compression can
    # also be applied below.
    if directive is False:
        encoding['_FillValue'] = None
    elif directive is True:
        fill = numpy_fillvals.get(getattr(var, 'dtype', None))
        encoding['_FillValue'] = fill
    else:
        # Default behavior (no explicit directive): detect NaNs lazily and
        # preserve explicit enc/attrs when present.
        fill = _decide_fill_value(var_name, var, numpy_fillvals, None)
        present_in_enc = '_FillValue' in var.encoding
        present_in_attrs = '_FillValue' in var.attrs
        if fill is not None or present_in_enc or present_in_attrs:
            encoding['_FillValue'] = var.encoding.get(
                '_FillValue', var.attrs.get('_FillValue', fill)
            )

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
