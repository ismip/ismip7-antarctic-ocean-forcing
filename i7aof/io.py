import os
import subprocess
from pathlib import Path

import netCDF4
import numpy
from dask.diagnostics.progress import ProgressBar


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

    encoding_dict = {}
    var_names = list(ds.data_vars.keys()) + list(ds.coords.keys())
    for var_name in var_names:
        var = ds[var_name]
        encoding_dict[var_name] = {}
        fill = _decide_fill_value(
            var_name, var, numpy_fillvals, has_fill_values
        )
        present_in_enc = '_FillValue' in var.encoding
        present_in_attrs = '_FillValue' in var.attrs
        if fill is not None or present_in_enc or present_in_attrs:
            # Preserve explicit None to suppress backend auto-fill
            encoding_dict[var_name]['_FillValue'] = var.encoding.get(
                '_FillValue', var.attrs.get('_FillValue', fill)
            )

    if 'time' in ds.dims:
        # make sure the time dimension is unlimited
        ds.encoding['unlimited_dims'] = {'time'}
    else:
        # make sure there are no unlimited dimensions
        ds.encoding['unlimited_dims'] = set()

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
            except Exception:
                # Fall back to default behavior
                pass

    # Default: detect NaNs using xarray (works well for chunked data)
    try:
        has_nan = bool(var.isnull().any().compute())
    except Exception:
        has_nan = False
    return candidate if has_nan else None
