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
        dtype = var.dtype

        # add fill values
        if dtype in numpy_fillvals:
            if numpy.any(numpy.isnan(var)):
                # only add fill values if they're needed
                fill = numpy_fillvals[dtype]
            else:
                fill = None
            encoding_dict[var_name]['_FillValue'] = fill

    if 'time' in ds.dims:
        # make sure the time dimension is unlimited
        ds.encoding['unlimited_dims'] = {'time'}

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
