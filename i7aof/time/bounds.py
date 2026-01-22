from typing import Optional

import xarray as xr


def capture_time_bounds(
    ds: xr.Dataset,
) -> Optional[tuple[str, xr.DataArray]]:
    """
    Capture time bounds from a dataset if they exist.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset from which to capture time bounds.

    Returns
    -------
    time_bounds: tuple[str, xr.DataArray], optional
        A tuple containing the name of the time bounds variable and the
        corresponding data array.
    """
    time_bounds: tuple[str, xr.DataArray] | None = None
    # Optionally capture time bounds (do not error if absent)
    if 'time' in ds:
        tcoord = ds['time']
        tbname = tcoord.attrs.get('bounds')
        if isinstance(tbname, str) and tbname in ds:
            time_bounds = (tbname, ds[tbname])
    return time_bounds


def inject_time_bounds(
    ds: xr.Dataset,
    time_bounds: tuple[str, xr.DataArray] | None,
) -> None:
    """
    Inject time bounds into a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset into which to inject time bounds.

    time_bounds: tuple[str, xr.DataArray], optional
        A tuple containing the name of the time bounds variable and the
        corresponding data array.
    """
    # Inject time bounds if available
    if time_bounds is not None and 'time' in ds:
        tbname, bda = time_bounds
        ds[tbname] = bda.load()
        ds['time'].attrs['bounds'] = tbname
