"""
Monthly-to-annual time averaging utilities and CLI.

Features
--------
- Accepts one or more monthly input files and produces annual-mean outputs.
- Output naming: if the input basename contains ``Omon_``, it is replaced
    with ``Oyr_``; otherwise, the suffix ``_ann`` is inserted before the
    extension.
- Safe writes: outputs are written to a temporary file in the same directory
    and atomically renamed to the final filename upon success, avoiding
    partial files and allowing interrupted workflows to resume.
- Averages all data variables that have a ``time`` dimension.
- Weights months by their number of days via ``time.dt.days_in_month``,
    respecting model calendars (gregorian, noleap, 360_day, etc.).
- Requires complete years (12 months); raises an error otherwise.
- Annual time is labeled at the start of the year, with CF-style
    ``time_bnds`` spanning the full year.

Examples
--------
- Python API:
        annual_average(["/path/to/file1.nc", "/path/to/file2.nc"])  # same dir
        annual_average(["/path/to/*.nc"], out_dir="/new/out/dir",
                                     overwrite=True)

- CLI:
        ismip7-antarctic-annual-average /path/to/*.nc --outdir /new/out \
                --overwrite

"""

from __future__ import annotations

import argparse
import glob
import os
import uuid
from typing import Iterable, Sequence

import cftime
import numpy as np
import xarray as xr

from i7aof.coords import strip_fill_on_non_data
from i7aof.io import read_dataset, write_netcdf

__all__ = [
    'annual_average',
    'main',
]


def annual_average(
    in_files: Sequence[str] | Iterable[str],
    out_dir: str | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> list[str]:
    """
    Compute weighted annual means from monthly inputs.

    For each input file, all data variables containing the ``time``
    dimension are averaged using month-length weights, yielding one value
    per year at the start-of-year time stamp. ``time_bnds`` is added with
    bounds from the start of the year to the start of the following year.

    Parameters
    ----------
    in_files : sequence of str
        Paths or glob patterns to monthly input files.
    out_dir : str, optional
        If provided, write outputs to this directory; otherwise, outputs
        are placed alongside their inputs.
    overwrite : bool, optional
        Overwrite output files if they already exist.

    Returns
    -------
    list of str
        The list of output file paths created (or that already existed
        when ``overwrite`` is False).
    """
    # Expand globs and preserve order while removing duplicates
    expanded: list[str] = _expand_files(in_files)
    if not expanded:
        raise FileNotFoundError(
            'No input files provided after glob expansion.'
        )

    outputs: list[str] = []
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    for in_path in expanded:
        out_path = _make_out_path(in_path, out_dir, suffix='_ann')
        outputs.append(out_path)
        if os.path.exists(out_path) and not overwrite:
            print(f'Output exists, skipping: {out_path}')
            continue
        _process_single_file_annual(
            in_path=in_path, out_path=out_path, progress=progress
        )

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Average monthly files to annual means with month-length weights. '
            "All data variables with a 'time' dimension are averaged."
        )
    )
    parser.add_argument(
        'files',
        nargs='+',
        help=(
            "Input files or glob patterns. Example: '/path/to/*.nc' "
            '(quotes recommended to avoid shell expansion issues).'
        ),
    )
    parser.add_argument(
        '-o',
        '--outdir',
        dest='outdir',
        default=None,
        help="Optional output directory (defaults to each input's directory).",
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing outputs.',
    )
    parser.add_argument(
        '--no-progress',
        dest='progress',
        action='store_false',
        help='Disable progress bars while writing NetCDF files.',
    )
    parser.set_defaults(progress=True)
    args = parser.parse_args()

    # Allow users to pass either explicit files or glob patterns
    in_files = _expand_files(args.files)
    if not in_files:
        raise SystemExit('No files found from provided arguments.')

    outputs = annual_average(
        in_files,
        out_dir=args.outdir,
        overwrite=args.overwrite,
        progress=args.progress,
    )
    for out_path in outputs:
        print(out_path)


# ---------------------------
# helpers (module-internal)
# ---------------------------


def _process_single_file_annual(
    *, in_path: str, out_path: str, progress: bool = True
) -> None:
    """Process a single monthly file into annual means (memory-aware)."""
    # Heuristic chunking to avoid loading entire dataset
    chunk_spec: dict[str, int] = {'time': 12}
    with read_dataset(in_path) as probe:
        if 'time' not in probe.sizes:
            raise ValueError(
                f"Dataset has no 'time' dimension (required): {in_path}"
            )
    # Only chunk along time; allow spatial chunking to follow storage layout
    ds = read_dataset(in_path, chunks=chunk_spec)
    try:
        months_per_year = ds['time'].dt.month.groupby('time.year').count()
        if int(months_per_year.min()) < 12 or int(months_per_year.max()) > 12:
            bad_years = months_per_year.where(months_per_year != 12, drop=True)
            years_str = (
                ', '.join(str(int(y)) for y in bad_years['year'].values)
                if bad_years['year'].size > 0
                else ''
            )
            raise ValueError(
                'Input contains non-12-month years. This workflow '
                'requires complete years. Offending years: '
                f'{years_str or "unknown"}'
            )
        w = ds['time'].dt.days_in_month.astype('float32')
        var_names = [
            name for name, da in ds.data_vars.items() if 'time' in da.dims
        ]
        if not var_names:
            raise ValueError(
                "No data variables with a 'time' dimension found in: "
                f'{in_path}'
            )
        den = w.groupby('time.year').sum(dim='time')
        data_vars_out = {}
        for name in var_names:
            da = ds[name]
            if da.dtype == 'float64':
                da = da.astype('float32')
            num = (da * w).groupby('time.year').sum(dim='time')
            ann = (num / den).rename({'year': 'time'})
            ann.attrs = dict(ds[name].attrs)
            data_vars_out[name] = ann
        ds_out = xr.Dataset(data_vars=data_vars_out)
        years = ds['time'].dt.year.groupby('time.year').first().values
        calendar = _get_calendar(ds)
        times, tbnds = _build_time_and_bounds(years, calendar)
        ds_out['time'] = ('time', times)
        ds_out['time_bnds'] = (('time', 'bnds'), tbnds)
        ds_out['time'].attrs['bounds'] = 'time_bnds'
        for cname, cda in ds.coords.items():
            if cname != 'time':
                ds_out = ds_out.assign_coords({cname: cda})
        for vname, vda in ds.data_vars.items():
            if vname not in data_vars_out and 'time' not in vda.dims:
                ds_out[vname] = vda
        ds_out.attrs = dict(ds.attrs)
        note = (
            'Annual mean computed with month-length weights; time at '
            'start-of-year; time_bnds spans full year.'
        )
        hist = ds_out.attrs.get('history')
        ds_out.attrs['history'] = f'{hist}\n{note}' if hist else note
        if calendar is not None:
            ds_out['time'].encoding['calendar'] = calendar
            ds_out['time_bnds'].encoding['calendar'] = calendar
        # Consistent with other workflows: no fill values on coords/bounds
        ds_out = strip_fill_on_non_data(ds_out, data_vars=var_names)
        # Build explicit fill-value policy: only 'ct', 'sa', and 'tf'
        # should carry _FillValue; all others (including coords/bounds)
        # should not. This avoids backend defaults and scanning.
        fill_and_compress = ['ct', 'sa', 'tf']

        # Write to a temporary file in the same directory, then atomically
        # replace the final file. This avoids read/write conflicts and leaves
        # no partial files if interrupted.
        tmp_path = f'{out_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}'
        # Clean up a stale temp file path if it already exists
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            write_netcdf(
                ds_out,
                tmp_path,
                progress_bar=progress,
                has_fill_values=fill_and_compress,
                compression=fill_and_compress,
            )
            # After successful write, replace/overwrite atomically
            os.replace(tmp_path, out_path)
        finally:
            # Ensure temp is removed if something went wrong
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    finally:
        ds.close()


def _expand_files(files: Sequence[str] | Iterable[str]) -> list[str]:
    """Expand glob patterns and deduplicate while preserving order."""
    seen: set[str] = set()
    expanded: list[str] = []
    for item in files:
        matches = glob.glob(item)
        if not matches:
            # treat as literal path if no matches
            matches = [item]
        for m in matches:
            if m not in seen:
                seen.add(m)
                expanded.append(m)
    return expanded


def _make_out_path(in_path: str, out_dir: str | None, suffix: str) -> str:
    """Determine annual-mean output path.

    If the basename contains 'Omon_', replace it with 'Oyr_'. Otherwise,
    insert the provided suffix before the last extension.
    """
    base = os.path.basename(in_path)
    if 'Omon_' in base:
        out_name = base.replace('Omon_', 'Oyr_')
    else:
        root, ext = os.path.splitext(base)
        if ext:
            out_name = f'{root}{suffix}{ext}'
        else:
            out_name = f'{base}{suffix}'
    return os.path.join(out_dir or os.path.dirname(in_path), out_name)


def _get_calendar(ds: xr.Dataset) -> str | None:
    """Best-effort extraction of the calendar string from the dataset."""
    # encoding is a dict and .get is safe; 'time' exists earlier in flow
    cal = ds['time'].encoding.get('calendar')
    if cal is None:
        cal = ds['time'].attrs.get('calendar')
    return cal


def _build_time_and_bounds(years: np.ndarray, calendar: str | None):
    """
    Create start-of-year time coordinates and [start, next-start] bounds.

    Parameters
    ----------
    years : array-like of int
        The years present in the averaged dataset.
    calendar : str or None
        CF calendar name. If None or unrecognized, falls back to
        proleptic gregorian via numpy datetime64.

    Returns
    -------
    times : np.ndarray
        Array of datetime-like objects (numpy or cftime) for 'time'.
    time_bnds : np.ndarray
        2D array of shape (ntime, 2) with bounds per time.
    """
    # Determine which datetime class to use (require cftime)
    cal_map: dict[str, type] = {
        'noleap': cftime.DatetimeNoLeap,
        '365_day': cftime.DatetimeNoLeap,
        '360_day': cftime.Datetime360Day,
        'gregorian': cftime.DatetimeGregorian,
        'standard': cftime.DatetimeGregorian,
        'proleptic_gregorian': cftime.DatetimeProlepticGregorian,
    }
    cal_key = calendar or 'proleptic_gregorian'
    dt_class = (
        cal_map[cal_key]
        if cal_key in cal_map
        else (cftime.DatetimeProlepticGregorian)
    )

    years = np.asarray(years, dtype=int)
    n = years.size
    times = np.array([dt_class(int(y), 1, 1) for y in years])
    nexts = np.array([dt_class(int(y) + 1, 1, 1) for y in years])

    tbnds = np.empty((n, 2), dtype=object)
    tbnds[:, 0] = times
    tbnds[:, 1] = nexts
    return times, tbnds
