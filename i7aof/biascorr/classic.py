"""
Bias correct the extrapolated CMIP data towards climatology.

This is the classical method with bias correction in geographic space.

Workflow
========

"""

import os
from typing import List, Tuple

import cftime
import gsw
import numpy as np
import xarray as xr
from mpas_tools.config import MpasConfigParser

from i7aof.cmip import get_model_prefix
from i7aof.grid.ismip import get_ismip_grid_filename, get_res_string
from i7aof.io import write_netcdf


def biascorr_cmip(
    model: str,
    scenario: str,
    clim_name: str,
    workdir: str | None = None,
    user_config_filename: str | None = None,
):
    """
    Bias correct CMIP ct/sa in two stages:

    1) extract the bias in ct and sa
    2) apply the bias correction in ct and sa

    Parameters
    ----------
    model: str
        Name of the CMIP model to bias correct
    scenario: str
        The name of the scenario ('historical', 'ssp585', etc.)
    clim_name: str
        The name of the reference climatology
    workdir : str, optional
        The base work directory within which the bias corrected files will be
        placed
    user_config_filename : str, optional
        The path to a file with user config options that override the
        defaults
    """

    # Read config
    (
        config,
        workdir,
        extrap_dir,
        outdir,
        ismip_res_str,
        model_prefix,
    ) = _load_config_and_paths(
        model=model,
        workdir=workdir,
        user_config_filename=user_config_filename,
        scenario=scenario,
        clim_name=clim_name,
    )

    # Collect files to bias correct
    ct_files, sa_files = _collect_extrap_outputs(extrap_dir, ismip_res_str)
    if not ct_files or not sa_files:
        raise FileNotFoundError(
            'No extrapolated files found. Run: ismip7-antarctic-extrap-cmip'
        )

    # Compute bias over historical period
    _compute_biases(
        config=config,
        workdir=workdir,
        model=model,
        scenario=scenario,
        ismip_res_str=ismip_res_str,
        extrap_dir=extrap_dir,
        clim_name=clim_name,
    )

    # Apply actual correction
    _apply_biascorrection(
        config=config,
        workdir=workdir,
        model=model,
        scenario=scenario,
        ismip_res_str=ismip_res_str,
        clim_name=clim_name,
        ct_files=ct_files,
        sa_files=sa_files,
        outdir=outdir,
    )

    # Compute thermal forcing
    _compute_thermal_forcing(
        config=config,
        workdir=workdir,
        model=model,
        scenario=scenario,
        ismip_res_str=ismip_res_str,
        clim_name=clim_name,
        ct_files=ct_files,
        sa_files=sa_files,
        outdir=outdir,
    )


# helper functions


def _load_config_and_paths(
    model,
    workdir,
    user_config_filename,
    scenario,
    clim_name,
):
    model_prefix = get_model_prefix(model)

    config = MpasConfigParser()
    config.add_from_package('i7aof', 'default.cfg')
    config.add_from_package('i7aof.cmip', f'{model_prefix}.cfg')
    config.add_from_package('i7aof.clim', f'{clim_name}.cfg')
    if user_config_filename is not None:
        config.add_user_config(user_config_filename)

    if workdir is None:
        if config.has_option('workdir', 'base_dir'):
            workdir = config.get('workdir', 'base_dir')
        else:
            raise ValueError(
                'Missing configuration option: [workdir] base_dir. '
                'Please supply a user config file that defines this option.'
            )
    assert workdir is not None, (
        'Internal error: workdir should be resolved to a string'
    )

    extrap_dir = os.path.join(
        workdir, 'extrap', model, scenario, 'Omon', 'ct_sa'
    )

    outdir = os.path.join(
        workdir, 'biascorr', model, scenario, clim_name, 'Omon', 'ct_sa'
    )

    os.makedirs(outdir, exist_ok=True)
    os.chdir(workdir)

    ismip_res_str = get_res_string(config, extrap=False)
    return config, workdir, extrap_dir, outdir, ismip_res_str, model_prefix


def _collect_extrap_outputs(
    extrap_dir: str, ismip_res_str: str
) -> Tuple[List[str], List[str]]:
    """Collect all extrapolated ct and sa files"""
    if not os.path.isdir(extrap_dir):
        return [], []
    ct_files: List[str] = []
    sa_files: List[str] = []
    allfiles = sorted(os.listdir(extrap_dir))
    for name in allfiles:
        if f'ismip{ismip_res_str}' in name and 'ct' in name:
            ct_name = name
            sa_name = ct_name.replace('ct', 'sa')
            if sa_name in allfiles:
                ct_files.append(os.path.join(extrap_dir, ct_name))
                sa_files.append(os.path.join(extrap_dir, sa_name))

    return ct_files, sa_files


def _compute_biases(
    config, workdir, model, ismip_res_str, extrap_dir, scenario, clim_name
):
    """Compute the bias if not already done"""

    biasdir = os.path.join(
        workdir, 'biascorr', model, 'intermediate', clim_name
    )
    os.makedirs(biasdir, exist_ok=True)

    modclimdir = os.path.join(workdir, 'biascorr', model, 'intermediate')

    climdir = os.path.join(workdir, 'extrap', 'climatology', clim_name)

    hist_dir = os.path.join(
        workdir, 'extrap', model, 'historical', 'Omon', 'ct_sa'
    )

    time_chunk = config.get('biascorr', 'time_chunk')

    for var in ['ct', 'sa']:
        # Get historical files
        hist_files: List[str] = []
        for name in sorted(os.listdir(hist_dir)):
            if f'ismip{ismip_res_str}' in name and var in name:
                hist_files.append(os.path.join(hist_dir, name))
        if not hist_files:
            raise FileNotFoundError(
                f'No historical extrapolated files available for {var}'
            )

        # Define filename for bias and skip if it's already present
        biasfile = os.path.join(biasdir, f'bias_{var}.nc')
        if os.path.exists(biasfile):
            print(f'Bias file already exists, skipping: {biasfile}')
            continue
        modclimfile = os.path.join(modclimdir, f'model_clim_{var}.nc')

        # Get climatology file for this variable
        climfile = os.path.join(
            climdir, f'OI_Climatology_ismip{ismip_res_str}_{var}_extrap.nc'
        )
        ds_clim = xr.open_dataset(climfile)

        # Get historical file(s)
        ds_hist = xr.open_mfdataset(hist_files, use_cftime=True)

        # Extract climatology period (only full annual for now)
        # TODO make dependent on clim
        ds_hist = ds_hist.sel(time=slice('1995-01-01', '2015-01-01'))
        # chunk just the variable because of issues chunking whole dataset
        da_hist = ds_hist[var].chunk({'time': time_chunk})

        # Compute time-average over climatology period
        dpm = ds_hist.time.dt.days_in_month
        weightedsum = (da_hist * dpm).sum(dim='time')
        modclim = weightedsum / dpm.sum()

        # Write out model climatology (preserve attrs) and overwrite
        # x/y/z (and bounds) from ISMIP grid
        ds_out = xr.Dataset()
        ds_grid = xr.open_dataset(
            get_ismip_grid_filename(config), decode_times=False
        )
        _assign_coord_with_bounds(ds_out, ds_grid, 'x')
        _assign_coord_with_bounds(ds_out, ds_grid, 'y')
        _assign_coord_with_bounds(ds_out, ds_grid, 'z')
        # data var with attrs
        ds_out[var] = modclim
        ds_out[var].attrs = ds_hist[var].attrs
        write_netcdf(
            ds_out,
            modclimfile,
            progress_bar=True,
            has_fill_values=lambda name, _v, v=var: name == v,
        )
        ds_out.close()
        ds_grid.close()

        # Compute bias in model climatology
        bias = modclim - ds_clim[var]

        # Write out bias (keep same attrs as variable) and coordinates
        ds_out = xr.Dataset()
        ds_grid = xr.open_dataset(
            get_ismip_grid_filename(config), decode_times=False
        )
        _assign_coord_with_bounds(ds_out, ds_grid, 'x')
        _assign_coord_with_bounds(ds_out, ds_grid, 'y')
        _assign_coord_with_bounds(ds_out, ds_grid, 'z')
        ds_out[var] = bias
        ds_out[var].attrs = ds_hist[var].attrs
        write_netcdf(
            ds_out,
            biasfile,
            progress_bar=True,
            has_fill_values=lambda name, _v, v=var: name == v,
        )
        ds_out.close()
        ds_grid.close()

        ds_clim.close()
        ds_hist.close()


def _apply_biascorrection(
    config,
    workdir,
    model,
    ismip_res_str,
    scenario,
    clim_name,
    ct_files,
    sa_files,
    outdir,
):
    """Apply bias correction to all in_files"""

    biasdir = os.path.join(
        workdir, 'biascorr', model, 'intermediate', clim_name
    )

    time_chunk = config.get('biascorr', 'time_chunk')

    for ct_file, sa_file in zip(ct_files, sa_files, strict=True):
        for var, file in zip(['ct', 'sa'], [ct_file, sa_file], strict=True):
            # Read biases
            biasfile = os.path.join(biasdir, f'bias_{var}.nc')
            ds_bias = xr.open_dataset(biasfile)

            # Read CMIP files
            ds_cmip = xr.open_dataset(file)
            da_cmip = ds_cmip[var].chunk({'time': time_chunk})

            # Define output filename
            outfile = os.path.join(outdir, os.path.basename(file))
            if os.path.exists(outfile):
                print(f'Corrected files already exist: {outfile}')
            else:
                # Output file doesn't exist yet, write out

                # Build dataset with ISMIP coordinates (and bounds) first
                ds_out = xr.Dataset()
                ds_grid = xr.open_dataset(
                    get_ismip_grid_filename(config), decode_times=False
                )
                _assign_coord_with_bounds(ds_out, ds_grid, 'x')
                _assign_coord_with_bounds(ds_out, ds_grid, 'y')
                _assign_coord_with_bounds(ds_out, ds_grid, 'z')
                # time coord comes from source; bounds will be added
                # after resample
                ds_out['time'] = ds_cmip['time']

                # Corrected variable and preserve attrs
                ds_out[var] = da_cmip - ds_bias[var]
                ds_out[var].attrs = ds_cmip[var].attrs

                # Convert to yearly output (keep CF time, add time_bnds)
                ds_out = ds_out.resample(time='1YE').mean()
                _assign_time_bounds_annual(ds_out, ds_cmip)
                # Re-apply variable attrs after resample (may be dropped)
                ds_out[var].attrs = ds_cmip[var].attrs

                write_netcdf(
                    ds_out,
                    outfile,
                    progress_bar=True,
                    has_fill_values=lambda name, _v, v=var: name == v,
                )
                ds_out.close()
                ds_grid.close()

            # Clean up
            ds_bias.close()
            ds_cmip.close()


def _compute_thermal_forcing(
    config,
    workdir,
    model,
    ismip_res_str,
    scenario,
    clim_name,
    ct_files,
    sa_files,
    outdir,
):
    """Apply bias correction to all in_files"""

    biasdir = os.path.join(
        workdir, 'biascorr', model, 'intermediate', clim_name
    )

    time_chunk = config.get('biascorr', 'time_chunk')

    # Read liquidus parameters
    lbd1 = config.getfloat('biascorr', 'lbd1')
    lbd2 = config.getfloat('biascorr', 'lbd2')
    lbd3 = config.getfloat('biascorr', 'lbd3')

    for ct_file, sa_file in zip(ct_files, sa_files, strict=False):
        # Read biases
        biasfile_ct = os.path.join(biasdir, 'bias_ct.nc')
        ds_bias_ct = xr.open_dataset(biasfile_ct)
        biasfile_sa = os.path.join(biasdir, 'bias_sa.nc')
        ds_bias_sa = xr.open_dataset(biasfile_sa)

        # Read cmip output
        ds_cmip_ct = xr.open_dataset(ct_file)
        da_cmip_ct = ds_cmip_ct['ct'].chunk({'time': time_chunk})
        ds_cmip_sa = xr.open_dataset(sa_file)
        da_cmip_sa = ds_cmip_sa['sa'].chunk({'time': time_chunk})

        # Compute corrected ct sa
        ct_corr = da_cmip_ct - ds_bias_ct['ct']
        sa_corr = da_cmip_sa - ds_bias_sa['sa']

        # Create 4D array of pressure
        pres = xr.ones_like(sa_corr)
        latitudes = ds_cmip_sa['lat'].values
        depths = pres.z.values
        pres_t = np.array([gsw.p_from_z(z, latitudes) for z in depths])
        pres[:, :, :, :] = pres_t[np.newaxis, :, :, :]

        # Compute freezing temperature
        ct_freeze = lbd1 * sa_corr + lbd2 + lbd3 * pres
        # ct_freeze = gsw.CT_freezing_poly(
        #    sa_corr, pres, saturation_fraction=1
        # )

        # Create dataset with thermal forcing; include ISMIP coords/bounds
        ds_tf = xr.Dataset()
        ds_grid = xr.open_dataset(
            get_ismip_grid_filename(config), decode_times=False
        )
        _assign_coord_with_bounds(ds_tf, ds_grid, 'x')
        _assign_coord_with_bounds(ds_tf, ds_grid, 'y')
        _assign_coord_with_bounds(ds_tf, ds_grid, 'z')
        ds_tf['time'] = ds_cmip_ct['time']
        ds_tf['tf'] = ct_corr - ct_freeze
        tf_attrs = {
            'units': 'degC',
            'long_name': 'Thermal Forcing',
            'comment': (
                'Computed as Conservative Temperature minus linearized '
                'freezing temperature using liquidus coefficients '
                '(Jourdain et al., 2017; doi:10.1002/2016JC012509).'
            ),
        }
        ds_tf['tf'].attrs = tf_attrs

        # Convert to yearly output; keep CF time and add time_bnds
        ds_tf = ds_tf.resample(time='1YE').mean()
        _assign_time_bounds_annual(ds_tf, ds_cmip_ct)
        # Re-apply variable attrs after resample (may be dropped)
        ds_tf['tf'].attrs = tf_attrs

        # Define output file
        file = ct_file.replace('ct', 'tf')
        outfile = os.path.join(outdir, os.path.basename(file))

        print(f'writing output: {outfile}')
        write_netcdf(
            ds_tf,
            outfile,
            progress_bar=True,
            has_fill_values=lambda name, _v: name == 'tf',
        )

        # Clean up
        ds_tf.close()
        ds_grid.close()
        ds_bias_ct.close()
        ds_bias_sa.close()
        ds_cmip_ct.close()
        ds_cmip_sa.close()


# -----------------------------------------------------------------------------
# Helpers for attributes and CF bounds
# -----------------------------------------------------------------------------


def _assign_coord_with_bounds(
    ds_out: xr.Dataset, ds_grid: xr.Dataset, coord: str
) -> None:
    """Assign a 1D coordinate and its bounds to ds_out from ISMIP grid.

    - Copies the coordinate variable and its attributes from the canonical
      ISMIP grid
    - Copies the corresponding *_bnds variable and sets coord 'bounds'
    """
    if coord not in ds_grid:
        return
    ds_out[coord] = ds_grid[coord]
    # Determine bounds variable name and copy over
    bname = ds_grid[coord].attrs.get('bounds', f'{coord}_bnds')
    if bname in ds_grid:
        ds_out[bname] = ds_grid[bname]
        ds_out[bname].attrs = ds_grid[bname].attrs.copy()
    ds_out[coord].attrs['bounds'] = bname


## removed: local bounds recomputation; always use ISMIP grid


def _assign_time_bounds_annual(
    ds_yearly: xr.Dataset, ds_src: xr.Dataset
) -> None:
    """Create CF-compliant annual time bounds and set attributes on time.

    ds_yearly should be the result of resample(time='1YE').mean().
    We keep the datetime-like 'time' coordinate and attach 'time_bnds'.
    If monthly time bounds exist in ds_src as 'time_bnds' (with two columns),
    we derive yearly bounds from min(start) and max(end) across each year.
    Otherwise, we fall back to calendar-based [YYYY-01-01, (YYYY+1)-01-01).
    """
    if 'time' not in ds_yearly:
        return
    # Try to determine calendar
    cal = (
        ds_src['time'].encoding.get('calendar')
        if 'time' in ds_src and hasattr(ds_src['time'], 'encoding')
        else None
    )
    if cal is None:
        cal = (
            ds_src['time'].attrs.get('calendar', 'standard')
            if 'time' in ds_src
            else 'standard'
        )

    years = xr.DataArray(ds_yearly['time'].dt.year, dims=('time',))

    # Prefer deriving from source monthly time bounds if available
    if 'time_bnds' in ds_src and 'time' in ds_src['time_bnds'].dims:
        tb = ds_src['time_bnds']
        # Normalize bounds dim name to 'bnds' in output
        bdim = tb.dims[-1]
        if bdim != 'bnds':
            tb = tb.rename({bdim: 'bnds'})
        starts = []
        ends = []
        for y in years.values:
            mask = ds_src['time'].dt.year == int(y)
            if mask.any():
                tby = tb.sel(time=mask)
                starts.append(tby.isel(bnds=0).min().item())
                ends.append(tby.isel(bnds=1).max().item())
            else:
                s, e = _year_bounds_cftime(int(y), cal)
                starts.append(s)
                ends.append(e)
        bnds = np.stack(
            [
                np.array(starts, dtype=object),
                np.array(ends, dtype=object),
            ],
            axis=1,
        )
        ds_yearly['time_bnds'] = (('time', 'bnds'), bnds)
    else:
        # Fallback: calendar-year bounds
        starts = []
        ends = []
        for y in years.values:
            s, e = _year_bounds_cftime(int(y), cal)
            starts.append(s)
            ends.append(e)
        bnds = np.stack(
            [
                np.array(starts, dtype=object),
                np.array(ends, dtype=object),
            ],
            axis=1,
        )
    ds_yearly['time_bnds'] = (('time', 'bnds'), bnds)

    # Set coord attributes
    ds_yearly['time'].attrs = ds_src['time'].attrs.copy()
    ds_yearly['time'].attrs['bounds'] = 'time_bnds'
    # provide minimal attributes on time_bnds
    ds_yearly['time_bnds'].attrs = {}
    units = ds_yearly['time'].attrs.get('units')
    if units is not None:
        ds_yearly['time_bnds'].attrs['units'] = units
    cal_out = (
        ds_yearly['time'].encoding.get('calendar')
        if hasattr(ds_yearly['time'], 'encoding')
        else ds_yearly['time'].attrs.get('calendar')
    )
    if cal_out is not None:
        ds_yearly['time_bnds'].attrs['calendar'] = cal_out


def _year_bounds_cftime(year: int, calendar: str):
    """Return (start, end) cftime objects for the given year and calendar."""
    cal = (calendar or 'standard').lower()
    if cal in ('standard', 'gregorian'):
        cls = cftime.DatetimeGregorian
    elif cal == 'proleptic_gregorian':
        cls = cftime.DatetimeProlepticGregorian
    elif cal in ('noleap', '365_day'):
        cls = cftime.DatetimeNoLeap
    elif cal in ('all_leap', '366_day'):
        cls = cftime.DatetimeAllLeap
    elif cal == '360_day':
        cls = cftime.Datetime360Day
    else:
        cls = cftime.DatetimeGregorian
    return cls(year, 1, 1), cls(year + 1, 1, 1)
