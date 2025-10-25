import os

import numpy as np
import xarray as xr

from i7aof.grid.ismip import get_ismip_grid_filename
from i7aof.io import read_dataset


class VerticalResampler:
    """
    Conservative vertical resampling from a source Z coordinate to a
    destination Z coordinate using layer-overlap weights.

    This is designed to resample from ``z_extrap`` to ``z`` but works for
    any pair of coordinates present in the ISMIP grid file.

    Notes
    -----
    The ISMIP grid path is obtained from
    :func:`i7aof.grid.ismip.get_ismip_grid_filename`. If this path is
    relative and not found from the current working directory, we will
    attempt to resolve it against ``[workdir] base_dir`` from ``config``.
    If still not found, a FileNotFoundError is raised with the attempted
    paths. Ensure the ISMIP grid exists (see
    :func:`i7aof.grid.ismip.write_ismip_grid`) and use a consistent
    working directory or provide ``[workdir] base_dir`` in the config.
    """

    def __init__(self, src_valid, src_coord, dst_coord, config):
        grid_rel = get_ismip_grid_filename(config)
        grid_path = grid_rel
        if not os.path.isabs(grid_path) and not os.path.exists(grid_path):
            if config.has_option('workdir', 'base_dir'):
                candidate = os.path.join(
                    config.get('workdir', 'base_dir'), grid_rel
                )
                if os.path.exists(candidate):
                    grid_path = candidate
        if not os.path.exists(grid_path):
            raise FileNotFoundError(
                'ISMIP grid file not found. Tried: '
                f"'{grid_rel}' and '{grid_path}'. "
                'Ensure the grid is generated and paths are correct.'
            )
        ds_ismip = read_dataset(grid_path)
        self.threshold = config.getfloat('vert_interp', 'threshold')
        self.src_coord = src_coord
        self.dst_coord = dst_coord
        self.config = config

        # capture source validity mask (may be all True) and coords
        self.src_valid = src_valid.copy()

        # Bounds from ISMIP reference grid
        src_bnds_name = f'{src_coord}_bnds'
        dst_bnds_name = f'{dst_coord}_bnds'
        if src_bnds_name not in ds_ismip or dst_bnds_name not in ds_ismip:
            raise KeyError(
                'Missing required bounds in ISMIP grid: '
                f'{src_bnds_name} or {dst_bnds_name}'
            )
        z_src_bnds = ds_ismip[src_bnds_name]
        z_dst_bnds = ds_ismip[dst_bnds_name]

        # Precompute weights and destination thickness
        weights, dz_dst = _compute_overlap_weights(
            z_src_bnds, z_dst_bnds, src_dim=src_coord, dst_dim=dst_coord
        )
        # Store as attributes
        self.weights = weights  # dims: (src_coord, dst_coord)
        self.dz_dst = dz_dst

    def resample(self, da: xr.DataArray) -> xr.DataArray:
        """
        Resample an intensive field conservatively from src_coord to dst_coord.

        Parameters
        ----------
        da : xarray.DataArray
            Data with a vertical dimension ``src_coord``. Other dims such as
            time/y/x are preserved.

        Returns
        -------
        xr.DataArray
            Resampled data with vertical dimension ``dst_coord``.
        """
        src = self.src_coord
        dst = self.dst_coord

        # Ensure the input carries the expected vertical dim
        if src not in da.dims:
            raise ValueError(
                'Input data is missing required vertical dim '
                f"'{src}'. Dims: {da.dims}"
            )

        # Build a validity mask aligned to da (broadcast src_valid if needed)
        valid = self.src_valid
        # If valid has no time dim but da does, xarray will broadcast
        valid = valid.astype(da.dtype)

        # Numerator and denominator via weighted sums over src vertical dim
        # weights dims: (src, dst). Use xr.dot over the shared 'src' dim.
        # Align weight dims for dot: ensure name matches
        weights = self.weights

        # Sum over source layers: (other_dims, dst)
        num = xr.dot(da * valid, weights, dims=src)
        denom = xr.dot(valid, weights, dims=src)

        # Coverage fraction relative to destination thickness
        frac = denom / self.dz_dst
        mask = frac > self.threshold

        out = (num / denom).where(mask)
        # Enforce preferred dim order: (time?, dst, y, x) where present
        preferred = []
        if 'time' in da.dims:
            preferred.append('time')
        preferred.extend([dst, 'y', 'x'])
        # Keep only dims that are actually present in the result
        order = [d for d in preferred if d in out.dims]
        # Append any remaining dims in their current order to avoid dropping
        order += [d for d in out.dims if d not in order]
        out = out.transpose(*order)
        out.attrs = da.attrs
        # Ensure destination coordinate present
        if dst in self.weights.coords:
            out = out.assign_coords({dst: self.weights[dst]})
        return out


def _ensure_bounds_ascending(z_bnds: xr.DataArray) -> xr.DataArray:
    """Return bounds with ascending order (lower, upper) along last axis."""
    # works for any sign; ensures lower <= upper
    lower = xr.apply_ufunc(
        np.minimum, z_bnds.isel(bnds=0), z_bnds.isel(bnds=1)
    )
    upper = xr.apply_ufunc(
        np.maximum, z_bnds.isel(bnds=0), z_bnds.isel(bnds=1)
    )
    return xr.concat([lower, upper], dim='bnds').transpose(..., 'bnds')


def _compute_overlap_weights(
    z_src_bnds: xr.DataArray,
    z_dst_bnds: xr.DataArray,
    src_dim: str,
    dst_dim: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute conservative overlap weights between source and destination layers.

    Returns
    -------
    weights : xr.DataArray
        Overlap thickness in meters with dims (src_dim, dst_dim).
    dz_dst : xr.DataArray
        Destination layer thickness with dim (dst_dim,).
    """
    # Ensure bounds are ascending along the bnds axis: (lower, upper)
    src_b = _ensure_bounds_ascending(z_src_bnds)
    dst_b = _ensure_bounds_ascending(z_dst_bnds)

    # Prepare arrays for broadcasting: (src, 1, bnds) and (1, dst, bnds)
    src_lower = src_b.isel(bnds=0)
    src_upper = src_b.isel(bnds=1)
    dst_lower = dst_b.isel(bnds=0)
    dst_upper = dst_b.isel(bnds=1)

    # Broadcast to (src, dst)
    # Bring explicit names
    src_lower = src_lower.rename({src_lower.dims[0]: src_dim})
    src_upper = src_upper.rename({src_upper.dims[0]: src_dim})
    dst_lower = dst_lower.rename({dst_lower.dims[0]: dst_dim})
    dst_upper = dst_upper.rename({dst_upper.dims[0]: dst_dim})

    # Expand/broadcast to (src, dst)
    src_lower_b = src_lower.expand_dims({dst_dim: dst_lower.sizes[dst_dim]})
    src_upper_b = src_upper.expand_dims({dst_dim: dst_upper.sizes[dst_dim]})
    dst_lower_b = dst_lower.expand_dims(
        {src_dim: src_lower.sizes[src_dim]}
    ).transpose(src_dim, dst_dim)
    dst_upper_b = dst_upper.expand_dims(
        {src_dim: src_upper.sizes[src_dim]}
    ).transpose(src_dim, dst_dim)

    lower_overlap = xr.apply_ufunc(
        np.maximum, src_lower_b, dst_lower_b, dask='allowed'
    )
    upper_overlap = xr.apply_ufunc(
        np.minimum, src_upper_b, dst_upper_b, dask='allowed'
    )
    overlap = upper_overlap - lower_overlap
    weights = overlap.where(overlap > 0.0, other=0.0)
    weights.name = 'vert_overlap_weight'

    dz_dst = (dst_b.isel(bnds=1) - dst_b.isel(bnds=0)).rename(
        {dst_b.dims[0]: dst_dim}
    )
    dz_dst.name = 'dz_dst'
    return weights, dz_dst
