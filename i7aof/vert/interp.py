import xarray as xr

from i7aof.grid.ismip import get_ismip_grid_filename


class VerticalInterpolator:
    """
    A class to perform vertical interpolation to the ISMIP reference grid.

    Attributes
    ----------
    src_valid : xarray.DataArray
        A boolean array indicating valid source data points, same size as
        input data except for ``time`` dimension (if any)

    src_coord : str
        The source coordinate variable name in the dataset.

    dst_coord : {'z', 'z_extrap'}
        The destination coordinate variable from the ISMIP reference grid.

    config : mpas_tools.config.MpasConfigParser
        Configuration options, including threshold for normalization.

    threshold : float
        The threshold value for normalization after interpolation.

    src_frac_interp : xarray.DataArray
        The fraction of valid source data after interpolation.

    z_src : xarray.DataArray
        The source vertical coordinate from the dataset, corrected to be
        positive up if needed
    """

    def __init__(self, src_valid, src_coord, dst_coord, config):
        """
        Prepare vertical interpolation to the ISMIP reference grid.

        Parameters
        ----------
        src_valid : xarray.DataArray
            A boolean array indicating valid source data points, same size as
            input data except for ``time`` dimension (if any)

        src_coord : str
            The source coordinate variable name in the dataset.

        dst_coord : {'z', 'z_extrap'}
            The destination coordinate variable from the ISMIP reference grid.

        config :  mpas_tools.config.MpasConfigParser
            Configuration options
        """
        ds_ismip = xr.open_dataset(get_ismip_grid_filename(config))
        self.threshold = config.getfloat('vert_interp', 'threshold')
        self.dst_coord = dst_coord
        self.src_coord = src_coord
        self.config = config

        # Prepare source coordinate
        src_valid = src_valid.copy()
        self.z_src = src_valid[src_coord].copy()
        self.src_valid = src_valid

        # Prepare destination coordinate
        self.z_dst = ds_ismip[dst_coord]

        # Prepare src_frac and interpolate it
        src_frac = xr.where(src_valid, 1.0, 0.0)

        # Interpolate src_frac
        self.src_frac_interp = self._vert_interp(
            src_frac, src_coord, self.z_dst
        ).chunk()

    def mask_and_sort(self, da):
        """
        Mask the data array based on the source valid mask and sort so
        the source coordinate is in ascending order.

        Parameters
        ----------
        da : xarray.DataArray
            The data array to be masked.

        Returns
        -------
        da_masked : xarray.DataArray
            The masked data array.
        """
        da_masked = da.copy()
        src_coord = self.src_coord
        da_masked = da_masked.assign_coords(
            {src_coord: (src_coord, self.z_src.data)}
        )
        da_masked[src_coord].attrs = self.z_src.attrs
        da_masked = da_masked.where(self.src_valid, other=0.0)
        da_masked.attrs = da.attrs

        return da_masked

    def interp(self, da_masked):
        """
        Interpolate a data array to the ISMIP vertical coordinate.

        Parameters
        ----------
        da_masked : xarray.DataArray
            The masked data array to be interpolated.

        Returns
        -------
        da_interp : xarray.DataArray
            The data array after interpolation to the ISMIP vertical
            coordinate.
        """
        da_interp = self._vert_interp(da_masked, self.src_coord, self.z_dst)
        da_interp.attrs = da_masked.attrs
        da_interp = da_interp.drop_vars([self.src_coord])

        return da_interp

    def normalize(self, da_interp):
        """
        Normalize the data array following interpolation.  We remap a mask of
        where the data is valid, which becomes a fraction of how much of a
        destination cell overlapped with valid data.  For many "state" fields
        like temperature and salinity, we want to renormalize them by the valid
        fraction.  Otherwise, the values would be lower than expected by a
        factor of the valid fraction.  Locations that have less than a
        threshold fraction  of overlap (1e-3 by default) will be set to invalid
        values.

        Parameters
        ----------
        da_interp : xarray.DataArray
            The data array after interpolation to the ISMIP vertical
            coordinate.

        Returns
        -------
        da_normalized : xarray.DataArray
            The data array normalized by the valid source fraction
        """
        mask = self.src_frac_interp > self.threshold
        da_masked = da_interp.where(mask)
        denom_masked = self.src_frac_interp.where(mask)
        da_normalized = da_masked / denom_masked
        da_normalized.attrs = da_interp.attrs

        return da_normalized

    @staticmethod
    def _vert_interp(da, src_coord, z_dst):
        """
        Perform vertical interpolation on a DataArray.
        """
        da_interp = da.interp(
            coords={src_coord: z_dst},
            method='linear',
            kwargs={'fill_value': 'extrapolate'},
        )
        return da_interp


def fix_src_z_coord(ds, z_coord, z_bnds):
    """
    Invert the coordinate and convert to meters if necessary.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the vertical coordinates.

    z_coord : str
        The name of the vertical coordinate variable.

    z_bnds : str
        The name of the vertical coordinate bounds variable.

    Returns
    -------
    z_src : xarray.DataArray
        The source vertical coordinate from the dataset, corrected to be
        positive up if needed.

    z_bnds_src : xarray.DataArray
        The source vertical coordinate bounds from the dataset, corrected to be
        positive up if needed.
    """
    z_src = ds[z_coord].copy()
    z_bnds_src = ds[z_bnds].copy()
    attrs = z_src.attrs
    z_units = attrs.get('units', 'm').lower()
    if 'positive' in attrs:
        z_positive = attrs['positive'].lower()
    else:
        # we have to figure it out
        z_positive = 'down' if z_src.values[-1] > z_src.values[0] else 'up'

    bnds_attrs = z_bnds_src.attrs
    # typically, bnds have the same attributes as z itself but that isn't true
    # for the CESM2-WACCM units so we will only assume this as a fallback if
    # the attributes are missing
    bnds_units = bnds_attrs.get('units', z_units).lower()
    bnds_positive = bnds_attrs.get('positive', z_positive).lower()

    if z_units in ['cm', 'centimeters']:
        z_src = 1e-2 * z_src
    if bnds_units in ['cm', 'centimeters']:
        z_bnds_src = 1e-2 * z_bnds_src

    if z_positive == 'down':
        z_src = -z_src
    if bnds_positive == 'down':
        z_bnds_src = -z_bnds_src

    z_src.attrs = attrs
    z_src.attrs['units'] = 'm'
    z_src.attrs['positive'] = 'up'

    # bounds don't need attributes
    z_bnds_src.attrs = {}

    return z_src, z_bnds_src
