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
        self.z_src = self._fix_src_coord(src_valid[src_coord])
        src_valid = src_valid.copy()
        src_valid = src_valid.assign_coords(
            {src_coord: (src_coord, self.z_src.data)}
        )
        src_valid[src_coord].attrs = self.z_src.attrs
        self.src_valid = src_valid

        # Prepare destination coordinate
        self.z_dst = ds_ismip[dst_coord]

        # Prepare src_frac and interpolate it
        src_frac = xr.where(src_valid, 1.0, 0.0)
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
        Normalize the data array following interpolation

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
    def _fix_src_coord(z_src):
        """
        Invert the coordinate and convert to meters if necessary.
        """
        attrs = z_src.attrs
        if attrs.get('units', 'm').lower() in ['cm', 'centimeters']:
            z_src = 1e-2 * z_src
        if attrs.get('positive', 'down').lower() == 'down':
            z_src = -z_src
        z_src.attrs = attrs
        z_src.attrs['units'] = 'm'
        z_src.attrs['positive'] = 'up'
        return z_src

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
