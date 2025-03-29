"""
Much of this code has been adapted from the polaris package, specifically
https://github.com/E3SM-Project/polaris/blob/main/polaris/remap/mapping_file_step.py
"""  # noqa: E501

import subprocess

import pyproj
import xarray as xr
from pyremap import (
    LatLon2DGridDescriptor,
    LatLonGridDescriptor,
    PointCollectionDescriptor,
    ProjectionGridDescriptor,
    Remapper,
    get_lat_lon_descriptor,
)


class Mapping:
    """
    A class for creating a mapping file between grids

    Attributes
    ----------
    ntasks : int
        the target number of MPI tasks to use

    config : MpasConfigParser
        Configuration options

    src_grid_info : dict
        Information about the source grid

    dst_grid_info : dict
        Information about the destination grid

    method : {'bilinear', 'neareststod', 'conserve'}
        The method of interpolation used

    map_filename : str or None
        The name of the output mapping file

    expand_distance : float or None
        The distance in meters over which to expand destination grid cells for
         smoothing

    expand_factor : float or None
        The factor by which to expand destination grid cells for smoothing

    src_mesh_filename : str
        The name of the SCRIP file for the source mesh

    dst_mesh_filename : str
        The name of the SCRIP file for the destination mesh
    """

    def __init__(
        self,
        config,
        ntasks,
        map_filename=None,
        method='bilinear',
        src_mesh_filename='src_mesh.nc',
        dst_mesh_filename='dst_mesh.nc',
    ):
        """
        Create a new mapping object

        Parameters
        ----------
        ntasks : int
            the target number of MPI tasks to use

        config : MpasConfigParser
            Configuration options

        map_filename : str, optional
            The name of the output mapping file,
            ``map_{source_type}_{dest_type}_{method}.nc`` by default

        method : {'bilinear', 'neareststod', 'conserve'}, optional
            The method of interpolation used

        src_mesh_filename : str, optional
            The name of the SCRIP file for the source mesh
            ``src_mesh.nc`` by default

        dst_mesh_filename : str, optional
            The name of the SCRIP file for the destination mesh
            ``dst_mesh.nc`` by default
        """
        self.ntasks = ntasks
        self.config = config
        self.src_grid_info = dict()
        self.dst_grid_info = dict()
        self.map_filename = map_filename
        self.method = method
        self.expand_distance = None
        self.expand_factor = None
        self.src_mesh_filename = src_mesh_filename
        self.dst_mesh_filename = dst_mesh_filename

    def src_from_lon_lat(
        self, filename, mesh_name=None, lon_var='lon', lat_var='lat'
    ):
        """
        Set the source grid from a file with a longitude-latitude grid.  The
        latitude and longitude variables can be 1D or 2D.

        Parameters
        ----------
        filename : str
            A file containing the latitude-longitude grid

        mesh_name : str, optional
            The name of the lon-lat grid (defaults to resolution and units,
            something like "0.5x0.5degree")

        lon_var : str, optional
            The name of the longitude coordinate in the file

        lat_var : str, optional
            The name of the latitude coordinate in the file
        """
        src = dict()
        src['type'] = 'lon-lat'
        src['filename'] = filename
        src['lon'] = lon_var
        src['lat'] = lat_var
        if mesh_name is not None:
            src['name'] = mesh_name
        self.src_grid_info = src

    def dst_from_lon_lat(
        self, filename, mesh_name=None, lon_var='lon', lat_var='lat'
    ):
        """
        Set the destination grid from a file with a longitude-latitude grid.
        The latitude and longitude variables can be 1D or 2D.

        Parameters
        ----------
        filename : str
            A file containing the latitude-longitude grid

        mesh_name : str, optional
            The name of the lon-lat grid (defaults to resolution and units,
            something like "0.5x0.5degree")

        lon_var : str, optional
            The name of the longitude coordinate in the file

        lat_var : str, optional
            The name of the latitude coordinate in the file
        """
        dst = dict()
        dst['type'] = 'lon-lat'
        dst['filename'] = filename
        dst['lon'] = lon_var
        dst['lat'] = lat_var
        if mesh_name is not None:
            dst['name'] = mesh_name
        self.dst_grid_info = dst

    def dst_global_lon_lat(self, dlon, dlat, lon_min=-180.0, mesh_name=None):
        """
        Set the destination grid from a file with a longitude-latitude grid.
        The latitude and longitude variables can be 1D or 2D.

        Parameters
        ----------
        dlon : float
            The longitude resolution in degrees

        dlat : float
            The latitude resolution in degrees

        lon_min : float, optional
            The longitude for the left-hand edge of the global grid in degrees

        mesh_name : str, optional
            The name of the lon-lat grid (defaults to resolution and units,
            something like "0.5x0.5degree")
        """

        dst = dict()
        dst['type'] = 'lon-lat'
        dst['dlon'] = dlon
        dst['dlat'] = dlat
        dst['lon_min'] = lon_min
        if mesh_name is not None:
            dst['name'] = mesh_name
        self.dst_grid_info = dst

    def src_from_proj(
        self,
        filename,
        mesh_name,
        x_var='x',
        y_var='y',
        proj_attr=None,
        proj_str=None,
    ):
        """
        Set the source grid from a file with a projection grid.

        Parameters
        ----------
        filename : str
            A file containing the projection grid

        mesh_name : str
            The name of the projection grid

        x_var : str, optional
            The name of the x coordinate in the file

        y_var : str, optional
            The name of the y coordinate in the file

        proj_attr : str, optional
            The name of a global attribute in the file containing the proj
            string for the projection

        proj_str : str, optional
            A proj string defining the projection, ignored if ``proj_attr``
            is provided
        """
        src = dict()
        src['type'] = 'proj'
        src['filename'] = filename
        src['name'] = mesh_name
        src['x'] = x_var
        src['y'] = y_var
        if proj_attr is not None:
            src['proj_attr'] = proj_attr
        elif proj_str is not None:
            src['proj_str'] = proj_str
        else:
            raise ValueError('Must provide one of "proj_attr" or "proj_str".')
        self.src_grid_info = src

    def dst_from_proj(
        self,
        filename,
        mesh_name,
        x_var='x',
        y_var='y',
        proj_attr=None,
        proj_str=None,
    ):
        """
        Set the destination grid from a file with a projection grid.

        Parameters
        ----------
        filename : str
            A file containing the projection grid

        mesh_name : str
            The name of the projection grid

        x_var : str, optional
            The name of the x coordinate in the file

        y_var : str, optional
            The name of the y coordinate in the file

        proj_attr : str, optional
            The name of a global attribute in the file containing the proj
            string for the projection

        proj_str : str, optional
            A proj string defining the projection, ignored if ``proj_attr``
            is provided
        """
        dst = dict()
        dst['type'] = 'proj'
        dst['filename'] = filename
        dst['name'] = mesh_name
        dst['x'] = x_var
        dst['y'] = y_var
        if proj_attr is not None:
            dst['proj_attr'] = proj_attr
        elif proj_str is not None:
            dst['proj_str'] = proj_str
        else:
            raise ValueError('Must provide one of "proj_attr" or "proj_str".')
        self.dst_grid_info = dst

    def dst_from_points(
        self, filename, mesh_name, lon_var='lon', lat_var='lat'
    ):
        """
        Set the destination grid from a file with a collection of points.

        Parameters
        ----------
        filename : str
            A file containing the latitude-longitude grid

        mesh_name : str
            The name of the point collection

        lon_var : str, optional
            The name of the longitude coordinate in the file

        lat_var : str, optional
            The name of the latitude coordinate in the file
        """
        dst = dict()
        dst['type'] = 'points'
        dst['filename'] = filename
        dst['name'] = mesh_name
        dst['lon'] = lon_var
        dst['lat'] = lat_var
        self.dst_grid_info = dst

    def get_remapper(self):
        """
        Get the remapper object.  After the mapping file has been created, the
        remappper can be used to remap data between the source and destination
        grids by calling its ``remap_file()`` or ``remap()`` methods.

        Returns
        -------
        remapper : pyremap.Remapper
            The remapper between the source and destination grids
        """
        src = self.src_grid_info
        dst = self.dst_grid_info

        if 'type' not in src:
            raise ValueError('None of the "src_from_*()" methods were called')

        if 'type' not in dst:
            raise ValueError('None of the "dst_from_*()" methods were called')

        in_descriptor = _get_descriptor(src)
        out_descriptor = _get_descriptor(dst)

        if self.map_filename is None:
            map_tool = self.config.get('remap', 'tool')
            prefixes = {'esmf': 'esmf', 'moab': 'mbtr'}
            suffixes = {
                'conserve': 'aave',
                'bilinear': 'bilin',
                'neareststod': 'neareststod',
            }
            suffix = f'{prefixes[map_tool]}{suffixes[self.method]}'

            self.map_filename = (
                f'map_{in_descriptor.meshName}_to_{out_descriptor.meshName}'
                f'_{suffix}.nc'
            )

        remapper = Remapper(in_descriptor, out_descriptor, self.map_filename)
        return remapper

    def build_map(self):
        """
        Make the mapping file

        Returns
        -------
        remapper : pyremap.Remapper
            The remapper between the source and destination grids
        """
        config = self.config
        remapper = self.get_remapper()
        map_tool = self.config.get('remap', 'tool')
        _check_remapper(remapper, self.method, map_tool=map_tool)

        src_descriptor = remapper.sourceDescriptor
        src_descriptor.to_scrip(self.src_mesh_filename)

        dst_descriptor = remapper.destinationDescriptor
        dst_descriptor.to_scrip(
            self.dst_mesh_filename,
            expandDist=self.expand_distance,
            expandFactor=self.expand_factor,
        )

        if map_tool == 'esmf':
            args = _esmf_build_map_args(
                remapper,
                self.method,
                src_descriptor,
                self.src_mesh_filename,
                dst_descriptor,
                self.dst_mesh_filename,
            )

            esmf_path = config.get('remap', 'esmf_path')
            if esmf_path.lower() == 'none':
                esmf_exe = 'ESMF_RegridWeightGen'
            else:
                esmf_exe = f'{esmf_path}/bin/ESMF_RegridWeightGen'
            args = [esmf_exe] + args

        elif map_tool == 'moab':
            moab_path = config.get('remap', 'moab_path')
            src_mesh_filename = self._moab_partition_scrip_file(
                self.src_mesh_filename, moab_path
            )
            dst_mesh_filename = self._moab_partition_scrip_file(
                self.dst_mesh_filename, moab_path
            )
            args = _moab_build_map_args(
                remapper, self.method, src_mesh_filename, dst_mesh_filename
            )

            if moab_path.lower() == 'none':
                moab_exe = 'mbtempest'
            else:
                moab_exe = f'{moab_path}/bin/mbtempest'
            args = [moab_exe] + args

        cores = config.getint('remap', 'cores')
        parallel_exec = config.get('remap', 'parallel_exec')

        args = [parallel_exec, '-n', str(cores)] + args

        _check_call(args)
        return remapper

    def _moab_partition_scrip_file(self, in_filename, moab_path):
        """
        Partition SCRIP file for parallel mbtempest use
        """
        ntasks = self.ntasks

        print(f'Partition SCRIP file {in_filename}')

        h5m_filename = in_filename.replace('.nc', '.h5m')
        h5m_part_filename = in_filename.replace('.nc', f'.p{ntasks}.h5m')

        if moab_path.lower() == 'none':
            mbconvert = 'mbconvert'
            mbpart = 'mbpart'
        else:
            mbconvert = f'{moab_path}/bin/mbconvert'
            mbpart = f'{moab_path}/bin/mbpart'

        # Convert source SCRIP to mbtempest
        args = [
            mbconvert,
            '-B',
            in_filename,
            h5m_filename,
        ]
        _check_call(args)

        # Partition source SCRIP
        args = [
            mbpart,
            f'{ntasks}',
            '-z',
            'RCB',
            h5m_filename,
            h5m_part_filename,
        ]
        _check_call(args)

        print('  Done.')

        return h5m_part_filename


def _check_call(args, log_command=True, **kwargs):
    """
    Call the given subprocess

    Parameters
    ----------
    args : list or str
        A list or string of argument to the subprocess.  If ``args`` is a
        string, you must pass ``shell=True`` as one of the ``kwargs``.

    log_command : bool, optional
        Whether to print the command that is running

    **kwargs : dict
        Keyword arguments to pass to subprocess.Popen

    Raises
    ------
    subprocess.CalledProcessError
        If the given subprocess exists with nonzero status

    """

    if isinstance(args, str):
        print_args = args
    else:
        print_args = ' '.join(args)

    if log_command:
        print(f'Running: {print_args}')

    subprocess.run(args, check=True, **kwargs)


def _check_remapper(remapper, method, map_tool):
    """
    Check for inconsistencies in the remapper
    """
    if map_tool not in ['moab', 'esmf']:
        raise ValueError(
            f'Unexpected map_tool {map_tool}. Valid '
            f'values are "esmf" or "moab".'
        )

    if isinstance(
        remapper.destinationDescriptor, PointCollectionDescriptor
    ) and method not in ['bilinear', 'neareststod']:
        raise ValueError(
            f'method {method} not supported for destination '
            f'grid of type PointCollectionDescriptor.'
        )

    if map_tool == 'moab' and method == 'neareststod':
        raise ValueError('method neareststod not supported by mbtempest.')


def _esmf_build_map_args(
    remapper,
    method,
    src_descriptor,
    src_mesh_filename,
    dst_descriptor,
    dst_mesh_filename,
):
    """
    Get command-line arguments for making a mapping file with
    ESMF_RegridWeightGen
    """

    args = [
        '--source',
        src_mesh_filename,
        '--destination',
        dst_mesh_filename,
        '--weight',
        remapper.mappingFileName,
        '--method',
        method,
        '--netcdf4',
    ]

    if src_descriptor.regional:
        args.append('--src_regional')

    if dst_descriptor.regional:
        args.append('--dst_regional')

    if src_descriptor.regional or dst_descriptor.regional:
        args.append('--ignore_unmapped')

    return [args]


def _moab_build_map_args(
    remapper, method, src_mesh_filename, dst_mesh_filename
):
    """
    Get command-line arguments for making a mapping file with mbtempest
    """
    fvmethod = {'conserve': 'none', 'bilinear': 'bilin'}

    map_filename = remapper.mappingFileName

    args = [
        '--type',
        '5',
        '--load',
        src_mesh_filename,
        '--load',
        dst_mesh_filename,
        '--file',
        map_filename,
        '--weights',
        '--gnomonic',
        '--boxeps',
        '1e-9',
        '--method',
        'fv',
        '--method',
        'fv',
        '--order',
        '1',
        '--order',
        '1',
        '--fvmethod',
        fvmethod[method],
    ]

    return [args]


def _get_descriptor(info):
    """
    Get a mesh descriptor from the mesh info
    """
    grid_type = info['type']
    if grid_type == 'lon-lat':
        descriptor = _get_lon_lat_descriptor(info)
    elif grid_type == 'proj':
        descriptor = _get_proj_descriptor(info)
    elif grid_type == 'points':
        descriptor = _get_points_descriptor(info)
    else:
        raise ValueError(f'Unexpected grid type {grid_type}')

    # for compatibility with mbtempest
    descriptor.format = 'NETCDF3_64BIT_DATA'
    return descriptor


def _get_lon_lat_descriptor(info):
    """
    Get a lon-lat descriptor from the given info
    """

    if 'dlat' in info and 'dlon' in info:
        lon_min = info['lon_min']
        lon_max = lon_min + 360.0
        descriptor = get_lat_lon_descriptor(
            dLon=info['dlon'],
            dLat=info['dlat'],
            lonMin=lon_min,
            lonMax=lon_max,
        )
    else:
        filename = info['filename']
        lon = info['lon']
        lat = info['lat']
        with xr.open_dataset(filename) as ds:
            lon_lat_1d = len(ds[lon].dims) == 1 and len(ds[lat].dims) == 1
            lon_lat_2d = len(ds[lon].dims) == 2 and len(ds[lat].dims) == 2
            if not lon_lat_1d and not lon_lat_2d:
                raise ValueError(
                    f'longitude and latitude coordinates {lon} '
                    f'and {lat} have unexpected sizes '
                    f'{len(ds[lon].dims)} and '
                    f'{len(ds[lat].dims)}.'
                )

        if lon_lat_1d:
            descriptor = LatLonGridDescriptor.read(
                fileName=filename, lonVarName=lon, latVarName=lat
            )
        else:
            descriptor = LatLon2DGridDescriptor.read(
                fileName=filename, lonVarName=lon, latVarName=lat
            )

    if 'name' in info:
        descriptor.meshName = info['name']

    return descriptor


def _get_proj_descriptor(info):
    """
    Get a ProjectionGridDescriptor from the given info
    """
    filename = info['filename']
    grid_name = info['name']
    x = info['x']
    y = info['y']
    if 'proj_attr' in info:
        with xr.open_dataset(filename) as ds:
            proj_str = ds.attrs[info['proj_attr']]
    else:
        proj_str = info['proj_str']

    proj = pyproj.Proj(proj_str)

    descriptor = ProjectionGridDescriptor.read(
        projection=proj,
        fileName=filename,
        meshName=grid_name,
        xVarName=x,
        yVarName=y,
    )

    return descriptor


def _get_points_descriptor(info):
    """
    Get a PointCollectionDescriptor from the given info
    """
    filename = info['filename']
    collection_name = info['name']
    lon_var = info['lon']
    lat_var = info['lat']
    with xr.open_dataset(filename) as ds:
        lon = ds[lon_var].value
        lat = ds[lat_var].values
        unit_attr = lon.attrs['units'].lower()
        if 'deg' in unit_attr:
            units = 'degrees'
        elif 'rad' in unit_attr:
            units = 'radians'
        else:
            raise ValueError(f'Unexpected longitude unit unit {unit_attr}')

    descriptor = PointCollectionDescriptor(
        lons=lon, lats=lat, collectionName=collection_name, units=units
    )

    return descriptor
