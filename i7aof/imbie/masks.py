import os

import numpy as np
import shapefile
import shapely.geometry
import shapely.ops
import skfmm
import xarray as xr
from inpoly import inpoly2
from tqdm import tqdm

from i7aof.grid.ismip import get_horiz_res_string, get_ismip_grid_filename
from i7aof.imbie.download import download_imbie
from i7aof.io import write_netcdf


def make_imbie_masks(config):
    """
    Generate IMBIE2 Antarctic basin masks on the ISMIP grid.

    This function downloads IMBIE2 basin shapefiles (if needed), rasterizes the
    polygons onto the ISMIP grid using a point-in-polygon test (via inpoly),
    and extends each basin mask into the ocean using a distance transform.
    The final basin map is saved as a NetCDF file.

    Parameters
    ----------
    config : mpas_tools.config.MpasConfigParser
        Configuration object providing resolution and file paths.
    """
    res = get_horiz_res_string(config)
    ismip_grid_filename = get_ismip_grid_filename(config)
    basin_file_name = 'imbie/ANT_Basins_IMBIE2_v1.6/ANT_Basins_IMBIE2_v1.6.shp'
    out_file_name = f'imbie/basinNumbers_{res}.nc'

    os.makedirs('imbie', exist_ok=True)
    download_imbie()

    if os.path.exists(out_file_name):
        print(f'{out_file_name} already exists. Skipping mask generation.')
        return

    basins = _get_basin_definitions()
    x, y, points, nx, ny = _load_ismip_grid(ismip_grid_filename)
    in_basin_data = _load_basin_shapes(basin_file_name)

    basin_number = _rasterize_basins(points, nx, ny, basins, in_basin_data)
    basin_number = _extend_basins_to_ocean(basin_number, len(basins))

    _write_basin_mask(x, y, basin_number, out_file_name)


def _get_basin_definitions():
    """Return a dictionary defining merged IMBIE basin groups."""
    return {
        'A-Ap': ['A-Ap'],
        'Ap-B': ['Ap-B'],
        'B-C': ['B-C'],
        'C-Cp': ['C-Cp'],
        'Cp-D': ['Cp-D'],
        'D-Dp': ['D-Dp'],
        'Dp-E': ['Dp-E'],
        'E-F': ['E-Ep', 'Ep-F'],
        'F-G': ['F-G'],
        'G-H': ['G-H'],
        'H-Hp': ['H-Hp'],
        'Hp-I': ['Hp-I'],
        'I-Ipp': ['I-Ipp'],
        'Ipp-J': ['Ipp-J'],
        'J-K': ['J-Jpp', 'Jpp-K'],
        'K-A': ['K-A'],
    }


def _load_ismip_grid(filename):
    """Load x, y coordinates and meshgrid points from ISMIP grid file."""
    ds_grid = xr.open_dataset(filename)
    x = ds_grid['x'].values
    y = ds_grid['y'].values
    nx, ny = len(x), len(y)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack((xx.ravel(), yy.ravel())).T
    return x, y, points, nx, ny


def _load_basin_shapes(shapefile_path):
    """Read and return basin geometries from a shapefile."""
    reader = shapefile.Reader(shapefile_path)
    fields = reader.fields[1:]
    field_names = [field[0] for field in fields]
    in_basin_data = {}
    for sr in reader.shapeRecords():
        atr = dict(zip(field_names, sr.record, strict=True))
        if atr['Subregion'] == '':
            continue
        geom = sr.shape.__geo_interface__
        in_basin_data[atr['Subregion']] = shapely.geometry.shape(geom)
    return in_basin_data


def _rasterize_basins(points, nx, ny, basins, in_basin_data):
    """Rasterize each basin polygon using point-in-polygon tests."""
    basin_number = -1 * np.ones((ny, nx), dtype=int)
    for index, (name, members) in enumerate(
        tqdm(basins.items(), desc='Processing basins')
    ):
        polygons = [in_basin_data[m] for m in members]
        combined = shapely.ops.unary_union(polygons)

        if combined.geom_type == 'Polygon':
            polys = [combined]
        elif combined.geom_type == 'MultiPolygon':
            polys = list(combined.geoms)
        else:
            print(f'Unsupported geometry type for basin {name}. Skipping.')
            continue

        mask = np.zeros(points.shape[0], dtype=bool)
        for poly in polys:
            coords = np.array(poly.exterior.coords)
            edges = np.column_stack(
                (np.arange(len(coords)), np.roll(np.arange(len(coords)), -1))
            )
            inside, _ = inpoly2(points, coords, edges)
            mask |= inside

        basin_number_flat = basin_number.ravel()
        basin_number_flat[mask] = index
        basin_number = basin_number_flat.reshape((ny, nx))

    return basin_number


def _extend_basins_to_ocean(basin_number, num_basins):
    """Extend basin masks into ocean areas using a distance transform."""
    print('Extending basins into the ocean...')
    min_distance = np.full_like(basin_number, np.inf, dtype=float)
    final_basin = -1 * np.ones_like(basin_number)

    for index in range(num_basins):
        mask = basin_number == index
        phi = np.where(mask, -1.0, 1.0)
        dist = skfmm.distance(phi)
        update_mask = dist < min_distance
        final_basin[update_mask] = index
        min_distance[update_mask] = dist[update_mask]

    return final_basin.astype(int)


def _write_basin_mask(x, y, basin_number, filename):
    """Write the basin number array to a NetCDF file."""
    ds_out = xr.Dataset(
        {'basinNumber': (('y', 'x'), basin_number)},
        coords={'x': ('x', x), 'y': ('y', y)},
    )
    write_netcdf(ds_out, filename)
    print(f'Extended IMBIE basin masks saved to {filename}.')
