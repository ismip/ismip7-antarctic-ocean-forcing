import configparser
import logging
import os
from collections import deque
from dataclasses import dataclass

import numpy as np
import skfmm
import xarray as xr
from scipy import ndimage

from i7aof.io import read_dataset, write_netcdf
from i7aof.topo import get_topo


@dataclass(frozen=True)
class ShelfParams:
    """Parameters controlling shelf-break constrained basin extension."""

    shelf_isobath_depth_m: float = 1500.0
    frac_threshold: float = 1.0e-3
    seed_dilation_iters: int = 10


def extend_imbie_basins(
    *,
    config,
    basin_number: np.ndarray,
    num_basins: int,
    workdir: str = '.',
) -> np.ndarray:
    """Extend IMBIE basins to the ocean using a shelf-break constraint.

    This is a two-stage extension:

    1. Within the continental shelf (defined by an isobath), assign basin
       labels by projecting from the shelf-break contour (effectively normal
       to the contour).
    2. Beyond the shelf, extend basins by nearest distance to the basin seeds
       (current behavior).

    The shelf is defined on the configured topography dataset on the ISMIP
    grid. Topography is built if missing.

    Parameters
    ----------
    config : mpas_tools.config.MpasConfigParser
        Configuration options.

    basin_number : numpy.ndarray
        A 2D array of basin labels from rasterization, with ``-1`` where
        undefined.

    num_basins : int
        The number of basins.

    workdir : str, optional
        Base working directory containing ``topo/`` (and where topography will
        be built if missing).

    Returns
    -------
    numpy.ndarray
        A 2D array of basin labels extended to the full domain.
    """

    params = _get_shelf_params(config)

    topo_file = _ensure_topography_on_ismip(config=config, workdir=workdir)
    ds_topo = read_dataset(topo_file)

    bed = ds_topo['bed'].values
    ocean_frac = ds_topo['ocean_frac'].values

    x = ds_topo['x'].values
    y = ds_topo['y'].values

    dx = float(config.getfloat('ismip_grid', 'dx'))
    dy = float(config.getfloat('ismip_grid', 'dy'))

    try:
        debug = config.getboolean('imbie', 'debug')
    except configparser.NoOptionError:
        debug = False

    basin_with_shelf = extend_imbie_basins_with_shelf_break(
        basin_number=basin_number,
        bed=bed,
        ocean_frac=ocean_frac,
        shelf_isobath_depth_m=params.shelf_isobath_depth_m,
        dx=dx,
        dy=dy,
        num_basins=num_basins,
        frac_threshold=params.frac_threshold,
        seed_dilation_iters=params.seed_dilation_iters,
        debug=debug,
        debug_dir=os.path.join(workdir, 'imbie2'),
        x=x,
        y=y,
    )

    if debug:
        debug_dir = os.path.join(workdir, 'imbie2')
        shelf_fill_mask = np.logical_and(
            basin_number < 0, basin_with_shelf >= 0
        )
        _write_debug_snapshot(
            debug_dir=debug_dir,
            filename='debug_13_basin_with_shelf.nc',
            fields={
                'basin_number_in': basin_number,
                'basin_with_shelf': basin_with_shelf,
                'shelf_fill_mask': shelf_fill_mask,
            },
            x=x,
            y=y,
        )

    # Now extend beyond the shelf (and fill any remaining holes) with the
    # original nearest-distance approach.
    basin_final = extend_basins_to_ocean_nearest(
        basin_number=basin_with_shelf,
        num_basins=num_basins,
        dx=dx,
        dy=dy,
    )

    if debug:
        debug_dir = os.path.join(workdir, 'imbie2')
        deep_fill_mask = basin_with_shelf < 0
        _write_debug_snapshot(
            debug_dir=debug_dir,
            filename='debug_14_basin_final.nc',
            fields={
                'basin_with_shelf': basin_with_shelf,
                'basin_final': basin_final,
                'deep_fill_mask': deep_fill_mask,
            },
            x=x,
            y=y,
        )

    return basin_final


def extend_imbie_basins_with_shelf_break(
    *,
    basin_number: np.ndarray,
    bed: np.ndarray,
    ocean_frac: np.ndarray,
    shelf_isobath_depth_m: float,
    dx: float,
    dy: float,
    num_basins: int,
    frac_threshold: float = 1.0e-3,
    seed_dilation_iters: int = 10,
    debug: bool = False,
    debug_dir: str = 'imbie2',
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> np.ndarray:
    """Fill basin labels within the continental shelf using shelf-break rules.

    This function does not do the final deep-ocean fill; call
    :func:`extend_basins_to_ocean_nearest` after this step.

    Notes
    -----
    - ``ocean_frac`` is used directly, which includes floating-ice cavities.
      This ensures shelf regions exist and can be seeded even if the shelf
      break passes under ice shelves.
    """

    if debug:
        _write_debug_snapshot(
            debug_dir=debug_dir,
            filename='debug_00_inputs.nc',
            fields={
                'basin_number_in': basin_number,
                'bed': bed,
                'ocean_frac': ocean_frac,
            },
            x=x,
            y=y,
        )

    if (
        basin_number.shape != bed.shape
        or basin_number.shape != ocean_frac.shape
    ):
        raise ValueError(
            'basin_number, bed, and ocean_frac must have same shape'
        )

    ocean_mask = ocean_frac > frac_threshold
    depth = np.maximum(0.0, -bed)

    shelf_candidate = ocean_mask & (depth <= shelf_isobath_depth_m)

    if debug:
        _write_debug_snapshot(
            debug_dir=debug_dir,
            filename='debug_01_masks.nc',
            fields={
                'depth': depth,
                'ocean_mask': ocean_mask,
                'shelf_candidate': shelf_candidate,
            },
            x=x,
            y=y,
        )

    shelf_ocean = _select_shelf_region_seeded_by_basins(
        shelf_candidate=shelf_candidate,
        basin_number=basin_number,
        seed_dilation_iters=seed_dilation_iters,
    )

    if debug:
        _write_debug_snapshot(
            debug_dir=debug_dir,
            filename='debug_02_shelf_ocean.nc',
            fields={
                'shelf_ocean': shelf_ocean,
            },
            x=x,
            y=y,
        )

    if not np.any(shelf_ocean):
        # Fallback: if the shelf definition is somehow empty, just return
        # input.
        return basin_number.astype(int, copy=True)

    shelf_break = _extract_shelf_break(
        shelf_ocean=shelf_ocean, ocean_mask=ocean_mask
    )

    if debug:
        _write_debug_snapshot(
            debug_dir=debug_dir,
            filename='debug_03_shelf_break_raw.nc',
            fields={
                'shelf_break_raw': shelf_break,
            },
            x=x,
            y=y,
        )

    # Enforce a single continuous contour by keeping the largest connected
    # component. This avoids small-island shelves contributing boundaries.
    shelf_break = _largest_connected_component(shelf_break)

    if debug:
        _write_debug_snapshot(
            debug_dir=debug_dir,
            filename='debug_04_shelf_break_largest.nc',
            fields={
                'shelf_break_largest': shelf_break,
            },
            x=x,
            y=y,
        )

    if not np.any(shelf_break):
        # If there is no shelf break, we cannot do the contour-normal fill.
        return basin_number.astype(int, copy=True)

    (
        break_label,
        boundary_mask,
        connectors_mask,
    ) = _label_shelf_break_by_basin(
        shelf_break=shelf_break,
        shelf_ocean=shelf_ocean,
        basin_number=basin_number,
        num_basins=num_basins,
        dx=dx,
        dy=dy,
        seed_dilation_iters=seed_dilation_iters,
        debug=debug,
        debug_dir=debug_dir,
        x=x,
        y=y,
    )

    shelf_label_from_break = _extrude_labels_from_break(
        shelf_break=shelf_break,
        shelf_ocean=shelf_ocean,
        basin_number=basin_number,
        break_label=break_label,
    )

    if debug:
        # Combined seed dataset (for verification): original valid basins,
        # and the filled shelf-break. Connectors are walls (no labels).
        seed_dataset = -1 * np.ones_like(basin_number, dtype=int)
        seed_dataset[basin_number >= 0] = basin_number[basin_number >= 0]
        break_on_shelf = np.logical_and(shelf_break, break_label >= 0)
        seed_dataset[break_on_shelf] = break_label[break_on_shelf]
        _write_debug_snapshot(
            debug_dir=debug_dir,
            filename='debug_09_seed_dataset.nc',
            fields={
                'seed_dataset': seed_dataset,
                'shelf_ocean': shelf_ocean,
                'shelf_break': shelf_break,
                'connectors_mask': connectors_mask,
            },
            x=x,
            y=y,
        )

    # Boundary-aware flood fill to get combined shelf label
    shelf_label = _fill_shelf_with_boundaries(
        shelf_ocean=shelf_ocean,
        basin_number=basin_number,
        break_label=break_label,
        boundary_mask=boundary_mask,
        debug=debug,
        debug_dir=debug_dir,
        x=x,
        y=y,
    )

    if debug:
        _write_debug_snapshot(
            debug_dir=debug_dir,
            filename='debug_11_shelf_label.nc',
            fields={
                'shelf_label': shelf_label,
                'shelf_label_from_break_only': shelf_label_from_break,
            },
            x=x,
            y=y,
        )

    basin_out = basin_number.astype(int, copy=True)
    # Never overwrite any valid basin labels from the input; only extend into
    # previously-unlabeled ocean cells.
    fill_mask = (basin_number < 0) & shelf_ocean & (shelf_label >= 0)
    basin_out[fill_mask] = shelf_label[fill_mask]

    if debug:
        _write_debug_snapshot(
            debug_dir=debug_dir,
            filename='debug_12_basin_out_shelf.nc',
            fields={
                'basin_out_shelf': basin_out,
                'fill_mask': fill_mask,
            },
            x=x,
            y=y,
        )
    return basin_out


def extend_basins_to_ocean_nearest(
    *,
    basin_number: np.ndarray,
    num_basins: int,
    dx: float = 1.0,
    dy: float = 1.0,
) -> np.ndarray:
    """Extend basin masks everywhere by nearest-distance to a defined basin.

    This matches the existing behavior in ``i7aof.imbie.masks`` but allows grid
    spacing to be specified.
    """

    # Only fill previously-unlabeled cells; never overwrite existing labels.
    fill_mask = basin_number < 0
    final_basin = basin_number.astype(int, copy=True)

    min_distance = np.full_like(basin_number, np.inf, dtype=float)
    # Only track distances where we intend to fill
    min_distance[~fill_mask] = -np.inf

    # Axis order is (y, x) for (dy, dx)
    spacing = [dy, dx]

    for index in range(num_basins):
        mask = basin_number == index
        phi = np.where(mask, -1.0, 1.0)
        dist = skfmm.distance(phi, dx=spacing)
        update_mask = fill_mask & (dist < min_distance)
        final_basin[update_mask] = index
        min_distance[update_mask] = dist[update_mask]

    return final_basin.astype(int)


def _write_debug_snapshot(
    *,
    debug_dir: str,
    filename: str,
    fields: dict[str, np.ndarray],
    x: np.ndarray | None,
    y: np.ndarray | None,
) -> None:
    """Write intermediate fields to NetCDF for ParaView debugging.

    ParaView behaves best with float (double) variables, so all fields are
    converted to float64. Boolean masks are written as 0.0/1.0.
    """

    os.makedirs(debug_dir, exist_ok=True)
    out_path = os.path.join(debug_dir, filename)

    any_field = next(iter(fields.values()))
    ny, nx = any_field.shape

    if x is None:
        x_coord = np.arange(nx, dtype=np.float64)
    else:
        x_coord = np.asarray(x, dtype=np.float64)
    if y is None:
        y_coord = np.arange(ny, dtype=np.float64)
    else:
        y_coord = np.asarray(y, dtype=np.float64)

    data_vars: dict[str, tuple[tuple[str, str], np.ndarray]] = {}
    for name, arr in fields.items():
        if np.ma.isMaskedArray(arr):
            arr = np.ma.filled(arr, np.nan)
        arr64 = np.asarray(arr, dtype=np.float64)
        data_vars[name] = (('y', 'x'), arr64)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'x': ('x', x_coord),
            'y': ('y', y_coord),
        },
    )
    write_netcdf(ds, out_path)


def _get_shelf_params(config) -> ShelfParams:
    shelf_isobath_depth_m = config.getfloat('imbie', 'shelf_isobath_depth')
    if shelf_isobath_depth_m is None:
        raise ValueError('shelf_isobath_depth must be set in [imbie] section')

    frac_threshold = config.getfloat('imbie', 'frac_threshold')
    if frac_threshold is None:
        raise ValueError('frac_threshold must be set in [imbie] section')
    seed_dilation_iters = config.getint('imbie', 'seed_dilation_iters')
    if seed_dilation_iters is None:
        raise ValueError('seed_dilation_iters must be set in [imbie] section')

    return ShelfParams(
        shelf_isobath_depth_m=shelf_isobath_depth_m,
        frac_threshold=frac_threshold,
        seed_dilation_iters=seed_dilation_iters,
    )


def _ensure_topography_on_ismip(*, config, workdir: str) -> str:
    """
    Ensure configured topography exists on the ISMIP grid and return path.
    """

    logger = logging.getLogger(__name__)
    cwd = os.getcwd()
    try:
        os.makedirs(os.path.join(workdir, 'topo'), exist_ok=True)
        os.chdir(workdir)
        topo_obj = get_topo(config, logger)
        topo_rel_path = topo_obj.get_topo_on_ismip_path()
        if not os.path.exists(topo_rel_path):
            topo_obj.download_and_preprocess_topo()
            topo_obj.remap_topo_to_ismip()
        if not os.path.exists(topo_rel_path):
            raise FileNotFoundError(
                f'Failed to build topography file: {topo_rel_path}'
            )
        return os.path.join(workdir, topo_rel_path)
    finally:
        os.chdir(cwd)


def _select_shelf_region_seeded_by_basins(
    *,
    shelf_candidate: np.ndarray,
    basin_number: np.ndarray,
    seed_dilation_iters: int,
) -> np.ndarray:
    """Select shelf cells connected to the basins (avoids island shelves)."""

    structure = ndimage.generate_binary_structure(2, 2)

    basin_defined = basin_number >= 0
    # Seed shelf points that are near any defined basin region.
    basin_near_shelf = ndimage.binary_dilation(
        basin_defined, structure=structure, iterations=seed_dilation_iters
    )
    shelf_seed = shelf_candidate & basin_near_shelf

    labeled, num = ndimage.label(shelf_candidate, structure=structure)
    if num == 0:
        return np.zeros_like(shelf_candidate, dtype=bool)

    if np.any(shelf_seed):
        seed_labels = np.unique(labeled[shelf_seed])
        seed_labels = seed_labels[seed_labels != 0]
        if seed_labels.size > 0:
            return np.isin(labeled, seed_labels)

    # Fallback: keep the largest component if we couldn't find a seeded one.
    return _largest_connected_component(shelf_candidate)


def _extract_shelf_break(
    *, shelf_ocean: np.ndarray, ocean_mask: np.ndarray
) -> np.ndarray:
    """
    Extract shelf break cells (boundary between shelf ocean and deep ocean).
    """

    structure = ndimage.generate_binary_structure(2, 2)
    deep_ocean = ocean_mask & (~shelf_ocean)
    adjacent_to_deep = ndimage.binary_dilation(deep_ocean, structure=structure)
    shelf_break = shelf_ocean & adjacent_to_deep
    return shelf_break


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a boolean mask."""

    structure = ndimage.generate_binary_structure(2, 2)
    labeled, num = ndimage.label(mask, structure=structure)
    if num <= 1:
        return mask

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest = int(np.argmax(counts))
    return labeled == largest


def _label_shelf_break_by_basin_old(
    *,
    shelf_break: np.ndarray,
    shelf_ocean: np.ndarray,
    basin_number: np.ndarray,
    num_basins: int,
    dx: float,
    dy: float,
    seed_dilation_iters: int,
) -> np.ndarray:
    """Original nearest-distance fallback algorithm for labeling shelf-break.

    Kept as a module-level helper to be used as a robust fallback.
    """

    structure = ndimage.generate_binary_structure(2, 2)
    break_idx = np.flatnonzero(shelf_break.ravel())
    best_dist = np.full(break_idx.shape[0], np.inf, dtype=float)
    best_label = np.full(break_idx.shape[0], -1, dtype=int)
    spacing = [dy, dx]
    for basin_id in range(num_basins):
        basin_mask = basin_number == basin_id
        source = shelf_ocean & ndimage.binary_dilation(
            basin_mask, structure=structure, iterations=seed_dilation_iters
        )
        if not np.any(source):
            continue
        phi = np.where(source, -1.0, 1.0)
        phi = np.ma.MaskedArray(phi, mask=~shelf_ocean)
        dist = skfmm.distance(phi, dx=spacing)
        dist_flat = dist.reshape(-1)
        d_break = dist_flat[break_idx]
        if hasattr(d_break, 'filled'):
            d_break = d_break.filled(np.inf)
        update = d_break < best_dist
        best_dist[update] = d_break[update]
        best_label[update] = basin_id
    out = -1 * np.ones_like(basin_number, dtype=int)
    out_flat = out.reshape(-1)
    out_flat[break_idx] = best_label
    return out


def _find_triple_points(basin: np.ndarray):
    """Find 2x2 triple points: center locations and the two valid basin ids.

    Returns a list of tuples: (center_yx, (a,b), (i,j)) where center is
    fractional (y,x) coordinate (i+0.5,j+0.5) and (i,j) is the top-left of
    the 2x2 block.
    """
    triples = []
    ny, nx = basin.shape
    for i in range(ny - 1):
        for j in range(nx - 1):
            block = basin[i : i + 2, j : j + 2].ravel()
            unique = np.unique(block)
            if unique.size == 3 and np.any(unique < 0):
                valid = tuple(sorted([int(v) for v in unique if v >= 0]))
                if len(valid) == 2:
                    triples.append(((i + 0.5, j + 0.5), valid, (i, j)))
    return triples


def _closest_shelf_points_for_triples(
    triples, sb_y: np.ndarray, sb_x: np.ndarray, dx: float, dy: float
):
    """For each triple point find the closest shelf-break pixel.

    Returns list of (tp_center, pair, (y,x), distance).
    """
    closest = []
    sx = sb_x.astype(float)
    sy = sb_y.astype(float)
    for tp_center, pair, _ in triples:
        ty, tx = tp_center
        dyv = (sy - ty) * dy
        dxv = (sx - tx) * dx
        d2 = dxv * dxv + dyv * dyv
        idx = int(np.argmin(d2))
        closest.append(
            (tp_center, pair, (int(sy[idx]), int(sx[idx])), np.sqrt(d2[idx]))
        )
    return closest


def _shortest_path_on_binary_contour(mask: np.ndarray, start, goal):
    """BFS on mask (True = traversable). Returns list of (y,x) coords or []."""
    if start == goal:
        return [start]
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    prev = -np.ones((h, w, 2), dtype=int)
    dq = deque()
    sy, sx = start
    gy, gx = goal
    if not mask[sy, sx] or not mask[gy, gx]:
        return []
    visited[sy, sx] = True
    dq.append((sy, sx))
    # 8-neighbor offsets
    neigh = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]
    found = False
    while dq:
        y, x = dq.popleft()
        if (y, x) == (gy, gx):
            found = True
            break
        for dyo, dxo in neigh:
            yy = y + dyo
            xx = x + dxo
            if (
                0 <= yy < h
                and 0 <= xx < w
                and mask[yy, xx]
                and not visited[yy, xx]
            ):
                visited[yy, xx] = True
                prev[yy, xx, 0] = y
                prev[yy, xx, 1] = x
                dq.append((yy, xx))
    if not found:
        return []
    path = []
    cur = (gy, gx)
    while cur != (-1, -1):
        path.append(cur)
        py, px = prev[cur]
        if py == -1:
            break
        cur = (int(py), int(px))
    path.reverse()
    return path


def _rasterize_line(
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """Rasterize a straight line between two integer grid points (Bresenham).

    Returns a list of (y, x) integer coordinates approximating a straight line.
    """
    y0, x0 = int(start[0]), int(start[1])
    y1, x1 = int(goal[0]), int(goal[1])
    points: list[tuple[int, int]] = []
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    if dx > dy:
        err = dx // 2
        while x0 != x1:
            points.append((y0, x0))
            err -= dy
            if err < 0:
                y0 += sy
                err += dx
            x0 += sx
    else:
        err = dy // 2
        while y0 != y1:
            points.append((y0, x0))
            err -= dx
            if err < 0:
                x0 += sx
                err += dy
            y0 += sy
    points.append((y1, x1))
    return points


def _rasterize_line_supercover(
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """Rasterize a line and include corner cells on diagonal steps.

    Bresenham produces an 8-connected path. For boundary walls we typically
    need to avoid diagonal corner gaps, so for each diagonal step we include
    the two orthogonal corner cells as well (a 2x2 "supercover" at corners).
    """

    path = _rasterize_line(start, goal)
    if len(path) <= 1:
        return path

    out: list[tuple[int, int]] = [path[0]]
    for (y0, x0), (y1, x1) in zip(path[:-1], path[1:], strict=True):
        dy = y1 - y0
        dx = x1 - x0
        if dy != 0 and dx != 0:
            # Diagonal step: add both orthogonal corner cells to avoid a
            # corner gap that an 8-neighbor fill could slip through.
            out.append((y0, x1))
            out.append((y1, x0))
        out.append((y1, x1))

    # Preserve order but drop duplicates
    seen: set[tuple[int, int]] = set()
    deduped: list[tuple[int, int]] = []
    for pt in out:
        if pt in seen:
            continue
        seen.add(pt)
        deduped.append(pt)
    return deduped


def _supercover_mask_corners(mask: np.ndarray) -> np.ndarray:
    """Add orthogonal corner pixels for diagonally-adjacent True cells.

    This turns an 8-connected 1-cell-wide contour into a "wall" that blocks
    diagonal corner-cutting when using 8-neighbor flood fills.

    The original mask is not modified; a new boolean array is returned.
    """

    if mask.dtype != bool:
        mask = mask.astype(bool, copy=False)

    out = mask.copy()

    # Down-right diagonals: (y,x) and (y+1,x+1) are True.
    diag = mask[:-1, :-1] & mask[1:, 1:]
    out[:-1, 1:] |= diag
    out[1:, :-1] |= diag

    # Down-left diagonals: (y,x) and (y+1,x-1) are True.
    diag = mask[:-1, 1:] & mask[1:, :-1]
    out[:-1, :-1] |= diag
    out[1:, 1:] |= diag

    return out


def _compute_best_per_pair(
    triples, sb_y: np.ndarray, sb_x: np.ndarray, dx: float, dy: float
):
    closest = _closest_shelf_points_for_triples(triples, sb_y, sb_x, dx, dy)
    best_per_pair = {}
    for tp_center, pair, sb_pt, dist in closest:
        key = tuple(pair)
        if key not in best_per_pair or dist < best_per_pair[key][3]:
            best_per_pair[key] = (tp_center, pair, sb_pt, dist)
    return best_per_pair


def _write_break_seeds_debug(
    *,
    best_per_pair: dict,
    shelf_break: np.ndarray,
    num_basins: int,
    debug: bool,
    debug_dir: str,
    x: np.ndarray | None,
    y: np.ndarray | None,
    filename: str = 'debug_06_break_seeds.nc',
) -> None:
    if not debug:
        return
    chosen_mask = np.zeros_like(shelf_break, dtype=bool)
    chosen_pair_a = -1 * np.ones_like(shelf_break, dtype=float)
    chosen_pair_b = -1 * np.ones_like(shelf_break, dtype=float)
    seed_label = -1 * np.ones_like(shelf_break, dtype=float)
    for k in range(num_basins):
        a = k
        b = (k + 1) % num_basins
        key = tuple(sorted((a, b)))
        sb_pt = best_per_pair[key][2]
        yy, xx = sb_pt
        chosen_mask[yy, xx] = True
        chosen_pair_a[yy, xx] = float(a)
        chosen_pair_b[yy, xx] = float(b)
        seed_label[yy, xx] = float((k + 1) % num_basins)
    _write_debug_snapshot(
        debug_dir=debug_dir,
        filename=filename,
        fields={
            'shelf_break': shelf_break,
            'break_seed_points': chosen_mask,
            'break_seed_label': seed_label,
            'chosen_pair_a': chosen_pair_a,
            'chosen_pair_b': chosen_pair_b,
        },
        x=x,
        y=y,
    )


def _grid_spacing_from_coords(
    *, x: np.ndarray | None, y: np.ndarray | None, dx: float, dy: float
) -> tuple[float, float, float]:
    """Return (dx_grid, dy_grid, min_grid) in physical units."""

    if x is not None and x.size >= 2:
        dx_grid = float(np.median(np.abs(np.diff(x))))
    else:
        dx_grid = float(abs(dx))
    if y is not None and y.size >= 2:
        dy_grid = float(np.median(np.abs(np.diff(y))))
    else:
        dy_grid = float(abs(dy))
    min_grid = float(min(dx_grid, dy_grid))
    return dx_grid, dy_grid, min_grid


def _interp_coord_at_index(coord: np.ndarray, idx: float) -> float:
    """Interpolate a 1D coord array at fractional index."""

    n = coord.size
    if n == 0:
        return float(idx)
    if idx <= 0:
        return float(coord[0])
    if idx >= n - 1:
        return float(coord[-1])
    i0 = int(np.floor(idx))
    i1 = i0 + 1
    t = float(idx - i0)
    return float((1.0 - t) * coord[i0] + t * coord[i1])


def _nearest_index(coord: np.ndarray, value: float) -> int:
    """Nearest index in a 1D coordinate array (works even if descending)."""

    return int(np.argmin(np.abs(coord - value)))


def _xy_to_ij(
    *,
    x: np.ndarray,
    y: np.ndarray,
    xp: float,
    yp: float,
) -> tuple[int, int]:
    """Map a physical (x,y) point to nearest (i,j) grid index."""

    j = _nearest_index(x, xp)
    i = _nearest_index(y, yp)
    return (i, j)


def _build_offset_rays_indices(
    *,
    tp_center: tuple[float, float],
    sb_pt: tuple[int, int],
    x: np.ndarray | None,
    y: np.ndarray | None,
    dx: float,
    dy: float,
    extend_steps: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Build left/right ray endpoints in index space.

    Uses continuous offsets (half a grid cell) in physical space.
    """

    ny = int(sb_pt[0])
    nx = int(sb_pt[1])
    # Build synthetic coords if not provided.
    if x is None:
        x_coord = (np.arange(0, 1 + nx, dtype=float) * float(dx)).astype(float)
    else:
        x_coord = np.asarray(x, dtype=float)
    if y is None:
        y_coord = (np.arange(0, 1 + ny, dtype=float) * float(dy)).astype(float)
    else:
        y_coord = np.asarray(y, dtype=float)

    _, _, min_grid = _grid_spacing_from_coords(
        x=x_coord, y=y_coord, dx=dx, dy=dy
    )
    offset_dist = 0.5 * min_grid
    extend_dist = float(extend_steps) * min_grid

    # Triple-point center in physical coords
    tp_y, tp_x = tp_center
    x0 = _interp_coord_at_index(x_coord, tp_x)
    y0 = _interp_coord_at_index(y_coord, tp_y)

    # Shelf-break target in physical coords
    x1 = float(x_coord[int(sb_pt[1])])
    y1 = float(y_coord[int(sb_pt[0])])

    d_x = x1 - x0
    d_y = y1 - y0
    norm = float(np.hypot(d_x, d_y))
    if norm == 0.0:
        # Degenerate; fall back to no offset.
        i0, j0 = _xy_to_ij(x=x_coord, y=y_coord, xp=x0, yp=y0)
        i1, j1 = _xy_to_ij(x=x_coord, y=y_coord, xp=x1, yp=y1)
        return (i0, j0), (i1, j1), (i0, j0), (i1, j1)

    u_x = d_x / norm
    u_y = d_y / norm

    # Left-normal in physical coordinates (x east, y north): n = (-dy, dx)
    n_x = -u_y
    n_y = u_x

    # Start/end points for the two rays
    x0_l = x0 + n_x * offset_dist
    y0_l = y0 + n_y * offset_dist
    x0_r = x0 - n_x * offset_dist
    y0_r = y0 - n_y * offset_dist

    x1e = x1 + u_x * extend_dist
    y1e = y1 + u_y * extend_dist
    x1_l = x1e + n_x * offset_dist
    y1_l = y1e + n_y * offset_dist
    x1_r = x1e - n_x * offset_dist
    y1_r = y1e - n_y * offset_dist

    ij0_l = _xy_to_ij(x=x_coord, y=y_coord, xp=x0_l, yp=y0_l)
    ij1_l = _xy_to_ij(x=x_coord, y=y_coord, xp=x1_l, yp=y1_l)
    ij0_r = _xy_to_ij(x=x_coord, y=y_coord, xp=x0_r, yp=y0_r)
    ij1_r = _xy_to_ij(x=x_coord, y=y_coord, xp=x1_r, yp=y1_r)
    return ij0_l, ij1_l, ij0_r, ij1_r


def _build_center_ray_indices(
    *,
    tp_center: tuple[float, float],
    sb_pt: tuple[int, int],
    x: np.ndarray | None,
    y: np.ndarray | None,
    dx: float,
    dy: float,
    extend_steps: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Build a single connector centerline in index space.

    This is the un-offset ray from the triple point toward the shelf-break,
    extended slightly past the break so it reliably intersects the contour.
    """

    ny = int(sb_pt[0])
    nx = int(sb_pt[1])
    if x is None:
        x_coord = (np.arange(0, 1 + nx, dtype=float) * float(dx)).astype(float)
    else:
        x_coord = np.asarray(x, dtype=float)
    if y is None:
        y_coord = (np.arange(0, 1 + ny, dtype=float) * float(dy)).astype(float)
    else:
        y_coord = np.asarray(y, dtype=float)

    _, _, min_grid = _grid_spacing_from_coords(
        x=x_coord, y=y_coord, dx=dx, dy=dy
    )
    extend_dist = float(extend_steps) * min_grid

    tp_y, tp_x = tp_center
    x0 = _interp_coord_at_index(x_coord, tp_x)
    y0 = _interp_coord_at_index(y_coord, tp_y)

    x1 = float(x_coord[int(sb_pt[1])])
    y1 = float(y_coord[int(sb_pt[0])])

    d_x = x1 - x0
    d_y = y1 - y0
    norm = float(np.hypot(d_x, d_y))
    if norm == 0.0:
        ij0 = _xy_to_ij(x=x_coord, y=y_coord, xp=x0, yp=y0)
        ij1 = _xy_to_ij(x=x_coord, y=y_coord, xp=x1, yp=y1)
        return ij0, ij1

    u_x = d_x / norm
    u_y = d_y / norm

    x1e = x1 + u_x * extend_dist
    y1e = y1 + u_y * extend_dist

    ij0 = _xy_to_ij(x=x_coord, y=y_coord, xp=x0, yp=y0)
    ij1 = _xy_to_ij(x=x_coord, y=y_coord, xp=x1e, yp=y1e)
    return ij0, ij1


def _record_crossing(
    *,
    yx: tuple[int, int],
    sb_pt: tuple[int, int],
    basin_id: int,
    k: int,
    cross_d2: list[float],
    cross_pt: list[tuple[int, int] | None],
    cross_label: np.ndarray,
    boundary_mask: np.ndarray,
) -> None:
    yy, xx = yx
    d2 = (yy - sb_pt[0]) ** 2 + (xx - sb_pt[1]) ** 2
    if d2 < cross_d2[k]:
        cross_d2[k] = d2
        cross_pt[k] = (yy, xx)
    cross_label[yy, xx] = basin_id
    boundary_mask[yy, xx] = True


def _record_crossing_single(
    *,
    yx: tuple[int, int],
    sb_pt: tuple[int, int],
    k: int,
    cross_d2: list[float],
    cross_pt: list[tuple[int, int] | None],
    cross_mask: np.ndarray,
    boundary_mask: np.ndarray,
) -> None:
    yy, xx = yx
    d2 = (yy - sb_pt[0]) ** 2 + (xx - sb_pt[1]) ** 2
    if d2 < cross_d2[k]:
        cross_d2[k] = d2
        cross_pt[k] = (yy, xx)
    cross_mask[yy, xx] = True
    boundary_mask[yy, xx] = True


def _build_connectors_as_walls(
    *,
    best_per_pair: dict,
    shelf_break: np.ndarray,
    shelf_ocean: np.ndarray,
    basin_number: np.ndarray,
    num_basins: int,
    dx: float,
    dy: float,
    x: np.ndarray | None,
    y: np.ndarray | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[tuple[int, int] | None],
    np.ndarray,
]:
    boundary_mask = _supercover_mask_corners(shelf_break)
    connectors_mask = np.zeros_like(shelf_break, dtype=bool)

    # Allow connector labels to extend slightly beyond the shelf-break so both
    # side labels reliably intersect the shelf-break contour.
    structure = ndimage.generate_binary_structure(2, 2)
    extend_past_break = 2
    connector_allowed = ndimage.binary_dilation(
        shelf_ocean, structure=structure, iterations=extend_past_break
    )

    # For each adjacent pair boundary k:(k,k+1), store the best (closest to
    # sb_pt) shelf-break crossing point of the connector wall.
    cross_pt: list[tuple[int, int] | None] = [None] * num_basins
    cross_d2 = [np.inf] * num_basins
    cross_mask = np.zeros_like(shelf_break, dtype=bool)

    for k in range(num_basins):
        a = k
        b = (k + 1) % num_basins
        key = tuple(sorted((a, b)))
        tp_center = best_per_pair[key][0]
        sb_pt = best_per_pair[key][2]
        ti = int(np.floor(tp_center[0]))
        tj = int(np.floor(tp_center[1]))
        candidates = [(ti, tj), (ti + 1, tj), (ti, tj + 1), (ti + 1, tj + 1)]
        start = None
        for ci, cj in candidates:
            if (
                0 <= ci < basin_number.shape[0]
                and 0 <= cj < basin_number.shape[1]
            ):
                if basin_number[ci, cj] < 0:
                    start = (ci, cj)
                    break
        if start is None:
            d2 = [
                (ci - sb_pt[0]) ** 2 + (cj - sb_pt[1]) ** 2
                for (ci, cj) in candidates
            ]
            start = candidates[int(np.argmin(d2))]

        # Connector wall centerline.
        ij0, ij1 = _build_center_ray_indices(
            tp_center=tp_center,
            sb_pt=sb_pt,
            x=x,
            y=y,
            dx=dx,
            dy=dy,
            extend_steps=extend_past_break,
        )
        path = _rasterize_line_supercover(ij0, ij1)

        for yy, xx in path:
            if (
                0 <= yy < basin_number.shape[0]
                and 0 <= xx < basin_number.shape[1]
            ):
                if not connector_allowed[yy, xx]:
                    continue
                connectors_mask[yy, xx] = True
                boundary_mask[yy, xx] = True
                if shelf_break[yy, xx]:
                    _record_crossing_single(
                        yx=(yy, xx),
                        sb_pt=sb_pt,
                        k=k,
                        cross_d2=cross_d2,
                        cross_pt=cross_pt,
                        cross_mask=cross_mask,
                        boundary_mask=boundary_mask,
                    )

    return (boundary_mask, connectors_mask, cross_pt, cross_mask)


def _label_shelf_break_segments_from_crossings(
    *,
    shelf_break: np.ndarray,
    basin_number: np.ndarray,
    cross_pt: list[tuple[int, int] | None],
    num_basins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label the shelf-break contour with constant basin values per segment.

    For basin k, the expected shelf-break segment is between connector
    crossings on the shelf-break for the adjacent boundaries:
    - west corner: crossing of boundary (k-1,k)
    - east corner: crossing of boundary (k,k+1)

    We label the shortest path along the shelf-break between these two points
    with basin k.

    Returns
    -------
    break_label : ndarray[int]
        Labels on shelf-break cells; -1 elsewhere.
    break_seed_mask : ndarray[bool]
        Mask of the two corner seed points per basin.
    break_seed_label : ndarray[int]
        Basin id at each seed point; -1 elsewhere.
    """

    break_label = -1 * np.ones_like(basin_number, dtype=int)
    break_seed_mask = np.zeros_like(shelf_break, dtype=bool)
    break_seed_label = -1 * np.ones_like(basin_number, dtype=int)

    for k in range(num_basins):
        west = cross_pt[(k - 1) % num_basins]
        east = cross_pt[k]
        if west is None or east is None:
            continue
        break_seed_mask[west] = True
        break_seed_mask[east] = True
        break_seed_label[west] = k
        break_seed_label[east] = k
        path = _shortest_path_on_binary_contour(shelf_break, west, east)
        for yy, xx in path:
            break_label[yy, xx] = k

    return break_label, break_seed_mask, break_seed_label


def _label_shelf_break_by_basin(
    *,
    shelf_break: np.ndarray,
    shelf_ocean: np.ndarray,
    basin_number: np.ndarray,
    num_basins: int,
    dx: float,
    dy: float,
    seed_dilation_iters: int,
    debug: bool = False,
    debug_dir: str = 'imbie2',
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign each shelf-break cell to the nearest basin along the shelf.

    New algorithm:
    - find 2x2 'triple points' with exactly three basin values where one is <0
    - for each triple point find closest shelf-break pixel
    - for each basin-pair keep only the triple+break pixel with shortest
      distance
    - for ordered pairs (0-1,1-2,...,n-1-0) extract shortest-path along
      shelf_break between consecutive break pixels and label that path with the
      right-hand basin (k+1 for segment between Pk and Pk+1)
    - fallback to older nearest-distance approach if construction fails
    """

    triples = _find_triple_points(basin_number)
    if debug:
        # Visualize triple points by marking their 2x2 blocks
        triple_mask = np.zeros_like(basin_number, dtype=bool)
        triple_pair_min = -1 * np.ones_like(basin_number, dtype=float)
        triple_pair_max = -1 * np.ones_like(basin_number, dtype=float)
        for _, pair, (ii, jj) in triples:
            triple_mask[ii : ii + 2, jj : jj + 2] = True
            triple_pair_min[ii : ii + 2, jj : jj + 2] = float(min(pair))
            triple_pair_max[ii : ii + 2, jj : jj + 2] = float(max(pair))
        _write_debug_snapshot(
            debug_dir=debug_dir,
            filename='debug_05_triples.nc',
            fields={
                'triple_mask': triple_mask,
                'triple_pair_min': triple_pair_min,
                'triple_pair_max': triple_pair_max,
            },
            x=x,
            y=y,
        )
    if len(triples) == 0:
        brk = _label_shelf_break_by_basin_old(
            shelf_break=shelf_break,
            shelf_ocean=shelf_ocean,
            basin_number=basin_number,
            num_basins=num_basins,
            dx=dx,
            dy=dy,
            seed_dilation_iters=seed_dilation_iters,
        )
        return (
            brk,
            shelf_break,
            np.zeros_like(shelf_break, dtype=bool),
        )

    sb_y, sb_x = np.nonzero(shelf_break)
    if sb_y.size == 0:
        brk = _label_shelf_break_by_basin_old(
            shelf_break=shelf_break,
            shelf_ocean=shelf_ocean,
            basin_number=basin_number,
            num_basins=num_basins,
            dx=dx,
            dy=dy,
            seed_dilation_iters=seed_dilation_iters,
        )
        return (
            brk,
            shelf_break,
            np.zeros_like(shelf_break, dtype=bool),
        )
    best_per_pair = _compute_best_per_pair(triples, sb_y, sb_x, dx, dy)

    ordered_points = []
    for k in range(num_basins):
        a = k
        b = (k + 1) % num_basins
        key = tuple(sorted((a, b)))
        if key not in best_per_pair:
            brk = _label_shelf_break_by_basin_old(
                shelf_break=shelf_break,
                shelf_ocean=shelf_ocean,
                basin_number=basin_number,
                num_basins=num_basins,
                dx=dx,
                dy=dy,
                seed_dilation_iters=seed_dilation_iters,
            )
            return (
                brk,
                shelf_break,
                np.zeros_like(shelf_break, dtype=bool),
            )
        ordered_points.append(best_per_pair[key][2])

    boundary_mask, connectors_mask, cross_pt, cross_mask = (
        _build_connectors_as_walls(
            best_per_pair=best_per_pair,
            shelf_break=shelf_break,
            shelf_ocean=shelf_ocean,
            basin_number=basin_number,
            num_basins=num_basins,
            dx=dx,
            dy=dy,
            x=x,
            y=y,
        )
    )

    if debug:
        _write_debug_snapshot(
            debug_dir=debug_dir,
            filename='debug_06_connectors.nc',
            fields={
                'connectors_mask': connectors_mask,
                'boundary_mask': boundary_mask,
                'connectors_cross_mask': cross_mask,
            },
            x=x,
            y=y,
        )

    # Build shelf-break labels from connector crossings.
    break_label, break_seed_mask, break_seed_label = (
        _label_shelf_break_segments_from_crossings(
            shelf_break=shelf_break,
            basin_number=basin_number,
            cross_pt=cross_pt,
            num_basins=num_basins,
        )
    )

    if debug:
        _write_debug_snapshot(
            debug_dir=debug_dir,
            filename='debug_07_break_seeds.nc',
            fields={
                'shelf_break': shelf_break,
                'break_seed_points': break_seed_mask,
                'break_seed_label': break_seed_label,
            },
            x=x,
            y=y,
        )

        _write_debug_snapshot(
            debug_dir=debug_dir,
            filename='debug_08_break_label.nc',
            fields={
                'break_label': break_label,
                'shelf_break': shelf_break,
            },
            x=x,
            y=y,
        )

    return (
        break_label,
        boundary_mask,
        connectors_mask,
    )


def _extrude_labels_from_break(
    *,
    shelf_break: np.ndarray,
    shelf_ocean: np.ndarray,
    basin_number: np.ndarray,
    break_label: np.ndarray,
) -> np.ndarray:
    """Build shelf labels using shelf-break and original basin labels.

    Returns a tuple:
    - combined_shelf_label: includes original valid `basin_number` where
        defined, and uses nearest shelf-break labels to fill unlabeled shelf
        cells.
    - shelf_label_from_break_only: labels derived solely from shelf-break
        projection (for debugging/inspection).
    """

    # Nearest shelf-break indices for every cell
    edt_result = ndimage.distance_transform_edt(
        np.logical_not(shelf_break), return_indices=True
    )
    # SciPy returns (distance, indices) where indices is a tuple of arrays
    if isinstance(edt_result, tuple) and len(edt_result) >= 2:
        indices = edt_result[1]
        if isinstance(indices, tuple) and len(indices) >= 2:
            nearest_y = indices[0]
            nearest_x = indices[1]
        else:
            yy, xx = np.nonzero(shelf_break)
            nearest_y = np.zeros_like(shelf_break, dtype=int)
            nearest_x = np.zeros_like(shelf_break, dtype=int)
            for i in range(shelf_break.shape[0]):
                for j in range(shelf_break.shape[1]):
                    d2 = (yy - i) ** 2 + (xx - j) ** 2
                    k = int(np.argmin(d2))
                    nearest_y[i, j] = yy[k]
                    nearest_x[i, j] = xx[k]
    else:
        yy, xx = np.nonzero(shelf_break)
        nearest_y = np.zeros_like(shelf_break, dtype=int)
        nearest_x = np.zeros_like(shelf_break, dtype=int)
        for i in range(shelf_break.shape[0]):
            for j in range(shelf_break.shape[1]):
                d2 = (yy - i) ** 2 + (xx - j) ** 2
                k = int(np.argmin(d2))
                nearest_y[i, j] = yy[k]
                nearest_x[i, j] = xx[k]

    shelf_label_from_break = -1 * np.ones_like(break_label, dtype=int)
    shelf_label_from_break[shelf_ocean] = break_label[
        nearest_y[shelf_ocean], nearest_x[shelf_ocean]
    ]

    return shelf_label_from_break


def _fill_shelf_with_boundaries(
    *,
    shelf_ocean: np.ndarray,
    basin_number: np.ndarray,
    break_label: np.ndarray,
    boundary_mask: np.ndarray,
    debug: bool = False,
    debug_dir: str = 'imbie2',
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> np.ndarray:
    """Fill shelf basins constrained by boundary lines.

    The shelf interior (``shelf_ocean & ~boundary_mask``) should ideally be
    partitioned into connected regions by walls (connectors + shelf break).
    When the original basin rasterization is locally messy, a multi-source
    flood fill can allow small contaminant labels to "win" and then propagate
    into a much larger region.

    Instead, we:
    - identify connected components of the *interior* (independent of any
      labels),
    - determine the dominant (most frequent) basin label on the 8-neighbor
      boundary of each component,
    - fill all previously-unlabeled interior cells in that component with
      that dominant label.

    Existing valid labels from the input are never overwritten.
    """

    structure = ndimage.generate_binary_structure(2, 2)
    interior = np.logical_and(shelf_ocean, np.logical_not(boundary_mask))

    # Working label state starts from the original basin labels everywhere.
    label_state = basin_number.astype(int, copy=True)

    # Add shelf-break anchors on the wall (but never overwrite existing
    # labels).
    break_anchor = np.logical_and(
        np.logical_and(boundary_mask, break_label >= 0), label_state < 0
    )
    label_state[break_anchor] = break_label[break_anchor]

    comp_id, _ = ndimage.label(interior, structure=structure)
    objects = ndimage.find_objects(comp_id)

    # For debug: record which component each cell belongs to, and which label
    # that component chose for filling.
    component_fill_label = -1 * np.ones_like(basin_number, dtype=int)

    ny, nx = basin_number.shape
    for cid, sl in enumerate(objects, start=1):
        if sl is None:
            continue

        y0 = max(0, sl[0].start - 1)
        y1 = min(ny, sl[0].stop + 1)
        x0 = max(0, sl[1].start - 1)
        x1 = min(nx, sl[1].stop + 1)
        esl = (slice(y0, y1), slice(x0, x1))

        comp_local = comp_id[esl] == cid
        if not np.any(comp_local):
            continue

        dil = ndimage.binary_dilation(comp_local, structure=structure)
        boundary_ring = np.logical_and(dil, np.logical_not(comp_local))
        boundary_labels = label_state[esl][boundary_ring]
        boundary_labels = boundary_labels[boundary_labels >= 0]
        if boundary_labels.size == 0:
            continue

        counts = np.bincount(boundary_labels.astype(int))
        dominant = int(np.argmax(counts))

        # Fill only previously-unlabeled interior cells.
        fill_local = np.logical_and(comp_local, label_state[esl] < 0)
        label_state[esl][fill_local] = dominant
        component_fill_label[esl][comp_local] = dominant

    shelf_label = -1 * np.ones_like(basin_number, dtype=int)
    shelf_label[shelf_ocean] = label_state[shelf_ocean]

    if debug:
        comp_id_float = np.zeros_like(basin_number, dtype=float)
        comp_id_float[interior] = comp_id[interior].astype(float)
        fill_label_float = -1 * np.ones_like(basin_number, dtype=float)
        fill_label_float[interior] = component_fill_label[interior].astype(
            float
        )
        _write_debug_snapshot(
            debug_dir=debug_dir,
            filename='debug_10_shelf_fill_seeds.nc',
            fields={
                'shelf_allowed': interior,
                'shelf_component_id': comp_id_float,
                'shelf_component_fill_label': fill_label_float,
            },
            x=x,
            y=y,
        )

    return shelf_label
