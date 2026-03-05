import argparse
import os

import numpy as np
import shapefile
import shapely
import shapely.geometry
import shapely.ops
from shapely.geometry import MultiPolygon, Polygon

from i7aof.config import load_config
from i7aof.grid.ismip import get_horiz_res_string
from i7aof.imbie.download import download_imbie
from i7aof.imbie.masks import (
    BASIN_DEFINITIONS,
    _load_basin_shapes,
    make_imbie_masks,
)
from i7aof.io import read_dataset


def make_imbie_polygon_shapefile(
    config,
    *,
    workdir='.',
    simplify_tolerance_m=0.0,
    remove_extension_holes=True,
    min_hole_area_m2=0.0,
    out_shapefile='imbie2/extended_basin_polygons.shp',
    debug_raster_only_shapefile=None,
    debug_raster_only_simplified_shapefile=None,
    validate=True,
):
    """Build polygon shapefiles for combined + extended IMBIE basins.

    The algorithm keeps the original IMBIE combined polygons exactly and
    appends only the extension outside the original IMBIE footprint:

    1. Build/load extended basin labels from the raster workflow.
    2. Convert each raster basin to polygons.
    3. Remove the full original IMBIE footprint from each raster basin.
    4. Optionally simplify extension polygons.
    5. Union exact original basin polygons with the extension polygons.

    Parameters
    ----------
    config : mpas_tools.config.MpasConfigParser
        Configuration object.

    workdir : str, optional
        Base working directory for reading/writing files.

    simplify_tolerance_m : float, optional
        Shapely simplification tolerance in meters, applied only to extension
        polygons. A value of 0 disables simplification.

    remove_extension_holes : bool, optional
        Whether to remove holes from extension polygons after simplification.

    min_hole_area_m2 : float, optional
        Minimum interior-ring area to retain when removing holes from
        extension polygons. With the default (0), all extension holes are
        removed.

    out_shapefile : str, optional
        Output shapefile path.

    debug_raster_only_shapefile : str, optional
        Optional path for a debug shapefile built directly from rasterized
        extended basin labels only (no reference to original IMBIE polygons).

    debug_raster_only_simplified_shapefile : str, optional
        Optional path for a debug shapefile built directly from rasterized
        extended basin labels, with simplification/hole-removal options
        applied and no reference to original IMBIE polygons.

    validate : bool, optional
        Whether to run basic topology checks (no overlaps and valid geometry)
        on the final basin polygons.

    Returns
    -------
    dict
        Mapping from basin name to final polygon geometry.
    """

    cwd = os.getcwd()
    if workdir is not None:
        os.makedirs(workdir, exist_ok=True)
        os.chdir(workdir)

    try:
        basin_polygons, original_combined = _build_imbie_polygons(
            config=config,
            simplify_tolerance_m=float(simplify_tolerance_m),
            remove_extension_holes=remove_extension_holes,
            min_hole_area_m2=float(min_hole_area_m2),
        )
        if debug_raster_only_shapefile is not None:
            raster_only_polygons = _build_raster_only_polygons(
                config=config,
                simplify_tolerance_m=0.0,
                remove_holes=False,
                min_hole_area_m2=0.0,
                enforce_partition=False,
            )
            _validate_partition_topology(
                raster_only_polygons,
                target_domain=_target_domain_from_basins(raster_only_polygons),
            )
            _write_shapefile(debug_raster_only_shapefile, raster_only_polygons)
            _write_projection_file(
                out_shapefile=debug_raster_only_shapefile,
                source_prj=(
                    'imbie2/ANT_Basins_IMBIE2_v1.6/ANT_Basins_IMBIE2_v1.6.prj'
                ),
            )
        if debug_raster_only_simplified_shapefile is not None:
            raster_only_simplified_polygons = _build_raster_only_polygons(
                config=config,
                simplify_tolerance_m=float(simplify_tolerance_m),
                remove_holes=remove_extension_holes,
                min_hole_area_m2=float(min_hole_area_m2),
                enforce_partition=True,
            )
            _write_shapefile(
                debug_raster_only_simplified_shapefile,
                raster_only_simplified_polygons,
            )
            _write_projection_file(
                out_shapefile=debug_raster_only_simplified_shapefile,
                source_prj=(
                    'imbie2/ANT_Basins_IMBIE2_v1.6/ANT_Basins_IMBIE2_v1.6.prj'
                ),
            )
        overlap_geom_by_pair = _compute_pairwise_overlap_geometries(
            original_combined
        )
        basin_polygons = _resolve_new_overlaps(
            basin_polygons,
            allowed_overlap_geom_by_pair=overlap_geom_by_pair,
        )
        if validate:
            allowed_overlap_by_pair = _compute_pairwise_overlap_areas(
                original_combined
            )
            _validate_basin_polygons(
                basin_polygons,
                allowed_overlap_by_pair=allowed_overlap_by_pair,
            )
        _write_shapefile(out_shapefile, basin_polygons)
        _write_projection_file(
            out_shapefile=out_shapefile,
            source_prj=(
                'imbie2/ANT_Basins_IMBIE2_v1.6/ANT_Basins_IMBIE2_v1.6.prj'
            ),
        )
    finally:
        os.chdir(cwd)

    return basin_polygons


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    config = load_config(
        workdir=args.workdir,
        user_config_filename=args.config,
    )

    make_imbie_polygon_shapefile(
        config,
        workdir=args.workdir if args.workdir is not None else '.',
        simplify_tolerance_m=args.simplify_tolerance_m,
        remove_extension_holes=args.remove_extension_holes,
        min_hole_area_m2=args.min_hole_area_m2,
        out_shapefile=args.out_shapefile,
        debug_raster_only_shapefile=args.debug_raster_only_shapefile,
        debug_raster_only_simplified_shapefile=(
            args.debug_raster_only_simplified_shapefile
        ),
        validate=args.validate,
    )


def _build_imbie_polygons(
    *,
    config,
    simplify_tolerance_m,
    remove_extension_holes,
    min_hole_area_m2,
):
    """Build final basin geometries as exact-original plus extension.

    Extension geometry is built first as a topologically repaired raster-only
    partition, then the exact original IMBIE footprint is removed from each
    extension basin, and finally exact original combined IMBIE polygons are
    unioned back in basin-by-basin.
    """

    download_imbie()
    shp_path = 'imbie2/ANT_Basins_IMBIE2_v1.6/ANT_Basins_IMBIE2_v1.6.shp'
    in_basin_data = _load_basin_shapes(shp_path)
    original_combined = _get_combined_original_polygons(in_basin_data)
    original_footprint = shapely.ops.unary_union(
        list(original_combined.values())
    )

    raster_extension = _build_raster_only_polygons(
        config=config,
        simplify_tolerance_m=simplify_tolerance_m,
        remove_holes=remove_extension_holes,
        min_hole_area_m2=min_hole_area_m2,
        enforce_partition=True,
    )

    final_polygons = {}
    for basin_name in BASIN_DEFINITIONS.keys():
        extension_geom = _make_valid(
            raster_extension[basin_name].difference(original_footprint)
        )
        final_geom = _make_valid(
            original_combined[basin_name].union(extension_geom)
        )
        final_polygons[basin_name] = _ensure_multipolygon(final_geom)

    return final_polygons, original_combined


def _build_raster_only_polygons(
    *,
    config,
    simplify_tolerance_m,
    remove_holes,
    min_hole_area_m2,
    enforce_partition,
):
    """Build polygons directly from rasterized extended basin labels."""

    res = get_horiz_res_string(config)
    out_mask = f'imbie2/basin_numbers_ismip{res}.nc'

    make_imbie_masks(config)
    if not os.path.exists(out_mask):
        raise FileNotFoundError(f'Extended basin mask not found: {out_mask}')

    download_imbie()
    ds = read_dataset(out_mask)
    x = np.asarray(ds['x'].values)
    y = np.asarray(ds['y'].values)
    basin_number = np.asarray(ds['basinNumber'].values, dtype=int)

    basin_polygons = {}
    for basin_id, basin_name in enumerate(BASIN_DEFINITIONS.keys()):
        geom = _label_to_polygon(
            basin_number=basin_number,
            basin_id=basin_id,
            x=x,
            y=y,
        )

        if simplify_tolerance_m > 0.0 and not geom.is_empty:
            geom = _make_valid(
                geom.simplify(simplify_tolerance_m, preserve_topology=True)
            )

        if remove_holes:
            geom = _make_valid(
                _remove_holes(geom, min_hole_area_m2=min_hole_area_m2)
            )

        basin_polygons[basin_name] = _ensure_multipolygon(_make_valid(geom))

    if enforce_partition:
        reference_polygons = _build_raster_only_polygons(
            config=config,
            simplify_tolerance_m=0.0,
            remove_holes=False,
            min_hole_area_m2=0.0,
            enforce_partition=False,
        )
        basin_polygons, target_domain = _enforce_topological_partition(
            basin_polygons,
            reference_polygons=reference_polygons,
        )
        _validate_partition_topology(
            basin_polygons,
            target_domain=target_domain,
        )

    return basin_polygons


def _enforce_topological_partition(
    basin_polygons,
    *,
    reference_polygons=None,
    area_tolerance_m2=1.0e-3,
):
    """Rebuild basins from shared boundaries to remove holes/overlaps."""

    source = {}
    for name, geom in basin_polygons.items():
        source[name] = _ensure_multipolygon(_make_valid(geom))

    if reference_polygons is None:
        reference = source
    else:
        reference = {}
        for name in source:
            geom = reference_polygons.get(name, MultiPolygon([]))
            reference[name] = _ensure_multipolygon(_make_valid(geom))

    # Use hole-free union boundary as the target domain so tiny interior holes
    # from simplification artifacts are filled during face assignment.
    target_domain = _make_valid(shapely.ops.unary_union(list(source.values())))
    target_domain = _make_valid(
        _remove_holes(target_domain, min_hole_area_m2=0.0)
    )

    boundaries = [target_domain.boundary]
    for geom in source.values():
        if not geom.is_empty:
            boundaries.append(geom.boundary)

    linework = _make_valid(shapely.ops.unary_union(boundaries))
    faces = []
    for face in shapely.ops.polygonize(linework):
        if face.area <= area_tolerance_m2:
            continue
        rp = face.representative_point()
        if not (rp.within(target_domain) or rp.touches(target_domain)):
            continue
        faces.append(face)

    names = list(source.keys())
    assigned = {name: [] for name in names}

    for face in faces:
        best_name = None
        best_overlap = 0.0
        for name in names:
            overlap_area = face.intersection(reference[name]).area
            if overlap_area > best_overlap:
                best_overlap = overlap_area
                best_name = name

        if best_name is None or best_overlap <= area_tolerance_m2:
            rp = face.representative_point()
            best_name = min(
                names, key=lambda name: reference[name].distance(rp)
            )

        assigned[best_name].append(face)

    repaired = {}
    for name in names:
        pieces = assigned[name]
        if len(pieces) == 0:
            repaired[name] = MultiPolygon([])
            continue
        repaired[name] = _ensure_multipolygon(
            _make_valid(shapely.ops.unary_union(pieces))
        )

    return repaired, target_domain


def _target_domain_from_basins(basin_polygons):
    """Build a hole-free domain polygon from current basin geometries."""

    target_domain = _make_valid(
        shapely.ops.unary_union(list(basin_polygons.values()))
    )
    target_domain = _make_valid(
        _remove_holes(target_domain, min_hole_area_m2=0.0)
    )
    return target_domain


def _validate_partition_topology(
    basin_polygons,
    *,
    target_domain,
    area_tolerance_m2=1.0e-3,
):
    """
    Validate that basin polygons form a complete non-overlapping partition.
    """

    holes_by_basin = {}
    for name, geom in basin_polygons.items():
        if geom.is_empty:
            continue
        geoms = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
        hole_count = sum(len(poly.interiors) for poly in geoms)
        if hole_count > 0:
            holes_by_basin[name] = hole_count

    if holes_by_basin:
        summary = ', '.join(
            f'{name}:{count}' for name, count in sorted(holes_by_basin.items())
        )
        raise ValueError(f'Partition contains basin interior holes: {summary}')

    _validate_basin_polygons(
        basin_polygons,
        overlap_tolerance_m2=area_tolerance_m2,
    )

    combined = _make_valid(
        shapely.ops.unary_union(list(basin_polygons.values()))
    )
    target = _make_valid(target_domain)

    uncovered = _make_valid(target.difference(combined))
    uncovered_area = uncovered.area
    if uncovered_area > area_tolerance_m2:
        raise ValueError(
            'Partition has uncovered gaps (holes) with total area '
            f'{uncovered_area:.6f} m^2'
        )

    excess = _make_valid(combined.difference(target))
    excess_area = excess.area
    if excess_area > area_tolerance_m2:
        raise ValueError(
            'Partition extends outside target domain with total area '
            f'{excess_area:.6f} m^2'
        )


def _get_combined_original_polygons(in_basin_data):
    """Combine original IMBIE polygons per merged basin definition."""

    combined = {}
    for basin_name, members in BASIN_DEFINITIONS.items():
        polys = [in_basin_data[name] for name in members]
        combined[basin_name] = _make_valid(shapely.ops.unary_union(polys))
    return combined


def _label_to_polygon(*, basin_number, basin_id, x, y):
    """Convert a basin label in the raster to polygon geometry.

    This uses run-length encoding along rows to build axis-aligned rectangles,
    then unions them into a polygon/multipolygon.
    """

    mask = basin_number == basin_id
    if not np.any(mask):
        return MultiPolygon([])

    dx = _uniform_spacing(x)
    dy = _uniform_spacing(y)

    rects = []
    ny, nx = mask.shape

    for j in range(ny):
        row = mask[j, :]
        i = 0
        while i < nx:
            if not row[i]:
                i += 1
                continue

            i0 = i
            while i < nx and row[i]:
                i += 1
            i1 = i

            x0 = x[i0] - 0.5 * dx
            x1 = x[i1 - 1] + 0.5 * dx
            y0 = y[j] - 0.5 * dy
            y1 = y[j] + 0.5 * dy
            rects.append(shapely.geometry.box(x0, y0, x1, y1))

    return _make_valid(shapely.ops.unary_union(rects))


def _uniform_spacing(coord):
    """Estimate grid spacing and validate it is approximately uniform."""

    diffs = np.diff(coord)
    if diffs.size == 0:
        raise ValueError('Coordinate array must have at least two points')

    spacing = float(np.mean(diffs))
    if not np.allclose(diffs, spacing, rtol=0.0, atol=1.0e-6):
        raise ValueError('Grid coordinates must be uniformly spaced')

    return spacing


def _remove_holes(geom, *, min_hole_area_m2=0.0):
    """Remove holes from polygons, optionally keeping large holes."""

    if geom.is_empty:
        return geom

    if isinstance(geom, Polygon):
        return _remove_holes_polygon(geom, min_hole_area_m2=min_hole_area_m2)

    if isinstance(geom, MultiPolygon):
        parts = [
            _remove_holes_polygon(poly, min_hole_area_m2=min_hole_area_m2)
            for poly in geom.geoms
            if not poly.is_empty
        ]
        if len(parts) == 0:
            return MultiPolygon([])
        return MultiPolygon(parts)

    if geom.geom_type == 'GeometryCollection':
        parts = [
            _remove_holes(part, min_hole_area_m2=min_hole_area_m2)
            for part in geom.geoms
            if part.geom_type in ('Polygon', 'MultiPolygon')
        ]
        if len(parts) == 0:
            return MultiPolygon([])
        return _make_valid(shapely.ops.unary_union(parts))

    return geom


def _remove_holes_polygon(poly, *, min_hole_area_m2):
    """Remove interior rings below area threshold from one polygon."""

    keep = []
    for ring in poly.interiors:
        ring_poly = Polygon(ring)
        # min_hole_area_m2 is the minimum area to retain.
        # A threshold of 0 means remove all holes.
        if min_hole_area_m2 > 0.0 and ring_poly.area >= min_hole_area_m2:
            keep.append(ring.coords)
    return Polygon(poly.exterior.coords, holes=keep)


def _make_valid(geom):
    """Return a valid geometry using Shapely make_valid/buffer fallback."""

    if geom.is_empty:
        return geom

    if geom.is_valid:
        return geom

    if hasattr(shapely, 'make_valid'):
        return shapely.make_valid(geom)

    return geom.buffer(0)


def _ensure_multipolygon(geom):
    """Normalize polygonal geometry to MultiPolygon."""

    if geom.is_empty:
        return MultiPolygon([])

    if isinstance(geom, Polygon):
        return MultiPolygon([geom])

    if isinstance(geom, MultiPolygon):
        return geom

    polys = []
    if geom.geom_type == 'GeometryCollection':
        for part in geom.geoms:
            if isinstance(part, Polygon):
                polys.append(part)
            elif isinstance(part, MultiPolygon):
                polys.extend(list(part.geoms))

    if len(polys) == 0:
        return MultiPolygon([])

    return MultiPolygon(polys)


def _write_shapefile(path, basin_polygons):
    """Write basin polygons to a shapefile."""

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with shapefile.Writer(path, shapeType=shapefile.POLYGON) as writer:
        writer.autoBalance = 1
        writer.field('index', 'N', decimal=0)
        writer.field('name', 'C', size=16)

        for basin_id, (name, geom) in enumerate(basin_polygons.items()):
            for poly in geom.geoms:
                parts = [list(poly.exterior.coords)]
                for ring in poly.interiors:
                    parts.append(list(ring.coords))
                writer.poly(parts)
                writer.record(basin_id, name)


def _write_projection_file(*, out_shapefile, source_prj):
    """Write a .prj file for the output shapefile.

    If the source .prj is not found, this function leaves the output without
    a .prj sidecar.
    """

    if not os.path.exists(source_prj):
        return

    out_prj = os.path.splitext(out_shapefile)[0] + '.prj'
    with open(source_prj) as src, open(out_prj, 'w') as dst:
        dst.write(src.read())


def _compute_pairwise_overlap_areas(basin_polygons):
    """Compute positive overlap area for each basin pair."""

    overlaps = {}
    items = list(basin_polygons.items())
    for i, (name_a, geom_a) in enumerate(items):
        if geom_a.is_empty:
            continue
        for name_b, geom_b in items[i + 1 :]:
            if geom_b.is_empty:
                continue
            overlap = geom_a.intersection(geom_b).area
            if overlap > 0.0:
                overlaps[(name_a, name_b)] = overlap
    return overlaps


def _compute_pairwise_overlap_geometries(basin_polygons):
    """Compute overlap geometry for each basin pair."""

    overlaps = {}
    items = list(basin_polygons.items())
    for i, (name_a, geom_a) in enumerate(items):
        if geom_a.is_empty:
            continue
        for name_b, geom_b in items[i + 1 :]:
            if geom_b.is_empty:
                continue
            overlap = _make_valid(geom_a.intersection(geom_b))
            if not overlap.is_empty and overlap.area > 0.0:
                overlaps[(name_a, name_b)] = overlap
    return overlaps


def _resolve_new_overlaps(
    basin_polygons,
    *,
    allowed_overlap_geom_by_pair,
    overlap_tolerance_m2=1.0e-6,
):
    """Remove overlaps that are not part of the original IMBIE overlaps.

    This keeps original-basin overlaps (if any) unchanged but clips away any
    new overlap introduced by extension/simplification. When new overlap is
    found, the later basin in iteration order is trimmed.
    """

    resolved = {name: geom for name, geom in basin_polygons.items()}
    names = list(resolved.keys())

    for i, name_a in enumerate(names):
        geom_a = resolved[name_a]
        if geom_a.is_empty:
            continue

        for name_b in names[i + 1 :]:
            geom_b = resolved[name_b]
            if geom_b.is_empty:
                continue

            pair = (name_a, name_b)
            allowed = allowed_overlap_geom_by_pair.get(pair, None)
            overlap = _make_valid(geom_a.intersection(geom_b))
            if overlap.is_empty or overlap.area <= overlap_tolerance_m2:
                continue

            if allowed is None or allowed.is_empty:
                excess = overlap
            else:
                excess = _make_valid(overlap.difference(allowed))

            if excess.is_empty or excess.area <= overlap_tolerance_m2:
                continue

            resolved[name_b] = _make_valid(geom_b.difference(excess))

    for name, geom in resolved.items():
        resolved[name] = _ensure_multipolygon(_make_valid(geom))

    return resolved


def _validate_basin_polygons(
    basin_polygons,
    *,
    allowed_overlap_by_pair=None,
    overlap_tolerance_m2=1.0e-3,
):
    """Run basic geometry validity and overlap checks."""

    items = list(basin_polygons.items())
    for name, geom in items:
        if geom.is_empty:
            continue
        if not geom.is_valid:
            raise ValueError(f'Invalid geometry for basin {name}')

    for i, (name_a, geom_a) in enumerate(items):
        if geom_a.is_empty:
            continue
        for name_b, geom_b in items[i + 1 :]:
            if geom_b.is_empty:
                continue
            overlap = geom_a.intersection(geom_b)
            overlap_area = overlap.area
            pair = (name_a, name_b)
            allowed = 0.0
            if allowed_overlap_by_pair is not None:
                allowed = allowed_overlap_by_pair.get(pair, 0.0)
            if overlap_area > allowed + overlap_tolerance_m2:
                raise ValueError(
                    'Overlapping basin polygons found for '
                    f'{name_a} and {name_b}: '
                    f'area={overlap_area:.6f} m^2, '
                    f'allowed={allowed:.6f} m^2'
                )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Build combined+extended IMBIE basin polygons '
        'as a shapefile.'
    )
    parser.add_argument(
        '-w',
        '--workdir',
        dest='workdir',
        default=None,
        help='Base working directory.',
    )
    parser.add_argument(
        '-c',
        '--config',
        dest='config',
        default=None,
        help='User config file.',
    )
    parser.add_argument(
        '--simplify-tolerance-m',
        dest='simplify_tolerance_m',
        type=float,
        default=0.0,
        help='Simplification tolerance (meters) for extension polygons.',
    )
    parser.add_argument(
        '--remove-extension-holes',
        dest='remove_extension_holes',
        action='store_true',
        default=True,
        help='Remove holes from extension polygons (default).',
    )
    parser.add_argument(
        '--keep-extension-holes',
        dest='remove_extension_holes',
        action='store_false',
        help='Keep holes in extension polygons.',
    )
    parser.add_argument(
        '--min-hole-area-m2',
        dest='min_hole_area_m2',
        type=float,
        default=0.0,
        help='Minimum hole area to keep when removing extension holes.',
    )
    parser.add_argument(
        '-o',
        '--out-shapefile',
        dest='out_shapefile',
        default='imbie2/extended_basin_polygons.shp',
        help='Output shapefile path.',
    )
    parser.add_argument(
        '--debug-raster-only-shapefile',
        dest='debug_raster_only_shapefile',
        default=None,
        help='Optional debug shapefile path for polygons built directly from '
        'rasterized extended basin labels.',
    )
    parser.add_argument(
        '--debug-raster-only-simplified-shapefile',
        dest='debug_raster_only_simplified_shapefile',
        default=None,
        help='Optional debug shapefile path for simplified polygons built '
        'directly from rasterized extended basin labels.',
    )
    parser.add_argument(
        '--validate',
        dest='validate',
        action='store_true',
        default=True,
        help='Validate output topology (default).',
    )
    parser.add_argument(
        '--no-validate',
        dest='validate',
        action='store_false',
        help='Skip topology validation.',
    )
    return parser
