import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from i7aof.imbie.polygons import (
    _build_arg_parser,
    _compute_pairwise_overlap_areas,
    _compute_pairwise_overlap_geometries,
    _enforce_topological_partition,
    _label_to_polygon,
    _remove_holes,
    _resolve_new_overlaps,
    _uniform_spacing,
    _validate_basin_polygons,
    _validate_partition_topology,
)


def test_uniform_spacing_ok():
    coord = np.array([0.0, 8_000.0, 16_000.0])
    spacing = _uniform_spacing(coord)
    assert spacing == 8_000.0


def test_label_to_polygon_area_single_cell():
    x = np.array([0.0, 8_000.0, 16_000.0])
    y = np.array([0.0, 8_000.0])
    basin = np.array([[1, -1, -1], [-1, -1, -1]])

    geom = _label_to_polygon(basin_number=basin, basin_id=1, x=x, y=y)

    assert np.isclose(geom.area, 8_000.0 * 8_000.0)


def test_remove_holes_removes_small_hole():
    shell = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    hole = [(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]
    poly = Polygon(shell=shell, holes=[hole])

    out = _remove_holes(poly, min_hole_area_m2=5.0)

    assert len(out.interiors) == 0


def test_remove_holes_zero_threshold_removes_all_holes():
    shell = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    hole = [(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]
    poly = Polygon(shell=shell, holes=[hole])

    out = _remove_holes(poly, min_hole_area_m2=0.0)

    assert len(out.interiors) == 0


def test_remove_holes_positive_threshold_keeps_large_holes():
    shell = [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)]
    small_hole = [(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]  # area 4
    big_hole = [(10, 10), (18, 10), (18, 18), (10, 18), (10, 10)]  # area 64
    poly = Polygon(shell=shell, holes=[small_hole, big_hole])

    out = _remove_holes(poly, min_hole_area_m2=50.0)

    assert len(out.interiors) == 1


def test_validate_basin_polygons_accepts_touching():
    poly_a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    poly_b = Polygon([(1, 0), (2, 0), (2, 1), (1, 1), (1, 0)])

    _validate_basin_polygons({'A': poly_a, 'B': poly_b})


def test_validate_basin_polygons_raises_on_overlap():
    poly_a = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    poly_b = Polygon([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)])

    try:
        _validate_basin_polygons({'A': poly_a, 'B': poly_b})
    except ValueError as err:
        assert 'Overlapping basin polygons' in str(err)
    else:
        raise AssertionError('Expected overlap validation to fail')


def test_validate_basin_polygons_allows_configured_overlap():
    poly_a = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    poly_b = Polygon([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)])

    allowed = _compute_pairwise_overlap_areas({'A': poly_a, 'B': poly_b})
    _validate_basin_polygons(
        {'A': poly_a, 'B': poly_b},
        allowed_overlap_by_pair=allowed,
    )


def test_resolve_new_overlaps_removes_excess_overlap():
    poly_a = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    poly_b = Polygon([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)])

    repaired = _resolve_new_overlaps(
        {'A': poly_a, 'B': poly_b},
        allowed_overlap_geom_by_pair={},
    )

    assert repaired['A'].intersection(repaired['B']).area == 0.0


def test_resolve_new_overlaps_preserves_allowed_overlap():
    baseline_a = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    baseline_b = Polygon([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)])
    allowed = _compute_pairwise_overlap_geometries(
        {'A': baseline_a, 'B': baseline_b}
    )

    # Add extra overlap that should be removed while keeping baseline overlap.
    poly_a = Polygon([(0, 0), (2.5, 0), (2.5, 2.5), (0, 2.5), (0, 0)])
    poly_b = baseline_b

    repaired = _resolve_new_overlaps(
        {'A': poly_a, 'B': poly_b},
        allowed_overlap_geom_by_pair=allowed,
    )

    overlap_area = repaired['A'].intersection(repaired['B']).area
    allowed_area = allowed[('A', 'B')].area
    assert np.isclose(overlap_area, allowed_area)


def test_enforce_topological_partition_removes_overlaps():
    poly_a = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    poly_b = Polygon([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)])

    repaired, _ = _enforce_topological_partition({'A': poly_a, 'B': poly_b})

    overlap = repaired['A'].intersection(repaired['B']).area
    assert np.isclose(overlap, 0.0)


def test_enforce_topological_partition_uses_reference_labels():
    source = {
        'A': Polygon(
            shell=[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
            holes=[[(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]],
        ),
        'B': Polygon([(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]),
    }
    reference = {
        'A': Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]),
        'B': MultiPolygon([]),
    }

    repaired, _ = _enforce_topological_partition(
        source,
        reference_polygons=reference,
    )

    a_geoms = (
        repaired['A'].geoms
        if repaired['A'].geom_type == 'MultiPolygon'
        else [repaired['A']]
    )
    a_holes = sum(len(poly.interiors) for poly in a_geoms)
    assert a_holes == 0


def test_validate_partition_topology_detects_gaps():
    target = Polygon([(0, 0), (2, 0), (2, 1), (0, 1), (0, 0)])
    basins = {
        'A': Polygon([(0, 0), (0.9, 0), (0.9, 1), (0, 1), (0, 0)]),
        'B': Polygon([(1.1, 0), (2, 0), (2, 1), (1.1, 1), (1.1, 0)]),
    }

    try:
        _validate_partition_topology(basins, target_domain=target)
    except ValueError as err:
        assert 'uncovered gaps' in str(err)
    else:
        raise AssertionError('Expected partition gap validation to fail')


def test_validate_partition_topology_passes_on_clean_partition():
    target = Polygon([(0, 0), (2, 0), (2, 1), (0, 1), (0, 0)])
    basins = {
        'A': Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
        'B': Polygon([(1, 0), (2, 0), (2, 1), (1, 1), (1, 0)]),
    }

    _validate_partition_topology(basins, target_domain=target)


def test_validate_partition_topology_detects_basin_holes():
    target = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    hole = [(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]
    basins = {
        'A': Polygon(
            shell=[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
            holes=[hole],
        )
    }

    try:
        _validate_partition_topology(basins, target_domain=target)
    except ValueError as err:
        assert 'basin interior holes' in str(err)
    else:
        raise AssertionError('Expected basin-hole validation to fail')


def test_cli_parser_defaults():
    parser = _build_arg_parser()
    args = parser.parse_args([])

    assert args.simplify_tolerance_m == 0.0
    assert args.remove_extension_holes
    assert args.min_hole_area_m2 == 0.0
    assert args.out_shapefile == 'imbie2/extended_basin_polygons.shp'
    assert args.debug_raster_only_shapefile is None
    assert args.debug_raster_only_simplified_shapefile is None
    assert args.validate


def test_cli_parser_toggle_flags():
    parser = _build_arg_parser()
    args = parser.parse_args(['--keep-extension-holes', '--no-validate'])

    assert not args.remove_extension_holes
    assert not args.validate


def test_cli_parser_scientific_notation_values():
    parser = _build_arg_parser()
    args = parser.parse_args(
        ['--simplify-tolerance-m', '20e3', '--min-hole-area-m2', '400e6']
    )

    assert args.simplify_tolerance_m == 20000.0
    assert args.min_hole_area_m2 == 400000000.0


def test_cli_parser_debug_raster_only_shapefile():
    parser = _build_arg_parser()
    args = parser.parse_args(
        ['--debug-raster-only-shapefile', 'imbie2/raster_only_debug.shp']
    )

    assert args.debug_raster_only_shapefile == 'imbie2/raster_only_debug.shp'


def test_cli_parser_debug_raster_only_simplified_shapefile():
    parser = _build_arg_parser()
    args = parser.parse_args(
        [
            '--debug-raster-only-simplified-shapefile',
            'imbie2/raster_only_simplified_debug.shp',
        ]
    )

    assert (
        args.debug_raster_only_simplified_shapefile
        == 'imbie2/raster_only_simplified_debug.shp'
    )
