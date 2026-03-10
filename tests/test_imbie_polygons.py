import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from i7aof.imbie.masks import BASIN_DEFINITIONS
from i7aof.imbie.polygons import (
    _build_arg_parser,
    _compute_pairwise_overlap_areas,
    _compute_pairwise_overlap_geometries,
    _connected_component_sizes,
    _enforce_topological_partition,
    _label_to_polygon,
    _normalize_polygonal_geometry,
    _reassign_small_enclosed_slivers,
    _remove_holes,
    _resolve_new_overlaps,
    _uniform_spacing,
    _validate_basin_polygons,
    _validate_partition_topology,
    _validate_raster_basin_connectivity,
    _validate_single_polygon_basins,
    _write_failure_debug_shapefile,
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


def test_connected_component_sizes_counts_components():
    mask = np.array(
        [
            [True, True, False],
            [False, False, False],
            [False, True, True],
        ]
    )

    sizes = _connected_component_sizes(mask)

    assert sizes == [2, 2]


def test_validate_raster_basin_connectivity_passes_contiguous():
    n_basins = len(BASIN_DEFINITIONS)
    basin_number = np.full((n_basins, 1), fill_value=-1, dtype=int)
    for basin_id in range(n_basins):
        basin_number[basin_id, 0] = basin_id

    _validate_raster_basin_connectivity(
        basin_number,
        stage='unit-test-raster',
    )


def test_validate_raster_basin_connectivity_rejects_disconnected():
    n_basins = len(BASIN_DEFINITIONS)
    basin_number = np.full((20, 1), fill_value=-1, dtype=int)
    for basin_id in range(n_basins):
        basin_number[basin_id, 0] = basin_id

    # Split basin 0 into two disconnected components.
    basin_number[17, 0] = 0

    try:
        _validate_raster_basin_connectivity(
            basin_number,
            stage='unit-test-raster',
        )
    except ValueError as err:
        assert 'disconnected basin labels found' in str(err)
    else:
        raise AssertionError('Expected disconnected-raster validation to fail')


def test_validate_raster_basin_connectivity_diagonal_not_connected():
    n_basins = len(BASIN_DEFINITIONS)
    basin_number = np.full((20, 2), fill_value=-1, dtype=int)
    for basin_id in range(n_basins):
        basin_number[basin_id, 0] = basin_id

    # Basin 0 has only diagonal touch between cells, which is disconnected
    # under 4-neighbor connectivity.
    basin_number[17, 0] = 0
    basin_number[18, 1] = 0

    try:
        _validate_raster_basin_connectivity(
            basin_number,
            stage='unit-test-raster',
        )
    except ValueError as err:
        assert 'disconnected basin labels found' in str(err)
    else:
        raise AssertionError('Expected diagonal-disconnect validation to fail')


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


def test_reassign_small_enclosed_slivers_moves_island_to_surrounding_basin():
    donor_main = Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])
    donor_island = Polygon(
        [(8.0, 8.0), (8.2, 8.0), (8.2, 8.2), (8.0, 8.2), (8.0, 8.0)]
    )
    donor = MultiPolygon([donor_main, donor_island])
    recipient = Polygon([(6, 6), (10, 6), (10, 10), (6, 10), (6, 6)])

    out = _reassign_small_enclosed_slivers(
        {'A': donor, 'B': recipient},
        max_sliver_area_m2=1.0,
        max_sliver_fraction=0.1,
    )

    assert isinstance(out['A'], Polygon)
    assert np.isclose(out['A'].area, donor_main.area)
    point = donor_island.representative_point()
    assert not out['A'].covers(point)
    assert out['B'].covers(point)


def test_reassign_small_enclosed_slivers_falls_back_to_nearest_basin():
    donor_main = Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])
    donor_island = Polygon(
        [(8.0, 8.0), (8.2, 8.0), (8.2, 8.2), (8.0, 8.2), (8.0, 8.0)]
    )
    donor = MultiPolygon([donor_main, donor_island])
    # Not containing the island, but close enough for nearest-distance
    # fallback.
    recipient = Polygon(
        [(8.4, 7.6), (10.0, 7.6), (10.0, 9.6), (8.4, 9.6), (8.4, 7.6)]
    )

    out = _reassign_small_enclosed_slivers(
        {'A': donor, 'B': recipient},
        max_sliver_area_m2=1.0,
        max_sliver_fraction=0.1,
        containment_buffer_m=0.0,
        max_nearest_distance_m=1.0,
    )

    point = donor_island.representative_point()
    assert not out['A'].covers(point)
    assert out['B'].distance(point) == 0.0


def test_validate_basin_polygons_accepts_touching():
    poly_a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    poly_b = Polygon([(1, 0), (2, 0), (2, 1), (1, 1), (1, 0)])

    _validate_basin_polygons({'A': poly_a, 'B': poly_b})


def test_write_failure_debug_shapefile_writes_projection(monkeypatch):
    calls = []

    def _fake_write_shapefile(path, basin_polygons):
        calls.append(('shp', path, basin_polygons))

    def _fake_write_projection_file(*, out_shapefile, source_prj):
        calls.append(('prj', out_shapefile, source_prj))

    monkeypatch.setattr(
        'i7aof.imbie.polygons._write_shapefile',
        _fake_write_shapefile,
    )
    monkeypatch.setattr(
        'i7aof.imbie.polygons._write_projection_file',
        _fake_write_projection_file,
    )

    basin_polygons = {'A': Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])}
    out_path = 'imbie2/debug_failed_example.shp'
    _write_failure_debug_shapefile(out_path, basin_polygons)

    assert calls[0][0] == 'shp'
    assert calls[0][1] == out_path
    assert calls[1][0] == 'prj'
    assert calls[1][1] == out_path
    assert calls[1][2].endswith('ANT_Basins_IMBIE2_v1.6.prj')


def test_validate_basin_polygons_raises_on_overlap():
    poly_a = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    poly_b = Polygon([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)])

    try:
        _validate_basin_polygons({'A': poly_a, 'B': poly_b})
    except ValueError as err:
        assert 'Overlapping basin polygons' in str(err)
    else:
        raise AssertionError('Expected overlap validation to fail')


def test_validate_basin_polygons_rejects_multipolygon():
    poly_a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    poly_b = Polygon([(3, 0), (4, 0), (4, 1), (3, 1), (3, 0)])
    multi = MultiPolygon([poly_a, poly_b])

    try:
        _validate_basin_polygons({'A': multi})
    except ValueError as err:
        assert 'expected single Polygon per basin' in str(err)
    else:
        raise AssertionError('Expected MultiPolygon validation to fail')


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


def test_validate_single_polygon_basins_rejects_empty_by_default():
    try:
        _validate_single_polygon_basins(
            {'A': Polygon()},
            stage='unit-test-stage',
        )
    except ValueError as err:
        assert 'empty basin geometries' in str(err)
    else:
        raise AssertionError('Expected empty-basin validation to fail')


def test_normalize_polygonal_geometry_collection_to_polygon():
    geom = _normalize_polygonal_geometry(
        MultiPolygon(
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1), (1, 0)]),
            ]
        )
    )

    assert geom.geom_type in ('Polygon', 'MultiPolygon')


def test_cli_parser_defaults():
    parser = _build_arg_parser()
    args = parser.parse_args([])

    assert args.simplify_tolerance_m == 0.0
    assert args.remove_extension_holes
    assert args.min_hole_area_m2 == 0.0
    assert args.out_shapefile == 'imbie2/extended_basin_polygons.shp'
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


def test_cli_parser_rejects_removed_debug_shapefile_flags():
    parser = _build_arg_parser()
    try:
        parser.parse_args(['--debug-raster-only-shapefile', 'foo.shp'])
    except SystemExit:
        pass
    else:
        raise AssertionError('Expected removed debug flag to be rejected')

    try:
        parser.parse_args(
            ['--debug-raster-only-simplified-shapefile', 'foo.shp']
        )
    except SystemExit:
        pass
    else:
        raise AssertionError(
            'Expected removed simplified debug flag to be rejected'
        )
