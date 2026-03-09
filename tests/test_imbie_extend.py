import numpy as np

from i7aof.imbie.extend import (
    component_island_stats,
    count_basin_components,
    enforce_basin_contiguity,
)


def test_count_basin_components_simple_case():
    basin = np.array(
        [
            [0, 0, 1],
            [0, 1, 1],
            [2, 2, 2],
        ],
        dtype=int,
    )

    counts = count_basin_components(basin_number=basin, num_basins=3)

    assert counts == {0: 1, 1: 1, 2: 1}


def test_component_island_stats_reports_island_cells():
    basin = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=int,
    )

    stats = component_island_stats(basin_number=basin, num_basins=2)

    assert stats[0]['components'] == 2
    assert stats[0]['largest_component_cells'] == 8
    assert stats[0]['island_cells'] == 1


def test_enforce_basin_contiguity_moves_single_cell_island():
    basin = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 0],
        ],
        dtype=int,
    )

    repaired = enforce_basin_contiguity(basin_number=basin, num_basins=2)

    assert repaired[1, 1] == 0
    counts = count_basin_components(basin_number=repaired, num_basins=2)
    assert counts == {0: 1, 1: 1}


def test_enforce_basin_contiguity_reassigns_disconnected_island():
    basin = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=int,
    )

    repaired = enforce_basin_contiguity(basin_number=basin, num_basins=2)

    assert repaired[2, 3] == 1
    counts = count_basin_components(basin_number=repaired, num_basins=2)
    assert counts == {0: 1, 1: 1}
