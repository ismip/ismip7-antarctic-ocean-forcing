# Re-export common coordinate helpers for convenience
from i7aof.coords import (  # noqa: F401
    attach_grid_coords,
    dataset_with_var_and_bounds,
    propagate_time_from,
    strip_fill_on_non_data,
)
from i7aof.version import __version__  # noqa: F401

__all__ = [
    'attach_grid_coords',
    'dataset_with_var_and_bounds',
    'propagate_time_from',
    'strip_fill_on_non_data',
]
