# Fortran Extrapolation Tools

This directory contains the raw Fortran programs for horizontal and vertical
extrapolation of CMIP data on the ISMIP6 grid. They are intentionally kept
simple and currently remain outside the Python package build on PyPI. Conda
builds (or manual builds) can produce the executables.

## Building with CMake

```bash
mkdir -p build && cd build
cmake ..
cmake --build . --parallel
```

Upon success the executables will be located in `build/` (and installed to
`<prefix>/bin` if you run `cmake --install . --prefix <prefix>`):

* `i7aof_extrap_horizontal`
* `i7aof_extrap_vertical`

### Selecting NetCDF-Fortran
The `CMakeLists.txt` tries these discovery methods in order:
1. `pkg-config netcdf-fortran`
2. `nf-config` (classic helper script)
3. `find_package(NetCDF COMPONENTS Fortran)`

If discovery fails, set one of:
* Environment variable `NETCDF_ROOT` or `NETCDF_DIR` before configuring.
* Run `cmake -DNetCDF_ROOT=/path/to/netcdf ..` if a CMake config is present.

### Example
```bash
NETCDF_ROOT=$HOME/miniconda/envs/i7aof cmake ..
cmake --build .
```

## Namelist Usage
The programs read Fortran namelists:

* Horizontal: group `&horizontal_extrapolation`
* Vertical:   group `&vertical_extrapolation`

A combined template is shipped with the Python package as
`i7aof/extrap/namelist_template.nml.j2`.

## Future Directions
* Merge both steps into a single driver.
* Factor out shared utilities (error handling) into modules.
* Add lightweight regression tests with synthetic NetCDF fixtures.
