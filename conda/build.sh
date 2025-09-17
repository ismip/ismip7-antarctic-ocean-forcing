#!/usr/bin/env bash
set -euo pipefail

# Configure and build Fortran tools
pushd fortran
mkdir -p build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DBUILD_I7AOF_EXTRAP=ON
cmake --build . --parallel ${CPU_COUNT:-2}
cmake --install .
popd

# Install Python package (pure python; Fortran already installed to prefix)
$PYTHON -m pip install . --no-deps -vv
