# Developer Guide

Guidance for contributors and maintainers.

## Glossary

- **Package**: An importable directory with an `__init__.py` file (e.g.,
  `i7aof.remap`). Used to organize code.
- **Module**: A single Python file (`.py`) or a package treated as a module
  when imported (e.g., `i7aof.grid.ismip`, or the package itself `i7aof.io`).
- **Command-line tool**: An installed console script users run (e.g.,
  `ismip7-antarctic-remap-cmip`). Defined under `[project.scripts]` in
  `pyproject.toml` and backed by Python functions.
- **Python API**: Public functions and classes intended for import (e.g.,
  `i7aof.remap.cmip.main`). These provide programmatic access to functionality
  used by the CLI.

```{toctree}
:maxdepth: 2
:caption: Contents

contributing
architecture
packages/index
cli
releasing
```
