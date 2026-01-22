# Contributing

We welcome contributions! Please open issues and pull requests.

- Follow PEP8 and the configured Ruff rules.
- Add or update tests when changing behavior.
- Keep public APIs documented.
- Small PRs are easier to review.

## Dev setup

Set up your environment using the unified Getting Started guide:

- [Getting Started](../getting-started.md)

That page covers prerequisites, local package builds (including Fortran executables), environment creation, editable install, and verification.

If you don't already have a conda base installation set up, we recommend
starting with [Miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#requirements-and-installers).

To activate code linting/formatting hooks (required), ensure pre-commit is installed and run:

```bash
pre-commit install
```

Build docs locally when editing documentation:

```bash
sphinx-build -b html docs docs/_build/html
```

Add test scripts to `scripts/` that can be used to smoke test features.

## Running tests

Use pytest from the repository root:

```bash
pytest -q
```

## Fortran changes

If you modify the Fortran sources under `fortran/`, follow the “Rebuilding after Fortran source changes” section in the Getting Started guide to rebuild and refresh your environment. There is no Python fallback for the extrapolation step.
