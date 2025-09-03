# Contributing

We welcome contributions! Please open issues and pull requests.

- Follow PEP8 and the configured Ruff rules.
- Add or update tests when changing behavior.
- Keep public APIs documented.
- Small PRs are easier to review.

## Dev setup

Use conda-forge for the environment and create it directly from `dev-spec.txt`:

```bash
conda create -n ismip7_dev -c conda-forge --file dev-spec.txt
conda activate ismip7_dev
pip install --no-deps --no-build-isolation -e .
pre-commit install
```

If you don't already have a conda base installation set up, we recommend
starting with [Miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#requirements-and-installers).

Build docs locally when editing documentation:

```bash
sphinx-build -b html docs docs/_build/html
```

Add test scripts to `scripts/` that can be used to smoke test features.
