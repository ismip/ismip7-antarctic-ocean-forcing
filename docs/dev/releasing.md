# Releasing

This project targets conda-forge as the primary distribution channel.

## Prep

1. Update `i7aof/version.py` and `CHANGELOG.md`.
2. Ensure CI is green on `main`.
3. Create a release candidate branch or PR if needed.

## Tag

1. Tag the release in the repo, e.g., `v0.1.0`.
2. Push the tag to GitHub.

## conda-forge feedstock

1. Open (or let the bot open) a PR to the conda-forge feedstock updating the version and SHA256.
2. Review CI (Linux/macOS/Win) and merge when green.
3. Verify the new package version is available via `conda search -c conda-forge ismip7-antarctic-ocean-forcing`.

## Documentation

1. GitHub Pages is built from `docs/` via CI. After tagging, ensure the docs workflow completes.
2. Confirm the site updates and reflects the new version.

Note: PyPI publishing is not planned currently.
