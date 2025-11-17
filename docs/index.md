---
hide-toc: true
---

# ISMIP7 Antarctic Ocean Forcing (i7aof)

Welcome to the documentation for the `i7aof` package — tools to produce Antarctic ocean forcing for ISMIP7.

::: {card} Getting Started
:link: getting-started
:link-type: doc

One canonical install path for all users, including the required Fortran executables.
:::

::: {card} User Guide
:link: user/index
:link-type: doc

Install, configure data paths, run workflows, and reproduce datasets.
:::

::: {card} Developer Guide
:link: dev/index
:link-type: doc

Contribute code, design choices, testing, and release process.
:::

::: {card} API Reference
:link: api/index
:link-type: doc

Public modules, functions, and CLI entry points.
:::

::: {card} Changelog
:link: changelog
:link-type: doc

Notable changes between releases.
:::

:::

## About

`i7aof` aims to provide a reproducible toolkit to:

- download and remap CMIP output and other inputs to ISMIP grids;
- compute annual means from monthly data;
- extrapolate fields across shelves, cavities, ice, and bathymetry;
- perform bias correction using observational climatologies;
- provide helper scripts for MeltMIP-style experiments.

This project is under active development; interfaces may change.

```{toctree}
:hidden:

getting-started
user/index
dev/index
api/index
changelog
```
