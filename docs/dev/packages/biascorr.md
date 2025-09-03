# i7aof.biascorr

Purpose: Bias-correction methods (time-slice statistics and projection
workflows) to align model fields with a reference climatology, producing
corrected temperature (`thetao`) and salinity (`so`) on the ISMIP grid.

## Public Python API (by module)

Import paths and brief descriptions by module:

- Module: {py:mod}`i7aof.biascorr.timeslice`
  - {py:class}`Timeslice <i7aof.biascorr.timeslice.Timeslice>`: A container
    and helper for extracting per-year (or mean) gridded T and S, computing
    3D volume, binning volume in T–S space by basin, and building per-basin
    bias anomalies relative to a model reference period.
    - Key methods:
      - {py:meth}`get_all_data()` — convenience to call T/S extraction,
        volume, and binning.
      - {py:meth}`get_T_and_S()` — load `thetao` and `so` for a given time index
        or the first time.
      - {py:meth}`get_volume()` — compute grid-cell volume from uniform `dx`,
        `dy`, `dz` based on the first differences of `x`, `y`, `z`.
      - {py:meth}`get_bins()` — compute per-basin 2D histograms in (S, T)
        space: `Vb`, with edges `Sb`, `Tb`.
      - {py:meth}`compute_delta(modref)` — compute filled per-basin anomalies
        `deltaSf`, `deltaTf` on the model-ref bins.
      - {py:meth}`apply_anomaly(base)` — map filled anomalies back to 3D to
        produce `S_corrected`, `T_corrected` using the “base” fields.
    - Notable attributes: `T`, `S`, `V`, `Vb`, `Sb`, `Tb`, `deltaSf`,
      `deltaTf`, `S_corrected`, `T_corrected`, `Nbins`, `Nbasins`.

- Module: {py:mod}`i7aof.biascorr.projection`
  - {py:class}`Projection <i7aof.biascorr.projection.Projection>`: Orchestrates
    the bias-correction workflow across reference, model-reference, base
    (extrapolated), and future model periods; constructs continental-shelf
    basin masks; loops over years to write corrected outputs.
    - Key methods:
      - {py:meth}`read_reference()` — build `Timeslice` for reference (`ref`),
        model reference (`modref`), and base (`base`) and compute their data.
      - {py:meth}`compute_bias()` — compute salinity and temperature bin bias
        corrections (`Sc`, `Tc`) on the model reference bins.
      - {py:meth}`read_model()` — iterate over all model files in the future
        period, apply anomalies per time index, and write corrected outputs.
      - {py:meth}`read_model_timeslice()` — one-year helper used by
        `read_model()`.
      - {py:meth}`create_basin_masks()` — construct per-basin masks over the
        continental shelf using topography and IMBIE basins.
      - {py:meth}`plot_TS_diagrams(filename)` — quick-look T–S histograms per
        basin for ref/modref/bias-corrected bins.
    - Important attributes: `ref`, `modref`, `base`, `basinNumber`,
      `basinmask`, `basinmask_extrap`, `Nbins`, input filename patterns.

## Required config options

Section `[biascorr]` is required. Keys used by the workflow:

- Input datasets (NetCDF files or glob patterns):
  - `thetao_ref`, `so_ref` — reference dataset over the reference period
    (observational or reanalysis), on the ISMIP grid.
  - `thetao_modref`, `so_modref` — model over the reference period (non-
    extrapolated). Used to build model reference bins and anomalies.
  - `thetao_base`, `so_base` — model over the reference period with
    extrapolated fields, used as the base to which anomalies are applied.
  - `thetao_mod`, `so_mod` — model over the target/future period to be
    corrected (glob patterns; yearly or chunked files expected).

- Ancillary inputs and parameters:
  - `filename_topo` — ISMIP topography file (output of {doc}`topo`); used to
    mask out deep ocean and non-shelf regions.
  - `filename_imbie` — IMBIE basin mask file (output of {doc}`imbie`), with
    variable `basinNumber`.
  - `z_shelf` (float, meters) — maximum bed elevation for “continental shelf”
    mask (e.g., -1500 m).
  - `Nbins` (int) — number of equally spaced bins for both S and T per basin.

Assumptions:

- Input variables are named `thetao` and `so`, with dims `(time, z, y, x)`.
- Horizontal and vertical spacing are uniform and read from `x`, `y`, `z`
  coords. Volume currently ignores partial cell fractions (see Internals).

## Outputs

- Corrected NetCDF files written by {py:meth}`Projection.read_model` in the
  current working directory, one pair per input chunk:
  - `thetao_corrected_{startYear}_{endYear}.nc`
  - `so_corrected_{startYear}_{endYear}.nc`
- Optional: T–S diagram figure from
  {py:meth}`Projection.plot_TS_diagrams(filename)`.

## Data model

- Gridded variables: `T(thetao)`, `S(so)`, 3D volume `V(z,y,x)`.
- Per-basin T–S histograms: `Vb(Nbasins,Nbins,Nbins)` with bin edges
  `Sb(Nbasins,Nbins+1)`, `Tb(Nbasins,Nbins+1)`.
- Filled per-basin anomalies on model-ref bins: `deltaSf`, `deltaTf`.
- Corrected gridded fields: `S_corrected(z,y,x)`, `T_corrected(z,y,x)`.
- Basin metadata: `basinNumber(y,x)`, masks `basinmask(Nbasins,y,x)` and
  `basinmask_extrap(Nbasins,y,x)`.

## Runtime and external requirements

- Core: `numpy`, `xarray`, `tqdm`, `matplotlib` (for plotting).
- Inputs are expected on the ISMIP grid; see {doc}`grid`, {doc}`topo`, and
  {doc}`imbie` for producing required ancillary files.
- For the authoritative conda-forge environment, see `dev-spec.txt` (note
  `pyproject.toml` lists a PyPI-only subset).

## Usage

High-level workflow to bias-correct a model period:

```python
from i7aof.biascorr.projection import Projection

proj = Projection(config, logger)
proj.read_reference()   # loads ref, modref, base and computes bins/volumes
proj.compute_bias()     # derives per-basin Sc and Tc on model-ref bins
proj.read_model()       # loops over future files; writes corrected outputs

# Optional quick-look TS plots
proj.plot_TS_diagrams('ts_diagrams.png')
```

Notes:

- Ensure `filename_topo` and `filename_imbie` point to files on the same
  ISMIP grid as the inputs.
- Use sensible `z_shelf` so shelf basins contain finite volume.

## Internals (for maintainers)

- Timeslice volume and histograms:
  - Volume assumes uniform `dx, dy, dz` computed from the first grid steps; a
    TODO notes applying cell fractions in the future.
  - `get_bins()` builds 2D histograms in (S, T) per basin, weighting by grid
    cell volume masked by `basinmask`.
- Anomaly computation and filling:
  - `compute_delta(modref)` forms volume-weighted anomalies on the model-ref
    bins and fills missing bins iteratively by averaging neighbor bins until
    full coverage.
- Bias curves (Sc, Tc):
  - Salinity: per-basin search over scalings in [0.5, 1.5) to best match the
    reference S PDF while anchoring the chosen percentile (`perc=99` by
    default).
  - Temperature: scale the (T - Tmin) distribution so the `perc` percentile
    matches reference; anchor at Tmin.
- Apply to 3D fields: anomalies are looked up by bin indices derived from the
  base fields per basin and added to produce corrected fields.
- Basin masks: `create_basin_masks()` reads topography and IMBIE basins; the
  shelf mask uses `bed > z_shelf` to exclude deep ocean; an “extrapolated” mask
  includes full basins for applying anomalies to base fields.

## Edge cases / validations

- File pairing: `read_model()` assumes each `thetao` file has a matching `so`
  file via name replacement. An assertion will fail otherwise.
- Dimensions and names: variables must be named `thetao` and `so` and share the
  same time coordinate and (x,y,z) grids. The code uses `xr.testing.assert_equal`
  on `time` to enforce matching time axes.
- Uniform spacing: volume assumes uniform spacing; highly nonuniform grids
  aren’t supported in the current implementation.
- Binning robustness: empty bins lead to NaNs which are then filled
  iteratively; extreme cases could require tuning `Nbins`.

## Extension points

- Allow custom output paths/filenames for corrected fields instead of the
  current hard-coded names.
- Incorporate partial-cell fractions and exact grid metrics in volume.
- Support alternative binning strategies, percentiles, and scaling ranges.
- Generalize variable names/dimension names or pass them via config.
