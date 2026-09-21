# SuNeRF EUV density/temperature pipeline modernization

Date: 2026-08-28

> The per-point Gaussian temperature distribution described below was replaced
> on 2026-09-19 by one temperature and density per point with a response lookup
> table and consistent H/He absorption; see `plasma_pointwise_plan.md`. The ray,
> sampling, response-folding, data, and artifact infrastructure is unchanged.

## Executive outcome

The pipeline behind `config/plasma/all_2012_08.yaml` has completed its
correctness and modernization pass. The audit found the following defects in
the historical implementation:

- invalid EUV target pixels are retained by the dataset filter;
- configured time shuffling also runs during validation and inference;
- the implicit time normalization makes the initial jitter approximately 500
  days for a data sequence spanning days;
- the plasma line-of-sight quadrature overcounts the first interval;
- reported column density and DEM products are not their named physical
  quantities;
- response functions, image units, LOS units, and density scale do not close a
  physical unit equation;
- the response files mix different CHIANTI versions, ionization equilibria, and
  abundance assumptions;
- current Lightning cannot run the legacy validation hook, and the plasma
  evaluation loader has several immediate failures.

The implementation now uses the relevant SuNeRF-CME/Thomson patterns: exact
standard-WCS rays, shared nonuniform quadrature, cache fingerprints, current
Lightning strategy handling, deterministic distributed validation, explicit
instrument metadata, and strict preparation boundaries. Plasma and Thomson
physics remain separate renderers over the same ray/sampling/runtime core.

PSI synthetic observations no longer fit a neural surrogate before ray tracing.
A differentiable 4-D spherical-grid interpolator exposes the simulation through
the regular SuNeRF rendering interface. Density defaults to physical `cm^-3`,
LOS distances are converted to centimeters, and versioned responses cannot be
arbitrarily renormalized.

### Implemented status

| Contract | Implemented outcome |
|---|---|
| Configuration | One validated plasma schema v2; explicit distance/time/density/temperature units and deterministic holdout |
| Preparation | Unified AIA, EUVI-A/B, and EUI adapters producing calibrated prepared-EUV-v2 FITS and masks |
| Responses | Atomic version-1 NPZ artifacts, semantic response IDs, a shared-CHIANTI-basis folding API/CLI, unit and composition validation |
| Rays/sampling | Exact WCS directions, finite/validity masks, correct shell intersections, train-only perturbations, physical time jitter |
| Image formation | Normalized temperature PDF/DEM, temperature and LOS quadrature, `cm` path length, correct EM/DEM/column-density diagnostics |
| Training | Current Lightning hooks/devices/DDP, masked channel-balanced losses, physical regularization scales, identifiable gain corrections |
| Runtime/artifacts | Fingerprinted atomic caches, response/unit/channel checks, weights-only-safe states, artifact-driven evaluation |
| Synthetic PSI | Direct joint density/temperature grid interpolation with periodic longitude and regular SuNeRF rendering |
| Retirement | Scalar-emission runner/model/renderer and inconsistent response/preparation entry points are explicit tombstones |

The remaining science input is external by design: a released reconstruction
must supply immutable CHIANTI spectral-emissivity and instrument-throughput
artifacts made from authoritative calibration files. Those large database and
calibration products are not bundled in this repository.

## Scope and evidence

This audit traced code, configuration, bundled response products, preparation
scripts, tests, recent CME changes, and official instrument/CHIANTI guidance.
The absolute `/glade` inputs referenced by the 2012 configuration are not mounted
in this workspace, so their FITS headers and the exact configured response NPZ
files could not be inspected numerically. Findings that depend on those external
files are identified as provenance risks rather than asserted data defects.

## 1. How EUV temperature-response tomography works

For a pixel from channel \(c\), SuNeRF constructs a ray

\[
\mathbf{x}(s)=\mathbf{o}+s\mathbf{d}
\]

from the observer and the pixel's helioprojective world coordinate. It samples
that ray inside a spherical reconstruction shell and evaluates a shared neural
field at \((x,y,z,t)\).

The plasma model predicts three values per point: physical electron density, a
mean log-temperature, and a temperature width. Those values define a normalized
Gaussian temperature PDF on the explicitly configured grid. For instrument
channel \(c\), the local optically thin emissivity is

\[
\epsilon_c(s)=10^{g_c} n_e(s)^2\sum_j K_c(T_j,n_e)p_j(s)\,w_j,
\]

where \(K_c(T,n_e)\) is the channel temperature response, \(w_j\) is the
nonuniform \(d\log T\) quadrature weight, and \(g_c\) is an optional constrained
relative-channel gain. With absorption disabled in this configuration, image
formation is

\[
I_c=\int_{\mathrm{LOS}}\epsilon_c(s)\,ds.
\]

Training compares synthesized and observed channels after the same reversible
asinh ML transform. Invalid values remain masked. The shared neural field is
constrained simultaneously by AIA, EUVI-A, and EUVI-B viewing geometries.

### Implemented physical state

- predict physical electron density \(n_e\) in cm\(^{-3}\);
- predict a normalized temperature PDF \(p(\log T)\), initially a single
  Gaussian with \(\int p\,d\log T=1\);
- form
  \(\epsilon_c=n_e^2\int K_c(T,n_e)p(\log T)\,d\log T\);
- integrate with physical \(dl\) in cm and a common quadrature implementation;
- introduce a richer DEM basis only if residuals demonstrate that a unimodal
  temperature distribution is inadequate.

LOS quadrature converts model distances to centimeters. This formulation is
stable under temperature-grid refinement and gives density, DEM, emission
measure, and column density unambiguous meanings.

## 2. Exact trace of `config/plasma/all_2012_08.yaml`

The supported path is:

```text
config/plasma/all_2012_08.yaml
  -> validated schema v2 / normalized immutable config
  -> sunerf/run_plasma.py + fingerprinted DataModuleCache
  -> MultiInstrumentDataModule
       -> AIADataSet / EUVIDataSet(A) / EUVIDataSet(B)
       -> prepared FITS discovery + unique channel matching
       -> PreparedEUVObservation (image, mask, units, WCS)
       -> exact WCS rays + deterministic train/validation separation
  -> PlasmaSuNeRFModule
       -> PlasmaModel (shared physical-density SIREN field)
       -> valid shell intersections + spherical/hierarchical sampling
       -> PlasmaRadiativeTransfer per immutable response artifact
       -> masked, channel-balanced loss + calibrated regularizers
  -> Lightning checkpoint + atomic weights-only-safe artifact
  -> artifact-driven SuNeRFLoader / evaluation
```

`scripts/plasma.sh` now launches this canonical configuration. YAML overrides
are parsed structurally and the full schema/cross-field unit and channel
contract is validated before paths, caches, loggers, or models are created.

### Resolved configuration

| Item | Resolved behavior |
|---|---|
| AIA | 94, 131, 171, 193, 211, 335 Å; physical response artifact; constrained relative gains with A193 fixed |
| EUVI-A | 171, 195, 284 Å; physical response artifact; constrained relative gains with 195 fixed |
| EUVI-B | 171, 195, 284 Å; physical response artifact; constrained relative gains with 195 fixed |
| Reconstruction shell | 1.0–1.5 solar radii |
| Sampling | 64 coarse spherical samples plus the implicit default 128 hierarchical samples |
| Temperature grid | Explicit log10(T/K) 4–9, spacing 0.05 in schema |
| Time unit | Explicit `seconds_per_dt=86400` (one day per model-time unit) |
| Time shuffler | Train-only physical jitter, 43200 s down to 300 s |
| Absorption | disabled |
| Batch | 1024 training rays; 2048 validation rays |
| Network | four inputs, three outputs, width 512, eight SIREN layers |
| Optimizer | Adam with an explicit `1e-3 -> 1e-4` schedule over one million steps |
| Validation/checkpoint | deterministic held-out validation and atomic artifacts |

### Data path

The acquisition period remains 2012-08-01 through 2012-08-25. AIA preparation
uses pinned local aiapy pointing and correction tables, update-pointing,
registration, degradation correction, and exposure normalization. EUVI accepts
only an externally calibrated SECCHI product with an explicit spacecraft,
product level, calibration ID, unit, and exposure. EUI accepts a validated L2
product. Every path then uses the same geometry, mask, and FITS writer.

Prepared observations are discovered directly from configured FITS paths.
Channels are matched one-to-one within the configured tolerance; files cannot
be reused. Shape, sampled WCS, channel order, units, schema, and calibration
metadata are verified. Response compatibility is checked later when the loader
and renderer are assembled. Prepared FITS and cached targets stay in calibrated
units; the loss divides targets and physical predictions by one fixed
per-channel divisor of the instrument (`instruments[].scaling.divisor`).

The center holdout is selected deterministically and excluded from every
training dataset rather than being used for both optimization and validation.

### Rays and rendering

For the standard solar WCS path, `MapDataLoader` gets every pixel's world
coordinate from SunPy, constructs the observer pose in Carrington coordinates,
and converts the exact spherical helioprojective direction into the model frame.
A numerical comparison against Astropy/SunPy directions had a maximum finite
component error of approximately `1.3e-7`.

The spherical sampler computes the visible forward shell segment, terminates at
the near photosphere when appropriate, and returns an explicit validity mask for
misses or malformed rays. Perturbation and time shuffling occur only in training.
Hierarchical weights contain the physical emissivity contribution and
nonuniform path width.

The response loader verifies a versioned artifact, selects the configured
ordered channels, interpolates in log-temperature, and sets values outside
native support to zero. The renderer applies temperature quadrature, optional
density-response interpolation, abundance-consistent `n_H/n_e`, and nonuniform
LOS quadrature in centimeters.

### Loss, validation, and output

The loss applies the configured asinh transform to prediction and target, masks
invalid elements, averages each available channel, then applies explicit channel
and instrument weights. Static derivatives operate on dimensionless log density
and cannot cancel temperature derivatives. Radial density and calibration-gain
regularizers use explicit physical/prior scales.

The run writes Lightning checkpoints and one atomic weights-only reconstruction
artifact containing state dictionaries plus primitive
construction metadata. It records ordered channels, units, masks/scales,
temperature grid, response paths/IDs/hashes/provenance, source/config
fingerprints, and instrument routing. Plasma evaluation does not load executable
pickled modules.

## 3. Historical correctness audit and implemented corrections

The tables below preserve the evidence that motivated the refactor. Every listed
P0/P1 correction is implemented and covered by focused regression tests.

### Historical P0 defects

| Area | Confirmed defect | Consequence | Required correction |
|---|---|---|---|
| Invalid pixels | `TensorsDataset` drops a row only when every tensor is invalid. Since time is finite, NaN EUV targets survive. | NaN training loss or invalid supervision. | Carry an explicit per-channel validity mask; require finite rays/time; use masked losses and metrics. |
| Validation/inference | `BasicRenderingModule.forward()` applies the time shuffler regardless of `self.training`. | Validation, sanity checks, and inference evaluate different times from those requested. | Apply stochastic shuffling only in training; do not mutate caller batches in place. |
| Time scale | Plasma silently defaults to ten days per normalized unit while initial jitter is 50 units. | Initial samples are perturbed by roughly 500 days. | Make time scale required in schema; use seconds internally; set jitter in physical units. |
| Lightning | Plasma implements removed `validation_epoch_end`, logs a dict with `self.log`, uses legacy `dp`, and passes `devices=0` on CPU. | Current Lightning rejects or misconfigures the run. | Port the plasma module/runner to the current CME base hooks and strategy/device logic. |
| LOS integral | Plasma duplicates the first ray interval as an extra node width. | A constant three-node field on `[0,1]` is 50% too bright. | Share the composite trapezoidal node weights already used by the Thomson renderer. |
| Units/products | LOS lengths remain in model units; `total_ne` omits `dl`; `dem` sums `n_e`, not `n_e^2 dl`. | Physical labels and exported products are wrong. | Use cm path lengths and typed, tested output definitions. Remove misleading legacy names. |
| Evaluation | `PlasmaSuNeRFLoader` references undefined `device`. | Specialized plasma inference cannot instantiate. | Load through the base loader, then consume `self.state`. |
| Instrument dispatch | `load_image`/`load_observer_image` do not forward `instrument_key` to pose construction. | Non-default renders silently use the first response. | Carry one typed instrument/channel identity through pose, render, and map creation. |
| Channel metadata | Evaluation hard-codes seven AIA/four EUVI channels while this config produces six/three. | Index errors and mislabeled maps/video. | Persist ordered channel IDs and never infer them in evaluation. |

Relevant locations include `sunerf/data/loader/base_loader.py:355`,
`sunerf/rendering/base_tracing.py:184`, `sunerf/model/plasma.py:214`,
`sunerf/rendering/plasma.py:55`, and `sunerf/evaluation/loader.py:78-236,621-626`.

### Historical P1 defects and risks

| Area | Defect or unacceptable risk | Correction |
|---|---|---|
| Response interpolation | Interpolation is in linear T with constant endpoint extrapolation. | Validate/sort source axes, interpolate on log T, and use zero outside native support. |
| Response schema | The converter CLI writes aggregate/log arrays that its own runtime reader cannot consume. | Replace both with one versioned, validated response artifact. |
| Temperature discretization | Per-bin density has no normalized PDF/DEM definition or bin width. | Adopt the normalized temperature-PDF formulation above and test grid-refinement invariance. |
| Hierarchical PDF | Sampling weights omit interval width. | Use actual emissivity contribution including quadrature weight. |
| Calibration | AIA learns one shared scalar, degenerate with density; relative channel calibration cannot adjust. | Use small constrained per-channel nuisance gains with priors and fix one reference/zero-mean constraint. |
| Channel pairing | Nearest observations may be reused; non-reference WCS/shape are not verified. | Unique matching with tolerance; explicitly reproject every channel or reject mismatches. |
| Validation | Validation is not held out and maps NaNs to zero. | Define temporal/view holdouts and preserve masks in every metric. |
| Shell misses | Negative outer-sphere discriminants are unhandled. | Return a ray validity mask or reject rays outside the render domain. |
| Static regularization | Density and temperature derivatives are added before squaring and can cancel. | Penalize each normalized derivative separately. |
| Instrument weighting | Loss scales with channel count. | Average over valid channels, then apply explicit instrument/channel weights. |
| Artifact/cache | Cache has no source/config fingerprint and state is a full pickle. | Reuse CME fingerprints/atomic publication; save schema + state dict + immutable references. |

### CME integration findings corrected in the shared core

- The HAO azimuthal-equidistant coordinate helper transposes non-square images
  and treats projection-plane offsets as helioprojective angles. Wide-field ARC
  rays require the proper inverse projection or WCS-derived directions.
- Enabled CME alignment is currently a no-op because its learned angle is
  multiplied by zero, producing zero gradients.
- The Thomson prep path uses `np.percentile` rather than `np.nanpercentile` for
  a background operation, allowing one bad frame to poison a time-series pixel.

These are not caused by the plasma pipeline, but a common core must not import
them unchanged.

### Removed implementations

- The scalar-emission model, renderer, runner, configs, and evaluation entry
  points were deleted.
- The fitted PSI adapter and its dataset wrapper were deleted in favor of the
  direct interpolated-grid artifact.
- Duplicate per-instrument prep modules were deleted; the unified prepared-EUV
  adapters are the only supported input path.
- Hard-coded emission evaluation scripts and the evaluation stash were deleted.

## 4. Unified CHIANTI temperature responses

### Historical provenance was inconsistent

| Instrument | Inspected response basis |
|---|---|
| AIA | External/bare NPZ; the configured table records no abundance, ion balance, CHIANTI version, degradation/EVE convention, or input hashes. |
| EUVI-A/B | Old SRE products identify CHIANTI 7.00, Mazzotta ion balance, and Fludra abundance. A and B use different throughput products. |
| EUI/FSI | Bundled tables identify CHIANTI 10.1, `sun_coronal_2021_chianti`, `chianti.ioneq`, and a density dimension that the loader collapses. |

The current AIA `1e-21`, EUVI `1e6`, and target 10000/7000 normalizations are not
a unit conversion. They prevent a common absolute density interpretation.

There is also a concrete EUI loader unit bug: its default `target_ne=1e9` is
interpreted as m\(^{-3}\), equal to only \(10^3\) cm\(^{-3}\), not the likely
intended coronal \(10^9\) cm\(^{-3}\).

### Reproducible baseline

Build every instrument from the same pinned atomic specification:

- CHIANTI database 11.0.2, pinned by archive SHA-256;
- `sun_coronal_2021_chianti.abund`, passed explicitly rather than inherited from
  a library default;
- `chianti.ioneq`, also explicit;
- Fiasco 0.8.2 for the units-aware Python baseline;
- aiapy 0.12.1 for AIA wavelength response/calibration;
- a separately generated photospheric-abundance family for systematic-error
  experiments.

CHIANTI IDL 11.0.4 can be an optional backend for the version-11 advanced
density/charge-transfer ion-balance models. At the time of this audit, the
official CHIANTI documentation says those advanced models are available in the
IDL path while Python support is still developing. Do not silently mix a
standard file-based equilibrium for one instrument with an advanced equilibrium
for another.

This baseline is a reproducibility choice, not a claim that coronal abundance is
universal. Abundance uncertainty must be evaluated because Fe-dominated channel
responses are directly degenerate with inferred density. He II 304 Å should stay
excluded from the simple optically thin CIE model unless non-LTE,
resonant-scattering, and absorption effects are modeled explicitly.

### Builder design

1. Compute a shared spectral line-plus-continuum emissivity basis on
   `(log_temperature, log_density, wavelength)` using the pinned abundance and
   ionization equilibrium. Include all relevant ions, free-free, free-bound, and
   two-photon continua; the one-ion Fiasco tutorial is not a production builder.
2. Define the emission-measure convention. Recommended:

   \[
   I_c=\int K_c(T,n_e)n_e^2\,dl.
   \]

   If the atomic backend returns an \(n_en_H\) convention, convert using the
   composition-consistent \(n_H/n_e\), not a hard-coded constant.
3. Fold the same atomic basis through an instrument provider:
   - AIA: `aiapy.response.Channel.wavelength_response`, with crosstalk, EVE,
     calibration table, and degradation epoch explicit. Record but do not apply
     the old additive `chiantifix` to the CHIANTI 11 baseline;
   - EUVI-A/B: spacecraft-specific effective areas and the actual filter-wheel
     state; retain the old SRE only as a regression reference;
   - EUI: pinned public FSI V2 spectral calibration with an explicit
     `FILTER`/`FILTPOS` binding. A broadband table made with another abundance
     cannot be repaired by one multiplicative scalar.
4. Retain `K(channel, logT, logne)` in the artifact. A lean first runtime may use
   a documented \(n_e=10^9\) cm\(^{-3}\) slice if a sensitivity test proves that
   density dependence is negligible for selected channels.
5. Cache the expensive common atomic basis once. Throughput folding is then a
   cheap offline operation. Training must never download atomic or calibration
   data.

### Calibration convention

Choose one of these mutually consistent paths for each released dataset:

1. preprocess observations to exposure-normalized, reference-epoch sensitivity
   and use a reference-epoch response; or
2. retain native epoch sensitivity and evaluate an epoch-dependent response.

Never degradation-correct the image and also use a degraded epoch response.
Test the two equivalent paths on the same observation.

Use an instrument-native calibrated detector rate per solid angle, with exact
Astropy units, rather than forcing unlike DN systems into a nominally common
number. The matching response must carry the corresponding unit, for example
`DN cm5 s-1 sr-1`. Pixel-based responses are safe only if pixel solid angle is
fixed and preserved. ML scaling belongs in a reversible runtime transform and
must never be baked into prepared observations.

### Response artifact

The implemented format is a small compressed NPZ with a strict schema and
canonical JSON provenance. It contains:

- ordered `channel`, `log10_temperature_K`, and optional
  `log10_density_cm3` axes; calibration epoch is immutable provenance, so a new
  calibrated epoch produces a new artifact;
- response values and parseable unit;
- `ne2` versus `ne_nH` emission-measure convention;
- instrument, spacecraft, detector, channel, filter, gain, and crosstalk;
- CHIANTI database version/archive hash;
- abundance and ionization-equilibrium filenames/hashes;
- density/pressure and proton/electron assumptions;
- lines/continuum flags and backend/package versions;
- effective-area/calibration files and hashes;
- degradation, EVE, and empirical-correction conventions;
- schema version, canonical builder config, creation time, and source commit.

The semantic arrays, metadata, and immutable input hashes define `response_id`.
Configuration names the local artifact and ordered channel IDs; preparation and
runtime verify its ID and file hash rather than trusting the path. Physical
artifacts do not accept arbitrary normalization.

### Response validation

- schema, shape, unit, axis, non-negativity, and finite-value tests;
- explicitly zero sensitivity beyond native temperature/wavelength support;
- isothermal-slab analytic rendering tests;
- invariance under temperature-grid refinement and spatial resampling;
- reference comparisons against current aiapy/SSW AIA, legacy EUVI SRE, and
  released EUI response curves;
- abundance and ion-balance sensitivity reports;
- a calibration double-counting test;
- a channel permutation must fail rather than silently relabel output.

## 5. Unified per-instrument preparation

The shared in-memory contract is `PreparedEUVObservation`, rather than an
instrument-specific calibration path inside a training dataset:

```text
image[channel, y, x]        float32 calibrated measurement
valid_mask[channel, y, x]   bool
wcs[channel]                complete celestial WCS before alignment
channel_times[channel]      exposure times; obstime is their mean
channel_id[channel]         stable ordered identifiers
measurement_unit[channel]   parseable Astropy units
prepared_pixel_solid_angle[channel]
native_pixel_solid_angle[channel]
measurement_semantics[channel]
sensitivity_convention[channel]
source_path[channel]
```

Store prepared observations as standard multi-extension FITS. The data module
reads this contract directly; it does not calibrate instruments. Cache
fingerprints hash the configured FITS inputs, and calibration history remains
in the FITS headers.

### AIA adapter

1. validate quality/exposure and input level;
2. apply `update_pointing`, then registration in the documented aiapy order;
3. exposure-normalize exactly once;
4. apply pinned reference-epoch degradation correction;
5. crop/reproject once, propagate the validity mask, bind response radiometry,
   and write provenance.

### EUVI-A/B adapter

1. start from a documented SECCHI-prepped level;
2. require an explicit upstream calibration ID, product level, spacecraft, unit,
   exposure, and sensitivity convention;
3. verify the FITS spacecraft/product identity;
4. apply geometry only, propagate the validity mask, and bind the matching
   response/radiometry artifact.

### EUI adapter

1. prefer the released calibrated L2 product;
2. validate instrument identity, level, calibration ID when present, exposure,
   and `BUNIT`;
3. never reapply L1-to-L2 calibration to L2 data;
4. apply the chosen reference/native epoch convention;
5. reproject with mask propagation and bind an EUI response produced
   from the same atomic basis.

### Multi-channel assembly

- match observations uniquely; do not reuse a nearest file unless explicitly
  configured;
- retain per-channel original WCS, then reproject every channel to the selected
  common WCS;
- validate shape, WCS, observer, units, and calibration convention;
- use exposure midpoints and record every temporal offset;
- make tolerance/cadence/holdout policy part of the data-cache fingerprint.

## 6. Implemented grid-based synthetic data path

The historical `sunerf.train.fit_psi` command trained a second `PlasmaModel` for
100 epochs to approximate PSI density and temperature, then ray-traced that
approximation. It also called the current radiative-transfer API without the now
required `log_T_range`, and embedded a response path different from its CLI
argument.

The replacement is:

- `sunerf/model/spherical_grid.py`: differentiable nonuniform interpolation in
  time, radius, latitude, and periodic longitude;
- `sunerf/data/psi/spherical_grid.py`: density/temperature pairing by numeric
  frame ID, grid validation/sorting, duplicate periodic-endpoint removal, and
  normalized time metadata;
- `sunerf/data/psi/build_synthetic.py`: a CLI that creates an atomically written,
  SuNeRF-compatible render state using the standard spherical/hierarchical ray
  machinery;
- `scripts/fit_psi.sh`: migrated invocation.

The field-only PSI reader avoids constructing full spherical and Cartesian
coordinate volumes for every frame. Frames can stream from a bounded worker pool
into one compact `(time,r,lat,lon,field)` array.

At a query point, density and temperature are linearly interpolated in log space.
The local single temperature is deposited into the two adjacent renderer bins
with square-root weights. Therefore the sum of squared per-bin densities preserves
\(n_e^2\) while linearly interpolating the channel response. Non-finite simulation
corners are excluded and remaining interpolation weights are renormalized.

The weights-only-safe saved state records the grid tensors/axes, every paired
source density/temperature path and SHA-256 digest,
response path/hash/ID/channel order, normalized times, interpolation convention,
explicit source-to-`cm^-3` and source-to-kelvin scales, reference frame ID,
Carrington longitude convention, and renderer construction metadata. These
source conventions have no supported defaults and are applied exactly once.
The response NPZ is validated before construction, LOS distances are in
centimeters, and the state is published atomically.

Verification completed:

- interpolation on nonuniform 4-D grids;
- periodic longitude seam;
- emission-measure preservation across temperature bins;
- spatial out-of-domain behavior;
- PSI file pairing, field/time interpolation, and duplicate seam removal;
- the full repository regression suite passing (see the final implementation
  handoff for the exact run count);
- Ruff and `git diff --check` passing;
- end-to-end synthetic state creation, reload, ray sampling, and finite positive
  rendered image on a two-frame HDF5 smoke cube.

## 7. Implemented package design

The package uses composition and a few stable contracts:

```text
instrument prep adapters
  -> prepared FITS + PreparedEUVObservation
  -> MultiInstrumentDataModule + shared ray dictionaries

shared CHIANTI atomic basis
  + instrument throughput adapters
  -> immutable ResponseArtifact / response_id

PlasmaModel (reconstruction) | SphericalGridPlasmaField (simulation)
  -> BasicRenderingModule / MultiResolutionRenderingModule
  -> PlasmaRadiativeTransfer

MapDataLoader rays -> shell sampler -> PlasmaRadiativeTransfer products

Trainer -> versioned checkpoint (schema + state_dict + response IDs)
Evaluator -> artifact metadata, never hard-coded channels
```

### Shared primitives to extract from recent CME work

- exact WCS ray construction and observer pose;
- common ray/shell validity masks;
- one nonuniform quadrature implementation;
- cache fingerprints and atomic cache/state publication;
- current DDP/CPU device handling;
- distributed validation gathering and rank-zero writes;
- explicit instrument/channel metadata;
- background/prep validation patterns.

Physics renderers remain separate while sharing ray construction, shell
sampling, cache publication, and validity conventions. Instrument calibration
does not live in either renderer.

### Lean supported surface

- one supported train entry point and one config schema;
- one EUV prep interface with AIA/EUVI/EUI adapters;
- one response schema/loader/builder;
- one reconstruction plasma field plus the grid simulation field;
- one shared ray/sampler/quadrature core;
- one artifact-driven evaluator;
- archive or remove broken emission code, positional IDL/NPZ loaders, duplicated
  prep paths, and channel-specific evaluation scripts after regression capture.

Avoid a dynamic plugin framework here. A typed registry mapping stable instrument
IDs to three small adapters is sufficient.

### Implemented efficiency changes

- co-registered simultaneous channels reuse one ray bundle;
- training requests only lean renderer outputs, while explicit diagnostics are
  produced for validation/evaluation;
- worker budgets are capped across instrument loaders;
- cached training arrays are memory-mapped, content-fingerprinted, and published
  atomically;
- safe artifacts contain state dictionaries and immutable metadata rather than
  requiring executable pickles.
- EUV validation plots are selected in schema-v2 YAML; callbacks register only
  their required tensors, while an explicitly empty selection retains no LOS
  diagnostics. The supported products are channel residuals, physical LOS
  quantities, thermal-distribution summaries, response/gain monitoring, and
  opt-in ray sampling.

### Remaining efficiency work

1. Do not materialize every image and ray in RAM before creating mmap storage.
   Stream prepared observations and cache geometry by WCS hash.
2. Replace the `[rays,samples,temperature]` diagnostic intermediate with a compact
   normalized temperature
   basis and differentiable response lookup/convolution. A single-Gaussian field
   can evaluate a compact preconvolved response table.
3. Include `dl` in hierarchical sampling weights and reuse the same merged sample
   ordering for every renderer.
4. Avoid callback retention of any per-sample diagnostic that can be integrated
   per batch.

## 8. Implementation record and release gates

The original ordered plan is retained as a release checklist. The supported
software path is implemented; external response releases, instrument-by-
instrument validation, performance streaming, and science-systematics studies
remain release work because atomic/calibration archives and representative
observations are not vendored here.

### Phase 0 — trustworthy baseline (complete)

- fix invalid-pixel masks, train-only time jitter, explicit time units, current
  Lightning hooks/strategies, LOS quadrature, instrument dispatch, and metadata;
- repair shell-miss handling and the two CME issues found during integration;
- add analytic constant-field/isothermal-slab and instrument-routing tests.

Exit gate: current Lightning launches; all train/validation inputs and outputs are
finite; validation is deterministic; constant emissivity integrates exactly on
uniform and nonuniform samples.

### Phase 1 — common data and rendering contracts (complete)

- introduce `PreparedEUVObservation` and explicit renderer product dictionaries;
- migrate plasma to the current CME data/cache/validation infrastructure;
- make channel order, units, masks, and instrument identity mandatory;
- make evaluation wholly artifact-driven.

Exit gate: AIA/EUVI-A/EUVI-B batches can be permuted without changing identity,
and an incorrect channel order/WCS/unit fails at ingestion.

### Phase 2 — response and preparation system (baseline release implemented)

- implement versioned response, spectral-emissivity, and throughput schemas with
  semantic hashes;
- define a provider-neutral offline boundary for one shared CHIANTI atomic basis
  and instrument throughput inputs;
- implement the three preparation adapters and shared FITS writer;
- require explicit sensitivity and pixel-radiometry conventions.

The implemented release command now fetches and byte-verifies CHIANTI 11.0.2,
AIA V8/V10, both SECCHI SRA-001 files, and all four public EUI FSI V2 tables;
builds the FIASCO HDF5 database with per-file hash checks; retains exact line
wavelengths while integrating resolved continua; and exports the default AIA,
EUVI-A/B, and EUI/FSI response artifacts from one atomic calculation.

Still required for a broader systematics release: generate selected abundance
families, add reference-comparison reports beyond the provider golden tests,
and quantify density and abundance sensitivity for each science selection.

Exit gate: training validates every prepared channel against an immutable
compatible response; there are no downloads in training; provenance and units
are sufficient to rebuild every response.

### Phase 3 — physical plasma image formation (complete)

- migrate the field to `n_e + normalized p(logT)`;
- include `dlogT`, cm LOS lengths, and correct nonuniform quadrature;
- define local density, temperature PDF, DEM, emission measure, column density,
  absorption, and uncertainty products with explicit units;
- add bounded instrument-wide and constrained per-channel calibration nuisance
  parameters with an explicit global gauge;
- keep response artifacts in physical units while applying the same fixed,
  reversible per-channel training divisor to observations and predictions.

Exit gate: an analytic isothermal slab recovers its known intensity and density;
results are invariant to LOS and temperature-grid refinement within tolerance.

### Phase 4 — science workflow migration (code complete; science-systematics study pending)

- migrate `all_2012_08.yaml` to schema v2 response artifacts and prepared FITS paths;
- use temporal/view holdouts, masks, and explicit channel weighting;
- adopt the grid-backed PSI synthetic path and compare direct-grid images against
  analytic/simple cases;
- quantify abundance, ionization equilibrium, density-response, calibration, and
  temporal-matching systematics.

Exit gate: AIA/EUVI joint reconstruction has a closed unit chain and a documented
systematic-error report; the PSI synthetic generator contains no fitted surrogate.

### Phase 5 — simplify and deprecate (supported surface complete; cleanup ongoing)

- remove or archive duplicate prep, broken emission, legacy converter, hard-coded
  evaluation, and obsolete dataset paths;
- document one supported package workflow and enforce config schemas in CI.

Exit gate: no supported code imports archived modules; configuration and artifacts
are versioned; CI covers prep contracts, rays, quadrature, responses, training
smoke, and evaluation routing.

## Authoritative references

- [CHIANTI downloads and versions](https://www.chiantidatabase.org/chianti_download.html)
- [CHIANTI database history](https://www.chiantidatabase.org/chianti_database_history.html)
- [CHIANTI user guide: abundances and ionization equilibrium](https://www.chiantidatabase.org/cug.html)
- [CHIANTI version 11 advanced-model paper](https://arxiv.org/abs/2403.16922)
- [Fiasco ion API and emissivity convention](https://fiasco.readthedocs.io/en/stable/api/fiasco.Ion.html)
- [Fiasco/aiapy AIA response construction example](https://fiasco.readthedocs.io/en/latest/generated/gallery/user_guide/aia_response.html)
- [aiapy AIA wavelength-response API](https://aiapy.readthedocs.io/en/latest/api/aiapy.response.Channel.html)
- [aiapy AIA preparation order](https://aiapy.readthedocs.io/en/stable/preparing_data.html)
- [Official AIA response routine](https://hesperia.gsfc.nasa.gov/ssw/sdo/aia/idl/response/aia_get_response.pro)
- [SECCHI/EUVI calibration document](https://stereo-ssc.nascom.nasa.gov/publications/CMAD/secchi/STEREO_SECCHI_EUVI_CMAD_20211206.pdf)
- [SECCHI/EUVI response and preparation notes](https://stereo-ssc.nascom.nasa.gov/instruments/documentation/secchi/EUVI/DOCUMENTS/data_analysis_euvi2.html)
- [EUI current release notes](https://www.sidc.be/EUI/data/latest_release_notes.html)
- [EUI data-product definitions](https://www.sidc.be/EUI/data/documents/SP-ROB-SOEUI-19001-DPDD_1.8.pdf)
- [SunPy map resampling/reprojection API](https://docs.sunpy.org/en/latest/generated/api/sunpy.map.GenericMap.html)
