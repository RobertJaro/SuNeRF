# SuNeRF

SuNeRF reconstructs time-dependent coronal plasma and CME electron density from
multi-viewpoint images using neural fields and explicit line-of-sight physics.
The supported pipelines are:

- EUV density/temperature tomography with calibrated AIA, SECCHI/EUVI, and
  Solar Orbiter/EUI observations;
- white-light CME tomography with Thomson scattering;
- direct synthetic EUV image formation from interpolated PSI plasma grids.

EUV work uses the plasma pipeline, which models electron density, a normalized
temperature distribution, instrument response, and physical path length
together.

## Installation

Python 3.10 or newer is required for the core package. The pinned current
`aiapy` extra requires Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install only the optional capabilities needed by a workflow:

```bash
pip install -e ".[euv-prep]"        # Python 3.12+: pinned aiapy AIA preparation
pip install -e ".[evaluation,test]" # evaluation tools and development checks
```

The response workflow converts every camera to one throughput schema and folds
one separately prepared atomic-emissivity table through all instruments. See
[`sunerf/response/README.md`](sunerf/response/README.md).

## Data downloads

Every supported instrument downloader uses the same core interface:

```text
--start <ISO-UTC> --end <ISO-UTC> --output <directory>
[--overwrite] [--dry-run]
```

Instrument options extend this common contract. Cadence values consistently
use durations such as `30m`, `6h`, or `1d`; `all` selects every available
record. For example:

```bash
python -m sunerf.data.download.download_euvi \
  --start 2012-08-01 --end 2012-08-02 \
  --output data/euvi_a --cadence 6h \
  --channels 171 195 284 --sources STEREO_A --dry-run
```

The retained AIA, EUI, EUVI, SUVI, SECCHI/COR, CCOR, LASCO, PUNCH, PSP, and
Solar Orbiter modules also expose `build_parser()`, `download(request, ...)`,
and `main(argv=None)`. Shared request/result types live in
`sunerf.data.download`; import-time downloads and hard-coded-path downloader
scripts are intentionally unsupported.

## EUV plasma tomography

The physical image model for channel `c` is

```text
I_c = integral_LOS n_e^2 * integral_logT K_c(logT, n_e) p(logT) dlogT dl
```

where `n_e` is in `cm^-3`, `p(logT)` is normalized per dex, `K_c` is a
versioned instrument response, and `dl` is converted from model coordinates to
centimeters. For an `n_e n_H` response convention, the abundance-derived
`n_H/n_e` factor is carried by the response artifact and applied explicitly.

The supported data flow is deliberately strict:

```text
raw/calibrated instrument files
  -> one instrument prep adapter
  -> prepared-EUV-v2 FITS
  -> ray construction and deterministic train/validation holdout
  -> shared plasma field + instrument response renderers
  -> weights-only-safe reconstruction artifact
```

### 1. Build consistent responses offline

Download and standardize the instrument wavelength responses, then fold one
common atomic-emissivity table through all of them:

```bash
sunerf-responses \
  --root data/response_calibration prepare
sunerf-responses \
  --root data/response_calibration build \
  --spectral-emissivity responses/common_coronal_0p1A.spectral.npz
```

The pipeline never imports an atomic package. The single emissivity NPZ records
the chosen CHIANTI version, abundance, ionization equilibrium, density, and
emission-measure convention. Training remains fully offline.

Trusting the nominal absolute response amplitude is optional at training time.
Each instrument's `scaling.divisor` supplies fixed positive per-channel divisors,
either explicitly or as a table written once by
`python -m sunerf.data.euv.estimate_scaling --config <config>` (robust percentile
of the training observations). Datasets stay in physical units and all datasets
of an instrument share the vector. The loss applies the same divisor to the
observation and the physical prediction before its transform; the renderer learns a bounded
instrument-wide gain plus a constrained relative channel gain. Divisors
condition—and, especially with a nonlinear transform, effectively weight—the
loss; they do not remove the requirement that the gain-adjusted prediction
match the observation. Exactly one
`temperature_response.global_reference` fixes the otherwise unidentifiable
global density/gain gauge. Prepared FITS files and saved renderer outputs remain
in their calibrated native units; never normalize each image independently.

### 2. Prepare each instrument into one contract

All adapters preserve calibrated values and invalid pixels, write a
`VALID_MASK` extension, and stamp only preparation and calibration metadata.
They do not load or validate temperature-response artifacts.

AIA level-1 data are calibrated with pinned local aiapy correction and pointing
tables:

```bash
python -m sunerf.data.euv.prepare aia \
  --input 'raw/aia/*.fits' \
  --output-dir prepared/aia \
  --correction-table calibration/aia_correction.ecsv \
  --pointing-table calibration/aia_pointing.ecsv \
  --shape 512 512 --hpc-bounds -1560 -1560 1560 1560
```

EUVI starts from a documented, externally SECCHI-calibrated product; SuNeRF
does geometry only:

```bash
python -m sunerf.data.euv.prepare euvi \
  --input 'calibrated/euvi_a/*.fts' \
  --output-dir prepared/euvi_a \
  --spacecraft A \
  --product-level SECCHI-L1-calibrated \
  --sensitivity-convention static_assumed \
  --shape 512 512 --hpc-bounds -1560 -1560 1560 1560
```

EUI starts from a calibrated level-2 product:

```bash
python -m sunerf.data.euv.prepare eui \
  --input 'raw/eui/*.fits' \
  --output-dir prepared/eui \
  --calibration-id eui-l2-release-id \
  --sensitivity-convention static_assumed \
  --shape 512 512 --hpc-bounds -1560 -1560 1560 1560
```

Use the same shape and helioprojective bounds for simultaneous channels. A
reference FITS can instead define an exact target grid with
`--reproject-reference`. AIA refuses nonzero quality, missing exposure/unit
metadata, mutable online calibration inputs, and already rate-normalized input.
EUVI and EUI refuse products whose declared spacecraft/level/calibration does
not match the requested adapter.

### 3. Train

The canonical multi-instrument example is `config/plasma/all_2012_08.yaml`.
Update its response paths and prepared-FITS globs, then run:

```bash
python -m sunerf.run_plasma --config config/plasma/all_2012_08.yaml
# or: sunerf-plasma --config config/plasma/all_2012_08.yaml
```

The complete August 2012 release workflow is organized like the CME runs in
[`scripts/plasma_2012_08`](scripts/plasma_2012_08/README.md): response folding,
observation download, external calibration and prepared-v2 generation,
training, and artifact-driven evaluation are separate executable stages.

Schema-v2 validation occurs before caches, loggers, or models are created. The
runner checks channel order, physical units, fixed reversible image divisors,
WCS alignment, per-channel masks, held-out observations, sensitivity epoch,
and native-pixel radiometry against the immutable source response artifact.
Cache generations hash every configured prepared FITS product and publish
atomically.

Validation figures are configured under `callbacks.euv_tomography`. The
canonical configuration enables four science products: per-channel
observation/prediction/residual comparisons, LOS plasma diagnostics, the
spatial thermal-distribution summary, and response-function/learned-gain
monitoring. Ray-sampling inspection is available as an opt-in debugging plot.

```yaml
callbacks:
  euv_tomography:
    every_n_validations: 1
    datasets:
      AIA: {channels: [171, 193, 211]}
    products:
      channel_comparison:
        rows: [observation, prediction, residual]
        stretch: log
      plasma_diagnostics:
        quantities: [mean_log_temperature, column_electron_density,
                     emission_measure, emission_height, absorption_fraction]
      thermal_distribution: {enabled: true}
      response_and_gains: {enabled: true}
      ray_sampling: {enabled: false}
```

Dataset and channel selections are validated against the configured response
artifacts before any logger or model is created. Callback tensor retention is
product-aware, so disabled diagnostics are not accumulated across validation.
Plots are logged as `euv_tomography.<dataset>.<product>`.

The reconstruction state is a single `*.safe.pt` weights-only artifact. The
evaluation loader verifies referenced response hashes and reconstructs the
model without executing pickled Python.

Create an artifact-driven reconstruction movie with:

```bash
sunerf-plasma-video \
  --chk-path results/run/save_state.safe.pt \
  --video-path results/run/video \
  --instrument-key AIA \
  --channels 171 193 211
```

The evaluator resolves channel aliases against the immutable artifact metadata,
writes numbered JPEG frames, and encodes `plasma.mp4`. Use `--frames-only` when
only the image sequence is needed; video encoding requires the `evaluation`
optional dependencies.

## Direct PSI synthetic EUV observations

Synthetic images are formed directly from the PSI density and temperature
grids. SuNeRF interpolates the four-dimensional `(time, radius, latitude,
periodic longitude)` field and uses the same rays, shell samplers, quadrature,
and plasma renderer as reconstruction. No neural surrogate is fitted first.

```bash
python -m sunerf.data.psi.build_synthetic \
  --data-path data/psi \
  --temperature-response-artifact responses/aia.sunerf.npz \
  --response-channels A193 \
  --out-path results/psi \
  --source-density-scale-cm3 1e8 \
  --source-temperature-scale-k 2.807066716734894e7 \
  --reference-frame-id 1813 \
  --longitude-frame carrington \
  --seconds-per-dt 86400
```

The two source scales are dataset-specific conversions to `cm^-3` and kelvin;
the values above document the historical PSI archive used by this project.
They have no defaults and are applied exactly once. The reference frame is the
frame at `--reference-date`, and the longitude frame must also be declared.
Every paired source path and SHA-256 digest, along with these conventions, is
stored in the resulting grid artifact. Versioned physical responses cannot be
rescaled.

## White-light CME tomography

Prepare supported coronagraph data with the instrument routines in
`sunerf.data.prep`, then run a schema-appropriate configuration:

```bash
python -m sunerf.run_thomson --config config/cme/hao_2view.yaml
```

The EUV and Thomson renderers remain separate physics modules but share ray
construction, validity masks, shell sampling, runtime caching, device handling,
and artifact metadata.

## Design and migration notes

The detailed pipeline trace, scientific audit, response design, implemented
fixes, and migration notes are in
[`docs/plasma_pipeline_modernization.md`](docs/plasma_pipeline_modernization.md).
Historical plasma configurations and duplicate preparation/response commands
are retained only as explicit migration tombstones or archival examples; the
schema-v2 config and unified entry points above are the supported surface.
