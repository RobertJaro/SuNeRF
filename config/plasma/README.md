# Plasma configuration support

`all_2012_08.yaml` is the canonical schema-v2 multi-instrument example. It is
validated before SuNeRF creates output directories, loads observations, or
initializes a model.

A supported configuration must bind:

- strict prepared-EUV FITS inputs with ordered channels, `BUNIT`, sensitivity
  convention, and native-pixel radiometry;
- a versioned response artifact built from one shared CHIANTI spectral basis;
- explicit solar-radius and time normalization;
- fixed, explicit per-channel image divisors that belong to the instrument and
  are shared by all of its datasets (never estimated while loading data);
- one global calibration-reference instrument when response gains are
  learnable, plus bounded instrument-wide and channel-relative gains;
- training on every prepared observation of every training dataset, and one
  validation frame per validation dataset: the observation with all channels
  closest to its ISO `time` (`data.valid_datasets[].time`), or without one to
  the middle of the sequence. A validation dataset may read the files of a
  training dataset, so its frame is also trained on, or its own files (a view
  that is never trained on). A dataset may restrict its prepared files to a
  time window with ISO `start` and `end` timestamps (observations in
  `[start, end)` of the mean channel time) and thin them to a fixed
  `cadence_minutes` (an observation is kept once that much time has passed
  since the last kept one); a validation dataset sets either `time` or the
  window, not both; and
- explicit bounds of the pointwise temperature (`model.temperature_grid`; its
  `step_dex` is the bin width of validation emission-measure histograms) and a
  physical regularization scale. `lambda.regularization` weights the
  emission-measure prior on the temperature edges: the LOS emission measure of
  every sample, times a weight that is zero inside
  `module.regularization_log_T_range` and rises as a cubic smoothstep to one at
  the `temperature_grid` bounds, in units of
  `module.regularization_density_scale_cm3` squared over one solar radius. Plasma
  pushed toward the bounds costs its emission measure, so the model empties the
  hot and cool tails rather than piling plasma up at a bound.

With `model.cool_absorber: true`, `lambda.cool_absorber` weights a sparsity
prior on the LOS hydrogen column N of the cool absorber, the ray mean of
`log1p(N / module.cool_column_scale_cm2)`. The default scale of 1e18 cm^-2 is
an optical depth of about 0.1 at 171-193 A. Below the scale a column costs
linearly, above it only logarithmically, so a thin haze over many rays costs
more than the same column concentrated in a few compact absorbers.

The loss-space transform and its divisor are configured per instrument; datasets
define no scaling and stay in physical units:

```yaml
instruments:
  - key: "AIA"
    scaling: {type: "log", divisor: "/path/to/image_scaling.yaml"}
```

`divisor` is a scalar, an ordered list, a channel mapping, or the path of a
table written once before training:

```bash
python -m sunerf.data.euv.estimate_scaling --config config/plasma/all_2012_08.yaml
```

For each channel, the estimator takes the 99.5th absolute-valid-pixel percentile
of each training observation and then the
median over time. The constants only set the loss scale, so it opens only 200
evenly spaced prepared files per dataset (`--max-files 0` opens all; no header
scan or channel matching) and prints each entry as YAML; paste the values into the packaged table
`sunerf/resources/calibration/image_scaling.yaml` and reference it with
`divisor: "builtin:image_scaling"` to reuse them across runs. An existing table is reused unless `--overwrite` is given, so
resumed runs keep their constants. Every dataset mapped to the instrument
shares the vector, including validation-only views and mixed sequences of one
instrument; for a short high-cadence event sequence next to a background
sequence, restrict the estimate with `--datasets <background key>`. Prepared
FITS rates and cached targets remain unchanged, observations and predictions
are divided by the same vector only inside the loss, and saved rendering stays
in native units. With learnable response gains, set
`temperature_response.global_reference: true` for exactly one instrument;
unless that reference is independently calibrated, physical density amplitude
is fixed only by this declared gauge. `module.calibration_iterations: N`
trains the gains for the first N steps only and then freezes them. Use it
for absorption runs: there the gains of some channels keep drifting and
trade off against the absorber instead of staying a fixed calibration.

## Time jitter

`shuffle.type: "normal_time"` adds Gaussian noise to the observation times.
The width stays at `start_seconds` for `cold_iterations` steps, then decays
linearly to `end_seconds` over `iterations` steps.

## Saturated pixels

Saturated pixels are masked while the data load, with one optional level per
instrument in detector DN:

```yaml
instruments:
  - key: "AIA"
    saturation_dn: 13000   # null or absent: no saturation mask
```

Pixels whose `PEAK_DN` reaches the level get `VALID_MASK=0` and are not
supervised; the image loss ignores them per channel, so the other channels of
the same ray still constrain the plasma. `PEAK_DN` is written by
`sunerf.data.euv.prepare`: for every prepared pixel the highest detector DN of
the native pixels that can affect it, four pixels along the CCD readout
columns (bleed) and one across them, widened by the interpolation support.
Detector DN are AIA level-1 DN, SECCHI_PREP values on the 14-bit scale of the
header `DSATVAL`, and EUI level-2 rates times the exposure. Set the level below
the true saturation so the nonlinear range just under it is excluded too: AIA
level-1 pixels saturate at 15000-16383 DN, and 13000 masks only a few dozen
extra bright pixels of a quiet frame. EUI headers declare no saturation level;
leave `null` until one is measured on an FSI flare frame.
`sunerf.data.frame_review` finds the saturation plateau of a dataset from the
prepared `PEAKDN` headers, recommends 0.8 of it and records the chosen level
(`saturation.csv`). The level is part of the data-cache fingerprint, so
changing it rebuilds the cache without preparing again. Products prepared
before `PEAK_DN` existed raise an error when a level is set: prepare them
again with `--overwrite`. During flares AIA alternates saturated long and
unsaturated short exposures; with the mask the long exposures constrain
everything but the flare core and its bleed, which the short exposures supply.

## Pointing alignment

Alignment is off by default. `alignment: true` on an instrument adds a learnable
pointing correction: an image
roll about the boresight and a helioprojective shift `(x, y)`, constant per
instrument, applied to its ray directions before rendering (observer positions
stay fixed). It absorbs systematic registration errors between views (e.g. a
residual EUVI roll or pointing offset relative to AIA) and is bounded through
`tanh` by `roll_limit_deg` (default 1) and `shift_limit_arcsec` (default 30):

```yaml
instruments:
  - key: "EUVI-A"
    alignment: true                      # or {roll_limit_deg: 1.0, shift_limit_arcsec: 30.0, trainable: true}
```

It may be enabled on every trained instrument: the observer positions are fixed
and the sampling shell is centered on the Sun, which hold the frame. It cannot
be enabled on validation-only instruments, which never train it. The learned
values are logged as
`alignment.<instrument>.{roll_deg,shift_x_arcsec,shift_y_arcsec}` and are stored
in the artifact with the other weights.

## Packaged physics resources

Response tables `G(log T, log n_e)` and the H/He absorption bundle ship with the
package and are selected with `builtin:` references:

```yaml
temperature_response:
  artifact: "builtin:aia"        # euvi_a, euvi_b, eui_fsi
absorption:
  type: photoionization
  artifact: "builtin:h_he_photoionization"
```

`sunerf-resources verify` checks them against their manifest;
`sunerf-resources build --install` rebuilds them from CHIANTI and the instrument
calibrations. `all_2012_08.yaml` is the portable science
configuration. `temperature_response.temperature_cutoff: {T_cut_K, delta_T_K}`
suppresses emission (never opacity) below a transition-region temperature and
is intended for thermodynamic MHD cubes with a broadened transition region.

## Validation metrics

Every dataset listed under `callbacks.euv_tomography.datasets` logs full-image
metrics at each validation, independent of `every_n_validations`:
`validation/{loss,psnr,ssim}.<dataset>` (averaged over channels) and
`validation/{loss,psnr,ssim}.<dataset>.<channel>`. They are computed in loss
space (data range 1, so the PSNR is comparable to `train/psnr`) over the valid
pixels; SSIM uses the full image with masked pixels zeroed and is averaged over
the valid pixels. With `normalize: true` the metrics use the normalized
prediction, as in the plots.

## North-pole view

`callbacks.north_pole_view` adds a validation plot rendered from above the
solar north pole (Carrington latitude +90 deg), logged as
`euv_tomography.north_pole.<dataset>`:

```yaml
callbacks:
  north_pole_view:
    enabled: true
    dataset: "AIA"
```

`dataset` names a validation dataset (default: the first one). Its
validation frame sets the instrument, time, observer distance and field of view;
the view is disk-centered and rendered at 256x256 pixels. The top row shows
the predicted channels in loss space (default: the `euv_tomography` channels
of that dataset, or `channels`), the bottom row mean temperature, emission
measure and column density. Dotted circles mark 60, 70 and 80 deg latitude on
the photosphere; Carrington longitude 0 points right. Cadence and figure
resolution follow `euv_tomography`. The view is rendered on rank zero after
each validation; `polar_2012_08.yaml` enables it.
