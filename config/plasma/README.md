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
- a non-overlapping temporal holdout for every validation dataset that reads the
  files of a training dataset (their entries must match). Validation-only
  datasets or instruments with their own files need no training counterpart,
  and a training-only sequence (e.g. a short high-cadence event next to the
  background sequence) may set `holdout: null` to train on all observations; and
- explicit bounds of the pointwise temperature (`model.temperature_grid`; its
  `step_dex` is the bin width of validation emission-measure histograms) and a
  physical regularization scale.

The loss-space transform and its divisor are configured per instrument; datasets
define no scaling and stay in physical units:

```yaml
instruments:
  - key: "AIA"
    scaling: {type: "asinh", divisor: "/path/to/image_scaling.yaml"}
```

`divisor` is a scalar, an ordered list, a channel mapping, or the path of a
table written once before training:

```bash
python -m sunerf.data.euv.estimate_scaling --config config/plasma/all_2012_08.yaml
```

For each channel, the estimator takes the 99.5th absolute-valid-pixel percentile
of each training observation (held-out observations excluded) and then the
median over time. An existing table is reused unless `--overwrite` is given, so
resumed runs keep their constants. Every dataset mapped to the instrument
shares the vector, including validation-only views and mixed sequences of one
instrument; for a short high-cadence event sequence next to a background
sequence, restrict the estimate with `--datasets <background key>`. Prepared
FITS rates and cached targets remain unchanged, observations and predictions
are divided by the same vector only inside the loss, and saved rendering stays
in native units. With learnable response gains, set
`temperature_response.global_reference: true` for exactly one instrument;
unless that reference is independently calibrated, physical density amplitude
is fixed only by this declared gauge.

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
