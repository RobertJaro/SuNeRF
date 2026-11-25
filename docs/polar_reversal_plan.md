# Polar field reversal: SuNeRF EUV campaign plan

Goal: a continuous EUV plasma reconstruction of both polar coronae through the
Cycle 24 polar field reversal from SDO/AIA, STEREO-A/EUVI and STEREO-B/EUVI.
One full reconstruction of the first window is followed by a series run that
moves forward in 5-day steps, warm-started from the previous step.

## Interval

**2011-02-06 .. 2014-10-01** (1333 days).

| Date | Event |
|---|---|
| 2011-02-06 | STEREO-A/B 180 deg apart: the three views surround the Sun |
| 2012-06 .. 2012-11 | North polar field crosses zero (WSO mid-2012; HMI >60 deg 2012-11), near zero until late 2014 |
| 2012-08 | Most even spacing (all pairs ~120 deg) |
| 2013-06 .. 2014-03 | South polar field crosses zero (WSO mid-2013; HMI >60 deg 2014-03) |
| 2013-10 | Axial dipole changes sign (HMI) |
| 2014-04 | Cycle 24 maximum (smoothed SSN 116.4) |
| ~2014-08-20 | STEREO-A sidelobe operations: 2-3 EUVI 171/284 frames per day |
| 2014-10-01 | STEREO-B lost |

The last ~6 weeks have SDO + STEREO-B at full cadence plus sparse STEREO-A.
New-polarity polar fields build up after the interval (2015-2016) and are out
of reach with three views.

## Windows

- Fixed time origin: `data.ref_date: 2011-02-06T00:00:00` for every run.
- Window: 27 days (one rotation) = 11 days context + 5-day focus + 11 days context.
- Focus: a second dataset per instrument restricted to the 5-day focus
  (`start`/`end`), as the CME `time_range` focus; validation `time` = focus center.
- Step: 5 days. First focus 2011-02-17 .. 02-22, last 2014-09-15 .. 09-20;
  262 windows. Extending the download 11 days at both ends (2011-01-26 ..
  2014-10-01, STEREO-B is the hard end) moves the first focus to 2011-02-06.

## Cadence

EUVI 171 and 284 are recorded every 2 h throughout 2011-2014 (195 every 5 min),
so EUVI cannot go below 2 h. Solar rotation is 13.2 deg/day at the equator.

| Cadence | Rotation between frames | Per 27-day window (AIA obs / EUVI sets per spacecraft) | Per 5-day focus | Raw + prepared volume (EUVI L1 transient) |
|---|---|---|---|---|
| 1 h | 0.55 deg | 648 / 324 (EUVI at 2 h) | 120 / 60 | 4.0 TB (+1.6 TB) |
| 2 h | 1.1 deg | 324 / 324 | 60 / 60 | 2.6 TB (+1.6 TB) |
| 6 h | 3.3 deg | 108 / 108 | 20 / 20 | 0.9 TB (+0.5 TB) |
| 12 h | 6.6 deg | 54 / 54 | 10 / 10 | 0.4 TB (+0.3 TB) |

- **1 h:** AIA only; EUVI stays at 2 h. The model time encoding resolves about
  3 h, so sub-3 h sampling adds data but no temporal resolution. It also forces
  the model to fit hour-scale low-latitude activity (flares, AR loops) that is
  noise for the polar science.
- **2 h:** native EUVI cadence, AIA at the same times; the densest sampling all
  three views share.
- **6 h:** the cadence of the working 2012-08 pilot; 20 frames per observer in
  the focus at 3.3 deg rotation steps, enough for rotational tomography of
  coronal holes and plume clusters. A single rejected frame leaves a 12 h gap.
- **12 h:** 10 frames per observer in the focus; too sparse for the focus,
  adequate for the context.

**Decision (2026-11-13):** 2 h for every instrument, downloaded, prepared and
trained at 2 h. The background window and the focus use the same 2 h frames;
the focus is a second dataset over the central 5 days, so those frames are
sampled more often.

## Data volume (2 h download)

| Product | Files | Size per file | Volume |
|---|---|---|---|
| AIA level 1, 6 channels | ~96 000 | 12 MB | 1.15 TB |
| AIA prepared 512x512 (f8 data, mask, PEAK_DN) | ~96 000 | 3.4 MB | 0.31 TB |
| EUVI-A/B L0.5, 171/195/284 | ~95 000 | 8.4 MB | 0.80 TB |
| EUVI L1 (SECCHI_PREP, float32) | ~95 000 | 16.8 MB | 1.59 TB |
| EUVI prepared 512x512 | ~95 000 | 3.4 MB | 0.32 TB |
| Frame review (CSV, 192 px JPEG) | ~190 000 | ~10 kB | ~2 GB |
| Training cache per window (scratch) | | | ~10-20 GB |
| Save state per step | 262 | ~10-40 MB | ~3-10 GB |

Raw and calibrated files are deleted once prepared: the data are processed in
15 quarterly blocks (download.sh, prepare.sh, clear.sh: download, SECCHI_PREP, preparation, deletion of every raw
and calibrated file with a prepared product). Peak ~0.9 TB (all prepared
products plus one quarter of intermediates, ~0.25 TB), ~0.65 TB kept.
Measured sizes: AIA lev1 11.5-12.3 MB (2012), SECCHI_PREP L1 16.8 MB, prepared
products f8 primary + uint8 mask + f4 PEAK_DN. Flagged AIA slot records at
2 h: 0-1.2% per week in the sampled 2012-03, 2013-09 and 2014-06 weeks; they
are skipped by the preparation and left as gaps. Automatic frame
rejections: ~7% AIA and ~5% EUVI (polar_2025_10 frame review).

Data root: `/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2011_2014/{raw,calibrated,prepared,logs,review}`.

## Implementation

Scripts in `scripts/polar_reversal/` (README there), configs
`config/plasma/polar_reversal.yaml` (first window, full run) and
`config/plasma/polar_reversal_series.yaml` (series), driver
`sunerf.run_plasma_series`.

1. Per quarter `download.sh START END` (AIA and EUVI-A/B at 2 h in parallel),
   `prepare.sh` (SECCHI_PREP, preparation with PEAK_DN) and `clear.sh`
   (deletion of the raw and calibrated files with a prepared product).
   All slot records are downloaded; flagged AIA frames are skipped by the
   preparation (minor gaps); a week without EUVI records is listed instead
   of stopping the download.
2. `clean.sh`, after the last quarter: frame statistics per instrument and
   year, automatic rejections moved to `rejected/`.
3. `overview.sh`: observer overview of the cleaned set.
4. `run.sh`: one image-scaling table for the whole interval, then the full run
   of window 1.
5. `run_series.sh`: the series from the window-1 save state; per step a
   weights-only warm start (`training.init_state`), `training.max_steps`
   20000, learning rate 2e-4 -> 1e-4, 1-day time jitter; the job resubmits
   itself until `series.done`.

## Open decisions

- Training steps per window (20000 is a first guess) and the per-step
  learning-rate and time-jitter schedules: calibrate on the first steps.
- EUVI sensitivity: `static_assumed` over 3.5 years; the learnable gains
  (`gain_limit_dex` 0.15) can follow slow degradation through the warm start.
- Compute: 262 sequential steps; at 1-2 h per step on 4 GPUs about 11-22 days
  of wall time. Calibrate on the first steps.
