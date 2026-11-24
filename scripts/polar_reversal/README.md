# Polar field reversal 2011-2014

EUV plasma reconstruction of both polar coronae through the Cycle 24 polar
field reversal from SDO/AIA, STEREO-A/EUVI and STEREO-B/EUVI, as a series of
27-day windows moving forward in 5-day steps. Background and plan:
`docs/polar_reversal_plan.md`.

| | |
|---|---|
| Interval | 2011-02-06 .. 2014-10-01 (STEREO-A/B 180 deg apart .. STEREO-B lost) |
| Cadence | 2 h for every instrument (native EUVI 171/284; AIA at the same rate) |
| Window | 27 days (one rotation), all frames at 2 h |
| Focus | central 5 days, a second dataset per instrument (oversampled), same 2 h frames |
| Step | 5 days; 262 windows, step 0 is the full run |
| Time origin | `data.ref_date: 2011-02-06T00:00:00` in every step |
| Data root | `/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2011_2014` |

## Workflow

| Script | Input | Output |
|---|---|---|
| `all_data.sh` (PBS) | | the three scripts below per quarter, `logs/<quarter>.done` |
| `download.sh START END` | JSOC, VSO | `raw/{aia,euvi_a,euvi_b}`, logs in `logs/` |
| `prepare.sh` | every file in `raw/` | `calibrated/euvi_{a,b}` (SECCHI_PREP, A and B in parallel, `logs/secchi_prep_euvi_{a,b}.log`), `prepared/{aia,euvi_a,euvi_b}` |
| `clean.sh` | `prepared/` | `review/<dataset>_<year>`, rejected frames moved to `prepared/*/rejected/` |
| `clear.sh` | `raw/`, `calibrated/` | deletes every file with a prepared product |
| `overview.sh` | `prepared/` | `prepared/observer_overview.jpg` |

1. `qsub scripts/polar_reversal/all_data.sh` (PBS, 8 CPUs): per quarter
   `download.sh`, `prepare.sh`, `clear.sh`, so only one quarter of raw and
   calibrated files (~0.25 TB) exists at a time. A finished quarter is marked
   `logs/<quarter>.done` and not queried again; the job resubmits itself
   until all 15 quarters are marked, and an interrupted quarter resumes where
   it stopped: downloads skip records with a raw file or a prepared product
   (also in `rejected/`, so files deleted by `clear.sh` are not fetched again;
   an AIA chunk with every record present is not exported), SECCHI_PREP skips
   existing calibrated files (written through a staging directory, so a cut-off
   run leaves no truncated file) and the preparation existing products. Every
   slot record is downloaded; flagged AIA frames (eclipse,
   dark, calibration frames, ~1% at 2 h) are skipped by the preparation, leave
   a gap and stay in `raw/aia`. Weeks without EUVI records are
   listed at the end of the EUVI logs (expected for real gaps, e.g. STEREO-A
   sidelobe operations from ~2014-08-20).
2. `clean.sh`, once all quarters are prepared: frame statistics per
   instrument and year and the automatic rejections moved to `rejected/`.
   The level and structure jump flags compare neighboring frames, so they
   need the full time series. `clear.sh` does not depend on it (a rejected
   product counts as prepared). Check the printed saturation levels against
   `instruments[].saturation_dn` (13000) of the config.
3. `overview.sh`: observer positions, view directions and coverage of the
   cleaned set.
4. `run.sh` (PBS): image-scaling table over the whole data set, then the full
   run of the first window (`config/plasma/polar_reversal.yaml`); resubmit until
   it is converged.
5. `run_series.sh` (PBS): the series (`config/plasma/polar_reversal_series.yaml`)
   from the save state of step 4; the job resubmits itself until
   `series.done` exists. Check the windows first:
   `python -m sunerf.run_plasma_series --config config/plasma/polar_reversal_series.yaml --dry-run`.

Raw files are not kept: reprocessing a quarter (e.g. another field of view)
means moving its prepared products away and downloading it again.

## Series

`sunerf.run_plasma_series` shifts every `start`/`end` of the training datasets
and the validation `time` of the base config by `k * step_days` and runs
`sunerf.run_plasma` on the result with the series `overrides`:
`training.init_state` (weights of the previous step), `training.max_steps`,
a shorter learning-rate decay and a 1-day time jitter. Per step,
`<series base_path>/steps/<index>_<focus center>/` keeps `config.yaml` and
`save_state.snf`; checkpoints and the data cache of a finished step are removed.

The per-step budget (`max_steps: 20000`) is a first guess: time the first
steps and compare their validation residuals with the full run before
letting the chain run through.

## Data volume (2 h)

| Product | Files | Kept | Volume |
|---|---|---|---|
| AIA level 1, 6 channels | ~96 000 | one quarter at a time | 1.15 TB in total, ~0.08 TB per quarter |
| EUVI-A/B level 0.5 | ~95 000 | one quarter at a time | 0.80 TB in total, ~0.05 TB per quarter |
| EUVI calibrated | ~95 000 | one quarter at a time | 1.59 TB in total, ~0.11 TB per quarter |
| AIA prepared | ~96 000 | yes | 0.31 TB |
| EUVI prepared | ~95 000 | yes | 0.32 TB |
| Save states, 262 steps | 262 | yes | ~3-10 GB |

Peak disk use ~0.9 TB (all prepared products plus the intermediates of one
quarter); ~0.65 TB at the end.
