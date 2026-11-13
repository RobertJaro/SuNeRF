# Solar Orbiter polar validation, north pass October 2025

Held-out test of whether an ecliptic reconstruction (SDO/AIA + STEREO-A/EUVI)
predicts the north polar corona seen by Solar Orbiter/EUI FSI 174 from
+15 .. +16.8 deg heliographic latitude, 11 deg further north than any training
view. Geometry from JPL Horizons (queried 2026-09-20):

| | |
|---|---|
| Solar Orbiter above +15 deg | 2025-10-04 .. 2025-10-28, peak +16.8 deg on 10-14 |
| Solar Orbiter distance | 0.50 .. 0.73 au |
| Solar Orbiter longitude vs Earth | -1 .. +16 deg (almost on the Earth-Sun line) |
| STEREO-A longitude vs Earth | +48 deg |
| Earth B0 | +6.4 .. +4.8 deg |
| Validation time (SolO comparison) | 2025-10-14T09:00, Solar Orbiter latitude maximum (+16.84 deg, 8.3 deg east of Earth, 0.59 au); latitude stays above +16.5 deg 10-09 .. 10-21 |
| Training window | 2025-09-27 .. 2025-11-05 at 1 h (AIA 94/131/171/193/211/335, EUVI-A 171/195/284) |
| Focus window | 2025-10-11T21 .. 10-16T21 at 1 h, 5 days centered on the latitude maximum; trained 1:1 with the full window (one batch each per step) |
| Held-out window | 2025-10-05 .. 2025-10-23T18 at 6 h (FSI 174 L2; symmetric about the latitude maximum, SOAR holds no frame on 10-04) |

FSI 304 is chromospheric and the plasma model has no cool component, so only
174 is used. FSI has a single coronal channel: this validates emission and
geometry at the pole, not the temperature.

```bash
scripts/plasma_polar_2025_10/all_data.sh       # download.sh, then prepare.sh
scripts/plasma_polar_2025_10/review_frames.sh  # flag invalid frames (server)
qsub scripts/plasma_polar_2025_10/run.sh
qsub scripts/plasma_polar_2025_10/evaluation.sh  # Solar Orbiter validation figures
```

Data live under `/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2025_10` (`raw`, `calibrated`, `prepared`), the run under
`/glade/work/rjarolim/sunerf/polar_2025_10_plasma`. AIA and EUVI-A are sampled
at 1 h over the whole window, FSI as a single 6 h sequence. The focus
datasets (`AIA-focus`, `EUVI-A-focus`) reuse the same prepared files with a
`start`/`end` window, so each training step draws one batch from the full
window and one from the focus window. The validation
datasets render the AIA, EUVI-A and FSI frames closest to the latitude maximum.
The AIA export is one JSOC request per day with all channels, the EUVI
download is chunked one week per VSO query and the VSO/SOAR searches retry on
transient failures. The config
keeps the EUI gain fixed (`learnable: false`) so the FSI prediction is an
absolute test of the packaged FSI response; the validation-only dataset renders
the frame closest to its `time`, the latitude maximum, during training, and the
full sequence is rendered afterwards with the evaluation tools.

## Field of view

`prepare.sh` crops every instrument to 1.4 R_sun on a 512x512 grid, so AIA,
EUVI-A and FSI cover the same physical extent (about 180 pixels per R_sun).
An optional `max_radius_rsun` per dataset in
`config/plasma/polar_2025_10.yaml` narrows the field of view at load time
without preparing again.

## Frame review

`review_frames.sh` runs `sunerf.data.frame_review scan` on the three prepared
datasets (`aia`, `euvi_a`, `eui`) and writes one
review directory per dataset under
`/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2025_10/review`. The
flags, the review page and the saturation selection are documented in
`scripts/plasma_euv_wave/README.md`.

1. **Scan (server).** `scripts/plasma_polar_2025_10/review_frames.sh`.
2. **Review (local).**
   `rsync -av casper:/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2025_10/review/ data/plasma_polar_2025_10/review/`,
   then `python -m sunerf.data.frame_review review data/plasma_polar_2025_10/review/<dataset>`.
   Decisions go to `decisions.csv`, the chosen saturation level to
   `saturation.csv`; copy that value into `instruments[].saturation_dn` of the
   config.
3. **Apply (server).** Copy the `decisions.csv` files back and run
   `python -m sunerf.data.frame_review apply /glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2025_10/review`
   (first with `--dry-run`). Rejected files move into `rejected/` next to the
   prepared files, outside the dataset globs; `restore <prepared dir>` undoes
   it. A rejected FSI frame leaves the validation sequence; the validation
   `time` then picks the next closest frame to the latitude maximum.
