# EUV wave of 2022-03-28, three-view reconstruction

Reconstruction of the EUV wave associated with the M4.0 flare in AR 12975.
The wave interval is 11:05–11:45 UT; the focus window includes pre-event and
post-event context from 10:30 to 12:30 UT.

| Setting | Value |
|---|---|
| STEREO-A | ~33 deg east of Earth, 0.967 au |
| Solar Orbiter | ~83.5 deg west of Earth, 0.325 au; source near its east limb |
| Focus | 2022-03-28 10:30 .. 12:30 UT, requested cadence 1 min |
| Background | 2022-03-21 10:30 .. 2022-04-04 12:30 UT, 1 h cadence; focus included |
| Validation target | 2022-03-28 11:20 UT, nearest focus observation per view |
| Temporal jitter | Gaussian width 604800 s (7 days), linearly decaying to zero over 10000 steps |
| Channels | AIA 94/131/171/193/211/335; EUVI-A 171/195/284; EUI/FSI 174 |

Background coverage extends seven days before the focus start and seven days
after the focus end. Geometry is from the NASA STEREO orbit tool at
2022-03-28 11:20 UT. EUVI downloads complete channel sets anchored on 284;
the available native cadence limits EUVI and FSI sampling even when the
requested focus cadence is one minute. FSI 304 is chromospheric and excluded
from this plasma model.

```bash
scripts/plasma_euv_wave/all_data.sh     # download.sh, then prepare.sh
scripts/plasma_euv_wave/review_frames.sh # flag invalid frames (server)
python -m sunerf.data.frame_review review <review dir>          # local copy
python -m sunerf.data.frame_review apply <review root>              # server
qsub scripts/plasma_euv_wave/run.sh
qsub scripts/plasma_euv_wave/evaluation.sh   # videos of the focus window
```

`evaluation.sh` renders the focus window with the generic EUV video tool
`sunerf.evaluation.plasma.video` (the plasma counterpart of
`sunerf.evaluation.cme.video`): the camera follows waypoints in position and
time or a real spacecraft (`--observer earth|stereo_a|solo|...`, or a training
instrument key), and `--panels` picks the columns of every frame: rendered
channels of the training instruments (`aia193`, `euvi284`, `eui174`),
`density` (column electron density), `temperature` (emission-weighted
log T), `cool_column` (line-of-sight hydrogen column of the cool absorber,
only for models with `model.cool_absorber`), `em` (emission measure integrated radially above the footpoint of
every disk pixel over the rendered shell, `--em_radius_range`, with
`--em_n_radius` trapezoid samples, in two bins below and above log T = 6.1 over the
diagnostic grid or one explicit bin `em:5.8-6.3`; off limb is blank), `distance`
(emission-weighted distance from the observer relative to the Sun center)
and `observer` (3D sketch with the training spacecraft). Default:
`aia193 density em observer` with the two bins. The color scales are fixed by
the first frame, so a video that starts before the flare shows the wave as a
change against the quiet corona. The script produces the Earth view and the
near-quadrature Solar Orbiter view at 1 min over the focus window and a moving view
from Earth at the focus start to Solar Orbiter at the end of the window, time
and viewpoint advancing together; frames land under `${RUN_ROOT}/video_*` with the
encoded `.mp4` next to each directory. `sunerf.evaluation.plasma.video_em_maps`
adds a viewpoint-independent video: on a fixed equal-area Carrington grid
(`--n_lon` by `--n_lat`, uniform in longitude and sin(latitude)) the model is
integrated radially (`--radius_range`, default the rendered shell from
`sampling.min_distance`, `--n_radius` trapezoid samples) per
temperature bin (emission measure, `--em_bins` over the diagnostic grid or
`--em_edges`) together with the column electron density, between
`--time_start` and `--time_end` at `--cadence_minutes`.

Data live under `/glade/campaign/hao/radmhd/rjarolim/EUV_WAVE/data`, the run
under `/glade/work/rjarolim/sunerf/euv_wave_2022_03_plasma_v06`. Every
instrument is sampled in two tiers, each in its own directory (`<instrument>`,
`<instrument>_focus` under the raw, calibrated and prepared roots). The config
has one dataset per instrument and tier; only the focus datasets hold out a
validation frame, the observation closest to 11:20 UT, the
background is trained on completely. The AIA export is one JSOC request per
chunk (1 d background, 1 h focus) with all channels, the EUVI download is
chunked one week per VSO query and the VSO/SOAR searches retry on transient
failures.

The model is a Swish MLP on the isotropic positional encoding
(`positional_isotropic`, spatial and temporal maxima `2 ** 7.5 * pi`), as in
every plasma config. The spatial maximum is the Nyquist limit of the prepared
512 px images over 2.8 R_sun; the earlier `2 ** 10 * pi` (v03) left sub-pixel
ripples in the emission-measure bins.
Temporal jitter is applied during the first 10000 steps, then exact
observation times are used. The loss divisors are the packaged polar
2025-10 table; the flare brightens AIA 94/131 far beyond that table's
percentiles, so re-estimate with `sunerf.data.euv.estimate_scaling` if the
residuals are dominated by the flare pixels.

## Frame review

`review_frames.sh` runs `sunerf.data.frame_review scan` on every prepared
dataset. The scan is generic: it reads any 2D FITS sequence (raw, calibrated
or prepared, with the `VALID_MASK` extension), groups it by observatory,
detector and channel from the header, and computes per frame the valid-pixel
fraction, intensity percentiles, saturated pixels and three scores against the
three neighbouring frames of the same channel on each side: the jump of the
median and of the 99.9th-percentile log intensity, and the median absolute log
difference to the neighbours' pixelwise median (structure). Robust z-scores
(median and MAD over the channel) turn them into flags:

| Status | Condition |
|---|---|
| reject | unreadable file, nonzero `QUALITY`, valid fraction < 0.9 of the channel median, > 1e-4 of the detector saturated, level or structure \|z\| > 6 |
| warn | > 1e-6 saturated, valid fraction < 0.98 of the median, level or structure \|z\| > 4, peak \|z\| > 4 (flares), exposure < 0.5 of the channel median (AIA flare-mode exposure control), duplicate time |

Saturation is counted from the data only for detector counts (`BUNIT` without
a per-second unit) of AIA and SECCHI, otherwise from `NSATPIX`. Prepared
products are exposure normalized and degradation corrected, so the AIA scan
reads `NSATPIX` from the level-1 file named by `SRCFILE` (`--source-dir`);
products prepared from now on carry `NSATPIX` themselves. Saturated pixels
are masked by the reconstruction, not in the prepared products (see
`config/plasma/README.md`).

The review runs in three steps, so the FITS data never leave the server:

1. **Scan (server).** `review_frames.sh` writes one review directory per
   dataset: `frames.csv` (statistics, automatic status and reasons, one row
   per file, with its absolute server path), `scan.json`, `thumbs/` (JPEG
   previews, a few kB per frame) and `index.html`.
2. **Review (local).** Copy the review directories (`rsync -av
   casper:<review root> .`) and run
   `python -m sunerf.data.frame_review review <review dir>`. It serves the page
   on localhost and opens it; every decision is saved immediately to
   `<review dir>/decisions.csv` (`path, decision, auto_status, group_name,
   time, reasons`, `decision` is `reject` or `keep`), and reopening continues
   from it. The page shows the selected metric of every channel over the full
   sequence (scroll to zoom in time, drag to pan); the current observation with
   all channels side by side, the previous observation small above and the next
   below (a channel without a frame at that time is a blank panel; click a
   panel to select it), with the statistics and flags of the selected frame;
   and a grid of all frames, one row per observation time and one column per
   channel. Observations are anchored on the channel with the most frames;
   another channel's frame joins the nearest anchor within half its cadence.
   Every channel uses its SunPy instrument colormap (AIA, EUVI, EUI, SUVI,
   EIT; gray otherwise) on one log10 range for the whole sequence: the minimum
   of the 1st and the maximum of the 99.9th percentiles over all accepted
   frames, at most four decades deep (`scan.json` lists it per channel). Invalid (and, in prepared
   products, saturated) pixels are gray; saturated pixels of raw detector data
   are red.
   Rejected frames start as the automatic rejections; `x` toggles a frame, `r`
   a whole observation row, the arrow keys move through frames and rows,
   `n`/`p` jump to the next or previous flagged frame.
   **Saturation level.** The bar under the header selects
   `instruments[].saturation_dn` of the plasma config, which `run_plasma`
   applies while loading the data. Every frame's peak detector DN in those
   units comes from its prepared header (`PEAKDN`, `PEAKNPX`). Saturated frames
   share a ceiling: when at least two frames have a plateau of three or more
   native pixels within 1% of a channel's highest peak, that peak is the
   saturation, and 0.8 of it (rounded down to three digits) is recommended,
   the lowest over the channels of the dataset. The metric "Peak detector DN"
   plots every frame's peak with the detected saturation (dotted) and the
   selected level (dashed); the bar counts the frames per channel that reach
   it and shows the config line. **Save saturation.csv** stores the choice in
   the review directory; put the value into the config, no new preparation
   is needed. Without a plateau (no saturated frame) nothing is recommended.
3. **Apply (server).** Copy the `decisions.csv` files back (`rsync -av
   --include='*/' --include='decisions.csv' --exclude='*' review/
   casper:<review root>/`) and run
   `python -m sunerf.data.frame_review apply <review root>` (first with
   `--dry-run`). It walks every review directory below the folder: one with a
   `decisions.csv` applies it, one without applies the automatic flags of its
   `frames.csv`, and the report names the file used for each. Rejected files
   move into `rejected/` next to them, which the dataset globs no longer match,
   and the move is logged in `rejected/rejected.log`; `restore <prepared dir>`
   moves them back. A single review directory, or a single CSV with `path` and
   `decision` columns (e.g. `frames.csv` edited in a spreadsheet), works as
   well.
