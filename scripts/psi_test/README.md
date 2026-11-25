# PSI multi-view noise tests

This directory contains a split clear-preparation and detector-degradation
pipeline. tB and pB are always loaded, masked, saved, evaluated, and plotted as
separate observables. Python routines do not define an observing fleet: clear
views are discovered from the input folders, while shell arguments select any
degraded views and training YAML files select training views.

The clean tB/pB sequences are rendered from the PSI density cubes by
`scripts/psi_test/render.sh`, which must run before `prepare_data.sh`. Each
view is rendered directly in its own field of view, and `prepare_data.sh`
degrades the four instrument views over that same field:

- PUNCH at L5 covers 21–120 solar radii. Its tB and pB degradation uses the respective PUNCH detector patterns.
- L5 / CCOR pattern covers 3.7--17 solar radii. CCOR observes total brightness
  only: its tB pattern also degrades the rendered pB, and the training configs
  read tB only (`data_path_pB: null`).
- L4 / STEREO-A COR2 pattern covers 2.5--15 solar radii.
- L1 / LASCO C2 pattern covers 2.2--8.3 solar radii.
- The polar P1 view (80 degrees latitude, 2--30 solar radii) stays clear and held out.

The two-viewpoint configs train on PUNCH, L5, and L4; the three-viewpoint
configs add L1. Every view starts with 80 pre-event frames (20 days at 6 h).
PUNCH uses the first CORHEL cube in rigid corotation for those frames, and
the coronal views use the first coronal cube. Frame `0081` starts the native
sequence at 2021-10-28 15:30 UTC. Frame `0130` is the degradation reference
and renders dump 50 at each release’s own UTC time.

The synthetic forward model mimics real coronagraph preparation: a detector
background is added, photon-like noise is drawn, and the per-pixel minimum of
the noisy series is subtracted as a daily-min correction would do.

```text
raw[t]      = (clean[t] + mask) * multiplier + noise[t]
degraded[t] = raw[t] - min_t raw[t]
```

With a time-constant mask the minimum holds the mask plus the quiet-corona
floor, so the mask cancels and the images are over-subtracted by that floor.
The noise biases the minimum low and leaves a positive residual. The truth
directory stores the mask, the subtracted background, and `additive`, the
time-constant residual `mask - background / multiplier` that the learned
`tB_add`/`pB_add` fields should recover. `--keep-background` skips the
subtraction and `--noise-fraction 0` the noise.

The rendered views already hold their FOV, so `prepare_data.sh` passes no FOV
limits to `degrade_psi`.
The clean tB/pB image arrays, shapes, WCS, and pixel scales are retained without
cropping, resampling, resizing, normalization, or brightness clipping. Only the
noise masks are mapped to the existing image grid. The LASCO C2, PUNCH and STEREO-A
daily-min masks remain independent for tB and pB. Detector patterns are
registered by their valid annulus: the radii between the occulter edge and the
outer edge of the mask map linearly onto the same range of the image at equal
position angle. Plate scale, occulter size, observer location, and observation
time are deliberately ignored.
This avoids treating stationary detector artifacts as solar features and permits
the same degradation command to operate on any target folder.

After mapping, each observational detector mask is robustly normalized to
`[0, 1]` using its 1st and 99th percentiles. `--mask-fraction 0.1` creates only
a positive additive background capped at 10% of the reference-frame p99
brightness. `prepare_data.sh` writes two degraded sets. `degraded` applies no
gain. `degraded_gain` multiplies each view by an observation-independent gain:
the product of a scalar radiometric calibration (`--calibration`), an
axisymmetric radial vignetting profile
`g_inner + (g_outer - g_inner) * ((r - r_inner) / (r_outer - r_inner))**p`, and
a second-order azimuthal modulation `1 + a * cos(2 * position_angle)`
(`--azimuthal-gain a`), with per-view values: calibration 0.85/1.10/0.95/1.15,
radial profiles 1.3--0.7 with `p=3`, 0.8--1.4 with `p=2`, 1.5--1.1 with `p=2`,
0.6--1.2 with `p=1`, and azimuthal amplitudes 0.05/-0.05/0.08/-0.08 for
PUNCH/L5/L4/L1. `r_inner` and `r_outer` default to the radial extent of the
valid pixels of the reference frame. `--inner-rsun` and `--outer-rsun` are
optional: when given, they anchor the gain instead and pixels outside them
become `NaN` after both effects are applied. The truth directories
(`degradation_truth`, `degradation_truth_gain`) store the full multiplier per
channel.

The learned observation model of the `psi_degraded_*` configs uses
image-coordinate `tB_add` and `pB_add` fields (tB only for the CCOR-like L5
view). The `psi_degraded_gain_*` configs add `tB_mul` and `pB_mul` fields for
the injected gain, trained by `train_gain_2view.sh` and `train_gain_3view.sh`.
`config/cme/psi/psi_degraded_gain_2view_cold_correction.yaml` is the 2-view
degraded_gain fit with the cold phase extended to 20000 steps and
`module.correction_schedule: cold`: the image-plane fields are learned only while
the dynamic branch is inactive and frozen once it is ramped in (the observation
counterpart is `config/cme/2025_09/2025_09_cme_cold_correction.yaml`).

From the repository root, download the PSI inputs, render the five clean views,
then prepare the training data:

```bash
bash scripts/psi_test/download.sh
./scripts/psi_test/render.sh
./scripts/psi_test/prepare_data.sh
# after data preparation completes:
qsub scripts/psi_test/train.sh
```

`train.sh` runs the clean and degraded fits; `train_gain_2view.sh` and
`train_gain_3view.sh` run the degraded_gain fits. In addition to the clean and
two-view comparisons, it runs `../../config/cme/psi/psi_degraded_3view.yaml`, which trains
on degraded PUNCH/L5/L4/L1 and keeps the clear polar P1 image for held-out validation.
The fits retain the repository default of 200 epochs and are
checkpoint-resumable. Resubmit the script if a fit reaches its 12-hour walltime.

Evaluation requires the degraded run's `final.ckpt`, so run it only after the
degraded fit completes:

```bash
./scripts/psi_test/evaluate.sh
```

All synthetic image products live under `/glade/work/rjarolim/data/sunerf-cme/psi_obs_render`:

```text
clean/
degraded/
degraded_gain/
masks/
degradation_truth/
degradation_truth_gain/
degradation_previews/
degradation_previews_gain/
```

`prepare_data.sh` reads the clean sequences from `clean/<view>/` and calls the universal `extract_noise_mask` command separately for every detector
and brightness product, with explicit input and output paths. Mask files retain
the physical instrument names (`LASCO_C2`, `STEREO_A_COR2`,
`PUNCH`, and `CCOR`); only the subsequent degradation step associates them with synthetic
L1, L4, PUNCH, and L5. Every mask is the minimum over the daily medians of the
sequence; the pB masks are then smoothed around the Sun, with Gaussian widths of
`PB_SMOOTH_ANGLE_DEG` in position angle and `PB_SMOOTH_FRACTION` of the image
width along the radius, to suppress their radial streamer structure. The LASCO calls require four frames per daily
median to support the polarized cadence. Finally, `degrade_psi` is invoked once
each for PUNCH, L5, L4, and L1. Each invocation saves its own exact degradation
arrays and metadata.
After degradation, `plot_psi_degradation` writes three standalone preview figures
per degraded view to `${RENDER_ROOT}/degradation_previews`. Each figure contains
the matched-FOV clean image, actual degraded image, equivalent additive change
and the exact injected additive mask and shared radial gain for both tB and pB.
The preview does not require a trained
model or checkpoint.
No download or preprocessing is performed by `train.sh`.

The evaluation job creates separate tB/pB image comparisons, the density
histogram and fit, and the two-row radial-slice comparison at frame 050.

The supplied density HDF4 file has no composition metadata.  Density evaluation
therefore follows the repository's existing PSI convention and assumes fully
ionized pure hydrogen: `n_e = rho_code * 1e8 cm^-3`.  The metrics JSON records
this explicitly, including the helium correction formula, so a known helium
fraction can be substituted without ambiguity.

The HDF4 density evaluation uses the official `psi-io` reader.  On a fresh conda
environment, install HDF4 support before the Python package if a compatible
`pyhdf` wheel is unavailable:

```bash
conda install -c conda-forge hdf4 pyhdf
python -m pip install -r scripts/psi_test/requirements.txt
```

Rendering and degradation receive all input,
output, mask, and truth paths plus degradation keywords explicitly. The Python modules do
not select a number of viewpoints or local/`/glade` filesystem defaults.

## Rendering clean observations

`download.sh` fetches the 2021-10-28 coronal and CORHEL density series
and builds a separate UTC table for each. `render.sh` calls the common
`sunerf.data.psi.render_psi_thomson` renderer once per view and writes a
tB/pB quicklook for each frame.

- `--observer` is a Sun--Earth Lagrange point (`L1`--`L5`) or an explicit
  Stonyhurst position `LON_DEG LAT_DEG DISTANCE_AU`, evaluated at every frame
  time.
- `--inner-rsun`, `--outer-rsun`, and `--resolution` define a Sun-centred square
  image whose edge touches the outer impact parameter.
- Every cube yields one tB/pB pair at the observation time that `--dump-times`
  tabulates for its dump. The cubes carry no time stamp, so
  `sunerf.data.psi.dump_times` reads `SIM_DUMP` and `DATE_OBS` from the headers
  of PSI's own synthetic images of the run (header blocks only). For the coronal series
  the dumps follow at 5 min from 2021-10-28T15:30 to 2021-10-29T03:30.
- `--pre-duration` with `--pre-cadence` prepends frames before the first dump.
  They render the first cube, the relaxed steady state, in rigid corotation.

The rays are constructed by the same `MapDataLoader` that training uses, and the
output follows the PSI test layout, so `degrade_psi` applies detector effects
next:

```text
<out-dir>/tb/<KEY>_tb<NNNN>.fts   # chronological index; SIM_DUMP holds the cube
<out-dir>/pb/<KEY>_pb<NNNN>.fts
<out-dir>/manifest.json           # time and dump of every frame
<out-dir>/jpg/<tB>__<pB>.jpg      # quicklooks from quicklook_coronagraph_video
```

The quicklooks use `--a 1e-6`: the default asinh stretch leaves the outer corona
black over the dynamic range of a clean 2--30 solar-radius view.

Images are in mean solar brightness (`BUNIT = MSB`), the unit of the prepared
observations. PSI's own `getpb` images are lower by the constant factor
`1 - u/3 = 0.79`; apart from this normalization the renderer reproduces the PSI
L1 frame-050 tB and pB images to better than 1% per pixel.

### Pre-event frames

The cubes are stored in the corotating Carrington frame. A steady state that
corotates rigidly satisfies the continuity equation with
`v = v_corotating + Omega x r`, so rendering the first cube at earlier times is
physically consistent, can be arbitrarily long, and adds rotational parallax.
Cubes are never interpolated in time: cross-fading a moving front violates
continuity. Extending the series after the last dump requires further
simulation output.

### Limits of the coronal cubes

The mesh ends at 30.2 solar radii and every line of sight is integrated only
inside that sphere, exactly as in PSI's images and as in training with
`sampling.max_distance` near 30. Relative to an unbounded `r^-2` wind this loses
about 2% of the brightness at 10, 6% at 15, 15% at 20, and 33% at 25 solar radii.
The PUNCH view out to 120 solar radii uses the CORHEL cubes.

Without `psi-io`/`pyhdf` the cubes are read by a built-in parser for plain,
uncompressed big-endian float32 and float64 HDF4 data sets.

### Synthetic PUNCH white-light images from CORHEL

For PUNCH, `download.sh` fetches the 145 CORHEL cubes (44.8 GB) and reads
fakeHI FITS headers for UTC timestamps.

The download and render scripts use `/glade/work/rjarolim/data/sunerf-cme/psi_obs`
for coronal inputs and `/glade/work/rjarolim/data/sunerf-cme/psi_obs_corhel`
for CORHEL inputs. Each folder contains its own `rho/` and `dump_times.json`.
The full PUNCH run produces 225 L5-view tB/pB FITS pairs (80 pre-event frames
plus 145 native CORHEL dumps) on a
512 × 512 Sun-centered grid. The fixed 21–120 R☉ field approximates the
20,000″ inner cutoff at 1 AU. The output is under
`/glade/work/rjarolim/data/sunerf-cme/psi_obs_render/clean/PUNCH/{tb,pb}`,
with dump IDs and times in the adjacent `manifest.json`.

**Inspected input.** The October 2021 CORHEL release lists 145 density dumps.
Dump 50 is a 308,975,832-byte HDF4 file containing one big-endian float64 field
with dimension scales. In array order it has 360 longitudes (0 to 2π), 180
colatitudes (0 to π), and 596 nonuniform radii (1.00 to 236.46 solar radii).
All 38,620,800 density values are finite and positive, ranging from
`1.3193e-8` to `173.0765` in code units. With the pure-hydrogen convention
above, the median over angular grid nodes is 20.29 electrons/cm³ at 89.83
solar radii and 4.00 at 215.15 solar radii. This single grid covers the corona
and heliosphere along the same ray.
[CORHEL release](https://www.predsci.com/~epalmerio/issi_rho3d/20211028/corhel/)

**Time registration.** The CORHEL index gives model time 122.018378 for dump
50. PSI's fakeHI header independently gives 122.01838 and UTC
2021-10-30 16:30:23. The same-numbered fakeC3 image is dated 2021-10-28
19:35:00. The first and last fakeHI headers span 2021-10-28 15:30:00 to
2021-11-03 15:31:12 UTC, with roughly hourly dumps. Generate a dedicated
CORHEL dump-time table with `sunerf.data.psi.dump_times` pointed at the fakeHI
pB directory. Join any coronal or in-situ observations by physical time. Use
the native dump times for the synthetic movie.
[CORHEL time index](https://www.predsci.com/~epalmerio/issi_rho3d/20211028/corhel/mas_dumps_3d.txt)
and [PSI heliospheric reference frames](https://www.predsci.com/~epalmerio/getpb/20211028/fakeHI/fits_L1e/pb/)

**Pilot check.** Reading the float64 payload and scales directly allowed a
sparse check with the existing density interpolator and Thomson kernel.
On an 8 × 8 grid of pixels in each of the dump-50 L1 east and west fakeHI
images, the median modeled/reference pB ratios were 0.993 and 0.990 after
multiplying modeled MSB by 0.79. The respective 10th-percentile ratios were
0.942 and 0.904. Doubling LOS samples from 512 to 1024 changed tB and pB by
less than 0.06% on every sampled ray. The full-image comparison should locate
the low-ratio regions and check the absolute tB scale independently.

**Local render.** Use the CORHEL-only density directory and its fakeHI UTC
table. For the single downloaded dump, the table has the form
`{"dump_times": {"50": "2021-10-30T16:30:23.000"}}`. From the repository root:

```bash
python -m sunerf.data.psi.render_psi_thomson \
  --density-dir data/psi_test/corhel --dump-times data/psi_test/corhel/dump_times.json \
  --observer L5 --key PUNCH --inner-rsun 21 \
  --resolution 512 --outer-rsun 120 --n-samples 512 --ray-batch 512 --device cpu \
  --out-dir results/corhel/punch_l5_dump0050
```

1. **Read the density grid.** The plain HDF4 reader decodes the field and scales
   from the number-type descriptor for both coronal float32 and CORHEL float64
   files. Keep the CORHEL data and time table in their own directory because
   the two releases reuse filenames.
2. **Define the camera at L5.** Use the common coronagraph grid with a fixed
   21–120 R☉ annulus and compute the observer position at each dump time.
3. **Render absolute K-coronal brightness.** Reuse the finite-Sun Thomson
   kernel, Carrington density interpolation, and WCS-derived unit rays.
   Select pixels by impact parameter. Integrate from the
   observer, who lies inside the grid, to the far intersection with the
   236.46-solar-radius boundary. The 120-solar-radius image cut selects rays
   independently of this integration limit. Construct the sampling nodes
   inside each ray batch to bound memory. Write paired float32 tB/pB FITS in
   MSB with dump ID, UTC, observer, WCS, density scale and source provenance.
   In the PAM polarization basis the ideal Thomson signal has pB′ = 0.
4. **Validate dump 50 before scaling up.** Render in fakeHI's own geometry
   first. Compare absolute tB, pB and pB/tB, radial profiles and CME position
   angle with PSI for the same dump and observer. Account for the disk
   brightness normalization described above. At PUNCH elongations, compare
   512 and 1024 samples per ray and quantify sensitivity to the finite outer
   boundary. Check `0 ≤ pB ≤ tB` on clean pixels, then round-trip a synthetic
   pair through the training loader and verify its rays, units and footprint.
5. **Generate the time series and detector variant.** After the pilot,
   render each native dump once per requested view (44.8 GB of input for the
   full series). Keep clean truth and make the observational variant with
   the PUNCH channel-specific masks already extracted in `prepare_data.sh`.
   Map the detector patterns onto the synthetic coronagraph grid. Estimate noise
   from PAM uncertainties or residuals, and record any background subtraction
   with the degradation truth. For a single-frame pilot, render clean truth
   first since a temporal minimum needs a series. The training configs integrate PUNCH rays to 236.46 solar radii and
   the other views to 30.2 solar radii. The CME interval extends to the final
   CORHEL dump. The `punch_l5` run directory keeps this instrument set separate
   from previous checkpoints.

## Two-stage fit: background, then CME with fixed degradation

`config/cme/psi/psi_degraded_gain_2view_background.yaml` and
`config/cme/psi/psi_degraded_gain_2view_cme.yaml` split the degraded+gain 2-view
fit into two runs:

1. `train_background.sh` trains on the background datasets only: every view at
   a fixed 6 h cadence over the full series, eruption phase included, so the CME
   frames carry no extra weight. The physics samples cover the full series. This
   stage learns the corotating corona and the image-plane `tB_add`/`pB_add`/
   `tB_mul`/`pB_mul` fields.
2. `train_cme.sh` continues from `final.ckpt` of stage 1 (`training.init_path`
   loads the model weights only; optimizer and step count start fresh) with the
   degradation fields fixed (`correction.trainable: false`) and two datasets per
   view, the 6 h background series and every frame of the eruption phase. The
   dynamic branch is already trained, so `cold_steps` and `warm_steps` are 0.
   Submit it after stage 1 has written `final.ckpt`. Like every run, it resumes
   from its own `last.ckpt` when one exists and reads `init_path` only on the
   first start, so it can be resubmitted after the walltime.

Both stages share `ref_date`, instruments and model layout, so the checkpoint
loads strictly.

## Synthetic in-situ probe

`scripts/psi_test/prepare_insitu.sh` (after `download.sh`) writes two
virtual-spacecraft products with `sunerf.data.psi.synthetic_insitu` into
`psi_obs/insitu/`: `PSP_synth.npz`, a probe that follows the real PSP
encounter-25 trajectory (JPL Horizons, perihelion 2025-09-15T20:18 at 9.86
solar radii) shifted so that the perihelion falls at 2021-10-27T20:00 and
rotated so that the probe sits at the CME Carrington longitude (245 degrees) at
20:00 UT of the eruption day; and `PSP_synth_heldout.npz`, the same orbit
rotated 30 degrees east. During the eruption the training probe is outbound at
17--20 solar radii, where the CME body passes it. Every dump gives one sample
at its tabulated time; the 20 pre-event days are sampled at 1 h from the
corotating first cube, and samples outside the cubes are dropped, so the probe
enters the series about two days before the eruption.

The products use the format of the prepared PSP data and load through the `psp`
dataset type. `<probe>_truth.npz` holds the clean samples; the delivered file
carries lognormal density scatter (sigma 0.15), 25 km/s velocity errors and
velocity gaps in 20 % of the 30 min windows. The public PSI release holds no
velocity cubes, so `has_velocity` is 0 until `--velocity-dir` points to MAS
`vr`/`vt`/`vp` cubes (the tangential component then receives the corotation
term).

`config/cme/psi/psi_degraded_gain_2view_insitu.yaml` adds the training probe
to the degraded+gain 2-view setup (`insitu_density` and
`insitu_velocity_radial` at 1e-2, background/CME split as for the images) and
validates on both probes (`insitu_timeseries` callbacks). `train_insitu.sh`
runs it; `evaluate.sh` exports the reconstruction along both probes
(`export_insitu`) and draws plot 8 (`plot_insitu`).
