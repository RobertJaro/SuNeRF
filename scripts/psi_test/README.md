# PSI multi-view noise tests

This directory contains a split clear-preparation and detector-degradation
pipeline. tB and pB are always loaded, masked, saved, evaluated, and plotted as
separate observables. Python routines do not define an observing fleet: clear
views are discovered from the input folders, while shell arguments select any
degraded views and training YAML files select training views.

The clean tB/pB sequences are rendered from the PSI density cubes by
`scripts/psi_test/render_clean.sh`, which must run before `prepare_data.sh`. Each
view is rendered directly in its own field of view, and `prepare_data.sh`
degrades the three training views over that same field:

- L5_wide / STEREO-B COR2 pattern covers 15--30 solar radii.
- L5 / CCOR pattern covers 3.7--17 solar radii. CCOR observes total brightness
  only: its tB pattern also degrades the rendered pB, and the training configs
  read tB only (`data_path_pB: null`).
- L4 / STEREO-A COR2 pattern covers 2.5--15 solar radii.
- L1 / LASCO C2 pattern covers 2.2--8.3 solar radii.
- The polar P1 view (80 degrees latitude, 2--30 solar radii) stays clear and held out.

The two-viewpoint configs train on L5_wide, L5, and L4; the three-viewpoint
configs add L1. L5_wide, L5, L4, and L1 use radial gain profiles of 1.3--0.7
with `p=3`, 0.8--1.4 with `p=2`, 1.5--1.1 with `p=2`, and 0.6--1.2 with `p=1`,
respectively. The degradation reference frame is `0074`, i.e. cube 50 after the
24 pre-event frames.

The synthetic forward model is

```text
degraded[channel, view] = (clean[channel, view] + additive[channel, view])
                          * multiplier[channel, view]
```

The rendered views already hold their FOV, so `prepare_data.sh` passes no FOV
limits to `degrade_psi`.
The clean tB/pB image arrays, shapes, WCS, and pixel scales are retained without
cropping, resampling, resizing, normalization, or brightness clipping. Only the
noise masks are mapped to the existing image grid. The LASCO C2 and STEREO-A/B
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
brightness. The multiplicative term is independent of the observations and is
an axisymmetric radial gain
`g_inner + (g_outer - g_inner) * ((r - r_inner) / (r_outer - r_inner))**p`.
`r_inner` and `r_outer` default to the radial extent of the valid pixels of the
reference frame. `--inner-rsun` and `--outer-rsun` are optional: when given, they
anchor the gain instead and pixels outside them become `NaN` after both effects
are applied.

The learned degraded-data observation model uses `tB_add`, `pB_add`, and the
shared axisymmetric `calibration_gain`; it no longer learns image-coordinate
`tB_mul` or `pB_mul` fields for this experiment.

Install the HDF4 reader once, prepare all data in the CPU job, and then submit
the single training script:

```bash
bash scripts/psi_test/setup_density_reader.sh
./scripts/psi_test/render_clean.sh
./scripts/psi_test/prepare_data.sh
# after data preparation completes:
qsub scripts/psi_test/train.sh
```

`train.sh` is the only training entry point. In addition to the clean and
two-view comparisons, it runs `config/cme/psi_degraded_3view.yaml`, which trains
on degraded L5_wide/L5/L4/L1 and keeps the clear polar P1 image for held-out validation.
The fits retain the repository default of 200 epochs and are
checkpoint-resumable. Resubmit the script if a fit reaches its 12-hour walltime.

Evaluation requires the degraded run's `final.ckpt`, so run it only after the
degraded fit completes:

```bash
./scripts/psi_test/evaluate.sh
```

`prepare_data.sh` reads the clean sequences under
`/glade/work/rjarolim/data/sunerf-cme/psi_obs/clean` and downloads nothing. It
calls the universal `extract_noise_mask` command separately for every detector
and brightness product, with explicit input and output paths. Mask files retain
the physical instrument names (`LASCO_C2`, `STEREO_A_COR2`, `STEREO_B_COR2`,
`PUNCH`, and `CCOR`); only the subsequent degradation step associates them with synthetic
L1, L4, L5_wide, and L5. Every mask is the minimum over the daily medians of the
sequence; the pB masks are then smoothed around the Sun, with Gaussian widths of
`PB_SMOOTH_ANGLE_DEG` in position angle and `PB_SMOOTH_FRACTION` of the image
width along the radius, to suppress their radial streamer structure. The LASCO calls require four frames per daily
median to support the polarized cadence. Finally, `degrade_psi` is invoked once
each for L5_wide, L5, L4, and L1. Each invocation saves its own exact degradation
arrays and metadata.
After degradation, `plot_psi_degradation` writes three standalone preview figures
per degraded view to `${DATA_ROOT}/degradation_previews`. Each figure contains
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

`render_clean.sh` downloads the PSI/MAS 2021-10-28 density cubes
(`rho000001`--`rho000145`) and forward renders them with
`sunerf.data.psi.render_psi_thomson`. One call renders one observer:

- `--observer` is a Sun--Earth Lagrange point (`L1`--`L5`) or an explicit
  Stonyhurst position `LON_DEG LAT_DEG DISTANCE_AU`, evaluated at every frame
  time.
- `--inner-rsun`, `--outer-rsun`, and `--resolution` define a Sun-centred square
  image whose edge touches the outer impact parameter.
- Every cube yields one tB/pB pair at the observation time that `--dump-times`
  tabulates for its dump. The cubes carry no time stamp, so
  `sunerf.data.psi.dump_times` reads `SIM_DUMP` and `DATE_OBS` from the headers
  of PSI's own synthetic images of the run (header blocks only). For this run
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
A PUNCH-like field of view (21--90 solar radii in the prepared PAM mosaics)
requires the PSI heliospheric cubes.

Without `psi-io`/`pyhdf` the cubes are read by a built-in parser for plain,
uncompressed HDF4 data sets, which covers the MAS output.
