# SuNeRF-CME reconstruction of the September 2025 event

SuNeRF-CME uses white-light images from Earth’s vicinity and STEREO-A to reconstruct how coronal electron density changes in three dimensions. A single model explains six observing sequences through their individual viewing geometries. Total brightness (tB) constrains the electron column along a line of sight. Polarized brightness (pB) supplies a second dependence on scattering angle and helps localize material in depth. The model also infers a velocity field from density evolution and solar-wind priors. Parker Solar Probe (PSP) and Solar Orbiter supply independent trajectory comparisons in this configuration.

The event configuration is **`2025_09_cme_v25`**. The dates below are UTC. The observing counts come from the local frame-review files. Training settings and downstream products describe the configured v25 run. The local results include earlier inference states through `2025_09_cme_v23`. Completion on the cluster and numerical v25 results for the CME speed, mass and front position are unverified here. [Event configuration](../../config/cme/2025_09/2025_09_cme.yaml) · [Training entry point](../../sunerf/run_thomson.py)

## Observation intervals

| Purpose | Interval | How the pipeline uses it |
|---|---|---|
| Monthly observations | Sep 1 00:00 to Oct 1 00:00 | Request the September imaging and in-situ records. Each imager’s background loader reads its full reviewed series, including the CME. Sep 1 00:00 is the model time origin. |
| CME training window | Sep 21 12:00 through Sep 23 12:00 | A second loader per imager repeatedly samples these 48 hours. A second set of physics points covers the same interval. |
| Image validation | Sep 22 08:00 | Select the closest available observation separately for each sequence. These frames come from the training sources and measure reconstruction fidelity. |
| HI-1 comparison | Sep 21 00:00 to Sep 24 00:00 | Request STEREO-A HI-1 Level-0 FITS and SSC browse JPEGs at 4-hour cadence. Compare against the reconstruction over Sep 22 00:00 to Sep 23 06:00. HI-1 is independent of the fit. |
| Quantitative grid export | Sep 21 18:00 through Sep 23 06:00 | Export 73 half-hourly snapshots spanning the pre-event state and propagation. |

## Data products and instrument filters

[all_data.sh](all_data.sh) runs acquisition and preparation. The primary imagers use all available files matching their product selectors, with no additional cadence selection in the YAML. Each preparation script requests **512 × 512** FITS maps with pointing and observation time. Common preparation masks nonpositive brightness and the predicted positions of Mercury, Venus and the Moon, crops to the selected field where applicable, and resamples after masking. Angular limits below are measured from Sun center. [Shared map preparation](../../sunerf/data/coronagraph/prep_common.py)

| Observing sequence | Acquisition and prepared product | Instrument-specific selection |
|---|---|---|
| STEREO-A SECCHI/COR2 polarization | VSO SECCHI exposures. SolarSoft `SECCHI_PREP` forms tB and pB from the 0°, 120° and 240° triplets. Output under `prep/cor2/{tB,pB}`. | Match the other polarizers to the 0° exposure within 300 s. Require `SEB_PROG=NORMAL` in both derived maps. Reject the pair if either map has more than 50% NaN or nonpositive pixels. Retain 4,000–15,000 arcsec. |
| STEREO-A SECCHI/COR2 clear | Clear/double-exposure files from the same download, processed separately with `SECCHI_PREP`. Output tB under `prep/cor2_clear`. | Require `N_IMAGES` equal to 3 or 6. Use the same 4,000–15,000 arcsec annulus. This sequence has its own correction model and temporal sampling. |
| PUNCH Level-3 PAM | Select `0m.fits` products. From the three brightness planes, use B for tB and the quadrature magnitude of pB and pB′ for polarized brightness. Preparation writes `prep/punch_pam/{tB,pB}`. Review and v25 training read **`prep_stash/punch_pam/{tB,pB}`**. | Skip files with nonzero `OUTLIER` or `BADPKTS`. Mask nonpositive B or pB in its channel, set nonpositive pB′ to zero, and mask both channels where both primary uncertainty planes are zero. Use a 20,000 arcsec inner mask, a 120 R☉ projected outer limit, and 5,000 arcsec object masks. The preparation script does not populate the stash. |
| GOES-19 CCOR-1 | Level-2 science FITS, extension 1 tB and extension 2 quality mask. Output under `prep/ccor`. | Require `ISNORMAL`, `DCMPRS_Q`, `IMGBLK_Q` and `EARFOV_Q`. Mask pixels with nonzero quality-mask values or nonpositive brightness and undo the recorded quarter-turn rotation. The quality gate permits near-boresight Earth-limb intervals when Earth is outside the field. |
| SOHO/LASCO C2 and C3 | Download Level-0.5 clear exposures, SOHO roll information and monthly exposure factors. SolarSoft `REDUCE_LEVEL_1` calibrates the frames. Output tB under `prep/lasco_c2/tB` and `prep/lasco_c3/tB`. | Preserve detector orientation and mask raw zero/BLANK values before brightness scaling. Use 2,100–8,000 arcsec for C2 and 4,000–29,000 arcsec for C3. Mask pixels brighter than ten times the fitted radial brightness background. |

**In-situ products.** PSP acquisition requests SWEAP/SPC Level-3 ions, FIELDS quasi-thermal-noise (QTN) density and hourly COHO merged data. Preparation combines QTN and SPC density, giving QTN precedence at duplicate timestamps, and takes RTN velocity from SPC. Solar Orbiter uses hourly COHO merged plasma records, with proton density used as an electron-density proxy at factor 1.0. Both preparation routines attach interpolated heliocentric inertial (HCI) spacecraft positions and store time, density in cm⁻³, velocities in km s⁻¹, source and validity flags in the monthly NPZs. The nearest-measurement tolerances are 300 s for PSP and 3,600 s for Solar Orbiter. Training and validation loaders select samples at or inside 120 R☉. Both in-situ loss weights are zero, so the time series test the image-derived solution without steering it. [PSP preparation](../../sunerf/data/prep/psp_insitu.py) · [Solar Orbiter preparation](../../sunerf/data/prep/solo_insitu.py)

## Frame review and fixed background subtraction

After preparation, the server scans each sequence and channel. It flags unreadable frames, anomalous observing modes, pointing or exposure changes, and changes in coverage, intensity or image structure. The CME classifier compares valid area against the channel median (warning below 95%, rejection below 85%). It also tests per-frame statistics against a local six-hour baseline, iteratively excluding outliers beyond six robust standard deviations. The local dashboard lets the reviewer override frame decisions and mark individual bad pixels. Applying a review moves rejected FITS into a sibling `rejected/` directory and installs pixel masks for the loader. A saved decision file takes precedence over automatic flags. [Server review and apply workflow](review_frames.sh) · [Local review](review_local.sh) · [Outlier criteria](../../sunerf/data/coronagraph/detect_outliers.py)

The following counts are from the six saved review sets in `data/2025_09/review/`. “Kept” counts files marked keep before pairing and pixel-level exclusions. The event column uses the 48-hour training window.

| Prepared channel | September files scanned | Kept | Rejected | Kept in CME window |
|---|---:|---:|---:|---:|
| COR2 triplet tB | 1,966 | 1,955 | 11 | 48 |
| COR2 triplet pB | 1,966 | 1,917 | 49 | 48 |
| COR2 clear tB | 2,510 | 2,408 | 102 | 487 |
| PUNCH PAM tB | 1,260 | 1,260 | 0 | 91 |
| PUNCH PAM pB | 1,260 | 1,260 | 0 | 91 |
| CCOR-1 tB | 624 | 550 | 74 | 24 |
| LASCO C2 tB | 3,093 | 3,093 | 0 | 215 |
| LASCO C3 tB | 3,088 | 3,083 | 5 | 213 |

The loader omits a surviving partner when the other file has been moved to `rejected/`, then pairs tB and pB by unique FITS timestamps within one second. The local decisions imply **1,917 complete COR2 pairs** and **1,260 complete PUNCH pairs** for the month, including 48 and 91 respectively during the CME. Actual training-ray counts also depend on pixel masks and post-subtraction brightness. [Pair handling](../../sunerf/data/coronagraph/brightness_pairs.py) · [Thomson data loader](../../sunerf/data/loader/thomson_instrument.py)

Fixed backgrounds are computed **after review** from the remaining images with their pixel masks applied. COR2 tB and pB, COR2 clear, and both LASCO channels use the pixelwise minimum of daily median images, admitting days with at least six frames. For COR2 triplet products, a running angular minimum over ±60° followed by smoothing in angle and radius (radial width 2% of the image width) suppresses streamer imprints in the background. The loader subtracts the corresponding `.npy` arrays and masks nonpositive residuals independently per channel. The YAML leaves optional morphological cleaning off. CCOR’s full-series pixelwise minimum is used for its quicklook movie only. The CCOR and PUNCH training targets have no configured file subtraction. [Correction script](compute_correction.sh) · [Background estimator](../../sunerf/data/coronagraph/compute_correction.py) · [Quicklook products](review_products.sh)

## How a three-dimensional field predicts an image

For each valid pixel, SuNeRF constructs the observer position and viewing ray from the FITS world-coordinate system. It queries a continuous neural field at points along that ray. Each query uses the scattering time, equal to the detector time minus the light-travel time from the scattering point to the observer. Model coordinates use 100 R☉ per distance unit and one day per time unit.

The field has two sine-network branches. A static branch describes the background in a Carrington-corotating frame. A dynamic branch uses inertial position and time, combines these features with the static features, and contributes a scheduled residual. The combined outputs set a positive electron density, proportional to `exp(f) / r²`, and three velocity components around a 300 km s⁻¹ outward radial baseline, with learned deviations scaled by 100 km s⁻¹. This lets the model represent both persistent rotating structures and transient material crossing those structures. [Field model](../../sunerf/model/model.py)

The differentiable renderer integrates electron density times the finite-solar-disk Thomson-scattering kernels for tB and pB. The kernels depend on heliocentric distance and scattering angle and use a limb-darkening coefficient of 0.63. Each ray starts with 64 radially weighted samples through a sphere of radius 150 R☉. The initial brightness contributions guide 128 additional samples, and a trapezoidal integral over the combined grid gives each channel’s brightness. The outer integration radius extends past the 120 R☉ PUNCH mask so material along an edge pixel’s ray can still contribute. Brightness is normalized in units of 10⁻⁹ mean solar brightness (MSB). The saved Thomson normalization converts the learned density to cm⁻³. [Ray sampling](../../sunerf/train/sampling.py) · [Rendering](../../sunerf/rendering/base_tracing.py) · [Thomson kernels](../../sunerf/rendering/thomson.py) · [Physical normalization](../../sunerf/physics/thomson.py)

## Learned instrument response and image losses

The model first renders a clean electron-scattered image. To compare it with the prepared target, each instrument has a small network that learns smooth, time-independent fields in detector coordinates. For a channel with clean brightness `B`, the comparison uses **`B_pred = (B + F + A) × M`**. The signed additive field `A = s × a × exp(b)` can absorb residual stray light or an offset left by background subtraction. The positive multiplier `M = exp(0.01 × m)` can absorb spatial variations in sensitivity. These are fitted nuisance fields, so a feature in a correction map alone does not establish its instrumental origin.

| Stream | Trainable terms | Additive scale s and asinh parameter |
|---|---|---:|
| COR2 triplet | Independent A and M for tB and pB | 10⁻³ |
| PUNCH PAM | Independent A and M for tB and pB | 10⁻⁵ |
| COR2 clear and CCOR | A and M for tB, separate for each stream | 10⁻³ |
| LASCO C2 and C3 | A and M for tB plus a positive F-corona term F, separate for each camera | 10⁻³ |

The LASCO F-corona model uses helioprojective sky coordinates and observer distance. Its coordinate system allows a sky-fixed contribution to move relative to detector-fixed patterns during a roll. F is zero for the other streams. All configured correction penalties have zero weight. The YAML comment calls COR2 the photometric reference. In the executable configuration every imager has an implicit unit **scalar** gain because none enables a separate scalar calibration module. COR2’s **spatial** multipliers remain trainable, as do the other imagers’ multipliers. [Learned corrections and their application order](../../sunerf/train/correction.py) · [Instrument setup](../../sunerf/model/thomson.py)

The image loss (weight 1) sums the tB and pB mean squared residuals after instrument-specific asinh scaling. A pixel can contribute in either measured channel. A second loss (weight 1) uses pixels with both channels to compare observed pB with **observed tB × predicted degree of polarization**, in the same scaled pB space. The degree of polarization is computed from the instrument-corrected prediction, with a protected denominator, positive predicted tB and an accepted ratio range of 0–3. This formulation retains scattering-angle information while avoiding division by faint, noisy observed tB. [Image and polarization losses](../../sunerf/model/thomson.py)

## Physics regularization

Brightness directly measures an integral of density. Velocity enters the fit through how plasma motion must transport that density. At every optimizer step the model is differentiated at 2,048 random space-time points drawn across the imaging time span and 2,048 more drawn in the CME window. Both sets cover 1.5–160 R☉. The configured radial exponent of 1 gives uniform sampling in radius, and the angular sampler is uniform in solid angle. [Volume sampler](../../sunerf/data/loader/volume_sampling.py)

| Term | Physical meaning and v25 weight |
|---|---|
| Continuity | Penalize the squared residual of `∂t ln(nₑ) + v · ∇ln(nₑ) + ∇ · v = 0`, the electron continuity equation for positive density. Scale the residual by the crossing time of 10 R☉ at 300 km s⁻¹. The weight decays exponentially from 0.01 to 0.0005 over the first 50,000 optimizer steps and then holds. |
| Radial direction | Weight 1. Penalize flow outside a 30° cone around the outward radial direction. |
| Radial speed | Weight 1. Penalize radial speeds below 100 km s⁻¹. The configured upper bound is open. Both velocity priors increase quadratically in weight from 1 to 10 R☉ and have full weight farther out. |
| Ballistic acceleration | Weight 0. The fit permits acceleration during the CME. |
| In-situ density and radial velocity | Both weights 0. The model samples and logs residuals along the PSP and Solar Orbiter tracks for comparison. |

These are soft constraints. The inferred plasma velocity depends on the continuity and flow assumptions. The motion of an image feature, or the speed of a tracked CME front in the exported density, is a distinct observable. [Physics loss implementation](../../sunerf/model/thomson.py)

## Training cycle and saved states

1. **Build or reuse the ray cache.** The loader reads reviewed FITS and correction maps, constructs geometry and times, and caches batches of valid rays in the scratch work directory. The configuration requests 32 preprocessing workers and four data-loader workers. After updating reviewed inputs or fixed backgrounds, the Python entry point accepts `--reload`. The current shell wrapper does not forward command-line arguments. For a rebuild, run `python -m sunerf.run_thomson --config config/cme/2025_09/2025_09_cme.yaml --reload` in the cluster environment.
2. **Combine monthly and event observations every step.** Each of the six imagers contributes a full-series batch of 512 rays and a CME-window batch of 512 rays per process. The shorter loaders cycle to the length of the longest. Each step also receives the two physics batches and up to 512 trajectory samples per spacecraft per batch. The two image loaders for each instrument share the same response model. Repeated sampling gives the short eruption substantial weight while the month constrains persistent solar and detector structure.
3. **Steps 0–10,000, cold phase.** The dynamic coefficient is zero. Image gradients fit the corotating background and instrument corrections. The monthly and CME loaders and active physics losses already participate.
4. **Steps 10,000–15,000, warm phase.** The dynamic coefficient ramps linearly from zero to one, allowing time-dependent density and flow to explain the remaining structure.
5. **From step 15,000, joint fit.** The full dynamic residual contributes. Static and dynamic branches and the learned instrument terms continue updating. V25 sets no cold-only freeze for the corrections. Adam’s learning rate decays exponentially from 10⁻³ to 10⁻⁴ over 200,000 steps, then holds. The runner defaults to 200 epochs. The learning-rate horizon is independent of that stopping condition.
6. **Validate, checkpoint and export.** The PBS launcher requests four GPUs. Lightning uses distributed data parallel with multiple GPUs and gradient clipping at 0.5. Validation and periodic checkpointing are configured at 10,000-step intervals. W&B callbacks show target, instrument-corrected and clean renders, correction maps, density slices at 3, 5, 10, 20, 50 and 80 R☉, longitude slices across 2.5–100 R☉, and both in-situ time series. The runner resumes from `last.ckpt` when present, writes `save_state.snf` after validation, and writes `final.ckpt` and a final inference state on successful completion.

[PBS launcher](run.sh) · [Training loop and checkpoints](../../sunerf/run_thomson.py) · [Combined loaders](../../sunerf/data/loader/base_loader.py) · [Optimizer](../../sunerf/model/sunerf.py)

## Application to the September CME

- **Image fidelity and geometry.** The evaluation script generates a prepared-input montage, checks PUNCH and LASCO C2 alignment in the C3 frame near Sep 22 08:00, compares reference and rendered time series, and compares polarization. The montage and alignment use the PUNCH stash, while the reference-series and polarization commands read `prep/punch_pam`. Establishing which PUNCH copy was used matters for a like-for-like comparison. [Evaluation workflow](evaluation.sh)
- **Three-dimensional evolution.** The spherical export covers 3–100 R☉ with 0.5 R☉ radial spacing and 0.5° angular spacing. Each HCI grid stores electron density (cm⁻³), radial plasma velocity (km s⁻¹), time, coordinates and a corotating background density. The exporter estimates that background from the median of 15 model times spanning the export interval plus two days on each side. [Grid exporter](../../sunerf/evaluation/cme/export_density_grid.py)
- **Propagation, front shape and mass.** The tracking analysis uses the first exported frame, corotated to the time of each later frame, as its ambient reference. It tracks the core at the maximum of the radius-compensated density excess and the front and rear at half that excess. The mass-budget analysis integrates the reconstructed density in the tracked sector and compares with ambient mass. The script centers propagation and mass at HCI latitude −30°, longitude 5°, the resolved front at −30°, 15°, and density cuts at −25°, 5°. These are analysis directions supplied to the plots. [Propagation and tracking](../../sunerf/evaluation/cme/plot_cme_propagation.py) · [Mass budget](../../sunerf/evaluation/cme/plot_cme_mass_budget.py) · [Front morphology](../../sunerf/evaluation/cme/plot_cme_front.py)
- **Independent views and visualization.** Earth and STEREO-A synthetic views span 3–100 R☉ and display tB, pB, column density and the density-weighted line-of-sight offset from the plane of the sky (positive behind it). Additional products include tomography, radius maps, trajectory movies, VTK volumes for ParaView and a qualitative comparison with HI-1 browse imagery using its Level-0 geometry. The separate `video.sh` contains polar and virtual-flight examples pointing to `2025_09_v34`. The event’s `evaluation.sh` points to v25. [Evaluation workflow](evaluation.sh) · [Additional movie examples](video.sh)
