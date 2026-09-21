# SuNeRF EUV plasma pipeline: pointwise density/temperature with consistent absorption

Date: 2026-09-19

This plan supersedes the per-voxel Gaussian temperature distribution described in
`plasma_pipeline_modernization.md`. The ray, sampling, response-folding, data, and
artifact infrastructure of that document is kept.

## 1. Physical model

Every point of the reconstruction carries one electron density and one
temperature. For instrument channel `c`

```
eps_c   = g_c * f_cut(T) * G_c(log T, log n_e) * n_e^2
alpha_c = n_H * sum_s A_s f_s(T) sigma_eff[c, s]        s = H I, He I, He II
I_c     = int eps_c * exp(-tau_c) dl                    tau from the observer, dl in cm
```

- `G_c` is the channel response table on `(log10 T / K, log10 n_e / cm^-3)`,
  folded from one CHIANTI 11.0.2 / `sun_coronal_2021_chianti` spectral grid.
- `g_c` is the bounded learnable calibration gain (unchanged).
- `f_cut(T) = 0.5 * (1 + tanh((T - T_cut) / dT_cut))` is an optional
  transition-region cutoff applied to emission only (Downs / Mok et al. 2016
  convention, `T_cut = 4e5 K`, `dT_cut = 5e4 K`). It is required for
  thermodynamic MHD cubes with an artificially broadened transition region and
  is off by default for observations.
- `f_s(T)` are equilibrium ion fractions, `A_s` the abundance relative to
  hydrogen, and `sigma_eff` the photoionization cross sections of Verner et
  al. (1996) folded over the channel.
- The response tables, ion fractions, and abundances come from one atomic build,
  so emission and absorption are consistent by construction.

Known limitations that are recorded in every artifact:

- the recovered `n_e` is an RMS density (`sqrt(<n_e^2>)`, filling factor one);
- thermal width along a line of sight only comes from resolved structure;
- with `f_cut` enabled, genuine transition-region/moss emission is attributed
  to coronal plasma at the base;
- equilibrium ion fractions are not valid for photoionized prominence plasma;
  the v1 opacity is a baseline, not a prominence mass diagnostic;
- He II 304 and the AIA UV channels are excluded from the `n_e^2` path.

## 2. Table ranges

| Axis | Range | Step | Reason |
|---|---|---|---|
| `log10 T` | 4.0 - 8.0 | 0.05 dex | native CHIANTI ionization grid; 10^4 K is needed for the shared absorption temperature; nothing but continuum above 10^7.5 K |
| `log10 n_e` | 7.0 - 11.0 | 0.5 dex | coronal (density independent) limit below 10^7; sensitivity lies in 10^8 - 10^10; hotter-than-cutoff plasma above 10^11 only occurs in flares |
| wavelength | 10 - 1000 A | 1 A | lines are deposited conservatively; matches the throughput tables |

Runtime conventions: `G = 0` outside the temperature axis; the density is
clamped to the axis for the lookup only (`n_e^2` and the opacity use the
unclamped value). The model temperature is bounded to `log10 T` 4.0 - 7.5 by
`model.temperature_grid`.

## 3. Implementation steps

1. **Response table.** `LOG_DENSITY` in `sunerf/response/emissivity.py` becomes
   7.0 - 11.0 in 0.5 dex steps; `LOG_TEMPERATURE` 4.0 - 8.0.
2. **Pointwise renderer.** `PlasmaRadiativeTransfer` performs one bilinear
   lookup of `G` at `(mean_log_T, total_log_ne)`, applies the optional
   `temperature_cutoff`, and keeps the LOS quadrature, gains, optical depth,
   and photosphere termination unchanged.
3. **Two-output model.** `PlasmaModel` (siren or mlp backend) and
   `SphericalGridPlasmaField` return `total_log_ne` and `mean_log_T`; the
   Gaussian width and the per-bin outputs are removed. `temperature_grid`
   min/max bound the model temperature.
4. **Absorption.** `hydrogen_density_convention` in the bundle
   (`fully_ionized_proxy` default, `cie_electrons_per_hydrogen` optional and
   guarded by a minimum ionization); log-space interpolation of ion fractions;
   `sunerf-absorption ionization` builds the H/He table from the pinned CHIANTI
   database; cross sections are folded with emissivity x throughput when a
   spectral grid is supplied; the PSI synthetic path and grid evaluator accept
   absorption.
5. **Diagnostics.** DEM products become LOS histograms of `n_e^2 dl` over
   temperature, computed for validation/evaluation only.
6. **Verification (internal).**
   - isothermal slab: `I = G * n_e^2 * L`, stable under LOS refinement;
   - slab stack: far emission attenuated by `exp(-tau)`, near emission not,
     channel ratios follow the cross sections;
   - cross section per electron is ~1e-19 cm^2 at 1e4 K, ~5e-20 up to ~6e4 K and
     falls steeply above (plausibility only);
   - closed-loop PSI test (step 7);
   - short 2012-08 run: PSNR not worse, gains finite and bounded.
7. **PSI forward synthesis and first reconstruction** (`data/psi_data`, dumps
   1813 - 1815, `n_e = rho * 1e8 cm^-3`, `T = t * 2.807e7 K`, chromospheric
   base at 1.75e4 K, broadened TR at 1.005 - 1.02 R_sun).
   1. `PSI_DATA` is the directory that contains the `rho/` and `t/` folders;
      every snapshot pair is rendered as one frame (its own static grid). The
      HDF5 cubes carry no time information (no attributes or time axis), so the
      date of a reference snapshot and the snapshot cadence are required inputs;
   2. `sunerf-render-psi` writes prepared-EUV-v2 FITS for AIA and EUVI-A/B with
      the packaged tables, `f_cut` on, absorption on, radial-weighted sampling;
   3. forward checks: DN/s ranges, clean limb truncation, sample-count
      stability, absorption signature of the cool material;
   4. one render call per observer and instrument (`render_observer`, CLI
      `--instrument`, `--location`, `--out-path`); `LOCATION` is
      one of `L1` - `L5`, `mercury` - `neptune`, or Stonyhurst
      `longitude_deg,latitude_deg,distance_au`. L3/L4/L5 are the points 180 / +60 /
      -60 degrees along Earth's orbit, L1/L2 lie 1 % inside/outside Earth. The
      field of view is defined in solar radii so every observer frames the same
      region. The caller chooses each output directory;
   5. reconstruct with fixed gains, then with learnable gains;
   6. evaluate radial `log n_e` / `log T` errors, shell maps, held-out views,
      recovery of cool material, and document what is not recoverable;
   7. optional: noise, mis-set gains, time-dependent input.

   Gate for real data: the six-observer set of `scripts/psi_euv/render.sh`
   (Earth, L3, L4, L5, two polar views) recovers the corona above 1.02 R_sun with
   median `|d log n_e| < 0.1 dex` and `|d log T| < 0.05 dex`; a three-observer
   subset (Earth, L4, L5) quantifies the loss from realistic coverage; learnable
   gains return to within 0.05 dex of one.
8. **Packaged resources.** `sunerf/resources/` ships the abundance file, H/He
   ionization table, spectral grid, throughputs, response tables, absorption
   bundle, the required Verner rows, a `manifest.json`, and `LICENSES.md`.
   `builtin:<name>` resolves packaged artifacts in `temperature_response.artifact`
   and `absorption.artifact`. `sunerf-resources build|verify` rebuilds from
   scratch and checks the manifest. Builders remain the only producers.
9. **Science configuration.** `config/plasma/all_2012_08.yaml` uses
   the packaged tables with photoionization absorption.

Order: 1 -> 2 -> 3 (+5) -> slab tests -> 4 -> 8 -> 7 -> 9.

## 4. Deferred

Line opacity (Mok et al. 2016); He II 304 scattering renderer; AIA
1600/1700 boundary context; SUVI adapter; a thermal-width table axis if
multi-channel residuals become systematic.

## 5. Implementation status (2026-09-19)

| Step | State | Where |
|---|---|---|
| 1 response table axes | done; multi-density tables built and packaged | `sunerf/response/emissivity.py` |
| 2 pointwise renderer, `temperature_cutoff` | done | `sunerf/rendering/plasma.py` |
| 3 two-output models, PSI grid field | done (`initial_log_T`, bounded `log T`) | `sunerf/model/model.py`, `sunerf/model/spherical_grid.py` |
| 4 absorption | done: hydrogen-density conventions, log-space ion fractions, `sunerf-absorption ionization`, spectrum-weighted cross sections, grid artifact/evaluator support | `sunerf/absorption/`, `sunerf/data/psi/build_synthetic.py`, `sunerf/evaluation/loader.py` |
| 5 diagnostics | done: LOS emission-measure histogram on `model.temperature_grid` nodes | `sunerf/rendering/plasma.py` |
| 6 verification | slab, slab-stack, cutoff, support/clamping, histogram tests pass; cross section per hydrogen for AIA 171 is 6e-20 cm^2 at 10^4.5-10^4.7 K and 3e-23 cm^2 at 10^5.5 K | `tests/test_plasma_response_physics.py`, `tests/test_absorption_bundle.py` |
| 7 PSI synthesis | `sunerf-render-psi` writes prepared-EUV-v2 FITS for explicit observers (Lagrange points, planets, or Stonyhurst coordinates), one frame per snapshot pair; forward checks done on dump 1813; `sunerf-evaluate-psi-euv` compares a reconstruction with the cube; reconstruction run pending (GPU) | `sunerf/data/psi/render_euv.py`, `scripts/psi_euv/`, `config/plasma/psi_observers.yaml` |
| 8 packaged resources | `builtin:` resolution, manifest, `sunerf-resources build|verify` done; 13 files (6.2 MB) installed and verified | `sunerf/resources/` |
| 9 science configuration | done | `config/plasma/all_2012_08.yaml` |

`model.temperature_grid` keeps its three fields: min/max bound the pointwise
temperature and `step_dex` is the bin width of the diagnostic histogram.

Forward check on the thermodynamic cube (`docs/figures/psi_forward_cutoff_absorption.png`,
AIA view at Carrington longitude 0, interim single-density tables): with the
cutoff and absorption the quiet-Sun disk medians are 635 / 444 / 143 DN s^-1
pix^-1 in 171 / 193 / 211 A and the limb truncates cleanly. Without them the
broadened transition region dominates 171 - 335 A by factors of 20 - 800 and
aliases into concentric ring artifacts; off-limb intensities above 1.15 R_sun
are identical in both renderings.

Two defects found by the synthesis test were fixed: the PSI radial crop dropped
the layer between `min_radius` and the first retained node, and coincident
deterministic hierarchical samples were rejected by the LOS quadrature.

### Packaged table build (2026-09-19)

`sunerf-resources build --install` takes about one hour on a laptop. Two defects
of the multi-density build were fixed in the FIASCO provider: FIASCO 0.8.2
mis-broadcasts the two-photon continuum for more than one density (now evaluated
per density, identical to a single-density call), and Na IX level populations
underflow to NaN below log T = 4.3 where its ionization fraction is exactly zero
(those entries are zeroed and counted in `ion_selection` provenance; every other
non-finite value still fails). The n_e = 1e9 slice reproduces the earlier
single-density spectrum exactly. At the channel peak, G(1e7)/G(1e9) is 1.13 for
AIA 171/193 and G(1e11)/G(1e9) is 0.71 / 0.82 / 0.84 for 171 / 193 / 211, so
the density axis changes off-limb 171 intensities of the PSI cube by ~12 %.

Remaining before real data: run `scripts/psi_euv/render.sh`, `run.sh`, and
`evaluate.sh` on a GPU and check the recovery gate of step 7; then the short
2012-08 regression run.

### Light travel time (EUV)

`module.light_travel_time: true` evaluates every sample at its emission time,
`t_obs - |x - x_obs| / c`, using the FITS timestamps as detector times (the
implementation shared with the Thomson renderer). The model time axis is then
Sun time: observers at different heliocentric distances agree (8.3 min at 1 AU,
~2.5 min for Solar Orbiter at 0.3 AU) and the far side of the shell is seen a few
seconds earlier than the near side. The apparent Carrington longitude of each
observer already contains the light-travel correction of the rotating frame
(SunPy convention), so only the time coordinate changes. `sunerf-render-psi`
stamps the delayed detector time by default (`snapshot_time` is kept in
`observer.json`), and `sunerf-evaluate-psi-euv --time` takes the emission time.

The PSI closed loop runs with light travel time off on both sides
(`render.sh --no-light-travel-time`, `psi_observers.yaml` `light_travel_time: false`):
the frames are frozen snapshots, so a per-sample emission time has no truth to
compare with. Observations keep `light_travel_time: true`.

### Separate cool absorber (2026-09-19)

A toy inversion with the packaged tables showed that the single-field opacity
cannot be recovered from images: an emitting point would have to cross
10^5.6 - 10^6 K, where a dense point is bright, before it absorbs, so the
optimizer explains a filament by lowering the coronal density instead (fitted
log n_e 7.5 for a true 9.0, optical depth 2e-6 for a true 5.7). The forward
direction (PSI synthesis) is unaffected because the cube supplies the cold
plasma directly.

`model.cool_absorber: true` adds a third network output, the hydrogen density
`n_H,cool` of material that never emits:

```
alpha_c = alpha_c,hot(n_e, T)  +  n_H,cool * sum_s A_s x_s sigma_eff[c, s]
```

Only the amount and location of the absorber are reconstructed. The cross
sections and abundances are the packaged atomic data, and the ionization state
`absorption.cool_ion_fractions` (default H I 0.7, He I 0.7, He II 0.3) is a
fixed, recorded assumption for photoionized prominence plasma; the inferred
column scales roughly inversely with the neutral fractions. One density
therefore sets the opacity of every channel of every instrument with fixed
ratios (tau_94 : tau_131 : tau_171 : tau_193 : tau_211 ~ 0.2 : 0.5 : 1 : 1.35 : 1.7),
which separates an absorber from a calibration error or an emission deficit.
The field starts optically negligible (`cool_density_offset_log10_cm3: 7`), has
an L1 prior on its LOS column (`lambda.cool_absorber`, column unit
`module.cool_column_scale_cm2 = 1e19`), and the hierarchical sampling weights
now include the intensity each node removes so absorbers are refined.

In the same toy inversion the separate field reproduces all channels exactly,
returns the coronal density (9.03 for 9.00) and the hydrogen column within a
factor of two. A single ray cannot place the absorber along the line of sight;
that needs the multi-view closed loop, where `sunerf-evaluate-psi-euv` now
reports the shell-integrated equivalent cool hydrogen (alpha_171 / kappa_171)
of truth and reconstruction, separately for the limb layer below 1.02 R_sun and
the cool material above. An L1 weight of 1e-3 already biased the coronal
density low in the toy case, hence the 1e-4 default. Still open for
observations: a 304 A support constraint, and the chromospheric limb layer if
the cool field does not recover it.

### First-model scope (2026-09-19)

The first SuNeRF plasma model reconstructs the corona only. `all_2012_08.yaml`
and `psi_observers.yaml` use no absorption and no cool absorber, temperature
bounds log T 5.4 - 7.5, light travel time, and `sampling.min_distance: 1.007`:
a fixed opaque sphere at the EUV limb instead of the photosphere. In the PSI
cube the far-side transmission of a tangent ray is 0 at an impact parameter of
1.006 R_sun, 3 % at 1.008 and 98 % at 1.010, so the sphere reproduces the limb
truncation that the synthesis obtains from its physical H/He opacity; without
it a ring of ~0.007 R_sun above the limb is modelled a factor of two too bright.
The PSI synthesis keeps absorption on, which makes the closed loop a test of
this approximation (limb-ring residuals, density error in 1.02 - 1.1 R_sun).

Reasons for deferring the cool plasma: it adds a second, weakly constrained
field whose prior biases the coronal density; the PSI cube contains almost no
absorption above the limb (99.9 % of its 171 A opacity lies below 1.01 R_sun,
0.003 % above 1.02 R_sun), so it cannot validate a filament reconstruction; and
the 304 A support constraint does not exist yet. The cool absorber stays in the
package, off by default and tested; `all_2012_08.yaml` is the
stage-two configuration. Stage two requires a validated coronal baseline, a
filament-insertion option in `sunerf-render-psi`, and the 304 A constraint.

### Hydrostatic density baseline (2026-09-19)

`model.density_profile: {type: hydrostatic, scale_height_rsun: 0.1}` replaces the
`r**-2` factor of the SIREN plasma model by the isothermal stratification
`n ~ exp[-(1/H0)(1 - 1/r)]`; the network learns the deviation. `r**-2` is the
shape of a constant-speed wind and declines by 0.35 dex between 1.0 and 1.5
R_sun, whereas a 1.5 MK hydrostatic corona (`H0 = kT / (mu m_H g_sun)` = 0.068
R_sun per MK), the Newkirk and Baumbach-Allen models, and the median profile of
the PSI cube (1.66 dex, fitted H0 = 0.097 R_sun) all decline by 1.4 - 1.7 dex.
`{type: power_law, exponent: 2}` remains the default for other uses. With the
baseline in place `lambda.regularization` is 0 in both first-model
configurations: the radial density penalty mainly compensated the shallow prior
and is one-sided, so it biased the outer shell low. Open-field regions are not
hydrostatic; the network supplies those deviations.
