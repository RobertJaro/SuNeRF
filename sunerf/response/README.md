# EUV temperature responses

SuNeRF uses one small response pipeline:

```text
instrument calibration files -> common throughput NPZ
one common emissivity NPZ × each throughput NPZ -> temperature-response NPZ
```

Training reads only the final response NPZ files. It never imports CHIANTI,
FIASCO, ChiantiPy, aiapy, or SolarSoft.

The response is a table `G(log10 T, log10 n_e)` (log T 4.0-8.0 in 0.05 dex,
log n_e 7.0-11.0 in 0.5 dex). The renderer interpolates it bilinearly at the
pointwise plasma state, returns zero outside the temperature axis, and clamps
the density to its axis. A reference set built from CHIANTI 11.0.2 with
`sun_coronal_2021_chianti` abundances ships in `sunerf/resources` and is selected
with `artifact: builtin:aia` (`euvi_a`, `euvi_b`, `eui_fsi`);
`sunerf-resources build --install` runs the steps below plus the absorption
bundle and reinstalls it.

## 1. Download and standardize the instruments

Install `aiapy` for the AIA calibration adapter, then prepare every supported
instrument:

```bash
pip install -e ".[euv-prep]"
sunerf-responses --root data/response_calibration prepare
```

This downloads official calibration inputs and writes four files with the same
`InstrumentThroughput` schema:

```text
throughputs/aia.throughput.npz
throughputs/euvi_a.throughput.npz
throughputs/euvi_b.throughput.npz
throughputs/eui_fsi.throughput.npz
```

The adapters retain the instrument-specific details that affect the wavelength
shape: AIA calibration and degradation state, SECCHI spacecraft/filter state,
and EUI filter-wheel position. Absolute channel amplitudes remain physical in
the artifacts; fixed dataset divisors and constrained learned gains handle
training-scale differences.

## 2. Generate one shared atomic emissivity table

Install the optional atomic dependencies and run the repository script:

```bash
pip install -e ".[response-build]"
scripts/generate_spectral_emissivity.sh
```

This downloads the pinned CHIANTI 11.0.2 database and creates
`data/response_calibration/responses/chianti_coronal_2021.spectral.npz` with
FIASCO 0.8.2. The calculation uses the CHIANTI 2021 coronal abundances,
CHIANTI ionization equilibrium, log(T/K) 4--9 in 0.05 steps, log(ne/cm^-3)=9,
and the `ne2` emission-measure convention. If the output already exists, the
script validates and reuses it. Pass another shared calibration root as its
only argument when running on another system.

Inspect the input schemas with:

```bash
sunerf-responses schema
```

## 3. Fold every instrument uniformly

```bash
sunerf-responses --root data/response_calibration build \
  --spectral-emissivity data/response_calibration/responses/chianti_coronal_2021.spectral.npz \
  --label reference
```

Use `--instrument aia`, `--instrument euvi_a`, and so on to build only selected
instruments. Every selected instrument follows the same interpolation,
wavelength quadrature, unit conversion, provenance, and artifact-writing code.

## Instrument conventions

- AIA: channels 94, 131, 171, 193, 211, and 335 at the recorded reference
  calibration epoch. The throughput ends at 413 A, the range SolarSoft folds
  (`aia_bp_make_emiss.pro`); the instrument file's extrapolated 413-900 A
  Al-filter leak would otherwise add optically thin O III-O V/Ne VIII emission
  and move the A335 maximum from log T 6.4 to 5.3. The A335 row keeps its
  genuine A131 crosstalk peak (equal effective area, 2.6x the gain), which
  supplies up to ~40% of the A335 response near log T 5.7 and dominates above
  log T 7. A335 stays broad (within 2x of its peak over log T 5.6-6.6) and
  constrains plasma only together with the other channels.
- SECCHI/EUVI: spacecraft-specific 171, 195, and 284 S1 responses matching
  `SECCHI_PREP /NORMAL_OFF` products.
- EUI/FSI: default 174/N25 and 304/N4 filter-wheel modes. The 304 response is
  available, but an optically thin equilibrium model alone is not sufficient
  for general He II 304 interpretation.

Prepared observations must match the calibration/filter convention embedded in
their response artifact. Images use one fixed, reversible divisor per channel;
never normalize each image independently.
