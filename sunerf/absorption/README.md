# EUV absorption bundles

SuNeRF keeps atomic-data preparation outside training. The offline builder
produces one immutable, versioned NPZ bundle; training only validates and reads
that bundle. Rows are keyed by both `instrument_key` and `channel`, so channels
with the same nominal wavelength on different telescopes do not share an
effective cross section accidentally.

The physical opacity is

`alpha_c = n_H [f_HI sigma_c,HI + A_He f_HeI sigma_c,HeI + A_He f_HeII sigma_c,HeII]`.

The bundle contains the temperature grid, H/He equilibrium ion fractions,
electrons per hydrogen nucleus, assumed He/H abundance, and the H I/He I/He II
cross sections folded over each channel. Ion fractions are evaluated at the
pointwise plasma temperature and interpolated in log space.

The hydrogen density follows `absorption.hydrogen_density_convention`:

- `fully_ionized_proxy` (default): `n_H = n_e / (n_e/n_H)_fully ionized`. The
  density field traces total hydrogen as in a fully ionized single-fluid MHD
  model; neutral fractions only select how much of it absorbs.
- `cie_electrons_per_hydrogen`: `n_H = n_e / (n_e/n_H)(T)` from the equilibrium
  table, bounded below by `minimum_electron_per_hydrogen` (default 0.1) because
  the inversion is ill-conditioned for nearly neutral plasma (the equilibrium
  value is ~2e-3 at 10^4 K).

A model can instead provide `total_hydrogen_density` through the same opacity
interface. Equilibrium fractions do not describe photoionized prominence
plasma; the opacity is a consistent baseline, not a prominence mass diagnostic.

## Separate cool absorber

With `model.cool_absorber: true` the reconstruction carries a second density,
the hydrogen of cool material that does not emit in the EUV channels
(filaments, prominences, the chromospheric limb layer):

`alpha_c = alpha_c,hot + n_H,cool * sum_s A_s x_s sigma_c,s`.

Its ionization state `absorption.cool_ion_fractions` (default
`{H_I: 0.7, He_I: 0.7, He_II: 0.3}`) is a fixed assumption recorded in the
reconstruction artifact; cross sections and abundances are those of this
bundle, so the wavelength dependence of the absorber is not learned. The
single-field equilibrium opacity cannot be recovered from images because an
emitting point would have to pass through temperatures at which it is bright.

A packaged bundle built from the same CHIANTI release and abundances as the
packaged response tables is selected with `artifact: builtin:h_he_photoionization`.

## Offline workflow

Download inputs with an explicit expected SHA-256:

```bash
sunerf-absorption fetch \
  --url https://www.pa.uky.edu/~verner/dima/photo/photo.dat \
  --sha256 EXPECTED_SHA256 \
  --output data/atomic/verner-photo.dat
```

Export equilibrium ion fractions directly from the pinned FIASCO CHIANTI
database (this also reports the He/H abundance of the selected table):

```bash
sunerf-absorption ionization \
  --hdf5-database data/response_calibration/chianti/11.0.2/chianti_11.0.2.h5 \
  --abundance sun_coronal_2021_chianti --output ionization.npz
```

Tables from another provider can be supplied as CSV with columns
`log_temperature,h_i,h_ii,he_i,he_ii,he_iii` and optionally
`metal_electron_per_hydrogen`, then convert it to the provider-neutral input:

```bash
sunerf-absorption convert-ionization \
  --input ionization.csv --output ionization.npz \
  --provider fiasco --version 0.X --source-sha256 SOURCE_SHA256
```

Build a bundle from local inputs and existing SuNeRF throughput artifacts:

```bash
sunerf-absorption build \
  --verner-table data/atomic/verner-photo.dat \
  --ionization ionization.npz \
  --throughput aia=data/responses/aia-throughput.npz \
  --throughput euvi_a=data/responses/euvi-a-throughput.npz \
  --helium-abundance 0.085 \
  --abundance-name sun_coronal_2021_chianti \
  --abundance-version CHIANTI-11.0.2 \
  --abundance-sha256 ABUNDANCE_FILE_SHA256 \
  --spectral-emissivity data/response_calibration/responses/chianti_coronal_2021.spectral.npz \
  --output data/absorption/h-he-photoionization.npz
```

With `--spectral-emissivity` the cross sections are weighted by the detected
spectrum (emissivity x throughput at the channel's peak-response temperature),
which is the first-order attenuation of the channel. Without it the weight is
the throughput alone. `sunerf-resources build --install` runs this whole chain
and installs the result into the package.

No network access occurs during `build` or training. Input paths, hashes,
provider metadata, abundance, calibration epochs, and folding convention are
stored in bundle provenance.
The abundance name, version, and hash must match the abundance identity in each
temperature-response artifact; SuNeRF rejects a mismatch before training.

## Training configuration

```yaml
absorption:
  type: photoionization
  artifact: data/absorption/h-he-photoionization.npz

lambda:
  absorption: 0.0
```

The instrument keys and ordered channels in the bundle must match the renderer
configuration exactly (channel spelling is canonicalized). `type: null`
disables absorption. `learned` and `constant` remain available only as explicit
legacy modes for loading older experiments.

The spectral-weighting choice of the cross-section fold is recorded explicitly
in provenance.
