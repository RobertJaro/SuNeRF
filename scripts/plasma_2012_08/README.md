# August 2012 EUV plasma reconstruction

This folder is the run-oriented companion to
`config/plasma/all_2012_08.yaml`, following the `scripts/cme_*` layout.

Data live under `/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2012_08`
(`raw`, `calibrated`, `prepared`, `review`), the run under
`/glade/work/rjarolim/sunerf/all_2012_08_plasma_v03`.

Run stages from the repository root in this order. The scripts use
project-relative paths and do not search for the checkout:

1. `responses.sh` generates or reuses the shared CHIANTI emissivity table,
   downloads the instrument calibrations, and folds the table through AIA and
   EUVI-A/B.
2. `download.sh` downloads AIA level-1 data from JSOC and EUVI inputs from VSO.
3. `prepare.sh` calls the shared `scripts/secchi_prep.csh` for the external
   SolarSoft calibration (one call per spacecraft),
   then writes calibrated FITS products for all three instruments.
4. `review_frames.sh` scans the prepared datasets for invalid frames and the
   saturation level; review them locally and apply the decisions (see
   below).
5. Submit or execute `run.sh` to train with the schema-v2 configuration.
   `run_polar.sh` trains the polar reconstruction instead
   (`config/plasma/polar_2012_08.yaml`): the `polar_2025_10.yaml` setup on
   the same prepared data and image-scaling table, with a rendered view from
   above the solar north pole in the validation plots
   (`callbacks.north_pole_view`). Its run lives under
   `/glade/work/rjarolim/sunerf/polar_2012_08_plasma_v01`.
6. Run `evaluation.sh` to generate artifact-driven virtual-observer movies.

`all_data.sh` executes only the download and preparation stages. Response
construction is independent and can run as its own job.

## Fresh run

Run the scripts from an already configured Python 3.12 environment and provide
the external value that cannot be hard-coded safely:

```bash
pip install -e ".[download,euv-prep,response-build]"
export DRMS_EMAIL="registered@example.org"

scripts/plasma_2012_08/all_data.sh
scripts/plasma_2012_08/responses.sh
scripts/plasma_2012_08/review_frames.sh
qsub scripts/plasma_2012_08/run.sh
```

`DRMS_EMAIL` is the JSOC export credential. SolarSoft with the SECCHI package
and IDL must already be installed under `/glade/work/rjarolim/ssw`.

The atomic table is shared across runs under
`${SUNERF_RESPONSE_CALIBRATION_ROOT}/responses`. The first response build
downloads the pinned CHIANTI 11.0.2 database and generates it with FIASCO;
later runs validate and reuse it. The first build downloads about 600 MB and
creates an approximately 2.3 GB local HDF5 database.

- `scripts/generate_spectral_emissivity.sh` is the standalone entry point for
  generating the shared table.
- `responses.sh` runs that entry point, downloads the pinned AIA
  instrument/degradation files and STEREO-A/B EUVI spectral response areas,
  then builds all three temperature-response artifacts.
- `download.sh` downloads AIA level-1 images, both spacecraft's EUVI images,
  and the JSOC AIA master-pointing table. It also converts the pinned AIA
  degradation table to the ECSV consumed by preparation.
- `prepare.sh` runs `SECCHI_PREP /NORMAL_OFF` and prepares AIA and both EUVI
  datasets on a 512x512 grid cropped to 1.4 R_sun. It does not load or
  validate temperature-response artifacts. An optional `max_radius_rsun` per
  dataset in `config/plasma/all_2012_08.yaml` narrows the field of view at
  load time; pixels whose line of sight passes the Sun farther out are not
  supervised and do not enter the image-scaling table.
- `run.sh` first writes the per-instrument image-scaling table referenced by
  `instruments[].scaling.divisor` (`sunerf.data.euv.estimate_scaling`; an
  existing table is reused). It then loads the prepared FITS files directly and
  validates their units and calibration conventions against the response
  artifacts before training.

The official SRA release has no published time-dependent degradation
correction, so this workflow uses `static_assumed`. `prepare.sh` also runs
`SECCHI_PREP` with `/NORMAL_OFF`; this matches the exported S1 throughput
exactly and avoids applying the separate open-filter normalization scalar.

Paths and numerical settings are written directly in the component scripts.
If the prepared-data or response roots change, update the matching paths in
`config/plasma/all_2012_08.yaml` too. The scripts deliberately do not load
modules, activate Conda, change directories, or interpret reload flags.

## Frame review

`review_frames.sh` runs `sunerf.data.frame_review scan` on the prepared AIA,
EUVI-A and EUVI-B datasets and writes one review directory per dataset under
`/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2012_08/review`. The flags, the review page
and the saturation selection are documented in
`scripts/plasma_euv_wave/README.md`. The scan reads every frame's peak
detector DN and saturated-pixel count from the prepared headers (`PEAKDN`,
`NSATPIX`), so the products must come from the current `prepare.sh`.

1. **Scan (server).** `scripts/plasma_2012_08/review_frames.sh`.
2. **Review (local).**
   `rsync -av casper:/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2012_08/review/ data/plasma_2012_08/review/`,
   then `python -m sunerf.data.frame_review review data/plasma_2012_08/review/aia`
   (and `euvi_a`, `euvi_b`). Decisions go to `decisions.csv`, the chosen
   saturation level to `saturation.csv`; copy that value into
   `instruments[].saturation_dn` of `config/plasma/all_2012_08.yaml`.
3. **Apply (server).** Copy the `decisions.csv` files back and run
   `python -m sunerf.data.frame_review apply /glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2012_08/review`
   (first with `--dry-run`). Rejected files move into `rejected/` next to the
   prepared files, outside the training glob; `restore <prepared dir>` undoes
   it. Then re-estimate the image-scaling table with `--overwrite`, since
   `run.sh` reuses an existing one.
