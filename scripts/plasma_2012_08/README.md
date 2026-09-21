# August 2012 EUV plasma reconstruction

This folder is the run-oriented companion to
`config/plasma/all_2012_08.yaml`, following the `scripts/cme_*` layout.

Run stages from the repository root in this order. The scripts use
project-relative paths and do not search for the checkout:

1. `responses.sh` generates or reuses the shared CHIANTI emissivity table,
   downloads the instrument calibrations, and folds the table through AIA and
   EUVI-A/B.
2. `download.sh` downloads AIA level-1 data from JSOC and EUVI inputs from VSO.
3. `prepare.sh` calls `secchi_prep.csh` for the external SolarSoft calibration,
   then writes calibrated FITS products for all three instruments.
4. Submit or execute `run.sh` to train with the schema-v2 configuration.
5. Run `evaluation.sh` to generate artifact-driven virtual-observer movies.

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
qsub scripts/plasma_2012_08/run.sh
```

`DRMS_EMAIL` is the JSOC export credential. SolarSoft with the SECCHI package
and IDL must already be installed under `${HOME}/ssw`.

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
  datasets. It does not load or validate temperature-response artifacts.
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
