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
| Training window | 2025-09-27 .. 2025-11-05 at 1 h (AIA 94/131/171/193/211/335, EUVI-A 171/195/284) |
| Held-out window | 2025-10-04 .. 2025-10-29 at 6 h (FSI 174 L2; ~95 frames in SOAR, none on 10-04) |

FSI 304 is chromospheric and the plasma model has no cool component, so only
174 is used. FSI has a single coronal channel: this validates emission and
geometry at the pole, not the temperature.

```bash
scripts/plasma_polar_2025_10/all_data.sh     # download.sh, then prepare.sh
qsub scripts/plasma_polar_2025_10/run.sh
```

Data live under `/glade/work/rjarolim/data/sunerf/polar_2025_10*`, the run under
`/glade/work/rjarolim/sunerf/polar_2025_10_plasma`. The AIA export is one JSOC
request per day and channel, the EUVI download is chunked one week per VSO query and the VSO/SOAR searches
retry on transient failures. The config keeps the EUI gain fixed
(`learnable: false`) so the FSI prediction is an absolute test of the packaged
FSI response; the validation-only dataset renders its central frame during
training, the full sequence is rendered afterwards with the evaluation tools.
