# SuNeRF

3D reconstructions of the solar atmosphere using neural radiance fields.

## Overview

SuNeRF is a Python-based project for reconstructing the solar atmosphere in 3D using neural radiance fields (NeRFs). It leverages data from multiple solar instruments (AIA, EUVI-A, EUVI-B) and deep learning techniques.

## Features

- Multi-instrument data integration (AIA, EUVI-A, EUVI-B)
- Spherical sampling and 3D reconstruction
- Configurable training and evaluation pipelines
- PyTorch Lightning support

## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

## Usage
```markdown
### Training

To train a model using the provided configuration, run:

```bash
python -m sunerf.run_plasma --config config/plasma/all_2012_08.yaml
```

You can modify the configuration file to change instruments, data paths, or training parameters.

### Evaluation

After training, evaluate the model or generate visualizations:

```bash
python -m sunerf.evaluation.video --chk_path /glade/work/rjarolim/sunerf/all_v02/save_state.snf --video_path /glade/work/rjarolim/sunerf/all_v02/evaluation/video
```

### Data Preparation

Prepare AIA and EUVI data:

```bash
python -m sunerf.data.euv.prep_aia --data_path "/glade/work/rjarolim/data/sunerf/2012_08/aia/*.fits" --out_path "/glade/work/rjarolim/data/sunerf/2012_08_prep/aia" --resolution 512

python -m sunerf.data.euv.prep_euvi --data_path "/glade/work/rjarolim/data/sunerf/2012_08/euvi_prep/*.fts" --out_path "/glade/work/rjarolim/data/sunerf/2012_08_prep/euvi"
```

### Configuration

Edit YAML files in `config/plasma/` to set up instruments, data sources, and training options.

### Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```