# SuNeRF  
**Solar Neural Radiance Fields for 3D Tomography**

SuNeRF is a Python framework for **3D tomographic reconstructions of the solar atmosphere and heliosphere** using **Neural Radiance Fields (NeRFs)**. It performs forward modeling of optically thin observations with explicit spacecraft geometry and image-formation physics.

SuNeRF supports:
- **EUV emission tomography** of the solar corona  
- **White-light CME tomography** using **Thomson scattering**

---

## Installation
```bash
conda create -n sunerf python=3.10 -y  
conda activate sunerf  
pip install -r requirements.txt  
pip install -e .
```
---

## EUV Emission Tomography

Reconstructs a 3D **EUV emissivity** field from multi-viewpoint EUV observations (e.g., AIA + EUVI).

### Data preparation
```bash
python -m sunerf.data.prep.sdo  
  --sdo_file_path "/path/to/aia/*.fits"  
  --output_path "workspace/data/euv/aia_193"  
  --resolution 512  

python -m sunerf.data.prep.stereo  
  --stereo_file_path "/path/to/euvi/*.fts"  
  --output_path "workspace/data/euv/euvi_195"  
  --resolution 512  
```

### Training
```bash
python -m sunerf.run_emission  
  --config "config/emission/emission_2012_08-193.yaml"
```
---

## CME Tomography (White-Light)

Reconstructs the 3D **electron density** of CMEs using a full **Thomson-scattering** forward model.

### CME Challenge Dataset (HAO / PUNCH)

Dataset:  
https://download.hao.ucar.edu/pub/punch/cme_challenge_v2/

Download and place the data locally, for example:

```bash
mkdir -p data/cme_challenge_v2
```

---

### Data preparation

```bash
python -m sunerf.data.prep.prep_hao_cme  
  --resolution 512  
  --hao_path "data/cme_challenge_v2/**/*.fits"  
  --output_path "workspace/data/cme/prepared"  
  --check_matching  
```

---

### Training / reconstruction

```bash
python -m sunerf.run_thomson  
  --config "config/cme/hao_2view.yaml"
```

---

### Evaluation

```bash
python -m sunerf.evaluation.cme.evaluate_cme_parameters  
  --sunerf_path "workspace/outputs/cme/save_state.snf"  
  --data_path "data/cme_challenge_v2/**/*.sav"
```
---

## Configuration

All experiments are configured via YAML files in:

```
config/  
  cme/
  emission/  
```
---