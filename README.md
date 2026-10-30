# pCR Prediction in TNBC with Virtual Staining

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23788133.svg)](https://doi.org/10.5281/zenodo.23788133)



Repository for the dissertation on predicting pathological complete response (pCR) in triple-negative breast cancer (TNBC), combining:
- H&E images,
- Ki-67 and PHH3 stains (real or virtual),
- Biomarker-based spatial attention.


## Project Structure

- `bcs-kedro/`  
  Kedro project containing the complete training and evaluation pipeline for pCR prediction:
  - Image preprocessing
  - Generation of biomarker-based spatial attention maps
  - ResNet-based prediction model with an attention mechanism

- `virtual_staining/`  
  Implementation of the virtual staining models to generate Ki-67 and PHH3 markers from H&E:
  - **CycleGAN**: Unpaired image-to-image translation
  - **cGAN**: Conditional GAN
  - **Diffusion**: Diffusion-based models

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Lab-of-Applied-Computer-Science/pcr-classification
cd pcr-classification
```

### 2. Create a virtual environment

Python 3.8 or higher is recommended:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Execution

### pCR Prediction Pipeline (Kedro)

1. **Navigate to the Kedro project directory:**

```bash
cd bcs-kedro
```

Run the `.sh` script located at `bcs-kedro/scripts/run_train_cnn_pcr.sh`, changing only the pipeline and the user you want to execute:

```bash
username="henrique"
kedro_pipeline="train_cnn"
```

Also, adjust the SLURM execution settings (desired node/gorgona, allocation time, and job name). Example:

```bash
#SBATCH --job-name=trainCNNPCR_9_channels
#SBATCH --time=10:00:00
#SBATCH --nodes=1 
#SBATCH --nodelist=gorgona7
#SBATCH --output=/path/to/output/slurm-logs/slurm-%j.log
```

*Note: The available pipelines are located in `bcs-kedro/src/bcs_kedro/pipelines` (each folder represents an existing pipeline).*

### Virtual Staining

1. **Navigate to the virtual staining directory:**

```bash
cd virtual_staining
```

Run the `.sh` file, changing only the target `.py` script you want to execute and the SLURM execution settings (node, time, and job name).

## Additional Configuration

### Important Adjustments

- **Data paths**: Ensure that the paths in the configuration files (`conf/base/catalog.yml` in Kedro, and the `.sh` scripts in the virtual staining module) correctly point to your data.

- **GPU/CPU**: Training is optimized for GPU. If you are using a CPU, adjust the batch sizes in the training scripts to avoid memory issues.


## Citation

If you use this code, please cite the associated dissertation and the
archived software release.

> **Note:** The associated article is currently under review at
> *Bioinformatics*. Its citation will be added here upon publication.

### Dissertation

```bibtex
@mastersthesis{colonese_2026_thesis,
  author = {Colonese, Henrique},
  title  = {Virtual Immunohistochemical Staining for Pathological Complete
            Response Prediction in Triple-Negative Breast Cancer},
  school = {Universidade Federal de Minas Gerais},
  year   = 2026,
  url    = {https://repositorio.ufmg.br/items/d99b933c-323c-45bd-a228-5fd05507a7c6}
}
```

### Software

```bibtex
@software{colonese_2026_23788133,
  author    = {Colonese, Henrique and Ayala, Daniel and Teodoro, George},
  title     = {Source code for: Virtual Immunohistochemical Staining for
               Pathological Complete Response Prediction in
               Triple-Negative Breast Cancer},
  month     = oct,
  year      = 2026,
  publisher = {Zenodo},
  version   = {v1.0.0},
  doi       = {10.5281/zenodo.23788133},
  url       = {https://doi.org/10.5281/zenodo.23788133}
}
```
