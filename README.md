# GPU-Parallelized Inference of ABodyBuilder3

This repository provides GPU-parallelized inference for **ABodyBuilder3**, an antibody structure prediction model.

For details on ABB3, see the original paper:
[ABodyBuilder3: Improved and Scalable Antibody Structure Predictions](https://arxiv.org/abs/2405.20863).

## Installation

### Download Model Weights
Download the necessary model weights from Zenodo:
```bash
mkdir -p output/ zenodo/
wget -P zenodo/ https://zenodo.org/records/11354577/files/output.tar.gz
tar -xzvf zenodo/output.tar.gz -C output/
```

### Setup Environment
The Conda initialization script has been slightly modified for convenience:
```bash
./init_conda_venv.sh
```
After installation, activate the environment and install Ray:
```bash
conda activate abb3
pip install ray==2.44.1
```

## Inference and Evaluation
The input CSV file should contain the following columns:
- `Therapeutic`
- `HeavySequence`
- `LightSequence`

Run the example inference script from the `abodybuilder3` directory:
```bash
python parallelized/inference.py parallelized/example.csv parallelized/example_output/ --num_gpus 0.2
```
num_gpus is the number of GPUs used per task (per antibody).
If running from a different directory, update the model path in `inference.py` accordingly.

## Citation (as provided by ABB3 authors as of april 25')
Citing ABB3:

**ABodyBuilder3:**
```bibtex
@article{abodybuilder3,
    author = {Kenlay, Henry and Dreyer, Frédéric A and Cutting, Daniel and Nissley, Daniel and Deane, Charlotte M},
    title = "ABodyBuilder3: Improved and Scalable Antibody Structure Predictions",
    journal = {Bioinformatics},
    volume = {40},
    number = {10},
    pages = {btae576},
    year = {2024},
    month = {10},
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btae576}
}
```

**ImmuneBuilder (original foundation work):**
```bibtex
@article{immunebuilder,
  author = {Abanades, Brennan and Wong, Wing Ki and Boyles, Fergus and Georges, Guy and Bujotzek, Alexander and Deane, Charlotte M.},
  doi = {10.1038/s42003-023-04927-7},
  issn = {2399-3642},
  journal = {Communications Biology},
  number = {1},
  pages = {575},
  title = {ImmuneBuilder: Deep-Learning Models for Predicting the Structures of Immune Proteins},
  volume = {6},
  year = {2023}
}
```

---
This refined README improves readability, consistency, and formatting while maintaining all the essential details.

