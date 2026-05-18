# cPaiNN

cPaiNN (charge-PaiNN) is a graph neural network (GNN) model based on the [PaiNN architecture](https://doi.org/10.48550/arXiv.2102.03150) and inspired by [CHGNet](https://doi.org/10.1038/s42256-023-00716-3). The model is designed to predict energies, forces, magnetic moments, and atomic charges for atomistic systems.

In particular, cPaiNN can describe:

- Magnetic moments derived from [Mulliken population analysis](https://doi.org/10.1063/1.1740588)
- Atomic charges obtained from [Bader charge analysis](https://doi.org/10.1088/0953-8984/21/8/084204)

The model is intended for atomistic simulations and machine-learning-driven materials discovery workflows.

---

# Installation

Clone the repository and install cPaiNN locally:

```bash
pip install .
```

## Optional: Install Universal Machine Learning Potentials

The following packages are only required if you want to use the included interfaces to other machine learning potentials (MLPs).

### CHGNet

```bash
pip install chgnet
```

### M3GNet

```bash
pip install m3gnet
```

### MACE-MP-0

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

# Training a cPaiNN Model

Training parameters are defined in the `config.toml` file. At minimum, the dataset path must be specified.

Supported dataset formats include:

- `.xyz`
- `.traj`
- Any format supported by `ase.io.read`

Training parameters can either be:

1. Defined in `config.toml`
2. Passed directly as command-line arguments

Command-line arguments overwrite the values defined in `config.toml`.

## Run Training

Using the provided shell script:

```bash
bash run.sh
```

Or directly with Python:

```bash
python train.py --cfg config.toml
```

---

# Using Bader Charges

To train cPaiNN using atomic charges obtained from Bader charge analysis, the charges must be stored in the ASE `Atoms` object under the `arrays` dictionary:

```python
atoms.arrays["bader_charge"] = bader_charges
```

The Bader charges can be extracted from an `ACF.dat` file using either:

- [ASE](https://wiki.fysik.dtu.dk/ase/)
- The `attach_bader_charges` function located in `extract_data/utils.py`

An example workflow is provided in the notebook:

```text
attach_bader_charge.ipynb
```

---

# Pre-trained Models and Datasets

A pre-trained cPaiNN model trained on the [polyanionic sodium cathode dataset](https://doi.org/10.11583/DTU.27202446) is available together with the corresponding [test dataset](https://doi.org/10.11583/DTU.27411681).

Different model names correspond to different:

- Target properties
- Hidden dimensions
- Numbers of interaction layers

---

# Simulations with cPaiNN

cPaiNN can be used independently or together with other universal machine interatomic learning potentials (MLIPs).

Additional MLPs can easily be integrated as long as they provide an ASE-compatible calculator. To add a new MLP, modify:

```text
cPaiNN/relax.py
```

inside the method:

```python
ML_Relaxer.get_get_calc()
```

following the same structure as the existing implementations.

---

# Example Workflows

## Single-Point Calculations

```bash
python One_hot_calculation.py
```

## Structure Relaxation

```bash
python Structure_relax.py
```

## Molecular Dynamics (MD)

```bash
python MD_simulation.py
```

## Nudged Elastic Band (NEB)

```bash
python NEB.py
```

## Generative Algorithm (GA)

```bash
python NEB.py
```

---
# autocPaiNN: Active Learning Workflow

The repository also includes an active learning workflow called `autocPaiNN`, which enables efficient and autonomous training of cPaiNN models for specific simulation tasks.

The active learning framework is based on [Curator](https://doi.org/10.26434/chemrxiv-2024-p5t3l), with additional utilities included in the cPaiNN package to support active learning workflows.

GitHub repository:

```text
https://github.com/dtu-energy/autocPaiNN
```

---

# Citation

If you use cPaiNN in your work, please cite the relevant publication.
```text
https://doi.org/10.1002/aidi.202500065
```

The implementation is based on earlier versions of PaiNN and related active learning workflows. Please also cite:

```text
https://doi.org/10.1038/s41524-022-00863-y
https://doi.org/10.26434/chemrxiv-2024-p5t3l
```
