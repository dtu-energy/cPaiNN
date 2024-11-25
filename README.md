# cPaiNN

A new GNN model based on the [PaiNN architecture](https://doi.org/10.48550/arXiv.2102.03150) and inspired by [CHGNet](https://doi.org/10.1038/s42256-023-00716-3), which we call charge-PaiNN (cPaiNN), that is capable of describing both the magnetic moments derived from [Mulliken analysis](https://doi.org/10.1063/1.1740588) and atomic charges obtained through [Bader charge analysis](https://doi.org/10.1088/0953-8984/21/8/084204) based on the charge density.

To install cPaiNN use the commando, when in the repository;
```bash
pip install .
```
This will install cPaiNN along with [CHGNet](https://doi.org/10.1038/s42256-023-00716-3) and [M3GNet](https://doi.org/10.1038/s41524-024-01227-4).

To install [Mace-MP-0](https://doi.org/10.48550/arXiv.2401.00096) one need to use the commando:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Several examples in this repository used all four machine learning potentials and the loading of them can be inspected in "cPaiNN/relax.py"


To train a cPaiNN model the path to the dataset needs to be added in the "config.toml" file. The dataset needs to be XYZ format, .traj format or any other format which can be read by the ase.io.read function. Tunning other parameters for the cPaiNN training are also done in the "config.toml" file. One can also parse the parameters directly in the training scripts as arguments. This will overwrite the parameters to in the "config.toml" file.
To run the cPaiNN training use either the "run.sh" script or the commando:
```bash
python train.py --cfg config.toml
```

The pre-trained cPaiNN model on [the polyanaion sodium cathode dataset](https://doi.org/10.11583/DTU.27202446) can be found along with [the test dataset](https://doi.org/10.11583/DTU.27411681). The different name corresponds to the different properties it is trained on and the different hidden nodes and interaction layer used for the training.

cPaiNN can be used for different simualtion alone or along with other universal MLP. Adding new MLPs can easily be done as long as they have an ASE calcultator. Just add them in the "cPaiNN/relax.py" class object "ML_Relaxer.get_get_calc()" in the same way as the other MLPs are added.
Four examples are given in this repository:
One-hot calculation of atomic structures:
```bash
python One_hot_calculation.py
```

Structure optimization of atomic structures:
```bash
python Structure_relax.py
```

Molecular dynamic (MD) simulation of an atomic structure:
```bash
python MD_simulation.py
```

Nugded eleastic band (NEB) calculation of an ionic movement in an atomic structure:
```bash
python NEB.py
```

When using the code cite: 

The code is modfied from older version of PaiNN, so also cite: https://doi.org/10.1038/s41524-022-00863-y and https://doi.org/10.26434/chemrxiv-2024-p5t3l