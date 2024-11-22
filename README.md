# cPaiNN

A new GNN model based on the PaiNN architecture \cite{schutt2021equivariant} and inspired by CHGNet \cite{deng2023chgnet}, which we call charge-PaiNN (cPaiNN), that is capable of describing both the magnetic moments derived from Mulliken analysis \cite{mulliken1955electronic} and atomic charges obtained through Bader charge analysis \cite{bader_charge} based on the charge density. (Mention also that you have the other MLPs)
To use Mace-MP-0 one need to follow do download it 

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


To train a cPaiNN model the path to the dataset needs to be added in the "config.toml" file. The dataset needs to be XYZ format, .traj format or any other format which can be read by the ase.io.read function. Tunning other parameters for the cPaiNN training are also done in the "config.toml" file. One can also parse the parameters directly in the training scripts as arguments. This will overwrite the parameters to in the "config.toml" file.
To run the cPaiNN training use either the "run.sh" script or the commando:
```bash
python train.py --cfg config.toml
```

The pre-trained cPaiNN model on the polyanaion sodium cathode dataset !!CITE!! can be found along with the test dataset in !!CITE!!. The different name corresponds to the different properties it is trained on and the different hidden nodes and interaction layer used for the training.

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
