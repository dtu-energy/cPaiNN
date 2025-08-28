from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor
import sys 
print(sys.path)
import chgnet
from chgnet.data.dataset import StructureData, get_train_val_test_loader
from chgnet.trainer import Trainer
from chgnet.model import CHGNet
import torch
datapath = '/home/energy/mahpe/Published_code/polyanion_cathode_dataset/polyanion_cathode_dataset_with_optimization_steps/Combined.xyz'

# Load the data
atoms = read(datapath, index=':')

# Convert to pymatgen structure and store in a list
structures, energies_per_atom, forces, stresses, magmoms = [], [], [], [], []
for atom in atoms:
    structure = AseAtomsAdaptor.get_structure(atom)
    structures.append(structure)
    energies_per_atom.append(atom.get_potential_energy()/len(atom))
    forces.append(atom.get_forces())
    stresses.append(atom.get_stress())
    magmoms.append(atom.get_magnetic_moments())

# Create a StructureData object
dataset = StructureData(
    structures=structures,
    energies=energies_per_atom,
    forces=forces,
    stresses=stresses,  # can be None
    magmoms=magmoms,  # can be None
)

# split the dataset into train, val, test

train_loader, val_loader, test_loader = get_train_val_test_loader(
    dataset, batch_size=20, train_ratio=0.8, val_ratio=0.1
)

# train from sctrach
model_path = '/home/energy/mahpe/anaconda3/envs/env_sylg/lib/python3.11/site-packages/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar'
model = torch.load(model_path, map_location='cuda')

chgnet = CHGNet(**model['model']["model_args"])

# Define Trainer
trainer = Trainer(
    model=chgnet,
    targets="efm", #efsm
    optimizer="Adam",
    scheduler="CosLR",
    criterion="MSE",
    epochs=1000,
    learning_rate=0.001,
    use_device="cuda",
    print_freq=100,
)

trainer.train(train_loader, val_loader, test_loader)

model = trainer.model
best_model = trainer.best_model  # best model based on validation energy MAE

state = {   "model": best_model.as_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.scheduler.state_dict(),
            "training_history": trainer.training_history,
            "trainer_args": trainer.trainer_args,
        }
torch.save(state, 'best_model_saved.pth')