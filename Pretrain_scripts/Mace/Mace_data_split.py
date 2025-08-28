from ase.io import read, write

atoms = read('Combined.xyz', index=':')  # Read all structures from the database
print(len(atoms))

import numpy as np
import random
from sklearn.model_selection import train_test_split
seed =711
random.seed(seed)
np.random.seed(seed)

# Split the dataset into train, validation, and test sets
train_atoms, test_atoms = train_test_split(atoms, test_size=0.15, random_state=seed)
train_atoms, valid_atoms = train_test_split(train_atoms, test_size=0.15, random_state=seed)  # 0.15 x 0.8 = 0.12

print(f"Train set size: {len(train_atoms)}")
print(f"Validation set size: {len(valid_atoms)}")
print(f"Test set size: {len(test_atoms)}")

write('train.xyz', train_atoms, format='extxyz')
write('valid.xyz', valid_atoms, format='extxyz')
write('test.xyz', test_atoms, format='extxyz')
