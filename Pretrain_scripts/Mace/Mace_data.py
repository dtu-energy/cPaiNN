from ase.io import read, write
train_atoms = read('train.xyz', index=':')  # Read all structures from the database
train_atoms = train_atoms[100000:-1]
for atoms in train_atoms:
    energy = atoms.get_potential_energy()
    atoms.info['REF_energy'] = energy  # Store the energy in the info dictionary
    forces = atoms.get_forces()
    atoms.arrays['REF_forces'] = forces  # Store the forces in the arrays dictionary
    stress = atoms.get_stress()
    atoms.info['REF_stress'] = stress  # Store the stress in the info dictionary
    write('train_REF.xyz', atoms, format='extxyz', append=True)  # Append the modified atoms to the file
