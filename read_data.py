from ase.io import read
import matplotlib.pyplot as plt
import numpy as np

# Set the datafile path
data_file = 'NaMPO4_olivine_single_total.xyz'

# Load all the structures 
all_structures = read(data_file, index=':')

# Preset the property list
Energy = []
Forces = []
Stress = []
Magmom = []
Bader_charge = []

# Loop over all the structures and extract the properties
for i, structure in enumerate(all_structures):
    Energy.append(structure.get_potential_energy())
    Forces.append(structure.get_forces())
    Stress.append(structure.get_stress())
    Magmom.append(structure.get_magnetic_moments())
    Bader_charge.append(structure.arrays['bader_charge'])

## Plot the distribution of the properties ##
# Calculate the force vector size for each atom in each structure
Forces_plot = np.concatenate([np.sqrt(np.sum(np.square(force), axis=1)) for force in Forces]) 

# Collect the properties into a single list
Stress_plot = np.concatenate(Stress)
Magmom_plot = np.concatenate(Magmom)
Bader_charge_plot = np.concatenate(Bader_charge)

fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(25, 5),gridspec_kw={'wspace': 0.5})
Nbins = 50
font = 18
color = 'blue'
ax[0].hist(Energy, bins=Nbins, color=color)
ax[0].set_xlabel('Energy (eV)', fontsize=font)
ax[0].set_ylabel('Frequency', fontsize=font)
ax[0].set_title('Energy distribution', fontsize=font)
ax[0].tick_params(axis='both', which='major', labelsize=font)
ax[1].hist(Forces_plot, bins=Nbins, color=color)
ax[1].set_xlabel('Force (eV/A)', fontsize=font)
ax[1].set_ylabel('Frequency', fontsize=font)
ax[1].set_title('Force distribution', fontsize=font)
ax[1].tick_params(axis='both', which='major', labelsize=font)
ax[2].hist(Stress_plot, bins=Nbins, color=color)
ax[2].set_xlabel('Stress (GPa)', fontsize=font)
ax[2].set_ylabel('Frequency', fontsize=font)
ax[2].set_title('Stress distribution', fontsize=font)
ax[2].tick_params(axis='both', which='major', labelsize=font)

ax[3].hist(Magmom_plot, bins=Nbins, color=color)
ax[3].set_xlabel('Magnetic moment (mu_B)', fontsize=font)
ax[3].set_ylabel('Frequency', fontsize=font)
ax[3].set_title('Magnetic moment distribution', fontsize=font)
ax[3].tick_params(axis='both', which='major', labelsize=font)
ax[4].hist(Bader_charge_plot, bins=Nbins, color=color)
ax[4].set_xlabel('Bader charge (e)', fontsize=font)
ax[4].set_ylabel('Frequency', fontsize=font)
ax[4].set_title('Bader charge distribution', fontsize=font)
ax[4].tick_params(axis='both', which='major', labelsize=font)

