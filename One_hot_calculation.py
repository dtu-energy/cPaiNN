from ase.io import read
import torch
from cPaiNN.relax import ML_Relaxer
import numpy as np

# linear least square of the formula E = X + N_Na*Na + N_M*M + N_P*P + N_O*O + N_Fe*Fe + N_Mn*Mn + N_Co*Co + N_Ni*Ni + N_Si*Si + N_S*S with N_i being the number of atoms of element i
# and X being the energy of the system
def get_energy(traj,E_ml,element_diff):
    E = []
    for i,atom in enumerate(traj):
        E_i = float(E_ml[i])
        for element in element_diff.keys():
            if element == 'P':
                continue
            if element == 'O':
                N = 1
            else:
                N = len([a for a in atom if a.symbol == element])
            E_i += N*element_diff[element]
        E.append(E_i)
    return E

# Setup paths
# Model directory (can be a list of directories)
model_dir = '/home/energy/mahpe/Playground/Universal_FF/ML_models_stress/Polyanion_bader_magmom_stress_512_4'
device_global = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the machine learning potential calcualtors (Note I call the calculators from each ML_relaxer class)
ml_calculators = {}
ml_calculators['cPaiNN'] = ML_Relaxer(calc_name='cpainn',calc_paths=model_dir,device=device_global).calculator
ml_calculators['CHGNet'] = ML_Relaxer(calc_name='chgnet',calc_paths=None,device=device_global).calculator
ml_calculators['M3GNet'] = ML_Relaxer(calc_name='m3gnet',calc_paths=None,device=device_global).calculator
ml_calculators['Mace'] = ML_Relaxer(calc_name='mace_large',calc_paths=None,device=device_global).calculator # MACE needs to be last or else CHGNET will cause an error

# OUTCAR file
dft_path = 'Relax_examples/one_hot_example.xyz'
dft_traj = read(dft_path, index=':')

# Determine the A-matrix used to fit the energy to the number of atoms of each element
symbols = []
for atom in dft_traj:
    symbols+=atom.get_chemical_symbols()
unique_elements = np.unique(symbols)
N_elements = len(unique_elements) - 1 # PO/SiO/SO goes as one

A_matrix = np.zeros((len(dft_traj), N_elements))
# Loop over all structures in the trajectory
for j,atom in enumerate(dft_traj):
    # Loop over all unique elements
    for i, element in enumerate(unique_elements):
        if element == 'P' or element == 'Si' or element == 'S':
            continue
        
        if element == 'O':
            N=1
        else:
            N = len([a for a in atom if a.symbol == element])
        # Fill the A matrix with the number of atoms of each element for each structure in the trajectory
        A_matrix[j,i] = N

# Loop over all structures in the trajectory

# One-hot calculation for each ML calculator
for ml_name, ml_calc in ml_calculators.items():
    print(f'Using {ml_name} model')

    # Save the energy result for the scaled energy fitting
    ML_energies = []
    DFT_energies = []

    # One-hot calculation for each structure in the trajectory
    for dft_atom in dft_traj:
        print(f'Reading the initial structure of {dft_atom.get_chemical_formula()}')
        # Get the DFT energy, forces, stress, magnetic moments and bader charges
        dft_energy = dft_atom.get_potential_energy()/len(dft_atom)
        dft_force = np.sqrt(np.sum(np.square(dft_atom.get_forces()),axis=1))
        dft_stress = dft_atom.get_stress(voigt=False)
        dft_magmom = dft_atom.get_magnetic_moments()
        dft_bader = dft_atom.arrays['bader_charge']
        print(f'DFT energy: {dft_energy} eV/atom')
        DFT_energies.append(dft_energy)

        # Calculate the ML energy, forces, stress, magnetic moments and bader charges
        ml_atom = dft_atom.copy()
        ml_atom.set_calculator(ml_calc)
        ml_energy = ml_atom.get_potential_energy()/len(ml_atom)
        ml_force = np.sqrt(np.sum(np.square(ml_atom.get_forces()),axis=1))
        ml_stress = ml_atom.get_stress(voigt=False)
        # Try to get the magnetic moments and bader charges
        try:
            ml_magmom = ml_atom.get_magnetic_moments()
        except:
            ml_magmom = None
        try:
            ml_bader = ml_atom.calc.results['bader_charge']
        except:
            ml_bader = None
        print(f'ML energy: {ml_energy} eV/atom')
        ML_energies.append(ml_energy)

        # Get the error in the energy, forces, stress, magnetic moments and bader charges
        mae_energy = np.abs(ml_energy-dft_energy)
        mae_force = np.mean(np.abs(ml_force-dft_force))
        mae_stress = np.mean(np.abs(ml_stress-dft_stress))
        if ml_magmom is not None:
            mae_magmom = np.mean(np.abs(ml_magmom-dft_magmom))
        else:
            mae_magmom = None
        if ml_bader is not None:
            mae_bader = np.mean(np.abs(ml_bader-dft_bader))
        else:
            mae_bader = None
        
        print(f'ML energy error: {mae_energy} eV/atom')
        print(f'ML force error: {mae_force} eV/Å')
        print(f'ML stress error: {mae_stress} GPa')
        if ml_magmom is not None:
            print(f'ML magnetic moment error: {mae_magmom} mu_B')
        if ml_bader is not None:
            print(f'ML bader charge error: {mae_bader} e')
        print('')
    
    # Scale the energy to the number of atoms of each element
    E_diff = np.array(DFT_energies) - np.array(ML_energies)
    # Solve the linear least square E_diff = A_matrix * X
    X = np.linalg.lstsq(A_matrix, E_diff, rcond=None)[0]
    element_diff = dict(zip(unique_elements, X))

    # Calculate the scaled energy
    energy_ml_scaled = np.array(get_energy(dft_traj,np.array(ML_energies),element_diff))
    mae_energy_scaled = np.mean(np.abs(energy_ml_scaled-DFT_energies))
    print(f'ML energy error after scaling: {mae_energy_scaled} eV/atom')
