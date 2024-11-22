from ase.db import connect
from ase.io import read, write, Trajectory
import sys, os 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import itertools
import json
from tqdm import tqdm
from ase import units
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms

from ase.optimize import FIRE, BFGS
from datetime import datetime
    
from ase.neb import NEBTools,NEB
from pathlib import Path
from ase.neb import NEBTools
from cPaiNN.relax import ML_Relaxer

sys.path.append('/home/energy/mahpe/Structure_generation/Post_processing')
from forcecurve import  fit_images #import from local pyhton file
def plot_band(neb, ax,c,label):
    """Plots the NEB band on matplotlib axes object 'ax'. If ax=None
    returns a new figure object."""
    forcefit = fit_images(neb.images)
    ax = forcefit.plot(c,label,ax=ax)
    return ax.figure



def copy_with_properties(atoms: Atoms) -> Atoms:
    atoms_copy = atoms.copy()
    atoms_copy.calc = SinglePointCalculator(
        atoms_copy,
        energy=atoms.get_potential_energy(),
        forces=atoms.get_forces())
    return atoms_copy

def save_to_traj(atoms: Atoms, filename: str) -> None:
    with Trajectory(filename, mode='a') as traj:
        traj.write(atoms)


# Import arguments
model_name = 'cpainn'
model_path = '/home/energy/mahpe/Playground/Universal_FF/ML_models_stress/Polyanion_bader_magmom_stress_512_4' 
system_name = 'FePO4'
root_dir = '.'
neb_folder_path = 'NEB_example/FePO4' # NEB olders with initial and final structures. Note that the folder should conatin a "initial" and "final" folder with the initial and final structures in OUTCAR format

# Set class for ML calculator
relax_cell = False # the cell must not be relaxed for NEB calcualtions
optimizer = 'FIRE'
device_global = 'cuda' if torch.cuda.is_available() else 'cpu'
per_atom = True # If True, the energy is per atom
fmax = 0.05 # Maximum force for relaxation
steps = 2000 # Maximum steps for relaxation

# NEB parameters
nimages = 5 
climb = False
neb_fmax = 0.05
neb_steps = 2000

# Define the folder for the ML NEB calculation
ml_neb_folder = f'NEB_ML/{system_name}'
if not os.path.exists(ml_neb_folder):
    os.makedirs(ml_neb_folder)

# Read the initial and final structures from the OUTCAR files
dft_neb_traj = Trajectory(neb_folder_path+'/NEB/neb_last.traj' )
initial_prime = read(neb_folder_path+'/initial/OUTCAR',index=0)
final_prime = read(neb_folder_path+'/final/OUTCAR',index=0)

# Define the machine learning potentials
ml_model = {}
ml_model['cPaiNN'] = ML_Relaxer(calc_name='cpainn',calc_paths=model_path,
                          optimizer=optimizer,relax_cell=relax_cell,device=device_global)

ml_model['CHGNet'] = ML_Relaxer(calc_name='chgnet',calc_paths=None,
                          optimizer=optimizer,relax_cell=relax_cell,device=device_global)
ml_model['Mace'] = ML_Relaxer(calc_name='mace_large',calc_paths=None,
                          optimizer=optimizer,relax_cell=relax_cell,device=device_global)
ml_model['M3GNet'] = ML_Relaxer(calc_name='m3gnet',calc_paths=None,
                          optimizer=optimizer,relax_cell=relax_cell,device=device_global)# MACE needs to be last or else CHGNET will cause an error


# Plot DFT NEB
fig, ax = plt.subplots(figsize=(5.5, 4.0))
ax.plot(0,0,'.b',alpha = 0,label='DFT')
n_img = nimages+2
dft_atoms = dft_neb_traj[-n_img:]
dft_nebtools = NEBTools(dft_atoms) # images including initial and final
dft_Ef, dft_dE = dft_nebtools.get_barrier(fit=False) #fit = false means that you take the max energy for the images. fit True uses the interpolation
dft_max_force = dft_nebtools.get_fmax()
dft_Er = dft_Ef - dft_dE
dft_E_initial, dft_E_final = dft_atoms[0].get_potential_energy(),dft_atoms[-1].get_potential_energy()
dft_Ebarrier = dft_Ef+dft_E_initial - (dft_E_initial+dft_E_final)/2
plot_band(dft_nebtools,ax,'b','DFT');

# Perform NEB calculation for ML potentials
colors = ['r','g','c','m','y','k']
count = 0
for ml_name, ml_class in ml_model.items():
    print(ml_class.calc_name)
    print(f'Performing NEB calculation for {ml_class.calc_name}')
    
    # Setup the initial and final structures with the ML calculator
    print('Performing structure relaxation for the initial and final structures')
    initial = initial_prime.copy()
    final = final_prime.copy()
    
    # Perform geometry optimization for initial and final images
    initial_results=ml_class.relax(initial, fmax=fmax, steps=steps,
                                    traj_file=f'{ml_neb_folder}/{ml_name}_initial.traj', 
                                    log_file=f'{ml_neb_folder}/{ml_name}_initial.log', interval=1)
    initial_ml = initial_results['final_structure']
    final_results=ml_class.relax(final, fmax=fmax, steps=steps,
                                    traj_file=f'{ml_neb_folder}/{ml_name}_final.traj', 
                                    log_file=f'{ml_neb_folder}/{ml_name}_final.log', interval=1)
    final_ml = final_results['final_structure']

    # Make a band consisting of N images
    images = [initial]
    images += [initial.copy() for i in range(nimages)]
    images += [final]

    # Setup NEB
    neb_path = f'{ml_neb_folder}/{ml_name}_neb_ML.xyz'
    neb = NEB(images,allow_shared_calculator=True, climb=climb)#, parallel=parallel)#, method="improvedtangent")
    neb.interpolate(mic=True)

    # Set up the calculators
    for i, image in enumerate(images):
        image.calc = ml_class.calculator
        image.get_potential_energy()

    # Run the NEB
    print('Running NEB calculation')
    optimizer = ml_class.opt_class(neb,trajectory=neb_path,logfile=neb_path.replace('xyz','log'))
    optimizer.run(fmax=neb_fmax, steps=neb_steps)
    print('NEB calculation done')
    print('---------------------------------')
    
    # plot the results 
    ml_neb_traj = neb.images

    ml_nebtools = NEBTools(ml_neb_traj) # images including initial and final
    ml_Ef, ml_dE = ml_nebtools.get_barrier(fit=False) #fit = false means that you take the max energy for the images. fit True uses the interpolation
    ml_max_force = ml_nebtools.get_fmax(allow_shared_calculator=True)
    ml_Er = ml_Ef - ml_dE
    ml_E_initial, ml_E_final = ml_neb_traj[0].get_potential_energy(),ml_neb_traj[-1].get_potential_energy()
    ml_Ebarrier = ml_Ef+ml_E_initial - (ml_E_initial+ml_E_final)/2
    plot_band(ml_nebtools,ax,colors[count],ml_name);
    ax.legend(loc=0,prop = {'size':14},frameon=False)
    textstr = f'DFT vs {ml_name} \n'+ \
    r'$dE_\mathrm{{kinetic}}=${:.3f} eV'.format(np.abs(dft_Ebarrier-ml_Ebarrier))#+ ' \n'+ \
    ax.text(1.05, 0.98-0.3*count, textstr, transform=ax.transAxes, fontsize=14,  
    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    count += 1
        
    plt.savefig('NEB.png',dpi=300,bbox_inches='tight')
