
from ase.db import connect
from ase.io import read, write, Trajectory
import sys, os 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import itertools
from tqdm import tqdm
from ase import units
import seaborn as sns
from cPaiNN.relax import ML_Relaxer
import pandas as pd

# Setup paths
# Model directory (can be a list of directories)
model_dir = '/home/energy/mahpe/Playground/Universal_FF/ML_models_stress/Polyanion_bader_magmom_stress_512_4' 

# Set class for ML calculator
relax_cell = True
optimizer = 'FIRE'
device_global = 'cuda' #if torch.cuda.is_available() else 'cpu'
per_atom = True # If True, the energy is per atom
fmax = 0.05 # Maximum force for relaxation
steps = 2000 # Maximum steps for relaxation

# Define the machine learning potentials
ml_model = {}
ml_model['PAINN'] = ML_Relaxer(calc_name='painn_charge',calc_paths=model_dir,
                          optimizer=optimizer,relax_cell=relax_cell,device=device_global)

ml_model['chgnet'] = ML_Relaxer(calc_name='chgnet',calc_paths=None,
                          optimizer=optimizer,relax_cell=relax_cell,device=device_global)

ml_model['Mace'] = ML_Relaxer(calc_name='mace_large',calc_paths=None,
                          optimizer=optimizer,relax_cell=relax_cell,device=device_global)
ml_model['m3gnet'] = ML_Relaxer(calc_name='m3gnet',calc_paths=None,
                          optimizer=optimizer,relax_cell=relax_cell,device=device_global)


## OUTCAR file
list_root_dir = {'FePO4_init':'NEB_example/FePO4/initial/OUTCAR',
                 'FePO4_final':'NEB_example/FePO4/final/OUTCAR',}

file = 'OUTCAR'

# Create dictionary for the results
results = {}
for ml_class in ml_model.keys():
    results[ml_class] = {'Name':[],'System':[],'a':[],'b':[],'c':[],'alpha':[],'beta':[],'gamma':[],'volume':[],'energy':[],'iter':[]}
results['DFT_init'] = {'Name':[],'System':[],'a':[],'b':[],'c':[],'alpha':[],'beta':[],'gamma':[],'volume':[],'energy':[],'iter':[]}
results['DFT_final'] = {'Name':[],'System':[],'a':[],'b':[],'c':[],'alpha':[],'beta':[],'gamma':[],'volume':[],'energy':[],'iter':[]}

for system_name, root_dir in list_root_dir.items():
    for folder in os.listdir(root_dir):
        outcar_dir = os.path.join(root_dir,folder,file)
        if not os.path.exists(outcar_dir):
            continue
        print(outcar_dir)
        atom_init = read(outcar_dir,index=0)
        atom_final = read(outcar_dir,index=-1)
        results['DFT_init']['Name'].append(folder)
        results['DFT_init']['System'].append(system_name)
        results['DFT_init']['a'].append(atom_init.cell[0,0])
        results['DFT_init']['b'].append(atom_init.cell[1,1])
        results['DFT_init']['c'].append(atom_init.cell[2,2])
        results['DFT_init']['alpha'].append(atom_init.cell.angles()[0])
        results['DFT_init']['beta'].append(atom_init.cell.angles()[1])
        results['DFT_init']['gamma'].append(atom_init.cell.angles()[2])
        results['DFT_init']['volume'].append(atom_init.get_volume())
        results['DFT_init']['energy'].append(atom_init.get_potential_energy())
        results['DFT_init']['iter'].append(0)
        results['DFT_final']['Name'].append(folder)
        results['DFT_final']['System'].append(system_name)
        results['DFT_final']['a'].append(atom_final.cell[0,0])
        results['DFT_final']['b'].append(atom_final.cell[1,1])
        results['DFT_final']['c'].append(atom_final.cell[2,2])
        results['DFT_final']['alpha'].append(atom_final.cell.angles()[0])
        results['DFT_final']['beta'].append(atom_final.cell.angles()[1])
        results['DFT_final']['gamma'].append(atom_final.cell.angles()[2])
        results['DFT_final']['volume'].append(atom_final.get_volume())
        results['DFT_final']['energy'].append(atom_final.get_potential_energy())
        results['DFT_final']['iter'].append(-1)


        for ml_clas in ml_model.keys():
            print('Relaxing with ',ml_clas)
            relaxer = ml_model[ml_clas]
            atom = atom_init.copy()
            traj_path = ml_clas+'_traj.traj'
            log_path = ml_clas+'_log.txt'
            relax_results=relaxer.relax(atom, fmax=fmax, steps=steps,
                                        traj_file=traj_path, log_file=log_path, interval=1)
            atom_ml = relax_results['final_structure']
            results[ml_clas]['Name'].append(folder)
            results[ml_clas]['System'].append(system_name)
            results[ml_clas]['a'].append(atom_ml.cell[0,0])
            results[ml_clas]['b'].append(atom_ml.cell[1,1])
            results[ml_clas]['c'].append(atom_ml.cell[2,2])
            results[ml_clas]['alpha'].append(atom_ml.cell.angles()[0])
            results[ml_clas]['beta'].append(atom_ml.cell.angles()[1])
            results[ml_clas]['gamma'].append(atom_ml.cell.angles()[2])
            results[ml_clas]['volume'].append(atom_ml.get_volume())
            results[ml_clas]['energy'].append(atom_ml.get_potential_energy())

            # open txt file to write the results
            f = open(log_path, 'r')
            data = f.read()
            f.close()
            # read last line of the file
            last_line = data.split('\n')[-2] # 
            last_line = list(filter(None, last_line.split(' '))) # remove empty strings
            last_iter =int(last_line[1])
            results[ml_clas]['iter'].append(last_iter)
            # delete the files
            os.remove(log_path)
            os.remove(traj_path)
            print('Done with ',ml_clas)
            # print the mean absolute relative error for a,b,c and alpha,beta,gamma compared to the initial structure in procent
            
            #print the mean absolute relative error for the atomic positions compared to the initial structure in procent
        # Save the results
        results_df = {}
        for key in results.keys():
            results_df[key] = pd.DataFrame(results[key])
            results_df[key].to_csv(key+'_results.csv')