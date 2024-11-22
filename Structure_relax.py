
from ase.io import read
import torch
from cPaiNN.relax import ML_Relaxer
import pandas as pd

# Setup paths
# Model directory (can be a list of directories)
model_dir = '/home/energy/mahpe/Playground/Universal_FF/ML_models_stress/Polyanion_bader_magmom_stress_512_4' 

# Set class for ML calculator
relax_cell = True
optimizer = 'FIRE'
device_global = 'cuda' if torch.cuda.is_available() else 'cpu'
per_atom = True # If True, the energy is per atom
fmax = 0.05 # Maximum force for relaxation
steps = 2000 # Maximum steps for relaxation

# Define the machine learning potentials
ml_model = {}
ml_model['cPaiNN'] = ML_Relaxer(calc_name='cpainn',calc_paths=model_dir,
                          optimizer=optimizer,relax_cell=relax_cell,device=device_global)

ml_model['CHGNet'] = ML_Relaxer(calc_name='chgnet',calc_paths=None,
                          optimizer=optimizer,relax_cell=relax_cell,device=device_global)
ml_model['M3GNet'] = ML_Relaxer(calc_name='m3gnet',calc_paths=None,
                          optimizer=optimizer,relax_cell=relax_cell,device=device_global)

ml_model['Mace'] = ML_Relaxer(calc_name='mace_large',calc_paths=None,
                          optimizer=optimizer,relax_cell=relax_cell,device=device_global) # MACE needs to be last or else CHGNET will cause an error


## OUTCAR file
list_root_dir = {'Silicate':'Relax_examples/Na2FeMnNiSiO4/OUTCAR',
                 'Alluadite':'Relax_examples/NaFeMnCoNiSO4/OUTCAR',}

# Create dictionary for the results
results = {}
for ml_class in ml_model.keys():
    results[ml_class] = {'System':[],'a':[],'b':[],'c':[],'alpha':[],'beta':[],'gamma':[],'volume':[],'energy':[],'iter':[]}
results['DFT_init'] = {'System':[],'a':[],'b':[],'c':[],'alpha':[],'beta':[],'gamma':[],'volume':[],'energy':[],'iter':[]}
results['DFT_final'] = {'System':[],'a':[],'b':[],'c':[],'alpha':[],'beta':[],'gamma':[],'volume':[],'energy':[],'iter':[]}

for system_name, outcar_dir in list_root_dir.items():

    # Read the initial and final structure from the OUTCAR file
    atom_init = read(outcar_dir,index=0)
    atom_final = read(outcar_dir,index=-1)
    
    # Save the initial and final structure information
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


    # Relax the structure with the machine learning potentials
    for ml_class in ml_model.keys():
        print('Relaxing with ',ml_class)
        relaxer = ml_model[ml_class]
        atom = atom_init.copy()
        traj_path = ml_class+f'_{system_name}_traj.traj'
        log_path = ml_class+f'_{system_name}_log_.txt'
        relax_results=relaxer.relax(atom, fmax=fmax, steps=steps,
                                    traj_file=traj_path, log_file=log_path, interval=1)
        atom_ml = relax_results['final_structure']
        results[ml_class]['System'].append(system_name)
        results[ml_class]['a'].append(atom_ml.cell[0,0])
        results[ml_class]['b'].append(atom_ml.cell[1,1])
        results[ml_class]['c'].append(atom_ml.cell[2,2])
        results[ml_class]['alpha'].append(atom_ml.cell.angles()[0])
        results[ml_class]['beta'].append(atom_ml.cell.angles()[1])
        results[ml_class]['gamma'].append(atom_ml.cell.angles()[2])
        results[ml_class]['volume'].append(atom_ml.get_volume())
        results[ml_class]['energy'].append(atom_ml.get_potential_energy())

        # Open txt file to write the results
        f = open(log_path, 'r')
        data = f.read()
        f.close()

        # Read last line of the file to get the number of iterations
        last_line = data.split('\n')[-2] # 
        last_line = list(filter(None, last_line.split(' '))) # remove empty strings
        last_iter =int(last_line[1])
        results[ml_class]['iter'].append(last_iter)
        
        print('Done with ',ml_class)
        print(results[ml_class])
        
# Save the results
results_df = {}
for key in results.keys():
    print(results[key])
    results_df[key] = pd.DataFrame(results[key])
    results_df[key].to_csv(key+'_results.csv')