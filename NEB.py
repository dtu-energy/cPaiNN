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
sys.path.append('/home/energy/mahpe/Structure_generation/Post_processing')
from forcecurve import  fit_images #import from local pyhton file
def plot_band(neb, ax,c,label):
    """Plots the NEB band on matplotlib axes object 'ax'. If ax=None
    returns a new figure object."""
    forcefit = fit_images(neb.images)
    ax = forcefit.plot(c,label,ax=ax)
    return ax.figure

def perform_NEB(initial,final,calc_dict,ML_name,N_images,folder,climb=False,fire=True,fmax=0.03,max_iter=2000):
    # Set the calculator for initail and final images
    ML_calc = calc_dict[ML_name]
    initial.calc, final.calc = ML_calc, ML_calc
    initial.get_potential_energy()
    final.get_potential_energy()
    # Set up name for the NEB calculation
    name = ML_name

    # Define folder
    folder += f'{initial.get_chemical_formula()}'
    #folder = f'/home/energy/mahpe/Playground/Universal_FF/NEB_{initial.get_chemical_formula()}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    os.chdir(folder)
    
    # perform geometry optimization for initial and final images
    if fire:
        opt = FIRE(initial, trajectory=name+'_initial.xyz',logfile=name+'_initial.log')
        opt.run(fmax=fmax)
        opt = FIRE(final, trajectory=name+'_final.xyz',logfile=name+'_final.log')
        opt.run(fmax=fmax)
    else:
        opt = BFGS(initial, trajectory=name+'_initial.xyz',logfile=name+'_initial.log')
        opt.run(fmax=fmax)
        opt = BFGS(final, trajectory=name+'_final.xyz',logfile=name+'_final.log')
        opt.run(fmax=fmax)
    
    # Loading the optimized structures for initial and final image
    traj_inital = Trajectory(name+'_initial.xyz')
    traj_final = Trajectory(name+'_final.xyz')
    initial, final = traj_inital[-1], traj_final[-1]

    # Make a band consisting of N images
    images = [initial]
    images += [initial.copy() for i in range(N_images)]
    images += [final]

    # Setup NEB
    neb_path = name+ f'_neb_ML.xyz'
    neb = NEB(images,allow_shared_calculator=True, climb=climb)#, parallel=parallel)#, method="improvedtangent")
    neb.interpolate(mic=True)#,apply_constraint=True)
        

    # Set up the calculators
    for i, image in enumerate(images):
        image.calc = ML_calc
        image.get_potential_energy()

    # Run path relaxation
    # Add a boolean for FIRE, and call it from the function's variables.
    if fire:
        opt = FIRE(neb, trajectory=neb_path, logfile=f"{name}_neb_ML_log.log", downhill_check=True, force_consistent=True)
    else:
        opt = BFGS(neb, trajectory=neb_path, logfile=f"{name}_neb_ML_log.log")

    # Run the optimizer and print()
    print(f"Running NEB for {name}")
    opt.run(fmax=fmax,steps=max_iter)
    print(f"NEB calculation for {name} is done")
    
    # Return the NEB object
    return neb.images
    new_images = neb.images

    neb_new = NEB(new_images,allow_shared_calculator=True, climb=True)
    neb_path_new = name+ f'_climb_neb_ML.xyz'
    if fire:
        opt = FIRE(neb_new, trajectory=neb_path_new, logfile=f"{name}_neb_climb_ML_log.log", downhill_check=True, force_consistent=True)
    else:
        opt = BFGS(neb_new, trajectory=neb_path_new, logfile=f"{name}_neb_climb_ML_log.log")


    return neb_new.images

def ML_calc(ml_model:str,model_dir:str,device:str='cpu'):
    """
    Create a ML calculator object
    Args:
    ml_model: str
        The name of the model, currently support 'PAINN_charge', 'PAINN', 'chgnet', 'Mace', 'Mace_personal', 'm3gnet'
    model_dir: str
        The directory of the model
    device: str
        The device to run the model, default is 'cpu'
    Returns:
    encalc: object
        The ML calculator object

    """

    if ml_model == 'cPaiNN':
        from cPaiNN.model import PainnModel
        from cPaiNN.calculator import MLCalculator, EnsembleCalculator
        import torch
        model_pth = Path(model_dir).rglob('*best_model.pth')
        models = []
        for each in model_pth:
            print(each)
            state_dict = torch.load(each, map_location=torch.device(device)) 
            model = PainnModel(
                num_interactions=state_dict["num_layer"], 
                hidden_state_size=state_dict["node_size"], 
                cutoff=state_dict["cutoff"],
                compute_forces=state_dict["compute_forces"],
                compute_stress=state_dict["compute_stress"],
                compute_magmom=state_dict["compute_magmom"],
                compute_bader_charge=state_dict["compute_bader_charge"],
                )
            model.to(device)
            model.load_state_dict(state_dict["model"],)    
            models.append(model)
        if len(models)==1:
            print('Using single PAINN_charge model')
            ensemble = False
            encalc = MLCalculator(models[0])
        elif len(models)>1:
            print('Using ensemble PAINN_charge model')
            ensemble = True
            encalc = EnsembleCalculator(models)
        else:
            raise ValueError('No model found')
    elif ml_model == 'chgnet':
        from chgnet.model.dynamics import CHGNetCalculator
        from chgnet.model import CHGNet
        print('Using CHGNET model')
        ensemble = False
        model = CHGNet.load()
        device = 'cpu'
        encalc = CHGNetCalculator(model=model,use_device=device)

    elif ml_model =='Mace':
        from mace.calculators import mace_mp, MACECalculator
        print('Using MACE model')
        ensemble = False
        encalc = mace_mp(model="large", dispersion=False, default_dtype="float64", device=device)
    elif ml_model == 'Mace_personal':
        from mace.calculators import MACECalculator
        print('Using MACE model')
        ensemble = False
        encalc = MACECalculator(model_dir, device=device)
    elif ml_model == 'm3gnet':
        from m3gnet.models import Potential, M3GNet, M3GNetCalculator
        potential = Potential(M3GNet.load())
        encalc = M3GNetCalculator(potential=potential, stress_weight=0.01)
    else:
        raise ValueError('No model found')
    return encalc

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


####################
inches_to_cm = 2.54
figsize = (25/inches_to_cm, 25/inches_to_cm)#(20/inches_to_cm, 10/inches_to_cm)
plt.rcParams.update({'font.size': 16})

####################

# Setup paths
model_dir_chg_stress = 'Pretrained_models/Polyanion_bader_magmom_stress_512_4'

root_dir = 'NEB'
folder_list = ['FePO4']

for folder_name in folder_list:
    name = folder_name #'NaFePO4_122'
    print('NEB for ',name)
    nimages = 5

    dft_neb_traj = Trajectory(os.path.join(root_dir,'NEB',folder_name,'neb_last.traj') )
    initial_prime = read(os.path.join(root_dir,'initial',folder_name,'OUTCAR'),index=0)
    final_prime = read(os.path.join(root_dir,'final',folder_name,'OUTCAR'),index=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encalc = {}
    #encalc['cPaiNN'] = ML_calc('cPaiNN',model_dir,device)
    #encalc['CHGNet'] = ML_calc('chgnet',None,device)
    encalc['M3GNet'] = ML_calc('m3gnet',None,device)
    #encalc['Mace_0'] = ML_calc('Mace',None,device) # MACE needs to be last or else CHGNET will cause an error
    

    # Plot DFT NEB
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(0,0,'.b',alpha = 0,label=name)
    n_img = nimages+2
    dft_atoms = dft_neb_traj[-n_img:]
    dft_nebtools = NEBTools(dft_atoms) # images including initial and final
    dft_Ef, dft_dE = dft_nebtools.get_barrier(fit=False) #fit = false means that you take the max energy for the images. fit True uses the interpolation
    dft_max_force = dft_nebtools.get_fmax()
    dft_Er = dft_Ef - dft_dE
    dft_E_initial, dft_E_final = dft_atoms[0].get_potential_energy(),dft_atoms[-1].get_potential_energy()
    dft_Ebarrier = dft_Ef+dft_E_initial - (dft_E_initial+dft_E_final)/2
    plot_band(dft_nebtools,ax,'b','DFT');

    colors = ['r','g','c','m','y','k']
    count = 0
    for calc_name, calc in encalc.items():
        try:
            ml_neb_traj = perform_NEB(initial_prime,final_prime,encalc,calc_name,
                                  folder=f'ML_NEB',
                                  N_images=nimages,climb=False,fire=True,fmax=0.05)
        except:
            print(f'Failed for {calc_name}')
            continue
        # Save the final trajectory
        for a in ml_neb_traj:
            a = copy_with_properties(a)
            save_to_traj(a, f'{calc_name}_neb_final_traj.traj')
        
        ml_nebtools = NEBTools(ml_neb_traj) # images including initial and final
        ml_Ef, ml_dE = ml_nebtools.get_barrier(fit=False) #fit = false means that you take the max energy for the images. fit True uses the interpolation
        ml_max_force = ml_nebtools.get_fmax(allow_shared_calculator=True)
        ml_Er = ml_Ef - ml_dE
        ml_E_initial, ml_E_final = ml_neb_traj[0].get_potential_energy(),ml_neb_traj[-1].get_potential_energy()
        ml_Ebarrier = ml_Ef+ml_E_initial - (ml_E_initial+ml_E_final)/2
        plot_band(ml_nebtools,ax,colors[count],calc_name);
        ax.legend(loc=0,prop = {'size':14},frameon=False)
        textstr = f'DFT vs {calc_name} \n'+ \
        r'$dE_\mathrm{{initial-final}}=${:.3f} eV'.format(np.abs(dft_E_initial-dft_E_final) - np.abs(ml_E_initial-ml_E_final) ) +'\n '+ \
        r'$dE_\mathrm{{kinetic}}=${:.3f} eV'.format(np.abs(dft_Ebarrier-ml_Ebarrier))#+ ' \n'+ \
        ax.text(1.05, 0.98-0.3*count, textstr, transform=ax.transAxes, fontsize=14,  
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        count += 1
            
        plt.savefig('NEB.png',dpi=300,bbox_inches='tight')

    # Save Parameters
    params = {'nimages':nimages,'climb':False,'fire':True,'fmax':0.05,'max_iter':1000,
            'DFT_path':os.path.join(root_dir,'NEB',folder_name,'neb_last.traj')}
    # save the parameters as json file

    with open('params.json', 'w') as fp:
        json.dump(params, fp)