import numpy as np
import ase
from ase.visualize import view
from ase.md import Langevin
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.io import read
from ase import units 
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.calculators.vasp import Vasp
import numpy as np 
import weakref
import shutil
import os
import toml
from ase.db import connect
import argparse
from ase.calculators.calculator import CalculationFailed

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="General Active Learning", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the machine learning potential",
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        help="Path to the personal machine learning potential",
        default=None,
    )
    return parser.parse_args(arg_list)

def update_namespace(ns, d):
    for k, v in d.items():
        if not isinstance(v, dict):
            ns.__dict__[k] = v

def main():
    # Load parameters
    args = get_arguments()
    with open(args.cfg, 'r') as f:
        params = toml.load(f)
    update_namespace(args, params['MD'][args.Job_name])
    params_md = params['MD_params']

    # Import arguments
    M_ion = args.M_ion
    db_dir = args.db_dir
    name = str(args.Job_name)

    # Setting the particlur structure for this relaxation
    db = connect(db_dir)
    row = db.get(name = name)
    atom = row.toatoms()

    # setting and creating the directory for the saved files
    root_dir = params['root_dir']
    atom_name = name.split('_')[-1]
    #relax_directory = f'{root_dir}/md_sim'
    relaxsim_directory =os.getcwd()

    # Set th VASP calcualtor
    atom.set_calculator(calc)

    # Set the momenta corresponding to T=1000K
    T = params_md['temperature'] # The temperature of the MD simulation
    T0 = str(T)
    f = params_md['friction_term'] # Frictional term in the Langevin equation

    MaxwellBoltzmannDistribution(atom, temperature_K=T)

    mdsim_name_log= 'md_'+T0+'K_'+name+'.log'
    mdsim_name_traj= 'MD.traj'#'md_'+T0+'K_'+name+'.traj'
    
    md = Langevin(atom, params_md['time_step'] * units.fs,
              temperature_K=T,
              friction=f,
              logfile=relaxsim_directory + "/" + mdsim_name_log)

    traj = Trajectory(relaxsim_directory + "/" + mdsim_name_traj,
	    "w",atom)#properties=["magmoms","energy"])
   
   # Set and attach logger to save MD log file
    md.attach(traj.write, interval=params_md['dump_step']) 
    

    # Set and attach MD_saver to save vasp output files each step in MD. It is also used to limit the MD simulation
    md.attach(saver, interval=params_md['dump_step'])

    # Start MD simulation
    md.run(params_md['max_step']) # Number of steps we want the simulation to run for 

if __name__ == "__main__":
    main()