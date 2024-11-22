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
from chemical_values import*

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="General Active Learning", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--db_dir",
        type=str,
        help="Path to the data base",
    )
    parser.add_argument(
        "--db_ids", 
        type=int, 
        help="Id with chemical formula in the database for this particular structure",
    )
    parser.add_argument(
        "--M_ion", 
        type=str, 
        help="Metalic ion for this particular structure",
    )
    parser.add_argument(
        "--Job_name",
        type=str,
        help="Name of the releaxation to run",
    )
    parser.add_argument(
        "--oszicar",
        type=bool,
        default=True,
        help="Save all the oszicar file",
    )
    parser.add_argument(
        "--procar",
        type=bool,
        default=False,
        help="Save all the procar file",
    )
    parser.add_argument(
        "--chgcar",
        type=bool,
        default=False,
        help="Save all the chgcar file",
    )
    parser.add_argument(
        "--outcar",
        type=bool,
        default=False,
        help="Save all the outcar file",
    )
    parser.add_argument(
        "--max_unconverged",
        type=int,
        default=50,
        help="Maximuk number of unconverged steps in MD before it stops",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        help="Directory of the config file",
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

    # Vasp calculator
    vasp_params = params['VASP']
    tot_magmom = 0
    for a in atom:
        if a == 'Fe':
            a.magmom = 5
        if a == 'Co':
            a.magmom = 4
        print(a.symbol, a.magmom)
        tot_magmom += a.magmom
    if 'Fe' in atom.get_chemical_symbols():
        tot_magmom = 20
    elif 'Co' in atom.get_chemical_symbols():
        tot_magmom = 16
    vasp_params['nupdown'] = tot_magmom
    vasp_params['nsw'] = 0
    vasp_params['istart'] = 1 
    
    print(f'{name} has nupdown {tot_magmom}')

    calc = Vasp(**vasp_params)
    ldau_luj = {'ldau_luj':{}}
    if type(M_ion)==str:
        ldau_luj['ldau_luj'][M_ion] = {'L': 2, 'U': get_U_value(M_ion),'J':0}
        print(f'{name} has L, U, J values: (2, {get_U_value(M_ion)}, 0)')
    else:
        for m in M_ion:
             ldau_luj['ldau_luj'][m] = {'L': 2, 'U': get_U_value(m),'J':0}
             print(f'{name} has L, U, J values: (2, {get_U_value(m)}, 0)')
    calc.set(**ldau_luj)
    
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
    if vasp_params['nelm']:
        nelm = vasp_params['nelm']
    else:
        nelm = 60 # vasp default value
    saver = MD_Saver(md, calc,nelm=nelm, max_unconverged=args.max_unconverged,root_dir=relaxsim_directory,backup_name="", 
            oszicar=args.oszicar,procar=args.procar,chgcar=args.chgcar,outcar=args.outcar)
    md.attach(saver, interval=params_md['dump_step'])

    # Start MD simulation
    md.run(params_md['max_step']) # Number of steps we want the simulation to run for 

if __name__ == "__main__":
    main()