from ase.md import Langevin
from ase.io.trajectory import Trajectory
from ase.io import read
from ase import units 
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import os
from cPaiNN.relax import ML_Relaxer
import torch


# MD parameters
params_md = {'temperature': 1000, # The temperature of the MD simulation
             'friction_term': 0.02, # Frictional term in the Langevin equation
             'time_step': 1, # Time step for the MD simulation
             'dump_step': 100, # Step interval for saving the trajectory
             'max_step': 10000} # Number of steps we want the simulation to run for

# Set class for ML calculator
device_global = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import arguments
model_name = 'cpainn'
model_path = '/home/energy/mahpe/Playground/Universal_FF/ML_models_stress/Polyanion_bader_magmom_stress_512_4' 
system_path = 'Relax_examples/Na2FeMnNiSiO4/OUTCAR'
system_name = 'Na2FeMnNiSiO4'

# setting and creating the directory for the saved files
root_dir = '.'
relaxsim_directory = f'{root_dir}/md_sim/{system_name}'
if not os.path.exists(relaxsim_directory):
    os.makedirs(relaxsim_directory)

# Read the initial structure from the OUTCAR file
print('Reading the initial structure')
atom = read(system_path, index=-1)

print('Setting up the MD simulation')
# Define the calculator
ML_class = ML_Relaxer(calc_name=model_name,calc_paths=model_path,device=device_global)
calc = ML_class.calculator

# Set th VASP calcualtor
atom.set_calculator(calc)

# Set the momenta corresponding to T=1000K
T = params_md['temperature'] # The temperature of the MD simulation
T0 = str(T)
f = params_md['friction_term'] # Frictional term in the Langevin equation

MaxwellBoltzmannDistribution(atom, temperature_K=T)

mdsim_name_log= 'md_'+T0+'K_'+system_name+'.log'
mdsim_name_traj= 'MD.traj'#'md_'+T0+'K_'+name+'.traj'

md = Langevin(atom, params_md['time_step'] * units.fs,
            temperature_K=T,
            friction=f,
            logfile=relaxsim_directory + "/" + mdsim_name_log)

traj = Trajectory(relaxsim_directory + "/" + mdsim_name_traj,
    "w",atom)

# Set and attach logger to save MD log file
md.attach(traj.write, interval=params_md['dump_step']) 

# Start MD simulation
print('Starting MD simulation')
md.run(params_md['max_step']) # Number of steps we want the simulation to run for 
