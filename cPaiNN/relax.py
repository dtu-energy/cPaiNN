from ase.filters import ExpCellFilter, FrechetCellFilter
from ase.optimize.bfgs import BFGS
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.optimize.fire import FIRE
from ase.optimize.lbfgs import LBFGS, LBFGSLineSearch
from ase.optimize.mdmin import MDMin
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase import Atoms, units
from ase.calculators.calculator import Calculator

from ase.optimize.optimize import Optimizer
from pathlib import Path
from ase.neb import NEBTools,NEB
from ase.io import write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase import units
import numpy as np
import logging

OPTIMIZERS = {
    "FIRE": FIRE,
    "BFGS": BFGS,
    "LBFGS": LBFGS,
    "LBFGSLineSearch": LBFGSLineSearch,
    "MDMin": MDMin,
    "SciPyFminCG": SciPyFminCG,
    "SciPyFminBFGS": SciPyFminBFGS,
    "BFGSLineSearch": BFGSLineSearch,
}

class ML_Relaxer:
    """ML_Relaxer is a class for structural relaxation."""
    
    def __init__(
        self,
        calc_name: str | str = "mace_large",
        calc_paths: str | None = None,
        optimizer: Optimizer | str = "LBFGSLineSearch",
        device: str = "cuda",
        relax_cell: bool = True,
    ):
        """
        Args:
            calc_name (str): calculator name. Defaults to "mace_large".
            calc_paths (str): path to the calculator. Defaults to None.
            optimizer (str or ase Optimizer): the optimization algorithm. Defaults to "FIRE".
            device (str): device to use. Defaults to "cuda".
            relax_cell (bool): whether to relax the lattice cell. Defaults to True.
        """
        if isinstance(optimizer, str):
            optimizer_obj = OPTIMIZERS.get(optimizer, None)
        elif optimizer is None:
            raise ValueError("Optimizer cannot be None")
        else:
            optimizer_obj = optimizer
        
        self.opt_class: Optimizer = optimizer_obj
        self.calc_name = calc_name
        self.calc_paths = calc_paths
        self.ensemble = False # False unless using ensemble of cPaiNN models
        self.device = device
        self.calculator= self.get_calc()    
        self.relax_cell = relax_cell
    
    def predict(self, atoms: Atoms):
        """Predict the energy and forces of an Atoms object.
        
        Args:
            atoms (Atoms): the input Atoms object
        
        Returns:
            atoms (Atoms): the Atoms object with calculator set
        """
        atom_ml = atoms.copy()
        atom_ml.set_calculator(self.calculator)
        energy = atom_ml.get_potential_energy()
        forces = atom_ml.get_forces()
        return atom_ml

    def relax(
        self,
        atoms: Atoms,
        fmax: float = 0.1,
        steps: int = 500,
        traj_file: str | None = None,
        log_file: str = "opt.log",
        cell_relaxer='FrechetCellFilter',
        interval=1,
        verbose=False,
        **kwargs,
    ):
        """
        Relax an input Atoms.

        Args:
            atoms (Atoms): the atoms for relaxation
            fmax (float): total force tolerance for relaxation convergence. Defaults to 0.1.
            Here fmax is a sum of force and stress forces
            steps (int): max number of steps for relaxation. Defaults to 500.
            traj_file (str): the trajectory file for saving
            log_file (str): the log file for saving. Defaults to "opt.log"
            interval (int): the step interval for saving the trajectories. Defaults to 1.

            verbose (bool): Whether to have verbose output.
            kwargs: Kwargs pass-through to optimizer.
        """
        # Set the calculator
        atoms.set_calculator(self.calculator)
        if self.relax_cell:
            if cell_relaxer == 'ExpCellFilter':
                atoms = ExpCellFilter(atoms)
            elif cell_relaxer == 'FrechetCellFilter':
                atoms = FrechetCellFilter(atoms)
            else:
                raise ValueError(f"Unknown cell relaxer: {cell_relaxer}")
        optimizer = self.opt_class(atoms,trajectory=traj_file,logfile=log_file,**kwargs)
        optimizer.run(fmax=fmax, steps=steps)
        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms
        if isinstance(atoms, FrechetCellFilter):
            atoms = atoms.atoms
        if 'cpainn' in self.calc_name:
            atoms = self.predict(atoms)

        return {
            "final_structure": atoms,
        }

    def NEB(self,
            initial_image: Atoms,
            final_image: Atoms,
            relax_source=True, 
            fmax: float = 0.1,
            steps: int = 500,
            N_images: int = 5,
            climb: bool = False,
            traj_file: str = 'neb.traj',
            log_file: str = "NEB.log",
        ):

        if relax_source:
            print('Relaxing the initial and final images before NEB')
            # relax source
            initial_image = self.relax(initial_image, fmax=fmax, steps=steps,log_file='Initial_image.log')['final_structure']
            final_image = self.relax(final_image, fmax=fmax, steps=steps,log_file='Final_image.log')['final_structure']

        # Make a band consisting of N images
        images = [initial_image]
        images += [initial_image.copy() for i in range(N_images)]
        images += [final_image]

        neb = NEB(images,allow_shared_calculator=True, climb=climb)
        neb.interpolate(mic=True)

        # Set up the calculators
        for i, image in enumerate(images):
            image.calc = self.calculator
            image.get_potential_energy()
        
        print('Running NEB calculation')
        optimizer = self.opt_class(neb,trajectory=traj_file,logfile=log_file)
        optimizer.run(fmax=fmax, steps=steps)
        print('NEB calculation done')
        print('---------------------------------')
        return neb.images
    

    def MD(self,
           atoms: Atoms,
           relax_source: bool = True,
           temp: float = 300,
           friction: float = 0.02,
           time_step: float = 1, #fs
           dump_step: float = 1000, #1000*fs = 1ps
           max_steps: int = 1000000, #1000000*fs = 1ns
           print_step: float = 1000, #1000*fs = 1ps
           traj_file: str = "md.xyz",
           log_file: str = "MD.log",
           log_relax_file: str = "MD_relax.log",
           fmax: float = 0.1,
           steps: int = 500,
           **kwargs,
           ):
        if relax_source:
            atoms = self.relax(atoms, fmax=fmax, steps=steps, log_file=log_relax_file)['final_structure']

        # Set calculator 
        atoms.calc = self.calculator
        atoms.get_potential_energy()

        # Initialize velocities:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temp)

        dyn = Langevin(atoms, time_step * units.fs,
            temperature_K=temp,
            friction=friction,
            logfile=log_file)
        
        class CallsCounter:
            def __init__(self, func):
                self.calls = 0
                self.func = func
            def __call__(self, *args, **kwargs):
                self.calls += 1
                self.func(*args, **kwargs)
        # set logger
        logger = logging.getLogger(__file__)
        logger.setLevel(logging.DEBUG)

        runHandler = logging.FileHandler(log_file, mode='w')
        runHandler.setLevel(logging.DEBUG)
        runHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
        logger.addHandler(runHandler)
        logger.addHandler(logging.StreamHandler())
        logger.info = CallsCounter(logger.info)



        @CallsCounter
        def printenergy(a=atoms,ensemble=self.ensemble):  # store a reference to atoms in the definition.
            """Function to print the potential, kinetic and total energy."""
            # Calculate energy and temperature
            epot = a.get_potential_energy()
            ekin = a.get_kinetic_energy()
            temp = ekin / (1.5 * units.kB) / a.get_global_number_of_atoms()

            # Calculate ensemble properties
            if ensemble:
                ensemble = a.calc.results['ensemble'].copy()
                ensemble['forces_var_mean'] = np.mean(ensemble['forces_var'])
                ensemble['forces_sd'] = np.mean(np.sqrt(ensemble['forces_var']))
                ensemble['forces_l2_var'] = np.mean(ensemble['forces_l2_var'])

                # Format ensemble for logging
                ensemble_formatted = ", ".join(
                            ["{}={:.5f}".format(k, np.mean(v)) for (k, v) in ensemble.items()]
                        )
            else:
                ensemble_formatted = ""
            logger.info("Steps={:10d} Epot={:12.3f} Ekin={:12.3f} temperature={:8.2f} {} ".format(
                    printenergy.calls * print_step,
                    epot,
                    ekin,
                    temp,
                    ensemble_formatted,
                ))
            return
        
        dyn.attach(printenergy, interval=print_step)

        #traj = Trajectory(os.path.join(run_dir,'MD.traj'), 'w', atoms)

        def write_xyz(a=atoms,traj_name=traj_file):
                if 'bader_charge' in a.arrays:
                    a.arrays['bader_charge'] = a.calc.results['bader_charge']
                write(traj_name, a, append=True)

        dyn.attach(write_xyz, interval=dump_step)

        print('Starting MD simulation')
        dyn.run(max_steps)
        return


    def get_calc(self):
        """ Get calculator from the given name
        
        Args:
            calc_name (str): calculator name
            calc_paths (str): path to the calculator
            device (str): device to use
            
        Returns:
            calc (ase.calculators.calculator.Calculator): calculator object
        """
        if self.calc_name == 'cpainn':
            from cPaiNN.model import PainnModel
            from cPaiNN.calculator import MLCalculator, EnsembleCalculator
            import torch
            model_pth = Path(self.calc_paths).rglob('*best_model.pth')
            print(self.calc_paths)
            models = []
            for each in model_pth:
                state_dict = torch.load(each, map_location=torch.device(self.device)) 
                model = PainnModel(
                    num_interactions=state_dict["num_layer"], 
                    hidden_state_size=state_dict["node_size"], 
                    cutoff=state_dict["cutoff"],
                    compute_forces=state_dict["compute_forces"],
                    compute_stress=state_dict["compute_stress"],
                    compute_magmom=state_dict["compute_magmom"],
                    compute_bader_charge=state_dict["compute_bader_charge"],
                    )
                model.to(self.device)
                model.load_state_dict(state_dict["model"],)    
                models.append(model)
            if len(models)==1:
                print('Using single cPaiNN model')
                self.ensemble = False
                calc = MLCalculator(models[0])
            elif len(models)>1:
                print('Using ensemble of cPaiNN models')
                self.ensemble = True
                calc = EnsembleCalculator(models)
            else:
                raise ValueError('No model found')
        elif self.calc_name == 'chgnet':
            from chgnet.model.dynamics import CHGNetCalculator
            from chgnet.model import CHGNet
            print('Using CHGNet model')
            ensemble = False
            model = CHGNet.load()
            calc = CHGNetCalculator(model=model,use_device=self.device)
        elif self.calc_name == 'chgnet_model':
            from chgnet.model.dynamics import CHGNetCalculator
            from chgnet.model import CHGNet
            print('Using CHGNet model')
            ensemble = False
            model = CHGNet.from_file(self.calc_paths)
            calc = CHGNetCalculator(model=model,use_device=self.device)
        elif self.calc_name == 'mace_large':
            from mace.calculators import mace_mp
            print('Using Mace-MP-0 large model')
            try:
                calc = mace_mp(model="large", dispersion=False, default_dtype="float64", device=self.device, enable_cueq=True)
                print('Using Mace with cueq')
            except:
                calc = mace_mp(model="large", dispersion=False, default_dtype="float64", device=self.device, enable_cueq=False)
                print('Using Mace without cueq')
        elif self.calc_name == 'mace_medium':
            from mace.calculators import mace_mp
            print('Using Mace-MP-0 medium model')
            try:
                calc = mace_mp(model="medium", dispersion=False, default_dtype="float64", device=self.device, enable_cueq=True)
                print('Using Mace with cueq')
            except:
                calc = mace_mp(model="medium", dispersion=False, default_dtype="float64", device=self.device, enable_cueq=False)
                print('Using Mace without cueq')
        elif self.calc_name == 'mace_small':
            from mace.calculators import mace_mp
            print('Using Mace-MP-0 small model')
            try:
                calc = mace_mp(model="small", dispersion=False, default_dtype="float64",device=self.device,enable_cueq=True)
                print('Using Mace with cueq')
            except:
                calc =  mace_mp(model="small", dispersion=False, default_dtype="float64",device=self.device,enable_cueq=False)
                print('Using Mace without cueq')
        elif self.calc_name == 'mace_model':
            from mace.calculators import MACECalculator
            print('Using Mace personal model')
            try:
                calc =  MACECalculator(model_paths=self.calc_paths,device=self.device,enable_cueq=True, default_dtype="float64")
                print('Using Mace with cueq')
            except:
                calc =  MACECalculator(model_paths=self.calc_paths,device=self.device,enable_cueq=False, default_dtype="float64")       
                print('Using Mace without cueq')
        elif self.calc_name == 'm3gnet':
            from m3gnet.models import Potential, M3GNet, M3GNetCalculator
            potential = Potential(M3GNet.load())
            print('Using M3GNet model')
            calc = M3GNetCalculator(potential=potential, stress_weight=0.01)
        elif self.calc_name == 'mace_omat':
            from mace.calculators import mace_mp

            try:
                calc = mace_mp(model="medium-omat-0", dispersion=False, default_dtype="float64",device=self.device,enable_cueq=True)
                print('Using Mace with cueq')
            except:
                calc =  mace_mp(model="medium-omat-0", dispersion=False, default_dtype="float64",device=self.device,enable_cueq=False)
                print('Using Mace without cueq')
            calc = mace_mp(model="medium-omat-0", dispersion=False, default_dtype="float64",device=self.device)
        else:
            raise RuntimeError('Calculator not found!')
        return calc
