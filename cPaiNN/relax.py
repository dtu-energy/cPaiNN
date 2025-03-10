from ase.constraints import ExpCellFilter
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
        optimizer: Optimizer | str = "FIRE",
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
            atoms = ExpCellFilter(atoms)
        optimizer = self.opt_class(atoms,trajectory=traj_file,logfile=log_file,**kwargs)
        optimizer.run(fmax=fmax, steps=steps)
        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms

        if 'cpainn' in self.calc_name:
            atoms = self.predict(atoms)

        return {
            "final_structure": atoms,
        }
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
        elif self.calc_name == 'mace_large':
            from mace.calculators import mace_mp
            print('Using Mace-MP-0 large model')
            calc = mace_mp(model="large", dispersion=False, default_dtype="float64", device=self.device)
        elif self.calc_name == 'mace_medium':
            from mace.calculators import mace_mp
            print('Using Mace-MP-0 medium model')
            calc = mace_mp(model="medium", dispersion=False, default_dtype="float64", device=self.device)
        elif self.calc_name == 'mace_small':
            from mace.calculators import mace_mp
            print('Using Mace-MP-0 small model')
            calc = mace_mp(model="small", dispersion=False, default_dtype="float64", device=self.device)
        elif self.calc_name == 'mace_model':
            from mace.calculators import MACECalculator
            print('Using Mace personal model')
            calc =  MACECalculator(model_paths=self.calc_paths,device=self.device, default_dtype="float64")
        
        elif self.calc_name == 'm3gnet':
            from m3gnet.models import Potential, M3GNet, M3GNetCalculator
            potential = Potential(M3GNet.load())
            print('Using M3GNet model')
            calc = M3GNetCalculator(potential=potential, stress_weight=0.01)
        elif self.calc_name == 'mace_omat':
            from mace.calculators import mace_mp
            calc = mace_mp(model="medium-omat-0", dispersion=False, default_dtype="float64",device=self.device)
        else:
            raise RuntimeError('Calculator not found!')
        return calc
