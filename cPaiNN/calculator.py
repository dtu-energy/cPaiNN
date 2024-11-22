from ase.calculators.calculator import Calculator, all_changes
from cPaiNN.data import AseDataReader
import numpy as np
from ase import Atoms
import torch
from typing import List

class MLCalculator(Calculator):
    """
    Calculator class for PyTorch models.

    Args:
        model (torch.nn.Module): PyTorch model
        energy_scale (float): energy scaling factor
        forces_scale (float): forces scaling factor
        charge_scale (float): charge scaling factor
        stress_scale (float): stress scaling factor
    """
    implemented_properties = ["energy", "forces","stress","magmoms","bader_charge"]

    def __init__(
        self,
        model: torch.nn.Module,
        energy_scale:float=1.0,
        forces_scale:float=1.0,
        charge_scale:float=1.0,
        stress_scale:float=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.model = model
        self.model_device = next(model.parameters()).device
        self.cutoff = model.cutoff
        self.ase_data_reader = AseDataReader(self.cutoff)
        self.energy_scale = energy_scale
        self.forces_scale = forces_scale
        self.charge_scale = charge_scale
        self.stress_scale = stress_scale

    def calculate(self, atoms:Atoms=None, properties:List[str]=["energy"], system_changes:List[str]=all_changes) -> None:
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)

        if atoms is not None:
            self.atoms = atoms.copy()       
        
        model_inputs = self.ase_data_reader(self.atoms)
        
        model_inputs = {
            k: v.to(self.model_device) for (k, v) in model_inputs.items()
        }

        model_results = self.model(model_inputs)
        results = {}

        # Convert outputs to calculator format
        results["energy"] = (
            model_results["energy"][0].detach().cpu().numpy().item()
            * self.energy_scale
        )
        try:
            results["forces"] = (
            model_results["forces"].detach().cpu().numpy() * self.forces_scale
            )
        except KeyError:
            pass    
        
        try:
            results["stress"] = (
                model_results["stress"].detach().cpu().numpy()[0] * self.stress_scale
            )
        except KeyError:
            pass

        try:
            results["magmoms"] = (
                model_results["magmom"].detach().cpu().numpy() * self.charge_scale
            )
        except KeyError:
            pass

        try:
            results["bader_charge"] = (
                model_results["bader_charge"].detach().cpu().numpy()* self.charge_scale
            )
        except KeyError:
            pass

        if model_results.get("fps"):
            atoms.info["fps"] = model_results["fps"].detach().cpu().numpy()
    
        self.results = results

class EnsembleCalculator(Calculator):
    """
    Ensemble calculator class for PyTorch models.

    Args:
        models (list of torch.nn.Module): List of PyTorch models
        energy_scale (float): energy scaling factor
        forces_scale (float): forces scaling factor
        charge_scale (float): charge scaling factor
        stress_scale (float): stress scaling factor
    
    Returns:
        Calculator: ASE calculator
    """
    implemented_properties = ["energy", "forces","stress","magmoms","bader_charge"]

    def __init__(
        self,
        models: List[torch.nn.Module],
        energy_scale:float=1.0,
        forces_scale:float=1.0,
        charge_scale:float=1.0,
        stress_scale:float=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.models = models
        self.model_device = next(models[0].parameters()).device
        self.cutoff = models[0].cutoff
        self.ase_data_reader = AseDataReader(self.cutoff)
        self.energy_scale = energy_scale
        self.forces_scale = forces_scale
        self.charge_scale = charge_scale
        self.stress_scale = stress_scale

    def calculate(self, atoms:Atoms=None, properties:List[str]=["energy"], system_changes:List[str]=all_changes) -> None:
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)
        if atoms is not None:
            self.atoms = atoms.copy()       

        model_inputs = self.ase_data_reader(self.atoms)
        model_inputs = {
            k: v.to(self.model_device) for (k, v) in model_inputs.items()
        }

        predictions = {'energy': [], 'forces': [],'stress':[], 'magmom': [], 'bader_charge': []}
        
        for model in self.models:
            model_results = model(model_inputs)
    
            predictions['energy'].append(model_results["energy"][0].detach().cpu().numpy().item() * self.energy_scale)
    
            try:
                predictions['forces'].append(model_results["forces"].detach().cpu().numpy() * self.forces_scale)
            except KeyError:
                pass
            
            try:
                predictions['stress'].append(model_results["stress"].detach().cpu().numpy()[0] * self.stress_scale)
            except KeyError:
                pass

            try:
                predictions['magmom'].append(model_results["magmom"].detach().cpu().numpy() * self.charge_scale)
            except KeyError:
                pass

            try:
                predictions['bader_charge'].append(model_results["bader_charge"].detach().cpu().numpy() * self.charge_scale)
            except KeyError:
                pass

        results = {"energy": np.mean(predictions['energy'])}
        ensemble = {'energy_var': np.var(predictions['energy'])}

        try:
            results["forces"] = np.mean(np.stack(predictions['forces']), axis=0)
            ensemble['forces_var'] = np.var(np.stack(predictions['forces']), axis=0)
            ensemble['forces_l2_var'] = np.var(np.linalg.norm(predictions['forces'], axis=2), axis=0)
        except ValueError:
            pass

        try:
            results["stress"] = np.mean(np.stack(predictions['stress']), axis=0)
            ensemble['stress_var'] = np.var(np.stack(predictions['stress']), axis=0)
        except ValueError:
            pass
    
        try:
            results["magmoms"] = np.mean(np.stack(predictions['magmom']), axis=0)
            ensemble['magmom_var'] = np.var(np.stack(predictions['magmom']), axis=0)
        except ValueError:
            pass
        try:
            results["bader_charge"] = np.mean(np.stack(predictions['bader_charge']), axis=0)
            ensemble['bader_charge_var'] = np.var(np.stack(predictions['bader_charge']), axis=0)
        except ValueError:
            pass
        
        results['ensemble'] = ensemble

        self.results = results
