from ase.io import read, write, Trajectory
import torch
from typing import List, Union
import asap3
import numpy as np
from scipy.spatial import distance_matrix
from ase import Atoms

class AseDataReader:
    """
    Read ASE atoms object and return a dictionary of tensors

    Args:
    cutoff (float): cutoff distance for neighborlist
    compute_forces (bool): whether to compute forces
    compute_stress (bool): whether to compute stress
    charge_key (str): key for charge data

    Returns:
    atoms_data (dict): dictionary of tensors
        	num_atoms (torch.Tensor): number of atoms in the system
            elems (torch.Tensor): atomic numbers of atoms
            coord (torch.Tensor): atomic coordinates
            cell (torch.Tensor): unit cell
            pairs (torch.Tensor): pair indices
            n_diff (torch.Tensor): difference in coordinates between neighboring atoms
            num_pairs (torch.Tensor): number of pairs
            energy (torch.Tensor): potential energy
            forces (torch.Tensor): atomic forces
            stress (torch.Tensor): stress tensor
            magmom (torch.Tensor): magnetic moments
            bader_charge (torch.Tensor): bader charges

    
    """
    def __init__(self, cutoff:float=5.0,compute_forces:bool=True,compute_stress:bool=True,
                charge_key:Union[str,List[str]]='magmom') -> None:            
        self.cutoff = cutoff
        self.compute_forces = compute_forces
        self.compute_stress = compute_stress
        self.charge_key = charge_key

    def __call__(self, atoms:Atoms) -> dict:
        atoms_data = {
            'num_atoms': torch.tensor([atoms.get_global_number_of_atoms()]),
            'elems': torch.tensor(atoms.numbers),
            'coord': torch.tensor(atoms.positions, dtype=torch.float),
        }
        
        # Get neighborlist
        if atoms.pbc.any():
            pairs, n_diff = self.get_neighborlist(atoms)
            atoms_data['cell'] = torch.tensor(atoms.cell[:], dtype=torch.float)
        else:
            pairs, n_diff = self.get_neighborlist_simple(atoms)
            
        # Add neighborlist to atoms_data
        atoms_data['pairs'] = torch.from_numpy(pairs)
        atoms_data['n_diff'] = torch.from_numpy(n_diff).float()
        atoms_data['num_pairs'] = torch.tensor([pairs.shape[0]])
        
        # Get properties
        # Energy, if there is no calculator it will raise an exception and return atoms_data
        try:
            energy = torch.tensor([atoms.get_potential_energy()], dtype=torch.float)
            atoms_data['energy'] = energy
        except:
            return atoms_data
        
        # Forces
        if self.compute_forces:
            forces = torch.tensor(atoms.get_forces(apply_constraint=False), dtype=torch.float)
            atoms_data['forces'] = forces
        # Stress
        if self.compute_stress:
            stress = torch.tensor(atoms.get_stress(voigt=False), dtype=torch.float)
            atoms_data['stress'] = stress

        # Charges
        if isinstance(self.charge_key, list):
            magmom = torch.tensor(atoms.get_magnetic_moments(), dtype=torch.float)
            atoms_data['magmom'] = magmom
            bader_charge = torch.tensor(atoms.arrays['bader_charge'], dtype=torch.float)
            atoms_data['bader_charge'] = bader_charge
        elif isinstance(self.charge_key, str):
            if self.charge_key == 'magmom':
                magmom = torch.tensor(atoms.get_magnetic_moments(), dtype=torch.float)
                atoms_data['magmom'] = magmom
            if self.charge_key == 'bader_charge':
                bader_charge = torch.tensor(atoms.arrays['bader_charge'], dtype=torch.float)
                atoms_data['bader_charge'] = bader_charge
  
        return atoms_data
            
    
    def get_neighborlist(self, atoms:Atoms)->tuple:
        """
        Get neighborlist using ASAP3 FullNeighborList class

        Args:
            atoms (ASE atoms object): atoms object

        Returns:
            pairs (np.ndarray): pair indices
            n_diff (np.ndarray): difference in coordinates between neighboring atoms    
        """    

        nl = asap3.FullNeighborList(self.cutoff, atoms)
        pair_i_idx = []
        pair_j_idx = []
        n_diff = []
        for i in range(len(atoms)):
            indices, diff, _ = nl.get_neighbors(i)
            pair_i_idx += [i] * len(indices)               # local index of pair i
            pair_j_idx.append(indices)   # local index of pair j
            n_diff.append(diff)

        pair_j_idx = np.concatenate(pair_j_idx)
        pairs = np.stack((pair_i_idx, pair_j_idx), axis=1)
        n_diff = np.concatenate(n_diff)
        
        return pairs, n_diff
    
    def get_neighborlist_simple(self, atoms):
        """
        Get neighborlist using simple distance matrix

        Args:
            atoms (ASE atoms object): atoms object
        
        Returns:
            pairs (np.ndarray): pair indices
            n_diff (np.ndarray): difference in coordinates between neighboring atoms
        """

        pos = atoms.get_positions()
        dist_mat = distance_matrix(pos, pos)
        mask = dist_mat < self.cutoff
        np.fill_diagonal(mask, False)        
        pairs = np.argwhere(mask)
        n_diff = pos[pairs[:, 1]] - pos[pairs[:, 0]]
        
        return pairs, n_diff

class AseDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for ASE atoms objects

    Args:
        ase_db (str or ASE Trajectory): ASE database or trajectory
        cutoff (float): cutoff distance for neighborlist
        compute_forces (bool): whether to compute forces
        compute_stress (bool): whether to compute stress
        charge_key (str or list of str): key for charge data
    
    Returns:
        torch.utils.data.Dataset: PyTorch dataset
    """
    def __init__(self, ase_db:Trajectory, cutoff:float=5.0,
                compute_forces:bool=True,compute_stress:bool=True,
                charge_key:Union[str,List[str]]='magmom', **kwargs)->None:
        super().__init__(**kwargs)
        
        # Load ASE database
        if isinstance(ase_db, str):
            try:
                self.db = Trajectory(ase_db)
            except:
                self.db = read(ase_db,index=':')
        else:
            self.db = ase_db
        
        self.cutoff = cutoff
        self.atoms_reader = AseDataReader(cutoff, compute_forces, compute_stress, charge_key)
        
    def __len__(self):
        return len(self.db)
    
    def __getitem__(self, idx:int)->dict:
        """
        Get dictionary of tensors from ASE atoms object

        Args:
            idx (int): index of ASE atoms object in database
        
        Returns:
            atoms_data (dict): dictionary of tensors
        """
        atoms = self.db[idx]
        atoms_data = self.atoms_reader(atoms)
        return atoms_data

def cat_tensors(tensors: List[torch.Tensor])->torch.Tensor:
    """
    Concatenate list of tensors along first dimension

    Args:
        tensors (list of torch.Tensor): list of tensors
    
    Returns:
        torch.Tensor: concatenated tensor
    """
    if tensors[0].shape:
        return torch.cat(tensors)
    return torch.stack(tensors)

def collate_atomsdata(atoms_data: List[dict], pin_memory=True):
    """
    Collate list of atoms data dictionaries into a dictionary of tensors

    Args:
        atoms_data (list of dict): list of atoms data dictionaries
        pin_memory (bool): whether to pin memory for CUDA tensors
    
    Returns:
        collated (dict): dictionary of tensors
    """
    # convert from list of dicts to dict of lists
    dict_of_lists = {k: [dic[k] for dic in atoms_data] for k in atoms_data[0]}
    if pin_memory:
        pin = lambda x: x.pin_memory()
    else:
        pin = lambda x: x
        
    collated = {k: cat_tensors(v) for k, v in dict_of_lists.items()}
    return collated
