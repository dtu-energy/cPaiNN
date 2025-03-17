from ase.io import read, write, Trajectory, iread
import torch
from typing import List, Union
import numpy as np
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
            'elems': torch.tensor(atoms.numbers, dtype=torch.int),
            'coord': torch.tensor(atoms.positions, dtype=torch.float),
        }
        
        # Get neighborlist
        #if atoms.pbc.any():
            #pairs, n_diff = self.get_neighborlist(atoms)                
        #    pairs, n_diff = self.get_neighborlist_costum(atoms)         
        #    atoms_data['cell'] = torch.tensor(atoms.cell[:], dtype=torch.float)
        #else:
        #    pairs, n_diff = self.get_neighborlist_simple(atoms)

        # Get neighborlist using matscipy, which results the same as ASAP3 and is more or less as fast
        pairs, n_diff = self.get_neighborlist_matscipy(atoms,pbc=atoms.pbc)
        atoms_data['cell'] = torch.tensor(atoms.cell[:], dtype=torch.float)

        # Add neighborlist to atoms_data
        atoms_data['pairs'] = torch.from_numpy(pairs).int()
        atoms_data['n_diff'] = torch.from_numpy(n_diff).float()
        atoms_data['num_pairs'] = torch.tensor([pairs.shape[0]]).int()
        
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
            if 'bader_charge' in atoms.arrays:
                bader_charge = torch.tensor(atoms.arrays['bader_charge'], dtype=torch.float)
                atoms_data['bader_charge'] = bader_charge
            else:
                empty_array = np.zeros(len(atoms))
                empty_array[:] = np.nan
                atoms_data['bader_charge'] = torch.from_numpy(empty_array)
        elif isinstance(self.charge_key, str):
            
            if self.charge_key == 'magmom':
                magmom = torch.tensor(atoms.get_magnetic_moments(), dtype=torch.float)
                atoms_data['magmom'] = magmom
            
            if self.charge_key == 'bader_charge':
                if 'bader_charge' in atoms.arrays:
                    bader_charge = torch.tensor(atoms.arrays['bader_charge'], dtype=torch.float)
                    atoms_data['bader_charge'] = bader_charge
                else:
                    empty_array = np.zeros(len(atoms))
                    empty_array[:] = np.nan
                    atoms_data['bader_charge'] = torch.from_numpy(empty_array)
  
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
        import asap3

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
    

    def get_neighborlist_costum(self, atoms:Atoms)->tuple:
        """
        Get neighborlist using ASAP3 FullNeighborList class.
        Use the given cutoff unless, the cell is too small, then we redefine the cutoff to just below the height of the cell.

        Args:
            atoms (ASE atoms object): atoms object

        Returns:
            pairs (np.ndarray): pair indices
            n_diff (np.ndarray): difference in coordinates between neighboring atoms    
        """    
        import asap3

        try:
            nl = asap3.FullNeighborList(cutoff, atoms)
        except:
            # Define the cell matrix
            cell_matrix = atoms.get_cell().array

            # Compute the volume of the unit cell
            volume = atoms.get_volume()

            # Choose a base for the height calculation           
            base_area_1 = np.linalg.norm(np.cross(cell_matrix[0], cell_matrix[1]))
            base_area_2 = np.linalg.norm(np.cross(cell_matrix[1], cell_matrix[2]))
            base_area_3 = np.linalg.norm(np.cross(cell_matrix[2], cell_matrix[0]))
            base_area = np.max([base_area_1, base_area_2, base_area_3])

            # Set the cutoff to the height
            cutoff = volume / base_area
            cutoff = cutoff - cutoff*0.01
            # Create the neighbor list
            nl = asap3.FullNeighborList(cutoff, atoms)

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
        from scipy.spatial import distance_matrix

        pos = atoms.get_positions()
        dist_mat = distance_matrix(pos, pos)
        mask = dist_mat < self.cutoff
        np.fill_diagonal(mask, False)        
        pairs = np.argwhere(mask)
        n_diff = pos[pairs[:, 1]] - pos[pairs[:, 0]]
        
        return pairs, n_diff
    
    def get_neighborlist_matscipy(self, atoms,pbc):
        """
        Get neighborlist using matScipy neighborlist. Code taken from Mace: https://github.com/ACEsuit/mace

        Args:
            atoms (ASE atoms object): atoms object
            pbc (tuple): periodic boundary conditions
        
        Returns:
            edge_index (np.ndarray): edge indices
            shifts (np.ndarray): shifts
        """
        from matscipy.neighbours import neighbour_list
        # Set pbc if not provided
        if pbc is None:
            pbc = (False, False, False)

        positions = atoms.get_positions()
        cell = atoms.get_cell()

        if cell is None or cell.any() == np.zeros((3, 3)).any():
            cell = np.identity(3, dtype=float)

        assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
        assert cell.shape == (3, 3)

        pbc_x = pbc[0]
        pbc_y = pbc[1]
        pbc_z = pbc[2]
        identity = np.identity(3, dtype=float)
        max_positions = np.max(np.absolute(positions)) + 1
        # Extend cell in non-periodic directions
        # For models with more than 5 layers, the multiplicative constant needs to be increased.
        temp_cell = np.copy(cell)
        if not pbc_x:
            temp_cell[0, :] = max_positions * 5 * self.cutoff * identity[0, :]
        if not pbc_y:
            temp_cell[1, :] = max_positions * 5 * self.cutoff * identity[1, :]
        if not pbc_z:
            temp_cell[2, :] = max_positions * 5 * self.cutoff * identity[2, :]

        sender, receiver,tot_diff, n_diff, unit_shifts = neighbour_list(
            quantities="ijdDS",
            pbc=pbc,
            cell=temp_cell,
            positions=positions,
            cutoff=float(self.cutoff),
            # self_interaction=True,  # we want edges from atom to itself in different periodic images
            # use_scaled_positions=False,  # positions are not scaled positions
        )

        # Build output
        edge_index = np.stack((sender, receiver)).T  # [n_edges,2 ]

        # From the docs: With the shift vector S, the distances D between atoms can be computed from
        # D = positions[j]-positions[i]+S.dot(cell)
        #shifts = np.dot(unit_shifts, cell)  # [n_edges, 3]

        return edge_index, n_diff

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
    def __init__(self, ase_db, cutoff:float=5.0,
                compute_forces:bool=True,compute_stress:bool=True,
                charge_key:Union[str,List[str]]='magmom', **kwargs)->None:
        super().__init__(**kwargs)
        # Load ASE database
        self.ase_db = ase_db
        if isinstance(self.ase_db, str):
            try:
                self.db = Trajectory(self.ase_db )
            except:
                self.db = read(self.ase_db ,index=':')                
        else:
            self.db = ase_db
        
        self.cutoff = cutoff
        self.atoms_reader = AseDataReader(cutoff, compute_forces, compute_stress, charge_key)
        
    def __len__(self):
        import subprocess
        if self.ase_db.endswith('.xyz'):
            # Define the command
            cmd = f"grep -c '^Lattice' {self.ase_db}"

            # Run the command and capture output
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            # Extract the integer value from the output
            total_len = int(result.stdout.strip())

        elif self.ase_db.endswith('.traj'):
            total_len = len(Trajectory(self.ase_db))
        else:
            raise ValueError('File format not supported, please use .traj or .xyz')
    
        return total_len
                
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
# Takes the dict of properties for each atom and returns a total dict containing a concatenated tensor for each property
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
        
    collated = {k: cat_tensors(v) for k, v in dict_of_lists.items() }
    return collated
