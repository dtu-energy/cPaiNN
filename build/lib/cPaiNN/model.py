import torch
from torch import nn
from typing import Tuple, List

def sinc_expansion(edge_dist: torch.Tensor, edge_size: int, cutoff: float) -> torch.Tensor:
    """
    calculate sinc radial basis function:
    
    sin(n *pi*d/d_cut)/d
    where d is the distance between two atoms, n is the order of the sinc function, and d_cut is the cutoff distance.

    Args:
        edge_dist (torch.Tensor): distance between two atoms
        edge_size (int): number of radial basis functions
        cutoff (float): cutoff distance
    
    Returns:
        torch.Tensor: radial basis functions
    """
    n = torch.arange(edge_size, device=edge_dist.device) + 1
    return torch.sin(edge_dist.unsqueeze(-1) * n * torch.pi / cutoff) / edge_dist.unsqueeze(-1)

def cosine_cutoff(edge_dist: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Calculate cutoff value based on distance.
    This uses the cosine Behler-Parinello cutoff function:

    f(d) = 0.5*(cos(pi*d/d_cut)+1) for d < d_cut and 0 otherwise
    where d is the distance between two atoms and d_cut is the cutoff distance.

    Args:
        edge_dist (torch.Tensor): distance between two atoms
        cutoff (float): cutoff distance
    
    Returns:
        torch.Tensor: cutoff value
    """

    return torch.where(
        edge_dist < cutoff,
        0.5 * (torch.cos(torch.pi * edge_dist / cutoff) + 1),
        torch.tensor(0.0, device=edge_dist.device, dtype=edge_dist.dtype),
    )

class PainnMessage(nn.Module):
    """PaiNN message passing layer for node and edge states in a graph neural network.
    
    Args: 
        node_size (int): size of node state
        edge_size (int): size of edge state
        cutoff (float): cutoff distance
    
    Returns:
        nn.Module: PaiNN message passing layer
    
    """
    def __init__(self, node_size: int, edge_size: int, cutoff: float) -> None:
        super().__init__()
        
        self.node_size = node_size
        self.edge_size = edge_size
        self.cutoff = cutoff

        # message passing layer for node scalar and vector states
        self.scalar_message_mlp = nn.Sequential(
            nn.Linear(node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )
        # Filter layer
        self.filter_layer = nn.Linear(edge_size, node_size * 3)
        
    def forward(self, node_scalar:torch.tensor, node_vector:torch.tensor,
                edge:torch.tensor, edge_diff:torch.tensor, edge_dist:torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """
        Forward pass of the message passing layer.

        Args:
            node_scalar (torch.tensor): scalar state of nodes
            node_vector (torch.tensor): vector state of nodes
            edge (torch.tensor): edge indices
            edge_diff (torch.tensor): difference between two atoms
            edge_dist (torch.tensor): distance between two atoms

        Returns:
            Tuple[torch.tensor, torch.tensor]: new node scalar and vector states
        """
        # Filter weight
        # remember to use v_j, s_j but not v_i, s_i        
        filter_weight = self.filter_layer(sinc_expansion(edge_dist, self.edge_size, self.cutoff))
        filter_weight = filter_weight * cosine_cutoff(edge_dist, self.cutoff).unsqueeze(-1)
        
        # message passing for node scalar and vector states and filter weight
        scalar_out = self.scalar_message_mlp(node_scalar)        
        filter_out = filter_weight * scalar_out[edge[:, 1]]
        
        # split message layer into gate_state_vector, gate_edge_vector, message_scalar
        gate_state_vector, gate_edge_vector, message_scalar = torch.split(
            filter_out, 
            self.node_size,
            dim = 1,
        )
        
        # Get message vector from node_vector and edge_diff and edge_dist
        # num_pairs * 3 * node_size, num_pairs * node_size
        message_vector =  node_vector[edge[:, 1]] * gate_state_vector.unsqueeze(1) 
        edge_vector = gate_edge_vector.unsqueeze(1) * (edge_diff / edge_dist.unsqueeze(-1)).unsqueeze(-1)
        message_vector = message_vector + edge_vector
        
        # Sum up the message from all neighbors
        residual_scalar = torch.zeros_like(node_scalar)
        residual_vector = torch.zeros_like(node_vector)
        residual_scalar.index_add_(0, edge[:, 0], message_scalar)
        residual_vector.index_add_(0, edge[:, 0], message_vector)
        
        # Update node scalar and vector states
        new_node_scalar = node_scalar + residual_scalar
        new_node_vector = node_vector + residual_vector
        
        return new_node_scalar, new_node_vector

class PainnUpdate(nn.Module):
    """PAINN Update function for updating node states in a graph neural network. 

    Args:
        node_size (int): size of node state

    Returns:
        nn.Module: PaiNN update function
    
    """
    def __init__(self, node_size: int) -> None:
        super().__init__()
        
        # Update layer for node scalar and vector states
        self.update_U = nn.Linear(node_size, node_size)
        self.update_V = nn.Linear(node_size, node_size)
        
        # MLP for updating node scalar and vector states
        self.update_mlp = nn.Sequential(
            nn.Linear(node_size * 2, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )
        
    def forward(self, node_scalar:torch.tensor, node_vector:torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """
        Forward pass of the update function.

        Args:
            node_scalar (torch.tensor): scalar state of nodes
            node_vector (torch.tensor): vector state of nodes

        Returns:
            Tuple[torch.tensor, torch.tensor]: new node scalar and vector states
        """
        # Update node scalar and vector states
        Uv = self.update_U(node_vector)
        Vv = self.update_V(node_vector)
        
        # MLP for updating node scalar and vector states
        Vv_norm = torch.linalg.norm(Vv, dim=1)
        mlp_input = torch.cat((Vv_norm, node_scalar), dim=1)
        mlp_output = self.update_mlp(mlp_input)

        # Split the output of the MLP into delta_s and delta_v
        a_vv, a_sv, a_ss = torch.split(
            mlp_output,                                        
            node_vector.shape[-1],                                       
            dim = 1,
        )
        
        delta_v = a_vv.unsqueeze(1) * Uv
        inner_prod = torch.sum(Uv * Vv, dim=1)
        delta_s = a_sv * inner_prod + a_ss
        
        return node_scalar + delta_s, node_vector + delta_v

class PainnModel(nn.Module):
    """
       Polarizable atom interaction neural network (PaiNN) model without edge updating.

    PaiNN employs rotationally equivariant atomwise representations.

    Resources:
        PaiNN: https://arxiv.org/abs/2102.03150

    Args:
        num_interactions (int): number of interaction in the message parsing neural network
        hidden_state_size (int): size of the hidden layer in the message parsing neural network
        cutoff (int): Cutoff range from an atom in Ångstrom
        normalization (bool): If the energy should be normalize in the network or not
        target_mean (List(float)): Energy mean of the training dataset, used in normalization
        target_std (List(float)): Energy std of the training dataset, used in normalization
        atomwise_normalization (bool): If the Energy normalization is per atom or not
        compute_forces (bool): If the forces should be computed
        compute_stress (bool): If the stress should be computed
        compute_magmom (bool): If the magnetic moment should be computed
        compute_bader_charge (bool): If the Bader charge should be computed

        Return:
            nn.Module: PaiNN model
    """
    def __init__(
        self, 
        num_interactions:int, 
        hidden_state_size:int, 
        cutoff:float,
        normalization:bool=True,
        target_mean:List[float]=[0.0],
        target_stddev:List[float]=[1.0],
        atomwise_normalization:bool=True, 
        compute_forces:bool=False,
        compute_stress:bool=False,
        compute_magmom:bool=False,
        compute_bader_charge:bool=True,
        
        **kwargs,
    ):
        super().__init__()
        
        num_embedding = 119   # number of all elements
        self.cutoff = cutoff
        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.edge_embedding_size = 20
        self.compute_forces = compute_forces
        self.compute_stress = compute_stress
        self.compute_magmom = compute_magmom
        self.compute_bader_charge = compute_bader_charge
        
        # Setup atom embeddings
        self.atom_embedding = nn.Embedding(num_embedding, hidden_state_size)

        # Setup message-passing layers
        self.message_layers = nn.ModuleList(
            [
                PainnMessage(self.hidden_state_size, self.edge_embedding_size, self.cutoff)
                for _ in range(self.num_interactions)
            ]
        )

        # Setup update layers
        self.update_layers = nn.ModuleList(
            [
                PainnUpdate(self.hidden_state_size)
                for _ in range(self.num_interactions)
            ]            
        )
        
        # Setup readout function
        self.readout_mlp = nn.Sequential(
            nn.Linear(self.hidden_state_size, self.hidden_state_size),
            nn.SiLU(),
            nn.Linear(self.hidden_state_size, 1),
        )

        # Normalisation constants
        self.normalization = torch.nn.Parameter(
            torch.tensor(normalization), requires_grad=False
        )
        self.atomwise_normalization = torch.nn.Parameter(
            torch.tensor(atomwise_normalization), requires_grad=False
        )
        self.normalize_stddev = torch.nn.Parameter(
            torch.tensor(target_stddev[0]), requires_grad=False
        )
        self.normalize_mean = torch.nn.Parameter(
            torch.tensor(target_mean[0]), requires_grad=False
        )

        # Linear layer for the charge representation prediction
        if self.compute_magmom and self.compute_bader_charge:
            #self.linear = nn.Linear(self.hidden_state_size, 2)
            self.linear_magmom = nn.Linear(self.hidden_state_size, 1)
            self.linear_bader_charge = nn.Linear(self.hidden_state_size, 1)
        elif self.compute_magmom or self.compute_bader_charge:
            self.linear = nn.Linear(self.hidden_state_size, 1)
        
    def forward(self, input_dict:dict) -> dict:
        """
        Forward pass of the PaiNN model.

        Args:
            input_dict (dict): input dictionary containing the following keys:
                elems (torch.tensor): atomic numbers of the atoms
                coord (torch.tensor): atomic coordinates
                cell (torch.tensor): cell vectors
                pairs (torch.tensor): pair indices
                n_diff (torch.tensor): difference between two atoms
                num_atoms (torch.tensor): number of atoms in the structure
                num_pairs (torch.tensor): number of pairs in the structure
            

        Returns:
            dict: output dictionary containing the following
                energy (torch.tensor): energy of the structure
                forces (torch.tensor): forces of the structure, if compute_forces is True
                stress (torch.tensor): stress of the structure, if compute_stress is True
                magmom (torch.tensor): magnetic moment of the structure, if compute_magmom is True
                bader_charge (torch.tensor): Bader charge of the structure, if compute_bader_charge is True
        """
        # Set up the output dictionary
        result_dict = {}

        # store the computrational graph
    
        num_atoms = input_dict['num_atoms']
        num_pairs = input_dict['num_pairs']

        # edge offset. Add offset to edges to get indices of pairs in a batch but not a structure
        edge = input_dict['pairs']
        edge_offset = torch.cumsum(
            torch.cat((torch.tensor([0], 
                                    device=num_atoms.device,
                                    dtype=num_atoms.dtype,                                    
                                   ), num_atoms[:-1])),
            dim=0
        )
        edge_offset = torch.repeat_interleave(edge_offset, num_pairs)
        edge = edge + edge_offset.unsqueeze(-1)        

        # edge difference 
        edge_diff = input_dict['n_diff']
        # Enable gradient computation for edge_diff if forces are to be computed
        if self.compute_forces:
            edge_diff.requires_grad_()
        
        # distance between neighboring atoms
        edge_dist = torch.linalg.norm(edge_diff, dim=1)

        # Atom embeddings for the scalar and vector states    
        node_scalar = self.atom_embedding(input_dict['elems'])
        node_vector = torch.zeros((input_dict['coord'].shape[0], 3, self.hidden_state_size),
                                  device=edge_diff.device,
                                  dtype=edge_diff.dtype,
                                 )
        
        # Message passing and update layers
        count = 0 # count the number of interactions
        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            
            # Charge layer
            if count == self.num_interactions-2: # second last layer
                # Linear layer for the charge representation prediction
                if self.compute_magmom and self.compute_bader_charge:
                    #print("Both magnetic moment and Bader charge are computed at the same time")
                    #linear_layer = self.linear(node_scalar)
                    magmom = self.linear_magmom(node_scalar)
                    #magmom = linear_layer[:,0]
                    magmom = magmom.squeeze()
                    #bader_charge = linear_layer[:,1]
                    bader_charge = self.linear_bader_charge(node_scalar)
                    bader_charge = bader_charge.squeeze()
                    result_dict['magmom'] = magmom
                    result_dict['bader_charge'] = bader_charge
                
                elif self.compute_magmom:
                    #print("Computing magnetic moment")
                    magmom = self.linear(node_scalar)
                    magmom = magmom.squeeze()
                    result_dict['magmom'] = magmom
                elif self.compute_bader_charge:
                    #print("Computing Bader charge")
                    bader_charge = self.linear(node_scalar)
                    bader_charge = bader_charge.squeeze()
                    result_dict['bader_charge'] = bader_charge
                else:
                    pass

            node_scalar, node_vector = message_layer(node_scalar, node_vector, edge, edge_diff, edge_dist)
            node_scalar, node_vector = update_layer(node_scalar, node_vector)
            count += 1
        
        # Readout function to get the energy
        node_scalar = self.readout_mlp(node_scalar)
        node_scalar = node_scalar.squeeze()

    	# Sum up the energy of the atoms in the structure
        image_idx = torch.arange(input_dict['num_atoms'].shape[0],
                                 device=edge.device,
                                )
        image_idx = torch.repeat_interleave(image_idx, num_atoms)
        
        energy = torch.zeros_like(input_dict['num_atoms']).float()        
        energy.index_add_(0, image_idx, node_scalar)

        # Apply (de-)normalization
        if self.normalization:
            normalizer = self.normalize_stddev
            energy = normalizer * energy
            mean_shift = self.normalize_mean
            if self.atomwise_normalization:
                mean_shift = input_dict["num_atoms"] * mean_shift
            energy = energy + mean_shift

        # Add energy to the output dictionary
        result_dict['energy'] = energy
        
        # Compute forces
        if self.compute_forces:
            dE_ddiff = torch.autograd.grad(
                energy,
                edge_diff,
                grad_outputs=torch.ones_like(energy),
                retain_graph=True,
                create_graph=True,
            )[0]
            
            # diff = R_j - R_i, so -dE/dR_j = -dE/ddiff, -dE/R_i = dE/ddiff  
            i_forces = torch.zeros_like(input_dict['coord']).index_add(0, edge[:, 0], dE_ddiff)
            j_forces = torch.zeros_like(input_dict['coord']).index_add(0, edge[:, 1], -dE_ddiff)
            forces = i_forces + j_forces
            
            
            result_dict['forces'] = forces

        # Get the derivative of the energy w.r.t. the edge difference
        if not self.compute_forces and self.compute_stress:
            dE_ddiff = torch.autograd.grad(
                energy,
                edge_diff,
                grad_outputs=torch.ones_like(energy),
                retain_graph=True,
                create_graph=True,
            )[0]

        # Compute stress
        if self.compute_stress:
            # Reference: https://en.wikipedia.org/wiki/Virial_stress
            # This method calculates virials by giving pair-wise force components

            atomic_stress = torch.einsum("ij, ik -> ijk", edge_diff, dE_ddiff)
            cell = input_dict['cell'].view(-1, 3, 3)
            volumes = torch.sum(cell[:, 0] * cell[:, 1].cross(cell[:, 2]), dim=1)
            atomic_stress = torch.zeros(
                (forces.shape[0], 3, 3),                                         
                dtype=forces.dtype,
                device=forces.device).index_add(0, edge[:, 0], atomic_stress)
            virial = torch.zeros_like(cell).index_add(0, image_idx, atomic_stress) / 2
            result_dict['stress'] = virial / volumes[:, None, None]
            result_dict['virial'] = virial 
        return result_dict
