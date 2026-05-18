from ase import Atoms
from ase.ga.startgenerator import StartGenerator
import numpy as np

from ase.ga.population import RankFitnessPopulation, Population
from ase.ga.slab_operators import (
    CutSpliceSlabCrossover,
    RandomCompositionMutation,
    RandomSlabPermutation,
    SymmetrySlabPermutation,
)

import random
from pathlib import Path

from ase.build import fcc111
from ase.data import atomic_numbers, reference_states
from ase.ga.data import PrepareDB

from ase.ga import get_raw_score
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.data import DataConnection
from ase.ga.offspring_creator import OperationSelector
from ase.ga.ofp_comparator import OFPComparator
from ase.ga.soft_mutation import SoftMutation
from ase.ga.standardmutations import StrainMutation
from ase.ga.utilities import CellBounds, closest_distances_generator
from ase.io import write, read
from ase.ga import set_raw_score
from cPaiNN.relax import ML_Relaxer
import os
import argparse
import torch

def relax(structure_input,name,fmax=fmax,steps=1000):
    """
    Relax the structure using the ML relaxer and return the final energy.
    """
    # find X atoms
    structure = structure_input.copy()
    # append Na list to reference structure
    structure += reference_structure
    X_index = [atom.index for atom in structure if atom.symbol == 'X']
    del structure[X_index]

    # Relax the structure
    traj_file = f'{name}.xyz' 
    log_file = f'{name}.log'
    if os.path.exists(traj_file):
        os.remove(traj_file)
        os.remove(log_file)
    # Relax the structure and return the final energy
    final_struc = ML_class.relax(structure,fmax=fmax,steps=steps,traj_file=traj_file,log_file=log_file,cell_relaxer='FrechetCellFilter',)
    return final_struc["final_structure"].get_potential_energy()

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="GA generation", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed for reproducibility",
    )

    return parser.parse_args(arg_list)
def main():
    # Load argument Namespace
    args = get_arguments()
    model_dir_mace_scratch = 'large_scratch_compiled.model' 
    
    device_global = 'cuda' if torch.cuda.is_available() else 'cpu'

    fmax = 0.01
    optimizer = 'LBFGSLineSearch'    
    ML_class = ML_Relaxer(calc_name='mace_model',
                          optimizer=optimizer,
                          calc_paths=model_dir_mace_olvine_ft,
                          device=device_global,
                          relax_cell=True)
    
    # Specify the number of generations this script will run
    num_gens = 400 
    pop_size = 200 
    
    # set up the random number generator
    random_seed = args.random_seed
    rng = np.random.RandomState(random_seed)

    # Set up the initial structures and the database for the GA
    root_dir = 'GA_example/relaxations'
    cif_file = 'GA_example/NaFePO4_olivine_new.cif'
    # Define the supercell size, which determines the maximum number of atoms in the structures we will be working with. This is important for setting up the database and ensuring that we have enough space to accommodate all candidates.
    supercell_size = (1,3,2)
    db_name = f'NaFePO4_{supercell_size[0]}{supercell_size[1]}{supercell_size[2]}_{random_seed}_{pop_size}_{fmax}_Mace_olivine_ft.db'
    relax_file_name = f'relaxations/'+db_name.replace('.db','')

    # Load the reference structures, which are the fully desodiated case
    reference_structure = read(cif_file)*supercell_size
    Na_index = [atom.index for atom in reference_structure if atom.symbol == 'Na']
    del reference_structure[Na_index]

    # Load the initial structures for the GA, which are the fully sodiated and desodiated cases. We will relax these structures and use their energies as reference points for calculating the mixing energy of other candidates.
    NaFePO4 = read(cif_file)*supercell_size
    FePO4 = NaFePO4.copy()
    for atom in FePO4:
        if atom.symbol == 'Na':
            atom.symbol = 'X'
    not_Na_index = [atom.index for atom in NaFePO4 if atom.symbol != 'Na']
    del NaFePO4[not_Na_index]
    del FePO4[not_Na_index]

    fu = len([atom for atom in NaFePO4 if atom.symbol == 'Na'])
    NaFePO4_energy = relax(NaFePO4, root_dir+f'/NaFePO4_{random_seed}')/fu
    FePO4_energy = relax(FePO4, root_dir+f'/FePO4_{random_seed}')/fu
    print('NaFePO4 energy:', NaFePO4_energy)
    print('FePO4 energy:', FePO4_energy)

    # Set relative energy
    set_raw_score(NaFePO4,0.0)
    set_raw_score(FePO4,0.0)

    # Set total energy
    NaFePO4.info['key_value_pairs']['E_tot'] = NaFePO4_energy
    FePO4.info['key_value_pairs']['E_tot'] = FePO4_energy
    NaFePO4.info['key_value_pairs']['Na_conc'] = len([atom for atom in NaFePO4 if atom.symbol == 'Na'])/fu
    FePO4.info['key_value_pairs']['Na_conc'] = len([atom for atom in FePO4 if atom.symbol == 'Na'])/fu

    ## Get references
    refs = {
        'NaFePO4': NaFePO4_energy,
        'FePO4': FePO4_energy
    }
    lattice_constants = NaFePO4.get_cell_lengths_and_angles()
    
    metals = ['Na','X']
    print('Supercell size:', supercell_size)
    print('Total Na:',fu)
    print('Max total number of atoms:', len(NaFePO4))
    print('Lattice constants:', lattice_constants)
    print('Population size:', pop_size)
    print('Random seed:', random_seed)


    target = Path(db_name)
    if target.exists():
        target.unlink()
    db = PrepareDB(
        target,
        population_size=pop_size,
        reference_energies=refs,
        metals=metals,
        lattice_constants=lattice_constants,
        formula_unit=fu
    )

    db.add_relaxed_candidate(
            NaFePO4, atoms_string=''.join(NaFePO4.get_chemical_symbols())
        )
    db.add_relaxed_candidate(
            FePO4, atoms_string=''.join(FePO4.get_chemical_symbols())
        )

    # Now we create the rest of the candidates for the initial population
    # random concentrations of X
    np.random.seed(random_seed)
    Na_index = [atom.index for atom in NaFePO4 if atom.symbol == 'Na']
    ran_conc = np.random.random_integers(1,len(Na_index)-1, pop_size)

    for i in range(pop_size):
        new_structure = NaFePO4.copy()
        np.random.seed(i)
        Na_index_i = np.random.choice(Na_index, ran_conc[i], replace=False)
        for index in Na_index_i:
            new_structure[index].symbol = 'X'

        # Add these candidates as unrelaxed, we will relax them later
        atoms_string = ''.join(new_structure.get_chemical_symbols())
        db.add_unrelaxed_candidate(new_structure, atoms_string=atoms_string)

    # Connect to the database containing all candidates
    db = DataConnection(db_name)

    # Retrieve saved parameters
    pop_size = db.get_param('population_size')
    refs = db.get_param('reference_energies')
    fu = db.get_param('formula_unit')
    metals = db.get_param('metals')
    lattice_constants = db.get_param('lattice_constants')

    def get_comp(atoms):
        return atoms.get_chemical_formula()

    def get_mixing_energy(atoms):
        # Calculate the energy

        Na_conc = len([a.index for a in atoms if a.symbol == 'Na'])/fu
        e = relax(atoms,name=relax_file_name)/fu

        # Set total energy
        atoms.info['energy'] = e

        # Calculate relative energy
        e_rel = e - Na_conc * refs['NaFePO4'] - (1 - Na_conc) * refs['FePO4']
        return e_rel 

    oclist = [
        (3, CutSpliceSlabCrossover(rng=rng)),
        (1, RandomSlabPermutation(rng=rng)),
        (1, RandomCompositionMutation(rng=rng)),
        (1, SymmetrySlabPermutation(rng=rng)),
    ]
    operation_selector = OperationSelector(*zip(*oclist),rng=rng)
    pop = RankFitnessPopulation(
        data_connection=db, population_size=pop_size, variable_function=get_comp,
    )



    print('Evaluating initial candidates')
    while db.get_number_of_unrelaxed_candidates() > 0:
        a = db.get_an_unrelaxed_candidate()
        set_raw_score(a, -get_mixing_energy(a))
        a.info['key_value_pairs']['E_tot'] = a.info['energy']
        a.info['key_value_pairs']['Na_conc'] = len([atom for atom in a if atom.symbol == 'Na'])/fu
        db.add_relaxed_step(a)
    pop.update()



    # Below is the iterative part of the algorithm
    gen_num = db.get_generation_number()
    for i in range(num_gens):
        print(f'Creating and evaluating generation {gen_num + i}')
        new_generation = []
        for _ in range(pop_size):
            dup = True
            while dup:
                # Select parents for a new candidate
                parents = pop.get_two_candidates()

                # Select an operator and use it
                op = operation_selector.get_operator()
                offspring, desc = op.get_new_individual(parents)
                # An operator could return None if an offspring cannot be formed
                # by the chosen parents
                if offspring is None:
                    continue

                atoms_string = ''.join(offspring.get_chemical_symbols())
                dup = db.is_duplicate(atoms_string=atoms_string)
            try:
                set_raw_score(offspring, -get_mixing_energy(offspring))
            except Exception as e:
                print('Failing:', e)
                continue
            offspring.info['key_value_pairs']['E_tot'] = offspring.info['energy']
            offspring.info['key_value_pairs']['Na_conc'] = np.round(len([atom for atom in offspring if atom.symbol == 'Na'])/fu,3)
            new_generation.append(offspring)

        # We add a full relaxed generation at once, this is faster than adding
        # one at a time
        db.add_more_relaxed_candidates(new_generation)

        # update the population to allow new candidates to enter
        pop.update()

if __name__ == "__main__":
    main()