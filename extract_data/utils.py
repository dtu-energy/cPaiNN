from ase.data import atomic_numbers
from ase.units import Bohr
from ase import Atoms
import pandas as pd
from ase.io import read, write
def attach_bader_charges(atoms:Atoms, ACF_path:str, displacement:float=1e-3)->Atoms:
    """
    Function used to attached the Bader charges to the atoms object.
    Note: The zval is given by the POTCAR from the VASP 6.4

    Args:
        atoms (Atoms): ASE atoms object
        ACF_path (str): Path to the ACF.dat file
        displacement (float): Displacement to test if the positions match

    Returns:
        atoms (Atoms): ASE atoms object with the Bader charges attached
    
    """

    # Define the zval, given by the POTCAR from VASP 6.4 
    zval = {'Na':7,'Fe':14,'O':6,'P':5,'Mn':13,'Co':9,'Ni':16,'Si':4,'S':6}

    # Load ACF file 
    df = pd.read_csv(ACF_path,skiprows=2,skipfooter=4,sep='\s+',header=None,index_col=0,engine='python') # Read the ACF.dat file and skip the first two lines and the last 4 lines
    df.columns = ['X','Y','Z','CHARGE','MIN DIST','ATOMIC VOL'] # Add the header according to ACF.dat
    acf_charge = df['CHARGE'].values # Get the charges from the ACF.dat file
    acf_pos = df[['X','Y','Z']].values # Get the positions from the ACF.dat file

    # Add charges and test if the positions match
    for i, a in enumerate(atoms):
        a.charge = zval[a.symbol]- acf_charge[i] # Add the charge to the atom object
        # Test if the atom positions match
        if displacement is not None:
            norm = np.linalg.norm(a.position - acf_pos[i])
            norm2 = np.linalg.norm(a.position - acf_pos[i] * Bohr)
            assert norm < displacement or norm2 < displacement
    #atoms.set_initial_charge = df['CHARGE'].values # Add the charge to the atoms object
    return atoms


def combine_trajactories(list_of_traj:list, output_path:str='Combined.xyz')->None:
    """
    Function used to combine a list of trajectories into one trajectory.

    Args:
        list_of_traj (list): List of paths to the trajectories
        output_path (str): Path to the output trajectory

    Returns:
        None
    """
    # Read the trajectories
    combined_traj = []
    for traj_path in list_of_traj:
        combined_traj.extend(read(traj_path,':'))
    
    # Save the trajectory
    write(output_path,combined_traj)
