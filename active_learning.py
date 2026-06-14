import numpy as np
import os 
import json
import argparse, toml
from pathlib import Path
import logging
from ase.io import read, write

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="General Active Learning", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--run_path",
        type=str,
        help="Path to the run directory",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        help="How to get features",
    )
    parser.add_argument(
        "--selection",
        type=str,
        help="Selection method, one of `max_dist_greedy`, `deterministic_CUR`, `lcmd_greedy`, `max_det_greedy` or `max_diag`",
    )
    parser.add_argument(
        "--n_random_features",
        type=int,
        help="If `n_random_features = 0`, do not use random projections.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="How many data points should be selected",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Where to find the models",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model. Default is cpainn",
        default='cpainn',
    )
    parser.add_argument(
        "--pool_set", type=str, help="Path to MD trajectory obtained from machine learning potential",
    )
    parser.add_argument(
        "--train_set", type=str, help="Path to training set. Useful for pool/train based selection method",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed for this run",
    )
    return parser.parse_args(arg_list)

def update_namespace(ns:argparse.Namespace, d:dict) -> None:
    """

    Update the namespace with the dictionary.

    Args:
        ns: The namespace to update
        d: The dictionary to update the namespace with
    
    """
    for k, v in d.items():
        
        ns.__dict__[k] = v

def main():
    from cPaiNN.active_learning import GeneralActiveLearning
    from cPaiNN.data import AseDataset
    from cPaiNN.model import PainnModel
    from cPaiNN.relax import ML_Relaxer
    import torch
    from cPaiNN.utils import setup_seed

    #return True, {'system_name':system_name}
    # Create device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # Put a tensor on the device before loading data
    # This way the GPU appears to be in use when other users run gpustat
    torch.tensor([0], device=device)

    args = get_arguments()

    # Load the run path
    run_path = args.run_path 
    
    # Load the system parameters
    
    pool_set = args.pool_set

    setup_seed(args.random_seed)

    # System directory
    system_dir = os.path.join(run_path,'active_learning')
    # Create the iteration directory
    if not os.path.exists(system_dir):
        os.makedirs(system_dir)

    # Save parsed arguments
    with open(os.path.join(system_dir,"arguments.json"), "w") as f:
        json.dump(vars(args), f)

    # Load models
    if args.model_name != 'cpainn':
        raise NotImplementedError("Only cpainn model is supported at the moment")

    models = []
    model_pth = Path(args.model_path).rglob('*best_model.pth')
    print(args.model_path)
    models = []
    for each in model_pth:
        state_dict = torch.load(each, map_location=torch.device(device)) 
        model = PainnModel(
            num_interactions=state_dict["num_layer"], 
            hidden_state_size=state_dict["node_size"], 
            cutoff=state_dict["cutoff"],
            compute_forces=state_dict["compute_forces"],
            compute_stress=state_dict["compute_stress"],
            compute_magmom=state_dict["compute_magmom"],
            compute_bader_charge=state_dict["compute_bader_charge"],
            )
        model.to(device)
        model.load_state_dict(state_dict["model"],)    
        models.append(model)


    # Test if models is a list and there is a pool set and train set
    assert isinstance(models, list)
    assert args.pool_set and args.train_set 

    # set logger
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(system_dir,"al_log.log"), mode="w"
            ),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Loding the dataset from {args.pool_set} and {args.train_set}")
    # Load pool set and train set
    if isinstance(args.pool_set, list):
        dataset = []
        for traj in args.pool_set:
            if not os.path.exists(traj):
                logging.info(f"File {traj} does not exist!")
                continue
            if Path(traj).stat().st_size > 0:
                dataset += read(traj, index=':') 
    else:
        dataset = read(args.pool_set, index=':')
    
    # Only include structures with calculators
    dataset_new = []
    for atoms in dataset:
        try:
            atoms.get_potential_energy()
            dataset_new.append(atoms)
        except:
            pass
    logging.info(f"Loaded {len(dataset_new)} structures with calculators out of {len(dataset)} structures")
    # if there are structures without calculators, write them to a new file and update pool set
    if len(dataset) != len(dataset_new):

        args.pool_set = os.path.join(system_dir,'new_pool_set.xyz')
        write(args.pool_set,dataset_new)
        logging.info(f"Structures with calculators are written to {args.pool_set}")
    
    dataset = dataset_new
    

    data_dict = {
        'pool': AseDataset(dataset, cutoff=models[0].cutoff),
        'train': AseDataset(args.train_set, cutoff=models[0].cutoff),
        #'train': AseDataset(read(args.train_set, index=':1000'), cutoff=models[0].cutoff),
    }

    logging.info(f"Train set size: {len(data_dict['train'])}")
    logging.info(f"Pool set size: {len(data_dict['pool'])}")

    # Select structures
    al = GeneralActiveLearning(
        kernel=args.kernel, 
        selection=args.selection, 
        n_random_features=args.n_random_features,
    )
    logging.info(f"Selecting {args.batch_size} structures")
    
    al_idx = al.select(models, data_dict, al_batch_size=args.batch_size)
    al_info = {
        'kernel': args.kernel,
        'selection': args.selection,
        'dataset': args.pool_set,
        'selected': al_idx,
    }

    with open(os.path.join(system_dir,'selected.json'), 'w') as f:
        json.dump(al_info, f)

if __name__ == "__main__":
    main()
