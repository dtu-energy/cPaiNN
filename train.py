import os 
print(os.environ['CONDA_DEFAULT_ENV'])

import numpy as np
import math
import json, sys, toml
import argparse
import logging
import itertools
import torch
import time

from cPaiNN.data import AseDataset, collate_atomsdata
from cPaiNN.model import PainnModel

# Funciton to setup random seed
def setup_seed(seed):
     torch.manual_seed(seed)
     if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

# Define function to get arguments
def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Train graph convolution network", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--load_model",
        type=str,
        help="Load model parameters from previous run",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        help="Atomic interaction cutoff distance in Angstrom",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        help="Train/test/validation split file json",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        help="Ratio of validation set. Only useful when 'split_file' is not assigned",
    )
    parser.add_argument(
        "--num_interactions",
        type=int,
        help="Number of interaction layers used",
    )
    parser.add_argument(
        "--node_size", type=int, help="Size of hidden node states"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to output directory",
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        help="Path to ASE trajectory. ",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        help="Maximum number of optimisation steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Set which device to use for training e.g. 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        help="Number of molecules per minibatch",
    )
    parser.add_argument(
        "--initial_lr", 
        type=float, 
        help="Initial learning rate",
    )
    parser.add_argument(
        "--forces_weight",
        type=float,
        help="Tradeoff between training on forces (weight=1) and energy (weight=0)",
    )
    parser.add_argument(
        "--charge_weight",
        type=float,
        help="Weight for charge representation",
    )
    parser.add_argument(
        "--stress_weight",
        type=float,
        help="Weight for stress representation",
    )
    parser.add_argument(
        "--log_inverval",
        type=int,
        help="The interval of model evaluation",
    )
    parser.add_argument(
        "--plateau_scheduler",
        action="store_true",
        help="Using ReduceLROnPlateau scheduler for decreasing learning rate when learning plateaus",
    )
    parser.add_argument(
        "--normalization",
        action="store_true",
        help="Enable normalization of the model",
    )
    parser.add_argument(
        "--atomwise_normalization",
        action="store_true",
        help="Enable atomwise normalization",
    )
    parser.add_argument(
        "--stop_patience",
        type=int,
        help="Stop training when validation loss is larger than best loss for 'stop_patience' steps",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed for this run",
    )  
    parser.add_argument(
        "--compute_forces",
        type=bool,
        help="Compute forces",
        default=True,
    )
    parser.add_argument(
        "--compute_stress",
        type=bool,
        help="Compute stress",
        #default=True,
    )
    parser.add_argument(
        "--compute_magmom",
        type=bool,
        help="Compute magnetic moments",
    )     
    parser.add_argument(
        "--compute_bader_charge",
        type=bool,
        help="Compute bader charges",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        help="Path to config file. e.g. 'arguments.toml'",
        default='config.toml'
    )
    return parser.parse_args(arg_list)


def split_data(dataset:AseDataset, args:argparse.Namespace) -> dict:
    """
    Split the dataset into training and validation sets, if not already split.

    Args:
        dataset: The dataset to split
        args: The command line arguments
    
    Returns:
        A dictionary containing the training and validation sets
    """

    # Load or generate splits
    if args.split_file:
        with open(args.split_file, "r") as fp:
            splits = json.load(fp)
    else:
        datalen = len(dataset)
        num_validation = int(math.ceil(datalen * args.val_ratio))
        indices = np.random.permutation(len(dataset))
        splits = {
            "train": indices[num_validation:].tolist(),
            "validation": indices[:num_validation].tolist(),
        }

    # Save split file
    with open(os.path.join(args.output_dir, "datasplits.json"), "w") as f:
        json.dump(splits, f)

    # Split the dataset
    datasplits = {}
    for key, indices in splits.items():
        datasplits[key] = torch.utils.data.Subset(dataset, indices)
    return datasplits

def forces_criterion(predicted:torch.Tensor, target:torch.Tensor, reduction:str="mean") -> torch.Tensor:
    """
    Compute the mean squared error between predicted and target forces.

    Args:
        predicted: Predicted forces
        target: Target forces
        reduction: Reduction method, either 'mean' or 'sum'
    
    Returns:
        The mean squared error between predicted and target forces
    
    """

    # predicted, target are (bs, max_nodes, 3) tensors
    # node_count is (bs) tensor
    diff = predicted - target
    total_squared_norm = torch.linalg.norm(diff, dim=1)  # and not torch.sum(torch.square(diff),dim=1)
    if reduction == "mean":
        scalar = torch.mean(total_squared_norm)
    elif reduction == "sum":
        scalar = torch.sum(total_squared_norm)
    else:
        raise ValueError("Reduction must be 'mean' or 'sum'")
    return scalar

def get_normalization(dataset: AseDataset, per_atom:bool=True) -> tuple:
    """
    Compute the mean and standard deviation of the dataset.

    Args:
        dataset: The dataset to compute the mean and standard deviation of
        per_atom: Whether to normalize per atom or per sample
    
    Returns:
        A tuple containing the mean and standard deviation of the dataset
    
    """
    # Use double precision to avoid overflows
    x_sum = torch.zeros(1, dtype=torch.double)
    x_2 = torch.zeros(1, dtype=torch.double)
    num_objects = 0
    for i, sample in enumerate(dataset):
        if i == 0:
            # Estimate "bias" from 1 sample
            # to avoid overflows for large valued datasets
            if per_atom:
                bias = sample["energy"] / sample["num_atoms"]
            else:
                bias = sample["energy"]
        x = sample["energy"]
        if per_atom:
            x = x / sample["num_atoms"]
        x -= bias
        x_sum += x
        x_2 += x ** 2.0
        num_objects += 1
    # Var(X) = E[X^2] - E[X]^2
    x_mean = x_sum / num_objects
    x_var = x_2 / num_objects - x_mean ** 2.0
    x_mean = x_mean + bias

    default_type = torch.get_default_dtype()

    return x_mean.type(default_type), torch.sqrt(x_var).type(default_type)

def eval_model(model:PainnModel, dataloader:AseDataset, device:str, args:argparse.Namespace,criterion = torch.nn.MSELoss()) -> dict:
    """
    
    Evaluate the model on the validation set.

    Args:
        model: The model to evaluate
        dataloader: The validation set
        device: The device to use for evaluation
        args: The command line arguments
        criterion: The loss function to use
    
    Returns:
        A dictionary containing the evaluation metrics
    
    """

    # Decide what charge representation to use
    if args.compute_magmom and args.compute_bader_charge:
        charge_key = ['magmom', 'bader_charge']
    elif args.compute_magmom:
        charge_key = 'magmom'
    elif args.compute_bader_charge:
        charge_key = 'bader_charge'
    else:
        charge_key = None
    
    # Initialize running variables
    energy_running_ae = 0
    energy_running_se = 0

    if args.compute_forces:
        forces_running_l2_ae = 0
        forces_running_l2_se = 0
        forces_running_c_ae = 0
        forces_running_c_se = 0
        forces_running_loss = 0

        forces_count = 0   

    if args.compute_stress:
        stress_running_ae = 0
        stress_running_se = 0

        stress_count = 0

    if isinstance(charge_key,list):
        magmom_running_ae = 0
        magmom_running_se = 0
        magmom_count = 0

        bader_charge_running_ae = 0
        bader_charge_running_se = 0
        bader_charge_count = 0
    elif isinstance(charge_key,str):
        charge_running_ae = 0
        charge_running_se = 0

        charge_count = 0
    else:
        pass


    running_loss = 0
    count = 0
    test = 0
    # Loop over each batch
    print(len(dataloader))
    for batch in dataloader:
        test += 1
        
        device_batch = {
            k: v.to(device=device, non_blocking=True) for k, v in batch.items()
        }

        # Forward pass
        out = model(device_batch)

        # Update running counts 
        count += batch["energy"].shape[0]
        if args.compute_forces:
            forces_count += batch['forces'].shape[0]
        if args.compute_stress:
            stress_count += batch['stress'].shape[0]
        
        if isinstance(charge_key,list):
            magmom_count += batch['magmom'].shape[0]
            bader_charge_count += batch['bader_charge'].shape[0]
        elif isinstance(charge_key,str):
            charge_count += batch[charge_key].shape[0]

        # Use mean square loss here
        if args.compute_forces:
            forces_loss = forces_criterion(out["forces"], device_batch["forces"]).item()
        else:
            forces_loss = 0.0

        if args.compute_stress:
            # calculate stress
            device_batch['stress'] = torch.reshape(device_batch['stress'], (device_batch['energy'].shape[0], 3,3))
            stress_loss = criterion(out["stress"], device_batch["stress"]).item()
            
            if test == len(dataloader):
            #    stress_loss = 0
                print('Val_stress',out["stress"],device_batch["stress"])
                print('Val_stress',out["stress"].shape,device_batch["stress"].shape)
                print('Val_forces',out["forces"],device_batch["forces"])
                print('Val_forces',out["forces"].shape,device_batch["forces"].shape)
                print('Val_energy',out["energy"],device_batch["energy"])
                print('Val_energy',out["energy"].shape,device_batch["energy"].shape)
                print('Val_stress',out["stress"],device_batch["stress"])
                print('Val_stress',out["stress"].shape,device_batch["stress"].shape)
        else:
            stress_loss = 0.0

        if isinstance(charge_key,list):
            magmom_loss = criterion(out['magmom'], device_batch['magmom']).item()
            bader_charge_loss = criterion(out['bader_charge'], device_batch['bader_charge']).item()
            charge_loss = magmom_loss + bader_charge_loss
        elif isinstance(charge_key,str):
            charge_loss = criterion(out[charge_key], device_batch[charge_key]).item()
        else:
            charge_loss = 0.0

        energy_loss = criterion(out["energy"], device_batch["energy"]).item() 

        # Calculate total loss
        # Total loss
        total_loss = (
                args.forces_weight * forces_loss
                + (1 - args.forces_weight) * energy_loss
                + args.stress_weight * stress_loss
                #+ args.stress_weight * stress_loss
                + args.charge_weight * charge_loss
        )
        
        # Update running loss
        running_loss += total_loss * batch["energy"].shape[0]
        # Energy errors
        outputs = {key: val.detach().cpu().numpy() for key, val in out.items()}
        energy_targets = batch["energy"].detach().cpu().numpy()
        energy_running_ae += np.sum(np.abs(energy_targets - outputs["energy"]), axis=0)
        energy_running_se += np.sum(
            np.square(energy_targets - outputs["energy"]), axis=0
        )

        # Force errors
        if args.compute_forces:
            forces_targets = batch["forces"].detach().cpu().numpy()
            forces_diff = forces_targets - outputs["forces"]
            forces_l2_norm = np.sqrt(np.sum(np.square(forces_diff), axis=1))

            forces_running_c_ae += np.sum(np.abs(forces_diff))
            forces_running_c_se += np.sum(np.square(forces_diff))

            forces_running_l2_ae += np.sum(np.abs(forces_l2_norm))
            forces_running_l2_se += np.sum(np.square(forces_l2_norm))
        
        # Stress errors
        if args.compute_stress:
            stress_targets = batch["stress"].detach().cpu().numpy()
            stress_targets = np.reshape(stress_targets, (energy_targets.shape[0], 3, 3))
            stress_diff = stress_targets - outputs["stress"]
            
            stress_running_ae += np.mean(np.sum(np.abs(stress_diff), axis=0))
            stress_running_se += np.mean(np.sum(
                np.square(stress_diff), axis=0
            ))
        
        # Charge errors
        if isinstance(charge_key,list):
            magmom_targets = batch['magmom'].detach().cpu().numpy()
            magmom_diff = magmom_targets - outputs['magmom']

            magmom_running_ae += np.sum(np.abs(magmom_diff), axis=0)
            magmom_running_se += np.sum(
                np.square(magmom_diff), axis=0
            )

            bader_charge_targets = batch['bader_charge'].detach().cpu().numpy()
            bader_charge_diff = bader_charge_targets - outputs['bader_charge']

            bader_charge_running_ae += np.sum(np.abs(bader_charge_diff), axis=0)
            bader_charge_running_se += np.sum(
                np.square(bader_charge_diff), axis=0
            )

        elif isinstance(charge_key,str):
            charge_targets = batch[charge_key].detach().cpu().numpy()
            charge_diff = charge_targets - outputs[charge_key]

            charge_running_ae += np.sum(np.abs(charge_diff), axis=0)
            charge_running_se += np.sum(
                np.square(charge_diff), axis=0
            )
    
    # Calculate mean absolute error and root mean squared error
    evaluation = {}

    evaluation['energy_mae'] = energy_running_ae / count
    evaluation['energy_rmse'] = np.sqrt(energy_running_se / count)

    if args.compute_forces:
        evaluation['forces_l2_mae']= forces_running_l2_ae / forces_count
        evaluation['forces_l2_rmse'] = np.sqrt(forces_running_l2_se / forces_count)

        evaluation['forces_mae'] = forces_running_c_ae / (forces_count * 3)
        evaluation['forces_rmse'] = np.sqrt(forces_running_c_se / (forces_count * 3))

    if args.compute_stress:
        evaluation['stress_mae'] = stress_running_ae / stress_count
        evaluation['stress_rmse'] = np.sqrt(stress_running_se / stress_count)

    if isinstance(charge_key,list):
        evaluation['magmom_mae'] = magmom_running_ae / magmom_count
        evaluation['magmom_rmse'] = np.sqrt(magmom_running_se / magmom_count)
        evaluation['bader_charge_mae'] = bader_charge_running_ae / bader_charge_count
        evaluation['bader_charge_rmse'] = np.sqrt(bader_charge_running_se / bader_charge_count)

    elif isinstance(charge_key,str):
        evaluation[f"{charge_key}_mae"] = charge_running_ae / charge_count
        evaluation[f"{charge_key}_rmse"] = np.sqrt(charge_running_se / charge_count)

    # Save the validation loss
    evaluation['sqrt(val_loss)'] = np.sqrt(running_loss / count)

    return evaluation

def update_namespace(ns:argparse.Namespace, d:dict) -> None:
    """

    Update the namespace with the dictionary.

    Args:
        ns: The namespace to update
        d: The dictionary to update the namespace with
    
    """
    for k, v in d.items():
        if not ns.__dict__.get(k):
            ns.__dict__[k] = v

class EarlyStopping() :
    """
    Early stopping to stop the training when the loss does not improve after certain epochs.

    """
    def __init__(self, patience=5, min_delta=0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, best_loss):
        if val_loss - best_loss > self.min_delta:
            self.counter +=1
            if self.counter >= self.patience:  
                self.early_stop = True
                
        return self.early_stop

def main():
    # Load argument Namespace
    args = get_arguments()

    # Load parameters from config file
    if os.path.exists(args.cfg):
        with open(args.cfg, 'r') as f:
            params = toml.load(f)
    
        # Update namespace with parameters
        update_namespace(args, params)
    
    # Setup random seed
    setup_seed(args.random_seed)

    # Create output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "printlog.txt"), mode="w"
            ),
            logging.StreamHandler(),
        ],
    )

    # Decide what charge representation to use
    if args.compute_magmom and args.compute_bader_charge:
        logging.info("Computing magnetic moments and bader charges")
        charge_key = ['magmom', 'bader']
    elif args.compute_magmom:
        logging.info("Computing magnetic moments")
        charge_key = 'magmom'
    elif args.compute_bader_charge:
        logging.info("Computing bader charges")
        charge_key = 'bader_charge'
    else:
        charge_key = None

    # Save command line args
    with open(os.path.join(args.output_dir, "commandline_args.txt"), "w") as f:
        f.write("\n".join(sys.argv[1:]))

    # Save parsed command line arguments
    with open(os.path.join(args.output_dir, "arguments.json"), "w") as f:
        json.dump(vars(args), f)

    # Create device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # Put a tensor on the device before loading data
    # This way the GPU appears to be in use when other users run gpustat
    torch.tensor([0], device=device)

    # Setup dataset and loader
    logging.info("loading data %s", args.dataset)
    dataset = AseDataset(
        args.dataset,
        cutoff = args.cutoff,
        compute_forces=args.compute_forces,
        compute_stress=args.compute_stress,
        charge_key=charge_key,

    )

    datasplits = split_data(dataset, args)

    train_loader = torch.utils.data.DataLoader(
        datasplits["train"],
        args.batch_size,
        sampler=torch.utils.data.RandomSampler(datasplits["train"]),
        collate_fn=collate_atomsdata,
    )
    val_loader = torch.utils.data.DataLoader(
        datasplits["validation"], 
        args.batch_size, 
        collate_fn=collate_atomsdata,
    )
    
    logging.info('Dataset size: {}, training set size: {}, validation set size: {}'.format(
        len(dataset),
        len(datasplits["train"]),
        len(datasplits["validation"]),
    ))

    # compute normalization statistics if needed
    if args.normalization:
        logging.info("Computing mean and variance")
        target_mean, target_stddev = get_normalization(
            datasplits["train"], 
            per_atom=args.atomwise_normalization,
        )
        logging.debug("target_mean=%f, target_stddev=%f" % (target_mean, target_stddev))

    # Setup model
    net = PainnModel(
        num_interactions=args.num_interactions, 
        hidden_state_size=args.node_size,
        cutoff=args.cutoff,
        normalization=args.normalization,
        target_mean=target_mean.tolist() if args.normalization else [0.0],
        target_stddev=target_stddev.tolist() if args.normalization else [1.0],
        atomwise_normalization=args.atomwise_normalization,
        compute_forces=args.compute_forces,
        compute_stress=args.compute_stress,
        compute_magmom=args.compute_magmom, 
        compute_bader_charge=args.compute_bader_charge
    )
    net.to(device)

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=args.initial_lr)
    criterion = torch.nn.MSELoss()
    if args.plateau_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    else:
        scheduler_fn = lambda step: 0.96 ** (step / 100000)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_fn)
    early_stop = EarlyStopping(patience=args.stop_patience)    

    # Initialize running variables
    running_loss = 0
    running_loss_count = 0

    # used for smoothing loss
    prev_loss = None
    best_val_loss = np.inf
    step = 0
    training_time = 0    

    # Load model if needed
    if args.load_model:
        logging.info(f"Load model from {args.load_model}")
        state_dict = torch.load(args.load_model)
        net.load_state_dict(state_dict["model"])
        scheduler.load_state_dict(state_dict["scheduler"])
    
    # Train model 
    for epoch in itertools.count():

        # Loop over each batch in training set
        for batch_host in train_loader:
            # Start timer
            start = time.time()
            
            # Transfer to 'device'
            batch = {
                k: v.to(device=device, non_blocking=True)
                for (k, v) in batch_host.items()
            }
            print('batch',batch.keys())
            # Reset gradient
            optimizer.zero_grad()
            #print(batch['stress'].shape,batch['stress'])
            #print(batch.keys())
            #print('num_atoms',batch['num_atoms'].shape)
            #print('forces',batch['forces'].shape)
            #print('coord',batch['coord'].shape)
            #print('elems',batch['elems'].shape)
            #atch['cell'] = torch.reshape(batch['cell'], (args.batch_size, 3, 3))
            #strain = torch.zeros_like(batch['cell'], requires_grad=True)
            #batch['cell'] = batch['cell'] + torch.matmul(strain, batch['cell'])
            
            # Forward pass 
            outputs = net(batch)            
            #print(outputs[charge_key].shape, outputs['energy'].shape, outputs['forces'].shape)
            #print(batch[charge_key].shape, batch['energy'].shape, batch['forces'].shape)
            
            # Reshape stress tensor
            
            # Compute loss
            # Energy loss
            energy_loss = criterion(outputs["energy"], batch["energy"])

            # Forces loss
            if args.compute_forces:
                forces_loss = forces_criterion(outputs['forces'], batch['forces'])
            else:
                forces_loss = 0.0

            # Stress loss
            if args.compute_stress:
                batch['stress'] = torch.reshape(batch['stress'], (batch['energy'].shape[0], 3,3))
                
                stress_loss = criterion(outputs['stress'], batch['stress'])
            else:
                stress_loss = 0.0
            
            # Charge loss
            if isinstance(charge_key,list):
                magmom_loss = criterion(outputs['magmom'], batch['magmom'])
                bader_charge_loss = criterion(outputs['bader_charge'], batch['bader_charge'])
                charge_loss = magmom_loss + bader_charge_loss

            elif isinstance(charge_key,str):
                charge_loss = criterion(outputs[charge_key], batch[charge_key])
            else:
                charge_loss = 0.0

            # Total loss
            total_loss = (
                args.forces_weight * forces_loss
                + (1 - args.forces_weight) * energy_loss
                + args.stress_weight * stress_loss
                + args.charge_weight * charge_loss
            )

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Update running loss and time
            running_loss += total_loss.item() * batch["energy"].shape[0]
            running_loss_count += batch["energy"].shape[0]
            training_time += time.time() -  start

            # Validate and save model for each log step
            if (step % args.log_interval == 0) or ((step + 1) == args.max_steps):
                # start timer
                eval_start = time.time()
                
                # Calculate training loss
                train_loss = running_loss / running_loss_count # loss per sample
                running_loss = running_loss_count = 0 # reset running loss

                # Evaluate model on validation set
                eval_dict = eval_model(net, val_loader, device, args,criterion=criterion)
                eval_formatted = ", ".join(
                    ["{}={:.5f}".format(k, v) for (k, v) in eval_dict.items()]
                )

                # Loss smoothing
                eval_loss = eval_dict["sqrt(val_loss)"]
                smooth_loss = eval_loss if prev_loss == None else 0.9 * eval_loss + 0.1 * prev_loss
                prev_loss = smooth_loss

                # Log results
                logging.info(
                    "step={}, {}, sqrt(train_loss)={:.3f}, sqrt(smooth_loss)={:.3f}, patience={:3d}, training time={:.3f} min, eval time={:.3f} min".format(
                        step,
                        eval_formatted,
                        math.sqrt(train_loss),
                        math.sqrt(smooth_loss),
                        early_stop.counter,
                        training_time / 60,
                        (time.time() - eval_start) / 60,
                    )
                )

                # initialize training time
                training_time = 0

                # reduce learning rate
                if args.plateau_scheduler:
                    scheduler.step(smooth_loss)
                
                # Save checkpoint
                if not early_stop(math.sqrt(smooth_loss), best_val_loss):
                    best_val_loss = math.sqrt(smooth_loss)
                    torch.save(
                        {
                            "model": net.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "step": step,
                            "best_val_loss": best_val_loss,
                            "node_size": args.node_size,
                            "num_layer": args.num_interactions,
                            "cutoff": args.cutoff,
                            "compute_forces": args.compute_forces,
                            "compute_stress": args.compute_stress,
                            "compute_magmom": args.compute_magmom,
                            "compute_bader_charge": args.compute_bader_charge,
                        },
                        os.path.join(args.output_dir, "best_model.pth"),
                    )
                else:
                    sys.exit(0)

            step += 1

            # Check if max steps reached
            if not args.plateau_scheduler:
                scheduler.step()

            # Check if max steps reached
            if step >= args.max_steps:
                logging.info("Max steps reached, exiting")
                torch.save(
                    {
                        "model": net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step,
                        "best_val_loss": best_val_loss,
                        "node_size": args.node_size,
                        "num_layer": args.num_interactions,
                        "cutoff": args.cutoff,
                        "compute_forces": args.compute_forces,
                        "compute_stress": args.compute_stress,
                        "compute_magmom": args.compute_magmom,
                        "compute_bader_charge": args.compute_bader_charge,
                    },
                    os.path.join(args.output_dir, "exit_model.pth"),
                )
                sys.exit(0)

if __name__ == "__main__":
    main()
