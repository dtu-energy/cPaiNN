from cPaiNN.model import PainnModel
from cPaiNN.data import AseDataset
import torch
import numpy as np
import math
import json
import os
import argparse


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
            
            #if test == len(dataloader):
            #    stress_loss = 0
                #print('Val_stress',out["stress"],device_batch["stress"])
                #print('Val_stress',out["stress"].shape,device_batch["stress"].shape)
                #print('Val_forces',out["forces"],device_batch["forces"])
                #print('Val_forces',out["forces"].shape,device_batch["forces"].shape)
                #print('Val_energy',out["energy"],device_batch["energy"])
                #print('Val_energy',out["energy"].shape,device_batch["energy"].shape)
                #print('Val_stress',out["stress"],device_batch["stress"])
                #print('Val_stress',out["stress"].shape,device_batch["stress"].shape)
        else:
            stress_loss = 0.0

        if isinstance(charge_key,list):
            magmom_loss = criterion(out['magmom'], device_batch['magmom']).item()
            bader_charge_loss = bader_charge_loss_func(out['bader_charge'], device_batch['bader_charge']).item()
            charge_loss = magmom_loss + bader_charge_loss
        elif isinstance(charge_key,str):
            if charge_key == 'bader_charge':
                charge_loss = bader_charge_loss_func(out['bader_charge'], device_batch['bader_charge']).item()
            else:
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

            bader_charge_targets = batch['bader_charge'].detach().cpu()
            # Remove nan values
            mask = ~torch.isnan(bader_charge_targets)
            pred = bader_charge_targets[mask].numpy()
            target = outputs['bader_charge'][mask]
            if len(pred) == 0:
                bader_charge_diff = torch.tensor(0.0)
            else:
                bader_charge_diff = target - pred

            bader_charge_running_ae += np.sum(np.abs(bader_charge_diff), axis=0)
            bader_charge_running_se += np.sum(
                np.square(bader_charge_diff), axis=0
            )

        elif isinstance(charge_key,str):
            if charge_key == 'bader_charge':
                charge_targets = batch[charge_key].detach().cpu()
                # Remove nan values
                mask = ~torch.isnan(charge_targets)
                pred = charge_targets[mask]
                target = outputs[charge_key][mask]
                if len(pred) == 0:
                    charge_diff = torch.tensor(0.0)
                else:
                    charge_diff = target - pred
            else:
                charge_targets = batch[charge_key].detach().cpu()
                charge_diff = charge_targets - outputs[charge_key]
            charge_diff = charge_diff.detach().cpu().numpy()
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

def bader_charge_loss_func(pred, target):
    """
    Compute the mean squared error between predicted and target bader charges.
    The error function is simialr to torch.nn.MSELoss, but it ignores nan values.
    Since the value is a mean over all contribution it means the loss is not affected by the loss of atoms in the training.
    Args:
        pred: Predicted bader charges
        target: Target bader charges
    Returns:
        The mean squared error between predicted and target bader chargesW
    """
    # Remove nan values
    mask = ~torch.isnan(target)
    pred = pred[mask]
    target = target[mask]
    if len(pred) == 0:
        return torch.tensor(0.0)
    loss = torch.mean(torch.square(pred - target))

    return loss

def setup_seed(seed):
     torch.manual_seed(seed)
     if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

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