import os 
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
from cPaiNN.utils import setup_seed, get_normalization, eval_model, forces_criterion, bader_charge_loss_func, split_data, EarlyStopping

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
        "--target_mean",
        type=float,
        help="Target mean for normalization",
    )
    parser.add_argument(
        "--target_stddev",
        type=float,
        help="Target standard deviation for normalization",
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
    logging.info(args.target_mean, args.target_stddev)
    if args.normalization:
        if args.target_mean is not None and args.target_stddev is not None:
            target_mean = torch.tensor([args.target_mean])
            target_stddev = torch.tensor([args.target_stddev])
            logging.info("Using user provided mean and stddev")
            logging.info("target_mean=%f, target_stddev=%f" % (target_mean, target_stddev))
        else:
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
            # Reset gradient
            optimizer.zero_grad()

            # Forward pass 
            outputs = net(batch)            
          
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
                bader_charge_loss = bader_charge_loss_func(outputs['bader_charge'], batch['bader_charge'])
                charge_loss = magmom_loss + bader_charge_loss

            elif isinstance(charge_key,str):
                if charge_key == 'bader_charge':
                    charge_loss = bader_charge_loss_func(outputs['bader_charge'], batch['bader_charge'])
                else:
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
            #print('total_loss:',total_loss)
            #print('energy_loss:',energy_loss)
            #print('forces_loss:',forces_loss)
            #print('stress_loss:',stress_loss)
            #print('charge_loss:',charge_loss)

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
