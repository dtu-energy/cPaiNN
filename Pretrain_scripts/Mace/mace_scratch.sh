#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<mahpe@dtu.dk>  # The default value is the submitting user.
#SBATCH --partition=sm3090_devel
#SBATCH --job-name=MAT_SC
#SBATCH --output=mace_h5_mt_sc300_2.out
#SBATCH --error=mace_h5_mt_sc300_2.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --time=02:00:00 # 1 day, 2 hours, 0 min, 0 sec.
#SBATCH --begin=now+0hour # 0 is in seconds
##SBATCH --exclusive
#SBATCH --mem-per-gpu=64G


nvidia-smi

#module use /home/energy/modules/modules/all

#source /home/energy/chipa/softwares/install/anaconda3/etc/profile.d/conda.sh

conda activate defect_MLIP

mace_run_train \
    --name="Martin_exp_large_scratch" \
    --seed=111 \
    --log_dir="logs_runchkpt" \
    --model_dir="." \
    --checkpoints_dir="chkpt_h5" \
    --results_dir="results_runchkpt200" \
    --train_file="/home/energy/chipa/genModels/universal_mace/martin/mtrain_REF.xyz" \
    --valid_file="/home/energy/chipa/genModels/universal_mace/martin/mvalid_REF.xyz" \
    --test_file="/home/energy/chipa/genModels/universal_mace/martin/mtest_REF.xyz" \
    --energy_weight=1.0 \
    --forces_weight=99.0 \
    --E0s='{11:-0.00850613, 27:1.42231373, 26:-0.25147872, 28:5.19751556, 25:-4.56090422, 8:-0.01400954, 15:-0.01380083, 16:-0.01578118, 14:-0.01138382}' \
    --loss='universal' \
    --enable_cueq=True \
    --energy_weight=1 \
    --forces_weight=99 \
    --compute_stress=True \
    --stress_weight=10 \
    --stress_key='REF_stress' \
    --energy_key='REF_energy' \
    --forces_key='REF_forces' \
    --eval_interval=1 \
    --error_table='PerAtomMAE' \
    --model="MACE" \
    --interaction_first="RealAgnosticDensityInteractionBlock" \
    --interaction="RealAgnosticDensityResidualInteractionBlock" \
    --num_interactions=2 \
    --correlation=3 \
    --max_ell=3 \
    --r_max=5.0 \
    --max_L=2 \
    --num_channels=128 \
    --num_radial_basis=8 \
    --MLP_irreps="16x0e" \
    --scaling='rms_forces_scaling' \
    --lr=0.001 \
    --weight_decay=1e-8 \
    --ema \
    --ema_decay=0.995 \
    --scheduler_patience=5 \
    --batch_size=16 \
    --valid_batch_size=16 \
    --pair_repulsion \
    --distance_transform="Agnesi" \
    --max_num_epochs=301 \
    --patience=40 \
    --amsgrad \
    --device=cuda \
    --seed=111 \
    --clip_grad=100 \
    --keep_checkpoints \
    --save_all_checkpoints \
    --restart_latest \
    --default_dtype="float64" \
    --num_workers=4 \
    --save_cpu \
    --enable_cueq=True \
    --wandb \
    --wandb_project="Martin_scratch" \
    --wandb_entity="mhp27" \
    --wandb_name="martin_mft_large" 
