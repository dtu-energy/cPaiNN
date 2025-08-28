#!/bin/bash
##SBATCH --mail-type=ALL
##SBATCH --mail-user=<chipa@dtu.dk>  # The default value is the submitting user.
#SBATCH --partition=sm3090_devel #a100
#SBATCH --job-name=MAT_ft
#SBATCH --output=mace_h5_mtft.out
#SBATCH --error=mace_h5_mtft.err
#SBATCH -N 1 #-1
#SBATCH -n 8 #32
#SBATCH --gres=gpu:1 #A100:1
#SBATCH --time=0-02:00:00 # 1 day, 2 hours, 0 min, 0 sec.
#SBATCH --begin=now+0hour # 0 is in seconds
#SBATCH --mem-per-gpu=64G


nvidia-smi

module use /home/energy/modules/modules/all

source /home/energy/chipa/softwares/install/anaconda3/etc/profile.d/conda.sh

conda activate mace_latest

mace_run_train \
    --name="Martin_exp_large" \
    --seed=111 \
    --log_dir="logs_runchkpt" \
    --model_dir="." \
    --checkpoints_dir="chkpt_h5" \
    --results_dir="results_runchkpt" \
    --foundation_model="mace-large-density-agnesi-stress.model" \
    --multiheads_finetuning=False \
    --train_file="/home/energy/chipa/genModels/universal_mace/martin/mtrain_REF.xyz" \
    --valid_file="/home/energy/chipa/genModels/universal_mace/martin/mvalid_REF.xyz" \
    --test_file="/home/energy/chipa/genModels/universal_mace/martin/mtest_REF.xyz" \
    --energy_weight=1.0 \
    --forces_weight=99.0 \
    --E0s='{11:-0.00850613, 27:1.42231373, 26:-0.25147872, 28:5.19751556, 25:-4.56090422, 8:-0.01400954, 15:-0.01380083, 16:-0.01578118, 14:-0.01138382}' \
    --lr=0.001 \
    --batch_size=16 \
    --valid_batch_size=16 \
    --max_num_epochs=50 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --default_dtype="float64" \
    --scaling="rms_forces_scaling" \
    --compute_stress=True \
    --keep_checkpoints \
    --save_all_checkpoints \
    --restart_latest \
    --save_cpu \
    --device=cuda \
    --enable_cueq=True \
    --wandb \
    --wandb_project="Martin_FT" \
    --wandb_entity="charlescp" \
    --wandb_name="martin_mft_large" 
