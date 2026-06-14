#!/bin/bash -ex

python active_learning.py \
    --train_set='/home/energy/mahpe/Published_code/autocPaiNN/Example_data/NaFePO4_train.xyz' \
    --pool_set='/home/energy/mahpe/Published_code/test_cPaiNN/simulate/iter_0/NaFePO4_1000K/MD.xyz' \
    --run_path='/home/energy/mahpe/Published_code/cPaiNN' \
    --model_name=cpainn \
    --model_path='/home/energy/mahpe/Published_code/test_cPaiNN/train/iter_0/64_node_4_layer' \
    --kernel='full-g' \
    --selection='lcmd_greedy' \
    --n_random_features=500 \
    --batch_size=10 \
    --random_seed=42
    
