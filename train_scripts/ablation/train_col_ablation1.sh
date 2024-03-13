#!/bin/bash

export CUDA_VISIBLE_DEVICES=GPU_ID

config=./configs/ablation/col_sample_ablatoin_1.yaml

python train_ablation_1.py --config $config --port PORT_ID

