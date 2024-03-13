#!/bin/bash

export CUDA_VISIBLE_DEVICES=GPU_ID

config=./configs/ablation/col_sample_ablatoin_2.yaml

python train_ablation_2.py --config $config --port PORT_ID

