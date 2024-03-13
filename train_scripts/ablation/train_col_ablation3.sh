#!/bin/bash

export CUDA_VISIBLE_DEVICES=GPU_ID

config=./configs/ablation/col_sample_ablatoin_3.yaml

python train_ablation_3.py --config $config --port PORT_ID


