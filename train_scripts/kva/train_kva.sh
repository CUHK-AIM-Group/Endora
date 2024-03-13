#!/bin/bash

export CUDA_VISIBLE_DEVICES=GPU_ID

config=./configs/kva/kva_train.yaml

python train.py \
    --config $config \
    --port PORT_ID --mode type_cnn --prr_weight 0.5

