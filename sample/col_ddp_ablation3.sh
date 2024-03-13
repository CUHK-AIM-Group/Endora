#!/bin/bash
export CUDA_VISIBLE_DEVICES=GPU_ID

cd /path/to/EnDora
conda activate EnDora

python sample/sample_ddp.py \
    --config ./configs/ablation/col_sample_ablatoin_3.yaml  \
    --ckpt /path/to/ckpt \
    --port PORT_ID \
    --save_video_path /path/to/save
