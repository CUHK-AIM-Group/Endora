#!/bin/bash
export CUDA_VISIBLE_DEVICES=GPU_ID

python sample/sample.py \
    --config ./configs/cho/cho_sample.yaml \
    --ckpt /path/to/ckpt \
    --save_video_path /path/to/save
