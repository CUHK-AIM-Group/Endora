#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

python sample/sample_ddp.py \
    --config ./configs/col/col_sample.yaml \
    --ckpt /path/to/ckpt \
    --port PORT_ID \
    --save_video_path /path/to/save
