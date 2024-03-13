#!/bin/bash
export CUDA_VISIBLE_DEVICES=GPU_ID

python sample/sample_ddp.py \
    --config ./configs/kva/kva_sample.yaml \
    --ckpt /path/to/ckpt \
    --port PORT_ID \
    --save_video_path /path/to/save
