
CONFIG_PATH="/mnt/zhen_chen/Dora/LVDM/configs/lvdm_short/cholec.yaml"
BASE_PATH="/mnt/zhen_chen/Dora/LVDM/log/lvdm_short_cholec/checkpoints/last.ckpt"
AEPATH="/mnt/zhen_chen/Dora/LVDM/log/lvdm_videoae_cholec/checkpoints/last.ckpt"
OUTDIR="results/uncond_cholec_short/"

CUDA_VISIBLE_DEVICES=7 python scripts/sample_uncond.py \
    --ckpt_path $BASE_PATH \
    --config_path $CONFIG_PATH \
    --save_dir $OUTDIR \
    --n_samples 200 \
    --batch_size 1 \
    --seed 12352 \
    --show_denoising_progress \
    model.params.first_stage_config.params.ckpt_path=$AEPATH