CUDA_VISIBLE_DEVICES=0 python src/scripts/generate.py \
--network_pkl /mnt/zhen_chen/Dora/MoStGAN-V/experiments/col_128_col_128/experiments/col_128_col_128/output/network-snapshot-005120.pkl \
--num_videos 3125 \
--as_grids false \
--save_as_mp4 true \
--fps 25 \
--video_len 16 \
--batch_size 25 \
--outdir mostgan_col \
--truncation_psi 0.9