CUDA_VISIBLE_DEVICES=7 python src/infra/launch.py \
hydra.run.dir=. \
exp_suffix=col_128 \
env=local \
dataset=col \
dataset.resolution=128 \
num_gpus=1