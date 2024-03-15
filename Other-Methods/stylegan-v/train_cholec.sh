CUDA_VISIBLE_DEVICES=6 python src/infra/launch.py \
hydra.run.dir=. \
exp_suffix=cholec_128 \
env=local \
dataset=cholec \
dataset.resolution=128 \
num_gpus=1