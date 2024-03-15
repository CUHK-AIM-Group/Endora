CUDA_VISIBLE_DEVICES=7 python src/infra/launch.py \
hydra.run.dir=. \
exp_suffix=kvasir_128 \
env=local \
dataset=kvasir \
dataset.resolution=128 \
num_gpus=1