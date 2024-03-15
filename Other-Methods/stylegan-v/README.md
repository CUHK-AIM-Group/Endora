# StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2

This repo is modified from https://github.com/universome/stylegan-v for training, sampling and evaluation for the endoscopy scenario.

### [CVPR 2022] Official pytorch implementation
[[Project website]](https://universome.github.io/stylegan-v)
[[Paper]](https://kaust-cair.s3.amazonaws.com/stylegan-v/stylegan-v-paper.pdf)

## Installation
To install and activate the environment, run the following command:
```
conda env create -f environment.yaml -p env
conda activate ./env
```
For clip editing, you will need to install [StyleCLIP](https://github.com/orpatashnik/StyleCLIP) and `clip`.
This repo is built on top of [INR-GAN](https://github.com/universome/inr-gan), so make sure that it runs on your system.

If you have Ampere GPUs (A6000, A100 or RTX-3090), then use `environment-ampere.yaml` instead because it is based CUDA 11 and newer pytorch versions.

## System requirements

Our codebase uses the same system requirements as StyleGAN2-ADA: see them [here](https://github.com/NVlabs/stylegan2-ada-pytorch#requirements).
We trained all the 256x256 models on 4 V100s with 32 GB each for ~2 days.
It is very similar in training time to [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) (even a bit faster).

## Training

### Training StyleGAN-V
To train on FaceForensics 256x256, run:
```
sh train_col.sh / sh train_kvasir.sh / sh train_cholec.sh
```
for training col, kvasir, or cholec endoscopy datasets.

If you do not want `hydra` to create some log directories (typically, you don't), add the following arguments: `hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled`.

In case [slurm](https://slurm.schedmd.com/documentation.html) is installed on your system, you can submit the slurm job with the above training by adding `slurm=true` parameter.
Sbatch arguments are specified in `configs/infra.yaml`, you can update them with your required ones.
Also note that you can create your own environment in `configs/env`.

On older GPUs (non V100 and newer), custom CUDA kernels (bias_act and upfirdn2n) might fail to compile. The following two lines can help:
```
export TORCH_CUDA_ARCH_LIST="7.0"
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
```

### Inference
To sample from the model, launch the following command:
```
sh generate.sh
```
Note that `--network_pkl` should be specified.


## Evaluation
Please first generate the same number of videos as Endora and then use the same script used to evaluate the Endora.

## License
This repo is built on top of [INR-GAN](https://github.com/universome/inr-gan), which is likely to be restricted by the [NVidia license](https://nvlabs.github.io/stylegan2-ada-pytorch/license.html) since it's built on top of [StyleGAN2-ADA](https://github.com/nvlabs/stylegan2-ada).
If that's the case, then this repo is also restricted by it.


## Bibtex
```
@misc{stylegan_v,
    title={StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2},
    author={Ivan Skorokhodov and Sergey Tulyakov and Mohamed Elhoseiny},
    journal={arXiv preprint arXiv:2112.14683},
    year={2021}
}

@inproceedings{digan,
    title={Generating Videos with Dynamics-aware Implicit Generative Adversarial Networks},
    author={Sihyun Yu and Jihoon Tack and Sangwoo Mo and Hyunsu Kim and Junho Kim and Jung-Woo Ha and Jinwoo Shin},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=Czsdv-S4-w9}
}
```
