# MoStGAN-V

This repo is modified from https://github.com/xiaoqian-shen/MoStGAN-V for for training, sampling and evaluation for the endoscopy scenario.

> **MoStGAN-V: Video Generation with Temporal Motion Styles**, ***CVPR 2023***.
>
> Xiaoqian Shen, Xiang Li, Mohamed Elhoseiny
>
> <a href='https://arxiv.org/abs/2304.02777'><img src='https://img.shields.io/badge/arXiv-2304.02777-red'></a> <a href='https://xiaoqian-shen.github.io/MoStGAN-V'><img src='https://img.shields.io/badge/Project-Video-blue'></a>




## Installation

```
conda env create -f environment.yaml
```

And also make sure [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch#requirements) is runnable. 


## Training

```
sh train.sh
```
Note that `exp_suffix` and `dataset` should be identified in the script.
## Inference

+ evaluation

Please first generate the same number of videos as Endora and then use the same script used to evaluate the Endora.

+ generation

```
sh generation.sh
```
Note that `network_pkl` should be identified.
## Reference

This code is mainly built upon [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) and [StyleGAN-V](https://github.com/universome/stylegan-v) repositories.

Baseline codes are from [MoCoGAN-HD](https://github.com/snap-research/MoCoGAN-HD), [VideoGPT](https://github.com/wilson1yan/VideoGPT), [DIGAN](https://github.com/sihyun-yu/digan), [StyleGAN-V](https://github.com/universome/stylegan-v)

## Bibtex
```
@article{shen2023mostganv,
  author    = {Xiaoqian Shen and Xiang Li and Mohamed Elhoseiny},
  title     = {MoStGAN-V: Video Generation with Temporal Motion Styles},
  journal   = {arXiv preprint arXiv:2304.02777},
  year      = {2023},
}
```
