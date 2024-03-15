# MoStGAN-V

Official PyTorch implementation for the paper:

> **MoStGAN-V: Video Generation with Temporal Motion Styles**, ***CVPR 2023***.
>
> Xiaoqian Shen, Xiang Li, Mohamed Elhoseiny
>
> <a href='https://arxiv.org/abs/2304.02777'><img src='https://img.shields.io/badge/arXiv-2304.02777-red'></a> <a href='https://xiaoqian-shen.github.io/MoStGAN-V'><img src='https://img.shields.io/badge/Project-Video-blue'></a>

<div style="display: flex; flex-direction: row;">
  <div style="flex: 1;">
    <img src="assets/ffs.gif" alt="First GIF" style="width: 300pt;">
    <img src="assets/celebv.gif" alt="Second GIF" style="width: 300pt;">
  </div>
</div>
<div style="display: flex; flex-direction: row;">
  <div style="flex: 1;">
    <img src="assets/jelly.gif" alt="First GIF" style="width: 300pt;">
    <img src="assets/sky.gif" alt="Second GIF" style="width: 300pt;">
  </div>
</div>



## Installation

```
conda env create -f environment.yaml
```

And also make sure [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch#requirements) is runnable. 

## System requirements

4 32GB V100s are required, training time is approximately 2 days

## Data

+ [CelebV-HQ](https://celebv-hq.github.io)
+ [FaceForensics](https://github.com/ondyari/FaceForensics)

+ [SkyTimelapse](https://github.com/weixiong-ur/mdgan)
+ [RainbowJelly](https://www.youtube.com/watch?v=P8Bit37hlsQ)
+ [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

We follow the same procedure as [StyleGAN-V](https://github.com/universome/stylegan-v) to process all datasets

```
convert_videos_to_frames.py -s /path/to/source -t /path/to/target --video_ext mp4 --target_size 256
```

FaceForensics was preprocessed with `src/scripts/preprocess_ffs.py` to extract face crops, (result in a little bit unstable).

## Training

```
python src/infra/launch.py hydra.run.dir=. exp_suffix=my_experiment_name env=local dataset=ffs dataset.resolution=256 num_gpus=4
```

## Inference

+ evaluation

```
src/scripts/calc_metrics.py
```

+ generation

```
python src/scripts/generate.py --network_pkl /path/to/network-snapshot.pkl --num_videos 25 --as_grids true --save_as_mp4 true --fps 25 --video_len 128 --batch_size 25 --outdir /path/to/output/dir --truncation_psi 0.9
```
You can find the checkpoints from [here](https://drive.google.com/drive/folders/1ZlGmjRmjV4_ZzcU2t2RN0RdvFfeTniAW?usp=sharing)

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
