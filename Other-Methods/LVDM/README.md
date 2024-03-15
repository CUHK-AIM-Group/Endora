
<div align="center">

<h2> LVDM: <span style="font-size:12px">Latent Video Diffusion Models for High-Fidelity Long Video Generation </span> </h2> 

  <a href='https://arxiv.org/abs/2211.13221'><img src='https://img.shields.io/badge/ArXiv-2211.14758-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='https://yingqinghe.github.io/LVDM/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>


<div>
    <a href='https://github.com/YingqingHe' target='_blank'>Yingqing He <sup>1</sup> </a>&emsp;
    <a href='https://tianyu-yang.com/' target='_blank'>Tianyu Yang <sup>2</a>&emsp;
    <a href='https://yzhang2016.github.io/' target='_blank'>Yong Zhang <sup>2</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?hl=en&user=4oXBp9UAAAAJ&view_op=list_works&sortby=pubdate' target='_blank'>Ying Shan <sup>2</sup></a>&emsp;
    <a href='https://cqf.io/' target='_blank'>Qifeng Chen <sup>1</sup></a>&emsp; </br>
</div>
<br>
<div>
    <sup>1</sup> The Hong Kong University of Science and Technology &emsp; <sup>2</sup> Tencent AI Lab &emsp;
</div>
<br>
<br>

<b>TL;DR: An efficient video diffusion model that can:</b>  
1️⃣ conditionally generate videos based on input text;  
2️⃣ unconditionally generate videos with thousands of frames.

<br>
</div>


This repo is modified from https://github.com/YingqingHe/LVDM for training, sampling and evaluation for the endoscopy scenario.

---
## ⚙️ Setup

### Install Environment via Anaconda
```bash
conda create -n lvdm python=3.8.5
conda activate lvdm
pip install -r requirements.txt
```

## 💫 Training
<!-- tar -zxvf dataset/sky_timelapse.tar.gz -C /dataset/sky_timelapse -->
### Train video autoencoder
```
bash shellscripts/train_lvdm_videoae.sh 
```
- remember to set `PROJ_ROOT`, `EXPNAME`, `DATADIR`, and `CONFIG` to the endoscopy datasets (col, kvasir, and cholec).

### Train unconditional lvdm for short video generation
```
bash shellscripts/train_lvdm_short.sh
```
- remember to set `PROJ_ROOT`, `EXPNAME`, `DATADIR`, `AEPATH` and `CONFIG` to the endoscopy datasets (col, kvasir, and cholec).

---
## 💫 Inference 
- unconditional generation

```
bash shellscripts/sample_lvdm_short.sh
```

---
## 💫 Evaluation
Please first generate the same number of videos as Endora and then use the same script used to evaluate the Endora.

---

## 📃 Abstract
AI-generated content has attracted lots of attention recently, but photo-realistic video synthesis is still challenging. Although many attempts using GANs and autoregressive models have been made in this area, the visual quality and length of generated videos are far from satisfactory. Diffusion models have shown remarkable results recently but require significant computational resources. To address this, we introduce lightweight video diffusion models by leveraging a low-dimensional 3D latent space, significantly outperforming previous pixel-space video diffusion models under a limited computational budget. In addition, we propose hierarchical diffusion in the latent space such that longer videos with more than one thousand frames can be produced. To further overcome the performance degradation issue for long video generation, we propose conditional latent perturbation and unconditional guidance that effectively mitigate the accumulated errors during the extension of video length. Extensive experiments on small domain datasets of different categories suggest that our framework generates more realistic and longer videos than previous strong baselines. We additionally provide an extension to large-scale text-to-video generation to demonstrate the superiority of our work. Our code and models will be made publicly available.
<br>

## 🔮 Pipeline

<p align="center">
    <img src=assets/framework.jpg />
</p>

---
## 😉 Citation

```
@article{he2022lvdm,
      title={Latent Video Diffusion Models for High-Fidelity Long Video Generation}, 
      author={Yingqing He and Tianyu Yang and Yong Zhang and Ying Shan and Qifeng Chen},
      year={2022},
      eprint={2211.13221},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 🤗 Acknowledgements
We built our code partially based on [latent diffusion models](https://github.com/CompVis/latent-diffusion) and [TATS](https://github.com/SongweiGe/TATS). Thanks the authors for sharing their awesome codebases! We aslo adopt Xintao Wang's [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for upscaling our text-to-video generation results. Thanks for their wonderful work!