# SJEDD

This repository contains the official PyTorch implementation of the paper **"[Semantics-Oriented Multitask Learning for DeepFake Detection: A Joint Embedding Approach](https://ieeexplore.ieee.org/document/11010889)"** by Mian Zou, Baosheng Yu, Yibing Zhan, Siwei Lyu, and Kede Ma.

☀️ If you find this work useful for your research, please kindly star our repo and cite our paper! ☀️

### TODO
We are working hard on the following items.

- [x] Release [arXiv paper](https://arxiv.org/abs/2408.16305)
- [ ] Release training codes
- [x] Release inference codes
- [ ] Release checkpoints 
- [ ] Release datasets

## 📁 Datasets
### 1. Preprocessing 

## 🚀 Quick Start

### 1. Installation of base reqiurements
 - python == 3.8
 - PyTorch == 1.13
 - Miniconda
 - CUDA == 11.7

### 2. Download the pretrained model and our model

|      Model       |                                                               Download                                                                | |
|:----------------:|:-------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------:|
| CLIP-pretrained | [Google Drive]() |⬜|
| SJEDD model (FF++)    | [Google Drive]() |⬜|
| SJEDD model (FFSC)    | [Google Drive]() |⬜|

After downloading these checkpoints, put them into the folder ``pretrained``.

### 3. Inference on the test sets

```
CUDA_VISIBLE_DEVICES=5 python SJEDD_test.py --auc --test_log True --dataset CDF --batch_size 1 --n_frames 64 --resume [path to checkpoints, e.g., ./pretrained/ckpt_best.pth]
```



## Citation
If you find this repository useful in your research, please consider citing the following paper:
```
@article{zou2025sjedd,
  title={Semantics-oriented multitask learning for DeepFake detection: A joint embedding approach},
  author={Zou, Mian and Yu, Baosheng and Zhan, Yibing and Lyu, Siwei and Ma, Kede},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025}
}
```
