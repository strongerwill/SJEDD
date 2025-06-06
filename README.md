# Semantics-Oriented Multitask Learning for DeepFake Detection: A Joint Embedding Approach (SJEDD)

This repository contains the official PyTorch implementation of the paper **"[Semantics-Oriented Multitask Learning for DeepFake Detection: A Joint Embedding Approach](https://ieeexplore.ieee.org/document/11010889)"** by Mian Zou, Baosheng Yu, Yibing Zhan, Siwei Lyu, and Kede Ma.

☀️ If you find this work useful for your research, please kindly star our repo and cite our paper! ☀️

- [x] Release [arXiv paper](https://arxiv.org/abs/2408.16305)
- [x] Release inference codes
- [x] Release checkpoints 
- [x] Release datasets
- [x] Release training codes

## 📁 Datasets
Follow the links below to download the datasets (🛡️ Copyright of the datasets belongs to their original providers, and you may be asked to fill out some forms before downloading):

|  [FF++](https://github.com/ondyari/FaceForensics) | [CDF(v2)](https://github.com/yuezunli/celeb-deepfakeforensics)| [FSh](https://github.com/ondyari/FaceForensics/blob/master/dataset/FaceShifter/README.md) | [DF-1.0](https://github.com/EndlessSora/DeeperForensics-1.0/tree/master) | 
|:-:|:-:|:-:|:-:|
| [FFSC](https://github.com/MZMMSEC/SO-DFD) | [DiffusionFace (excluding the purely synthesized faces)](https://github.com/Rapisurazurite/DiffFace)| [DiFF (excluding the purely synthesized faces)](https://github.com/xaCheng1996/DiFF) |[DFDC (test set of the full version, not the Preview)](https://ai.meta.com/datasets/dfdc/) |

**Note**: For FF++ and FFSC, please download the full dataset for both training and testing. For the other datasets, if a separate test set is explicitly provided, please download the test set. Otherwise, if no specific split is mentioned, download the full dataset. We also provide [download link](https://pan.baidu.com/s/1Otk8pNiGVXeF5o7ZIjO1NA?pwd=t8d5) for Diffusion-based Face Swapping, i.e., DiffusionFace and DiFF, in which the face images have been organized and pre-processed for direct use in testing.

### Preprocessing (see [instructions](https://github.com/MZMMSEC/SJEDD/tree/main/preprocessing))

1) Extract the frames, and then detect and crop the faces (Optional for video datasets)

2) Rearrange the data for the test experiments


## 🚀 Quick Start

### 1. Installation of base reqiurements
 - python == 3.8
 - PyTorch == 1.13
 - Miniconda
 - CUDA == 11.7

### 2. Download the pretrained model and our model

|      Model       |    Training Dataset   |                                                        Download                                                                | |
|:----------------:|:----------------:|:-------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------:|
| SJEDD-FF++ | [FF++](https://github.com/ondyari/FaceForensics)  | [Google Drive](https://drive.google.com/file/d/1iXDzFrH4o3h4C7HA2jZRoxoxFr3G2Vly/view?usp=sharing) |✅|
| SJEDD-FFSC | [FFSC](https://github.com/MZMMSEC/SO-DFD)  | [Google Drive]() |⬜|

After downloading these checkpoints, put them into the folder ``pretrained``.

### 3. Inference on the test sets

```
CUDA_VISIBLE_DEVICES=5 python SJEDD_test.py --auc --test_log True --dataset [e.g., CDF, FSh, Deeper, DFDC, FFSC] --batch_size 1 --n_frames 64 --resume [path to checkpoints, e.g., ./pretrained/ckpt_best.pth]
```

### 4. Training (For reference)
```
CUDA_VISIBLE_DEVICES=4 python SJEDD_train_FFpp.py --name lr6e-7_lambdaInit_1.0_lambdaLr1e-3_aug0.3_bz32 --aug --aug_probs 0.3 \
--batch_size 32 --num_epoch 20 --output ./output/train_process/autoL-FFppHalf-SO-unnorm-noAnton \
--txt_path_train [path to train list txt file] \
--txt_path_val [path to val list txt file] \
--weighting_method auto-l --task binary \
--weight autol --gpu 0 --autol_lr 1e-3 --autol_init 1.0 --initial_lr 6e-7
```
