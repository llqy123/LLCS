<h2 align="center">
✨Learning Local and Consistent Semantics for Open-Vocabulary Dense Prediction
</h2>

This repository is the official implementation of [**Learning Local and Consistent Semantics for Open-Vocabulary Dense Prediction**](xxx) accepted by [**TCSVT 2026**](https://ieeexplore.ieee.org/xpl/mostRecentIssue.jsp?punumber=76).      
> Qiuyu Liang, Yongqiang Zhang*  

## 📍Installation
### Fine-tune CLIP and Open-Vocabulary Object Detection 
Our implementation follows the environment configuration of [CLIPSelf](https://github.com/wusize/CLIPSelf).
Please refer to the CLIPSelf repository for environment setup and dependency installation.


### Open-Vocabulary Semantic Segmentation 

Our implementation follows the environment configuration of [CAT-Seg](https://github.com/cvlab-kaist/CAT-Seg).
Please refer to the CAT-Seg repository for environment setup and dependency installation.


## 📦 Model Zoo

We provide F-ViT-based checkpoints with different EVA-CLIP backbones for **Open-Vocabulary Object Detection** on the OV-COCO and OV-LVIS benchmarks.

### OV-COCO

| Method              | Backbone |    Novel AP50   | Base AP50 | All AP50 |    Checkpoint   |
| :------------------ | :------: | :-------------: | :-------: | :------: | :-------------: |
| F-ViT + LLCS | ViT-B/16 | 42.8 |    54.9   |   51.7   | [Download](https://pan.baidu.com/s/1-kKMQVolsuC4w8JjNrGIuQ?pwd=LLCS) |
| F-ViT + LLCS | ViT-L/14 | 48.1 |    63.2   |   59.3   | [Download](https://pan.baidu.com/s/1Bq6B2mdTO_NNN9QuGSW8Fw?pwd=LLCS) |

### OV-LVIS

| Method              | Backbone |       mAP_r      | mAP_c | mAP_f |  mAP  |    Checkpoint   |
| :------------------ | :------: | :-------------: | :--: | :--: | :--: | :-------------: |
| F-ViT + LLCS | ViT-B/16 | 28.1 | 22.9 | 30.7 | 26.8 | [Download](https://pan.baidu.com/s/1Qp9MbXKjPq2uXTU6XOVKeA?pwd=LLCS) |
| F-ViT + LLCS | ViT-L/14 | 39.3 | 34.3 | 35.5 | 35.6 | [Download](https://pan.baidu.com/s/14kt3jEMBzHDrtS0U_VaUuw?pwd=LLCS) |

### OVSS

We also provide CAT-Seg-based checkpoints with different EVA-CLIP backbones for **Open-Vocabulary Semantic Segmentation**. 

| Method | Backbone | Training Set | ADE847 | Context459 | ADE150 | Context59 | VOC20 | VOC21 | Checkpoint | 
|:---|:---:|:---:|---:|---:|---:|---:|---:|---:|:---:| 
| CAT-Seg + LLCS | ViT-B/16 | COCO-Stuff | 15.4 | 23.0 | 37.5 | 61.1 | 97.0 | 81.1 | [Download](https://pan.baidu.com/s/1Pz2v6-jneoKa0VOyDVK5_g?pwd=LLCS) | 
| CAT-Seg + LLCS | ViT-L/14 | COCO-Stuff | 18.7 | 27.0 | 41.7 | 63.5 | 97.8 | 83.8 | [Download](https://pan.baidu.com/s/1AR1jSvVpgWO8MX7r6S86EA?pwd=LLCS) |

## 📚 Citation

If you find this project useful for your research, please use the following BibTeX entry.

```

```


## 🤝 Acknowledgement

This project builds on [OpenCLIP](https://github.com/mlfoundations/open_clip), [EVA-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP), [CLIPSelf](https://github.com/wusize/CLIPSelf), [CAT-Seg](https://github.com/cvlab-kaist/CAT-Seg), [DINO](https://github.com/facebookresearch/dino), [DINOv2](https://github.com/facebookresearch/dinov2), and [Segment Anything](https://github.com/facebookresearch/segment-anything).
