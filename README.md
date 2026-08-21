# LLCS

Official code for **Learning Local and Consistent Semantics for Open-Vocabulary Dense Prediction**.

This repository is adapted from [OpenCLIP v2.16.0](https://github.com/mlfoundations/open_clip/tree/v2.16.0) and adds LLCS training code for refining EVA-CLIP models with COCO region proposals, local semantic learning, and consistent semantic learning from visual foundation models.

## Installation

Create a Python environment and install the dependencies:

```bash
conda create -n llcs python=3.10 -y
conda activate llcs

pip install -r requirements.txt
pip install pycocotools panopticapi pillow numpy
```

Before running the scripts, make sure `src/` is visible to Python:

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH="$PWD\src;$env:PYTHONPATH"
```

## Repository Layout

```text
LLCS-main/
|-- metadata/
|   |-- coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy
|   |-- coco_panoptic_clip_hand_craft_EVACLIP_ViTL14x336.npy
|   |-- coco_pseudo_4764_clip_hand_craft_EVACLIP_ViTB16.npy
|   `-- coco_pseudo_4764_clip_hand_craft_EVACLIP_ViTL14x336.npy
|-- scripts/
|   |-- train_clipself+LLCS_coco_region_proposals_eva_vitb16_dino_b_8.sh
|   |-- train_clipself+LLCS_coco_region_proposals_eva_vitb16_dino_b_16.sh
|   |-- train_clipself+LLCS_coco_region_proposals_eva_vitb16_dinov2_b.sh
|   |-- train_clipself+LLCS_coco_region_proposals_eva_vitb16_sam_b.sh
|   |-- train_clipself+LLCS_coco_region_proposals_eva_vitl14_dinov2_l.sh
|   `-- train_clipself+LLCS_coco_region_proposals_eva_vitl14_sam_l.sh
`-- src/
    |-- open_clip/
    |-- segment_anything/
    `-- training/
```

## Data Preparation

The experiments use [COCO](https://cocodataset.org/#home) and [LVIS](https://www.lvisdataset.org/) style annotations. Organize the datasets as follows:

```text
LLCS-main/
|-- data/
|   |-- coco/
|   |   |-- annotations/
|   |   |   |-- instances_train2017.json
|   |   |   |-- panoptic_val2017.json
|   |   |   `-- panoptic_val2017/
|   |   |-- train2017/
|   |   |-- val2017/
|   |   |-- coco_pseudo_4764.json
|   |   `-- coco_proposals.json
|   `-- lvis_v1/
|       |-- annotations/
|       |   `-- lvis_v1_train.json
|       |-- train2017/
|       `-- val2017/
|-- metadata/
`-- checkpoints/
```

For LLCS with region proposals or RegionCLIP-style region-text pairs, download `coco_pseudo_4764.json` and `coco_proposals.json` from [Google Drive](https://drive.google.com/drive/folders/11zG4nJffm0MbvA0Ph19p6jvJFj6VwRAH?usp=sharing), then place them under `data/coco/`.

The provided `metadata/*.npy` files contain handcrafted text embeddings used by training and evaluation. Keep them under `metadata/`.

## Model Checkpoints

Download the EVA-CLIP checkpoints from [EVA-02-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP) and place them under `checkpoints/`:

```text
LLCS-main/
`-- checkpoints/
    |-- EVA02_CLIP_B_psz16_s8B.pt
    `-- EVA02_CLIP_L_336_psz14_s6B.pt
```

For SAM teachers, also place the SAM checkpoints under `checkpoints/` when using `--vfm-type sam-B` or `--vfm-type sam-L`:

```text
checkpoints/
|-- sam_vit_b_01ec64.pth
`-- sam_vit_l_0b3195.pth
```

DINO and DINOv2 teachers are loaded in `src/training/utils.py` through local Torch Hub repositories. Update the local paths in `build_vfm()` to match your machine, or switch the commented `torch.hub.load(..., source='github')` calls if you prefer downloading from GitHub.

## Training

Run one of the provided scripts. For example, to train EVA02-CLIP-B/16 with a DINOv2-B teacher:

```bash
bash scripts/train_clipself+LLCS_coco_region_proposals_eva_vitb16_dinov2_b.sh
```

Available scripts:

```text
scripts/train_clipself+LLCS_coco_region_proposals_eva_vitb16_dino_b_8.sh
scripts/train_clipself+LLCS_coco_region_proposals_eva_vitb16_dino_b_16.sh
scripts/train_clipself+LLCS_coco_region_proposals_eva_vitb16_dinov2_b.sh
scripts/train_clipself+LLCS_coco_region_proposals_eva_vitb16_sam_b.sh
scripts/train_clipself+LLCS_coco_region_proposals_eva_vitl14_dinov2_l.sh
scripts/train_clipself+LLCS_coco_region_proposals_eva_vitl14_sam_l.sh
```

The scripts use `torchrun` with 2 GPUs by default. Adjust `--nproc_per_node`, `--batch-size`, `--workers`, and `--master_port` according to your hardware.

## Key Arguments

- `--dataset-type`: training dataset type. Current LLCS scripts use `proposals_distill`.
- `--test-type`: evaluation dataset type. Current scripts use `coco_panoptic`.
- `--model`: EVA-CLIP model name, such as `EVA02-CLIP-B-16` or `EVA02-CLIP-L-14-336`.
- `--cache-dir`: path to the EVA-CLIP checkpoint file used by the scripts.
- `--train-data`: COCO proposal annotation file, usually `data/coco/coco_proposals.json`.
- `--val-data`: COCO panoptic validation annotation file.
- `--embed-path`: class text embeddings for evaluation.
- `--train-embed-path`: noun/text embeddings for LLCS training.
- `--lsl-weight`: weight for Local Semantic Learning.
- `--csl-weight`: weight for Consistent Semantic Learning.
- `--vfm-type`: VFM teacher type. Choices are `sam-B`, `sam-L`, `dinov2-L`, `dinov2-B`, `dino-B-8`, and `dino-B-16`.
- `--alpha`: student-teacher ensemble ratio used when saving/evaluating checkpoints. Values below `1.0` blend student and teacher weights.

## Outputs

Training logs and checkpoints are saved under `logs/<experiment-name>/` by default. Each script sets the experiment name with `--name`.

## Notes Before Uploading to GitHub

- Do not upload large datasets or model checkpoints. Keep `data/`, `checkpoints/`, `logs/`, and Python cache files out of version control.
- The current `setup.py` references `requirements-training.txt`. If you want editable installation with `pip install -e .`, add that file or update `setup.py`.
- The training entry imports `tools.k_means`. If that module is not part of your release, either add it or remove the unused import before publishing.
- If you publish pretrained or trained weights, add download links and checksums in this README.

## Acknowledgement

This project builds on [OpenCLIP](https://github.com/mlfoundations/open_clip), [EVA-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP), [DINO](https://github.com/facebookresearch/dino), [DINOv2](https://github.com/facebookresearch/dinov2), and [Segment Anything](https://github.com/facebookresearch/segment-anything).
