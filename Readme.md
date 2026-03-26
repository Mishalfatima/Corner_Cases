# Corner Cases: How Size and Position of Objects Challenge ImageNet-Trained Models

## Overview

This repository contains datasets, scripts, and experiments for the TMLR paper:  
**"Corner Cases: How Size and Position of Objects Challenge ImageNet-Trained Models"**  

We investigate corner cases in image classification — rare or challenging scenarios where standard ImageNet-trained models fail. We introduce **Hard Spurious ImageNet**, a dataset specifically designed to surface these failure modes by controlling object size and position.

---

## Getting Started

### Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/Mishalfatima/Corner_Cases.git
cd Corner_Cases
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch >= 1.8

---

## Dataset Generation

### Hard Spurious ImageNet

#### Step 1: Download ImageNet
Download the ImageNet dataset and place it under:
```
dataset/ImageNet/
```

#### Step 2: Inpaint Backgrounds
Remove objects from ImageNet images using inpainting. This can be done for both **val** and **train** splits:
```bash
# Validation split
python Inpaint-Anything/remove_anything_ImageNet.py \
    --pkl_file_path Inpaint-Anything/val_GT_bboxes.pkl \
    --imagenet_path dataset/ImageNet/val \
    --output_path dataset/Inpainted_ImageNet_GT/val

# Training split
python Inpaint-Anything/remove_anything_ImageNet.py \
    --pkl_file_path Inpaint-Anything/train_GT_bboxes.pkl \
    --imagenet_path dataset/ImageNet/train \
    --output_path dataset/Inpainted_ImageNet_GT/train
```

#### Step 3: Create Dataset

Run the dataset creation script for both splits. Two resizing options are available:

**Option A — Aspect-ratio preserving (short side resized):**
```bash
python create_dataset_preserve_AR.py --number 0
```

**Option B — Square resizing:**
```bash
python create_dataset.py --number 0
```

Both scripts support parallel processing across 10 jobs (job IDs 0–9). To run all jobs in parallel:
```bash
for i in {0..9}; do
    python create_dataset.py --number $i &
done
wait
```

### Script Arguments

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `--number` | int | ✅ | — | Job number (0–9) for parallel processing |
| `--random` | bool | ❌ | False | Random corner placement vs. fixed top-right |
| `--GT_bboxes` | str | ❌ | `val_GT_bboxes.pkl` | Path to GT bounding boxes pickle file |
| `--in_path` | str | ❌ | `Inpainted_ImageNet_GT/test/` | Path to inpainted images |
| `--images_path` | str | ❌ | `ImageNet/val/` | Path to original ImageNet images |
| `--output_path` | str | ❌ | `Hard-Spurious-ImageNet/test/` | Path to save generated dataset |

#### Step 4: Split into Train and Val

Once the dataset is generated, split the train files into train and val using the provided utility script. The script expects two text files listing the relative file paths for each split (one path per line, e.g. `/84/Group_3/n02113799/n02113799_1144_3_84.JPEG`):
```bash
python utils/move_files.py \
    --src_root dataset/Hard-Spurious-ImageNet/train \
    --dst_val dataset/Hard-Spurious-ImageNet/val \
    --train_list utils/train_files.txt \
    --val_list utils/val_files.txt
```

---

## Repository Structure
```
Corner_Cases/
├── Inpaint-Anything/
│   ├── remove_anything_ImageNet.py
│   ├── train_GT_bboxes.pkl
│   └── val_GT_bboxes.pkl
├── utils/
│   ├── move_files.py               # Train/val splitting utility
│   ├── train_files.txt             # List of train file paths
│   └── val_files.txt               # List of val file paths
├── create_dataset.py               # Square resizing
├── create_dataset_preserve_AR.py   # AR-preserving resizing
├── requirements.txt
└── README.md
```

---

## Citation

If you use this work, please cite:
```bibtex
@article{
fatima2025corner,
title={Corner Cases: How Size and Position of Objects Challenge ImageNet-Trained Models},
author={Mishal Fatima and Steffen Jung and Margret Keuper},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=Yqf2BhqfyZ},
note={}
}
```