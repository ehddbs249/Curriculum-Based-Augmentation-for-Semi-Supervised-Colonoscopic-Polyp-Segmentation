# Curriculum-Based Augmentation for Semi-Supervised Colonoscopic Polyp Segmentation

Supplementary code for the manuscript **Curriculum-Based Augmentation for Semi-Supervised Colonoscopic Polyp Segmentation.**

## Overview

This repository provides the Dice-based curriculum method for semi-supervised polyp segmentation built on AD-MT. Training begins with **Target Copy-Paste (TCP)** and switches to **CutMix** when the Student's Dice score on labeled training images reaches a plateau.

## Repository Status

- Includes Dice-based training, Kvasir-SEG testing, and external-dataset evaluation.
- Dataset images, original split lists, and pretrained weights are not included.

## Key Contributions

- **Target Copy-Paste:** transfers complete polyp regions from labeled images to unlabeled images.
- **Color and size constraints:** selects source polyps using masked RGB similarity and foreground area.
- **Dice-based curriculum:** progresses from TCP to CutMix according to labeled-training performance.

TCP compares the source polyp's mean RGB vector with the target image's mean RGB vector. The intended settings are color similarity >= 0.6 and polyp area between 5% and 45% of the image.

## Model Architecture

The framework contains **one Student and two alternating EMA Teachers**, all using a 2D **U-Net** with RGB input and two output classes: background and polyp. Teacher predictions provide pseudo-labels through entropy-weighted ensembling and AD-MT's Conflict-Combating Module (CCM).

The backbone and AD-MT helpers are in [`code/networks/`](code/networks/) and [`code/train_utils.py`](code/train_utils.py).

## Dataset

The manuscript uses [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/): **880 training images and 120 test images**.

| Labeled ratio | Labeled images | Unlabeled images |
|---|---:|---:|
| 5% | 44 | 836 |
| 10% | 88 | 792 |
| 15% | 132 | 748 |

Cross-dataset evaluation uses **612 CVC-ClinicDB images** and **300 selected CVC-ColonDB images**, without fine-tuning.

```text
data/Kvasir/
|-- train.list
|-- test.list
|-- train/images/    and train/masks/
`-- test/images/     and test/masks/
```

List files contain image IDs without extensions. Images and masks must be matching PNG files, with mask values 0 for background and 255 for polyp. The first `labeled_num` entries of `train.list` form the labeled subset, so retain the original ordering. The current loader expects mask files for all training images. Keep `Kvasir` in the dataset path for the RGB loader branch.

## Training Details

| Setting | Value |
|---|---|
| Input size | 256 x 256 |
| Training iterations | 30,000 |
| Batch size | 24: 12 labeled + 12 unlabeled |
| Optimizer | SGD, initial learning rate 0.01 |
| Supervised loss | Cross-entropy + Dice, averaged |
| Unsupervised loss | Confidence-filtered Dice |
| TCP color / area constraints | 0.6 / 5%-45% |
| Dice transition settings | Minimum improvement 0.01; patience 5 evaluations |

The manuscript reports Python 3.6.13, PyTorch 1.10.1, CUDA 11.3, and an RTX 3090. Training and evaluation require CUDA and the packages imported by the scripts, including torchvision, NumPy, SciPy, pandas, Pillow, h5py, PyYAML, matplotlib, scikit-image, MedPy, TensorBoard, and tqdm.

Run from the repository root. Use [`cfgs/config_2d_kvasir_5percent.yml`](cfgs/config_2d_kvasir_5percent.yml), set its `res_path` to `./results`, and configure other experiment parameters in the YAML. The command explicitly overrides its baseline color threshold with 0.6.

```sh
python code/train_post_2d_aut_Dice.py --cfg config_2d_kvasir_5percent.yml --root_path ./data/Kvasir --model unet --labeled_num 44 --color_threshold 0.6 --exp Kvasir/Dice_5p
```

For 10% or 15%, use `--labeled_num 88` or `132` and a separate experiment name. Configure the seed in the YAML for each repeat. Curriculum transitions and best-checkpoint selection use **labeled-training Dice**; final test scores are computed separately.

## Evaluation Metrics

The primary metrics are **Mean Dice** and **Mean IoU (Jaccard)**, calculated per image and averaged across the test set. The scripts also report HD95 and average surface distance.

**Kvasir-SEG:**

```sh
python code/test_kvasir_2d.py --root_path ./data/Kvasir --model_path ./results/Kvasir/Dice_5p_44_labeled/unet/unet_best_stu_model.pth --save_csv
```

**External evaluation:**

```sh
python code/test_cross_polyp.py --dataset_path ./data/CVC-ClinicDB --dataset_name CVC-ClinicDB --model_path ./results/Kvasir/Dice_5p_44_labeled/unet/unet_best_stu_model.pth --save_csv
```

For ColonDB, change the dataset path and name to the selected ColonDB set. Use the manuscript's image selection for a direct comparison. External folders may use `images/` + `masks/` or `im_resize/` + `gt_resize/`. The evaluator uses all discovered pairs by default.

Inference input size is 256 x 256; predictions are resized back to the loaded mask dimensions for scoring. Provide masks at 256 x 256 when evaluating on that grid.

---

Built on [AD-MT: Alternate Diverse Teaching for Semi-supervised Medical Image Segmentation](https://github.com/zhenzhao/AD-MT) (ECCV 2024). Please cite the original method and dataset publications, and retain the source copyright/license notices.
