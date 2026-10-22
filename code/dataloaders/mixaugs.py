import numpy as np
import random
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


# # # # # # # # # # # # # # # # # # # # #
# # 0. random box
# # # # # # # # # # # # # # # # # # # # #
def rand_bbox(size, lam=None):
    # past implementation
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception
    B = size[0]

    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W/8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H/8), high=H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)


    return bbx1, bby1, bbx2, bby2


# # # # # # # # # # # # # # # # # # # # #
# # 1. cutmix for 2d
# # # # # # # # # # # # # # # # # # # # #
# def cut_mix(unlabeled_image, unlabeled_mask, unlabeled_logits):
#     mix_unlabeled_image = unlabeled_image.clone()
#     mix_unlabeled_target = unlabeled_mask.clone()
#     mix_unlabeled_logits = unlabeled_logits.clone()

#     # get the random mixing objects
#     u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
#     # print(u_rand_index)

#     # get box
#     u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))

#     # cut & paste
#     for i in range(0, mix_unlabeled_image.shape[0]):
#         mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
#             unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
#         # label is of 3 dimensions
# #         mix_unlabeled_target[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
# #             unlabeled_mask[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
#         mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
#             unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

#         mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
#             unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

#     del unlabeled_image, unlabeled_mask, unlabeled_logits

#     return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits


def cut_mix(unlabeled_image, unlabeled_mask, unlabeled_logits, unlabeled_conflict=None):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    if unlabeled_conflict is not None:
        mix_unlabeled_conflict = unlabeled_conflict.clone()

    # get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
    # print(u_rand_index)

    # get box
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))

    # cut & paste
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        # label is of 3 dimensions
#         mix_unlabeled_target[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
#             unlabeled_mask[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        if unlabeled_conflict is not None:
            mix_unlabeled_conflict[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                unlabeled_conflict[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    if unlabeled_conflict is not None:
        del unlabeled_image, unlabeled_mask, unlabeled_logits, unlabeled_conflict
        return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits, mix_unlabeled_conflict

    del unlabeled_image, unlabeled_mask, unlabeled_logits
    return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits


# # # # # # # # # # # # # # # # # # # # #
# # 1.5 copy-paste for AD-MT integration
# # # # # # # # # # # # # # # # # # # # #

def copy_paste(img_ulb, pseudo_outputs, pseudo_logits, img_lb, target_lb):
    """
    Simple Copy-Paste augmentation for AD-MT integration
    Copies objects from labeled data to unlabeled data

    NOTE: Probability control should be done by the caller.
          This function always applies copy-paste to all samples.

    Args:
        img_ulb: unlabeled images [B, C, H, W]
        pseudo_outputs: pseudo labels [B, H, W]
        pseudo_logits: pseudo confidence [B, H, W]
        img_lb: labeled images [B, C, H, W]
        target_lb: ground truth labels [B, H, W]

    Returns:
        augmented (img_ulb, pseudo_outputs, pseudo_logits)
    """
    batch_size = img_ulb.shape[0]

    # Clone inputs to avoid modifying originals
    aug_img_ulb = img_ulb.clone()
    aug_pseudo_outputs = pseudo_outputs.clone()
    aug_pseudo_logits = pseudo_logits.clone()

    for i in range(batch_size):
        # Use the provided labeled sample (already filtered by size/color conditions externally)
        # Note: img_lb.shape[0] is typically 1 when called from simple_color_copy_paste/conflict_based_augmentation
        j = 0

        # Get object mask from labeled data (exclude background class 0)
        obj_mask = (target_lb[j] > 0).float()

        if obj_mask.sum() > 100:  # Minimum object size threshold (safety check)
            # Copy object pixels from labeled to unlabeled image
            aug_img_ulb[i] = aug_img_ulb[i] * (1 - obj_mask.unsqueeze(0)) + img_lb[j] * obj_mask.unsqueeze(0)

            # Update pseudo labels with ground truth labels in pasted regions
            aug_pseudo_outputs[i] = aug_pseudo_outputs[i] * (1 - obj_mask).long() + target_lb[j] * obj_mask.long()

            # Set high confidence for pasted regions (reliable ground truth)
            aug_pseudo_logits[i] = aug_pseudo_logits[i] * (1 - obj_mask) + obj_mask * 0.99

    return aug_img_ulb, aug_pseudo_outputs, aug_pseudo_logits


# # # # # # # # # # # # # # # # # # # # #
# # 2. copy-paste for 2d (based on Google Research paper)
# # # # # # # # # # # # # # # # # # # # #

def copy_paste_batch(unlabeled_image, unlabeled_mask, unlabeled_logits=None, unlabeled_conflict=None,
                    num_classes=4, paste_prob=0.5, blur_sigma=1.0, use_blending=False):
    """
    Apply Copy-Paste augmentation to a batch of images
    Based on "Simple Copy-Paste is a Strong Data Augmentation Method"

    Args:
        unlabeled_image: tensor [B, C, H, W]
        unlabeled_mask: tensor [B, H, W]
        unlabeled_logits: tensor [B, num_classes, H, W] (optional)
        unlabeled_conflict: tensor [B, H, W] (optional)
        num_classes: number of segmentation classes
        paste_prob: probability of pasting each object
        blur_sigma: gaussian blur sigma
        use_blending: whether to apply gaussian blending

    Returns:
        augmented batch tensors
    """

    batch_size = unlabeled_image.shape[0]

    # Clone tensors
    aug_image = unlabeled_image.clone()
    aug_mask = unlabeled_mask.clone()
    if unlabeled_logits is not None:
        aug_logits = unlabeled_logits.clone()
    if unlabeled_conflict is not None:
        aug_conflict = unlabeled_conflict.clone()

    # Create random pairs for copy-paste
    source_indices = torch.randperm(batch_size)

    for i in range(batch_size):
        source_idx = source_indices[i]
        target_idx = i

        if source_idx == target_idx:
            continue  # Skip self-pairing

        # Find unique objects in source mask (excluding background class 0)
        source_mask = unlabeled_mask[source_idx]
        target_mask = unlabeled_mask[target_idx]
        unique_classes = torch.unique(source_mask)
        paste_classes = [cls for cls in unique_classes if cls > 0]  # exclude background

        if len(paste_classes) == 0:
            continue

        # Randomly select subset of objects to paste (following paper's approach)
        num_to_paste = random.randint(1, len(paste_classes))
        selected_classes = random.sample(paste_classes, num_to_paste)

        for cls in selected_classes:
            # Skip with paste_prob
            if random.random() > paste_prob:
                continue

            # Create binary mask for current object
            object_mask = (source_mask == cls).float()  # [H, W]

            if object_mask.sum() < 50:  # Skip very small objects
                continue

            # Optional: Apply gaussian blending at boundaries
            if use_blending:
                # Convert to numpy for gaussian_filter
                object_mask_np = object_mask.cpu().numpy()
                blurred_mask = gaussian_filter(object_mask_np, sigma=blur_sigma)
                object_mask = torch.from_numpy(blurred_mask).to(object_mask.device)

            # Paste image using the mask
            for c in range(aug_image.shape[1]):  # channels
                aug_image[target_idx, c] = (
                    aug_image[target_idx, c] * (1 - object_mask) +
                    unlabeled_image[source_idx, c] * object_mask
                )

            # Paste mask
            pasted_mask = (object_mask > 0.5)
            if pasted_mask.sum() > 0:
                aug_mask[target_idx] = torch.where(
                    pasted_mask,
                    source_mask,
                    aug_mask[target_idx]
                )

            # Handle logits if provided - copy corresponding logits
            if unlabeled_logits is not None:
                pasted_mask = (object_mask > 0.5)
                if pasted_mask.sum() > 0:
                    for c in range(num_classes):
                        aug_logits[target_idx, c] = torch.where(
                            pasted_mask,
                            unlabeled_logits[source_idx, c],
                            aug_logits[target_idx, c]
                        )

            # Handle conflict if provided
            if unlabeled_conflict is not None:
                pasted_mask = (object_mask > 0.5)
                if pasted_mask.sum() > 0:
                    aug_conflict[target_idx] = torch.where(
                        pasted_mask,
                        unlabeled_conflict[source_idx],
                        aug_conflict[target_idx]
                    )

    # Clean up memory
    del unlabeled_image, unlabeled_mask
    if unlabeled_logits is not None:
        del unlabeled_logits
        if unlabeled_conflict is not None:
            del unlabeled_conflict
            return aug_image, aug_mask, aug_logits, aug_conflict
        return aug_image, aug_mask, aug_logits

    if unlabeled_conflict is not None:
        del unlabeled_conflict
        return aug_image, aug_mask, aug_conflict

    return aug_image, aug_mask





# # # # # # # # # # # # # # # # # # # # #
# # 4. cutmix for 3d
# # # # # # # # # # # # # # # # # # # # #

def rand_bbox_3d(size, lam=None):
    # img: B x C x H x W x D, lb: B x H x W x D
    if len(size) == 5:
        W = size[2]
        H = size[3]
        D = size[4]
    elif len(size) == 4:
        W = size[1]
        H = size[2]
        D = size[3]
    else:
        raise Exception
    B = size[0]

    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cut_d = int(D * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W/8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H/8), high=H)
    cz = np.random.randint(size=[B, ], low=int(D/8), high=D)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbz1 = np.clip(cz - cut_d // 2, 0, D)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    bbz2 = np.clip(cz + cut_d // 2, 0, D)


    return bbx1, bby1, bbz1, bbx2, bby2, bbz2



# def cut_mix_3d(unlabeled_image, unlabeled_mask, unlabeled_logits):
#     mix_unlabeled_image = unlabeled_image.clone()
#     mix_unlabeled_target = unlabeled_mask.clone()
#     mix_unlabeled_logits = unlabeled_logits.clone()

#     # get the random mixing objects
#     u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]

#     # get box
#     # img: B x C x H x W x D, lb: B x H x W x D
#     u_bbx1, u_bby1, u_bbz1, u_bbx2, u_bby2, u_bbz2 = rand_bbox_3d(unlabeled_image.size(), lam=np.random.beta(4, 4))

#     # cut & paste
#     for i in range(0, mix_unlabeled_image.shape[0]):
#         mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
#             unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

#         mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
#             unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

#         mix_unlabeled_logits[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
#             unlabeled_logits[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

#     del unlabeled_image, unlabeled_mask, unlabeled_logits

#     return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits


def cut_mix_3d(unlabeled_image, unlabeled_mask, unlabeled_logits, unlabeled_conflict=None):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    if unlabeled_conflict is not None:
        mix_unlabeled_conflict = unlabeled_conflict.clone()

    # get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]

    # get box
    # img: B x C x H x W x D, lb: B x H x W x D
    u_bbx1, u_bby1, u_bbz1, u_bbx2, u_bby2, u_bbz2 = rand_bbox_3d(unlabeled_image.size(), lam=np.random.beta(4, 4))

    # cut & paste
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

        mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

        mix_unlabeled_logits[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_logits[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

        if unlabeled_conflict is not None:
            mix_unlabeled_conflict[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
                unlabeled_conflict[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

    if unlabeled_conflict is not None:
        del unlabeled_image, unlabeled_mask, unlabeled_logits, unlabeled_conflict
        return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits, mix_unlabeled_conflict

    del unlabeled_image, unlabeled_mask, unlabeled_logits
    return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits


def copy_paste_single_labeled_polyp(unlabeled_image, unlabeled_mask, unlabeled_logits, unlabeled_conflict,
                                    labeled_image, labeled_mask,
                                    num_classes=4, paste_prob=0.5, blur_sigma=1.0, use_blending=False):
    """
    Copy a single labeled polyp to multiple unlabeled images

    Args:
        unlabeled_image: tensor [B, C, H, W]
        unlabeled_mask: tensor [B, H, W]
        unlabeled_logits: tensor [B, num_classes, H, W]
        unlabeled_conflict: tensor [B, H, W]
        labeled_image: tensor [1, C, H, W]
        labeled_mask: tensor [1, H, W]
        paste_prob: probability of pasting to each unlabeled image

    Returns:
        augmented batch tensors
    """

    batch_size = unlabeled_image.shape[0]

    # Clone tensors
    aug_image = unlabeled_image.clone()
    aug_mask = unlabeled_mask.clone()
    if unlabeled_logits is not None:
        aug_logits = unlabeled_logits.clone()
    if unlabeled_conflict is not None:
        aug_conflict = unlabeled_conflict.clone()

    # Get object mask from labeled data (exclude background class 0)
    source_mask = labeled_mask[0]
    unique_classes = torch.unique(source_mask)
    paste_classes = [cls for cls in unique_classes if cls > 0]  # exclude background

    if len(paste_classes) == 0:
        # No objects to paste
        if unlabeled_logits is not None:
            if unlabeled_conflict is not None:
                return aug_image, aug_mask, aug_logits, aug_conflict
            return aug_image, aug_mask, aug_logits

        if unlabeled_conflict is not None:
            return aug_image, aug_mask, aug_conflict

        return aug_image, aug_mask

    # For each unlabeled image
    for target_idx in range(batch_size):
        # Skip with paste_prob
        if random.random() > paste_prob:
            continue

        # Paste all objects from labeled image
        for cls in paste_classes:
            # Create binary mask for current object
            object_mask = (source_mask == cls).float()  # [H, W]

            if object_mask.sum() < 50:  # Skip very small objects
                continue

            # Optional: Apply gaussian blending at boundaries
            if use_blending:
                # Convert to numpy for gaussian_filter
                object_mask_np = object_mask.cpu().numpy()
                blurred_mask = gaussian_filter(object_mask_np, sigma=blur_sigma)
                object_mask = torch.from_numpy(blurred_mask).to(object_mask.device)

            # Paste image using the mask
            for c in range(aug_image.shape[1]):  # channels
                aug_image[target_idx, c] = (
                    aug_image[target_idx, c] * (1 - object_mask) +
                    labeled_image[0, c] * object_mask
                )

            # Paste mask
            pasted_mask = (object_mask > 0.5)
            if pasted_mask.sum() > 0:
                aug_mask[target_idx] = torch.where(
                    pasted_mask,
                    source_mask,
                    aug_mask[target_idx]
                )

            # Handle logits if provided - copy corresponding logits
            if unlabeled_logits is not None:
                pasted_mask = (object_mask > 0.5)
                if pasted_mask.sum() > 0:
                    for c in range(num_classes):
                        aug_logits[target_idx, c] = torch.where(
                            pasted_mask,
                            unlabeled_logits[source_idx, c],
                            aug_logits[target_idx, c]
                        )

            # Handle conflict if provided
            if unlabeled_conflict is not None:
                pasted_mask = (object_mask > 0.5)
                if pasted_mask.sum() > 0:
                    aug_conflict[target_idx] = torch.where(
                        pasted_mask,
                        unlabeled_conflict[source_idx],
                        aug_conflict[target_idx]
                    )

    # Clean up memory
    del unlabeled_image, unlabeled_mask
    if unlabeled_logits is not None:
        del unlabeled_logits
        if unlabeled_conflict is not None:
            del unlabeled_conflict
            return aug_image, aug_mask, aug_logits, aug_conflict
        return aug_image, aug_mask, aug_logits

    if unlabeled_conflict is not None:
        del unlabeled_conflict
        return aug_image, aug_mask, aug_conflict

    return aug_image, aug_mask



# # # # # # # # # # # # # # # # # # # # #
# # 5. Advanced Copy-Paste with Conditions (EXP6)
# # # # # # # # # # # # # # # # # # # # #

def simple_color_copy_paste(img_ulb, pseudo_outputs, pseudo_logits, img_lb, target_lb,
                           paste_prob=1.0, color_threshold=0.6,
                           min_size_ratio=0.05, max_size_ratio=0.15, iter_num=0):
    """
    색상 및 크기 조건부 Copy-Paste (EXP6)

    조건:
    1. 용종 크기: 이미지의 5%~45% (3277~29491 픽셀 at 256x256)
    2. 색상 유사도: color_threshold 이상

    조건 불충족 시 배치 내 모든 라벨 데이터를 시도
    """
    batch_size = img_ulb.shape[0]
    lb_batch_size = img_lb.shape[0]
    color_matches = 0

    # 이미지 크기 기반 픽셀 수 계산 (256x256 = 65536)
    _, _, H, W = img_ulb.shape
    total_pixels = H * W
    min_pixels = int(total_pixels * min_size_ratio)  # 5% = 3277
    max_pixels = int(total_pixels * max_size_ratio)  # 45% = 29491

    for i in range(batch_size):
        if np.random.random() < paste_prob:
            # RGB 채널별 평균 벡터 [C]
            ulb_mean_rgb = img_ulb[i].mean(dim=(1, 2))  # [C]

            # 배치 내 모든 라벨 데이터를 랜덤 순서로 시도
            indices = list(range(lb_batch_size))
            np.random.shuffle(indices)

            for j in indices:
                # 조건 1: 용종 크기 확인 (5%~45%)
                obj_mask = (target_lb[j] > 0)
                polyp_size = obj_mask.sum().item()

                if polyp_size < min_pixels or polyp_size > max_pixels:
                    # 🔥 DEBUG: 크기 조건 불충족 시에도 로깅 (100 iteration마다)
                    if iter_num % 100 == 0 and iter_num > 0:
                        print(f"[DEBUG Stage1 SKIP] iter={iter_num}, img={i}, lb={j}, "
                              f"polyp_size={polyp_size} ({polyp_size/total_pixels*100:.1f}%), "
                              f"min={min_pixels} ({min_size_ratio*100:.0f}%), max={max_pixels} ({max_size_ratio*100:.0f}%), "
                              f"REASON=size_out_of_range")
                    continue  # 크기 조건 불충족 -> 다음 라벨로

                # 조건 2: 색상 유사도 확인 (RGB L2 거리 기반)
                lb_mean_rgb = img_lb[j].mean(dim=(1, 2))  # [C]
                color_dist = torch.norm(ulb_mean_rgb - lb_mean_rgb).item() / np.sqrt(3)  # 0~1 정규화
                color_similarity = 1.0 - color_dist

                if color_similarity < color_threshold:
                    if iter_num % 100 == 0 and iter_num > 0:
                        print(f"[DEBUG Stage1 SKIP] iter={iter_num}, img={i}, lb={j}, "
                              f"polyp_size={polyp_size} ({polyp_size/total_pixels*100:.1f}%), "
                              f"color_sim={color_similarity:.3f}, threshold={color_threshold}, "
                              f"REASON=color_mismatch")
                    continue  # 색상 유사도 불충족 -> 다음 라벨로

                # 🔥 DEBUG: 두 조건 모두 충족 시 로깅 (100 iteration마다)
                if iter_num % 100 == 0 and iter_num > 0:
                    print(f"[DEBUG Stage1 APPLY] iter={iter_num}, img={i}, lb={j}, "
                          f"polyp_size={polyp_size} ({polyp_size/total_pixels*100:.1f}%), "
                          f"min={min_pixels} ({min_size_ratio*100:.0f}%), max={max_pixels} ({max_size_ratio*100:.0f}%), "
                          f"color_sim={color_similarity:.3f}")

                # 두 조건 모두 충족 -> Copy-Paste 실행
                aug_img, aug_pseudo_out, aug_pseudo_log = copy_paste(
                    img_ulb[i:i+1], pseudo_outputs[i:i+1], pseudo_logits[i:i+1],
                    img_lb[j:j+1], target_lb[j:j+1]
                )

                img_ulb[i] = aug_img[0]
                pseudo_outputs[i] = aug_pseudo_out[0]
                pseudo_logits[i] = aug_pseudo_log[0]
                color_matches += 1
                break  # 성공 -> 다음 언라벨 이미지로

    return img_ulb, pseudo_outputs, pseudo_logits, color_matches


def conflict_based_augmentation(img_ulb_w, pseudo_outputs, pseudo_logits,
                                 img_lb_w, target_lb, mtx_bool_conflict,
                                 color_threshold=0.6, min_size_ratio=0.05, max_size_ratio=0.45,
                                 iter_num=0, epoch_threshold=None):
    """
    이미지별 Conflict 기반 적응적 증강 (EXP6_CONFLICT_EPOCH)

    Stage 2 (CutMix 단계)에서 사용:
    - 이미지별 conflict가 임계값보다 높으면 → Copy-Paste (어려운 이미지 보호)
    - 이미지별 conflict가 임계값보다 낮으면 → CutMix (쉬운 이미지는 도전적 증강)

    Args:
        epoch_threshold: 에폭 평균 conflict (None이면 배치 평균 사용)

    Returns:
        img_ulb_s: 증강된 이미지
        pseudo_outputs: 업데이트된 pseudo label
        pseudo_logits: 업데이트된 confidence
        mtx_bool_conflict: 업데이트된 conflict map (CutMix 적용 시)
        stats: 통계 딕셔너리 (cp_count, cutmix_count, batch_avg_conflict, threshold_used, per_image_conflicts)
    """
    batch_size = img_ulb_w.shape[0]
    lb_batch_size = img_lb_w.shape[0]
    _, _, H, W = img_ulb_w.shape
    total_pixels = H * W

    # 이미지별 conflict 계산
    per_image_conflicts = []
    for i in range(batch_size):
        img_conflict = mtx_bool_conflict[i].float().sum().item()
        img_conflict_ratio = img_conflict / total_pixels
        per_image_conflicts.append(img_conflict_ratio)

    # 배치 평균 conflict
    batch_avg_conflict = np.mean(per_image_conflicts)

    # 🔥 EXP6_CONFLICT_EPOCH: 에폭 임계값 사용 (없으면 배치 평균)
    threshold = epoch_threshold if epoch_threshold is not None else batch_avg_conflict

    # 결과 저장용
    img_ulb_s = img_ulb_w.clone()
    cp_count = 0  # Copy-Paste 적용 이미지 수
    cutmix_count = 0  # CutMix 적용 이미지 수

    # 크기 조건
    min_pixels = int(total_pixels * min_size_ratio)
    max_pixels = int(total_pixels * max_size_ratio)

    for i in range(batch_size):
        img_conflict_ratio = per_image_conflicts[i]

        if img_conflict_ratio > threshold:
            # 어려운 이미지 → Copy-Paste (보호)
            # 색상/크기 조건 맞는 라벨 이미지 찾기
            ulb_mean_rgb = img_ulb_w[i].mean(dim=(1, 2))

            indices = list(range(lb_batch_size))
            np.random.shuffle(indices)

            cp_applied = False
            for j in indices:
                # 크기 조건
                obj_mask = (target_lb[j] > 0)
                polyp_size = obj_mask.sum().item()
                if polyp_size < min_pixels or polyp_size > max_pixels:
                    # 🔥 DEBUG: 크기 조건 불충족 시 로깅 (100 iteration마다)
                    if iter_num % 100 == 0 and iter_num > 0:
                        print(f"[DEBUG Stage2 SKIP] iter={iter_num}, img={i}, lb={j}, "
                              f"polyp_size={polyp_size} ({polyp_size/total_pixels*100:.1f}%), "
                              f"min={min_pixels} ({min_size_ratio*100:.0f}%), max={max_pixels} ({max_size_ratio*100:.0f}%), "
                              f"REASON=size_out_of_range")
                    continue

                # 색상 조건
                lb_mean_rgb = img_lb_w[j].mean(dim=(1, 2))
                color_dist = torch.norm(ulb_mean_rgb - lb_mean_rgb).item() / np.sqrt(3)
                color_similarity = 1.0 - color_dist
                if color_similarity < color_threshold:
                    # 🔥 DEBUG: 색상 조건 불충족 시 로깅 (100 iteration마다)
                    if iter_num % 100 == 0 and iter_num > 0:
                        print(f"[DEBUG Stage2 SKIP] iter={iter_num}, img={i}, lb={j}, "
                              f"polyp_size={polyp_size} ({polyp_size/total_pixels*100:.1f}%), "
                              f"color_sim={color_similarity:.3f}, threshold={color_threshold}, "
                              f"REASON=color_mismatch")
                    continue

                # 🔥 DEBUG: 두 조건 모두 충족 시 로깅 (100 iteration마다)
                if iter_num % 100 == 0 and iter_num > 0:
                    print(f"[DEBUG Stage2 APPLY] iter={iter_num}, img={i}, lb={j}, "
                          f"polyp_size={polyp_size} ({polyp_size/total_pixels*100:.1f}%), "
                          f"min={min_pixels} ({min_size_ratio*100:.0f}%), max={max_pixels} ({max_size_ratio*100:.0f}%), "
                          f"color_sim={color_similarity:.3f}")

                aug_img, aug_pseudo_out, aug_pseudo_log = copy_paste(
                    img_ulb_w[i:i+1], pseudo_outputs[i:i+1], pseudo_logits[i:i+1],
                    img_lb_w[j:j+1], target_lb[j:j+1]
                )
                img_ulb_s[i] = aug_img[0]
                pseudo_outputs[i] = aug_pseudo_out[0]
                pseudo_logits[i] = aug_pseudo_log[0]
                cp_applied = True
                cp_count += 1
                break

            # Copy-Paste 조건 불충족 시 원본 유지
            if not cp_applied:
                img_ulb_s[i] = img_ulb_w[i]

        else:
            # 쉬운 이미지 → CutMix (도전적 증강)
            # 다른 unlabeled 이미지와 CutMix
            other_idx = np.random.randint(0, batch_size)
            while other_idx == i and batch_size > 1:
                other_idx = np.random.randint(0, batch_size)

            # 단일 이미지 CutMix 적용
            img_pair = torch.stack([img_ulb_w[i], img_ulb_w[other_idx]])
            pseudo_pair = torch.stack([pseudo_outputs[i], pseudo_outputs[other_idx]])
            logits_pair = torch.stack([pseudo_logits[i], pseudo_logits[other_idx]])
            conflict_pair = torch.stack([mtx_bool_conflict[i], mtx_bool_conflict[other_idx]])

            mixed_img, mixed_pseudo, mixed_logits, mixed_conflict = cut_mix(
                img_pair, pseudo_pair, logits_pair, conflict_pair
            )

            img_ulb_s[i] = mixed_img[0]
            pseudo_outputs[i] = mixed_pseudo[0]
            pseudo_logits[i] = mixed_logits[0]
            mtx_bool_conflict[i] = mixed_conflict[0]
            cutmix_count += 1

    stats = {
        'cp_count': cp_count,
        'cutmix_count': cutmix_count,
        'cp_ratio': cp_count / batch_size,
        'cutmix_ratio': cutmix_count / batch_size,
        'batch_avg_conflict': batch_avg_conflict,  # 현재 배치 평균
        'threshold_used': threshold,  # 실제 사용된 임계값 (에폭 or 배치)
        'using_epoch_threshold': epoch_threshold is not None,  # 에폭 임계값 사용 여부
        'per_image_conflicts': per_image_conflicts
    }

    return img_ulb_s, pseudo_outputs, pseudo_logits, mtx_bool_conflict, stats




def copy_paste_with_translation(img_ulb, pseudo_outputs, pseudo_logits, img_lb, target_lb,
                                new_center_y, new_center_x):
    """
    위치를 이동하여 Copy-Paste 수행

    Args:
        img_ulb: unlabeled images [B, C, H, W]
        pseudo_outputs: pseudo labels [B, H, W]
        pseudo_logits: pseudo confidence [B, H, W]
        img_lb: labeled images [B, C, H, W]
        target_lb: ground truth labels [B, H, W]
        new_center_y: 새로운 중심 y 좌표
        new_center_x: 새로운 중심 x 좌표

    Returns:
        augmented (img_ulb, pseudo_outputs, pseudo_logits)
    """
    batch_size = img_ulb.shape[0]
    _, _, H, W = img_ulb.shape

    # Clone inputs
    aug_img_ulb = img_ulb.clone()
    aug_pseudo_outputs = pseudo_outputs.clone()
    aug_pseudo_logits = pseudo_logits.clone()

    for i in range(batch_size):
        j = 0  # Labeled sample index

        # Get object mask from labeled data
        obj_mask = (target_lb[j] > 0).float()

        if obj_mask.sum() > 100:
            # 1. 원본 polyp 중심 계산
            ys, xs = torch.where(obj_mask > 0)
            orig_center_y = int(ys.float().mean().item())
            orig_center_x = int(xs.float().mean().item())

            # 2. 이동 거리 계산
            dy = new_center_y - orig_center_y
            dx = new_center_x - orig_center_x

            # 3. 새로운 마스크 생성 (이동된 위치)
            new_mask = torch.zeros_like(obj_mask)

            # 원본 마스크의 각 픽셀을 새 위치로 이동
            for y in range(H):
                for x in range(W):
                    if obj_mask[y, x] > 0:
                        new_y = y + dy
                        new_x = x + dx
                        # 경계 체크
                        if 0 <= new_y < H and 0 <= new_x < W:
                            new_mask[new_y, new_x] = 1.0

            # 4. 새 위치에 paste
            aug_img_ulb[i] = aug_img_ulb[i] * (1 - new_mask.unsqueeze(0)) + img_lb[j] * new_mask.unsqueeze(0)
            aug_pseudo_outputs[i] = aug_pseudo_outputs[i] * (1 - new_mask).long() + target_lb[j] * new_mask.long()
            aug_pseudo_logits[i] = aug_pseudo_logits[i] * (1 - new_mask) + new_mask * 0.99

    return aug_img_ulb, aug_pseudo_outputs, aug_pseudo_logits
