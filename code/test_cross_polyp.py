"""
Cross-Dataset Polyp Segmentation Test Script
- Kvasir에서 학습한 모델을 다른 용종 데이터셋에서 평가
- 지원 데이터셋: CVC-ClinicDB, CVC-ColonDB, ETIS-LaribPolypDB, CVC-300 등
- 폴더 구조: images/ + masks/ (flat) 또는 train/images/ + test/images/ (split)
"""

import argparse
import os
import glob
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm
from PIL import Image
import pandas as pd

from networks.net_factory import net_factory

parser = argparse.ArgumentParser(description='Cross-Dataset Polyp Segmentation Test')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to target dataset')
parser.add_argument('--dataset_name', type=str, default='Unknown', help='Dataset name for display')
parser.add_argument('--model_path', type=str, required=True, help='Path to trained model (.pth)')
parser.add_argument('--model', type=str, default='unet', help='Model architecture')
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--patch_size', type=int, default=256, help='Input size for model')
parser.add_argument('--save_csv', action='store_true', help='Save per-case CSV results')
parser.add_argument('--use_all', action='store_true', default=True,
                    help='Use ALL images (train+test) for cross-dataset eval')
parser.add_argument('--use_test_only', action='store_true',
                    help='Use only test split images')


def find_all_images(dataset_path, use_all=True):
    """데이터셋 경로에서 이미지-마스크 쌍 자동 탐색"""
    pairs = []

    if use_all:
        # Case 1: flat structure (images/ + masks/)
        flat_img_dir = os.path.join(dataset_path, 'images')
        flat_mask_dir = os.path.join(dataset_path, 'masks')
        if os.path.isdir(flat_img_dir) and os.path.isdir(flat_mask_dir):
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']:
                for img_path in glob.glob(os.path.join(flat_img_dir, ext)):
                    name = os.path.splitext(os.path.basename(img_path))[0]
                    # 마스크 확장자 탐색
                    mask_path = None
                    for mext in ['.png', '.jpg', '.bmp', '.tif']:
                        candidate = os.path.join(flat_mask_dir, name + mext)
                        if os.path.exists(candidate):
                            mask_path = candidate
                            break
                    if mask_path:
                        pairs.append((img_path, mask_path, name))

        # Case 1-b: im_resize/ + gt_resize/ (CVC-ColonDB_300)
        resize_img_dir = os.path.join(dataset_path, 'im_resize')
        resize_mask_dir = os.path.join(dataset_path, 'gt_resize')
        if os.path.isdir(resize_img_dir) and os.path.isdir(resize_mask_dir):
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']:
                for img_path in glob.glob(os.path.join(resize_img_dir, ext)):
                    name = os.path.splitext(os.path.basename(img_path))[0]
                    # 마스크 확장자 탐색
                    mask_path = None
                    for mext in ['.png', '.jpg', '.bmp', '.tif']:
                        candidate = os.path.join(resize_mask_dir, name + mext)
                        if os.path.exists(candidate):
                            mask_path = candidate
                            break
                    if mask_path:
                        pairs.append((img_path, mask_path, name))

        # Case 2: split structure (train/images/ + test/images/)
        for split in ['train', 'test', 'val']:
            split_img_dir = os.path.join(dataset_path, split, 'images')
            split_mask_dir = os.path.join(dataset_path, split, 'masks')
            if os.path.isdir(split_img_dir) and os.path.isdir(split_mask_dir):
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']:
                    for img_path in glob.glob(os.path.join(split_img_dir, ext)):
                        name = os.path.splitext(os.path.basename(img_path))[0]
                        mask_path = None
                        for mext in ['.png', '.jpg', '.bmp', '.tif']:
                            candidate = os.path.join(split_mask_dir, name + mext)
                            if os.path.exists(candidate):
                                mask_path = candidate
                                break
                        if mask_path:
                            pairs.append((img_path, mask_path, f"{split}/{name}"))
    else:
        # test split only
        for test_dir in ['test', '.']:
            img_dir = os.path.join(dataset_path, test_dir, 'images')
            mask_dir = os.path.join(dataset_path, test_dir, 'masks')
            if os.path.isdir(img_dir) and os.path.isdir(mask_dir):
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']:
                    for img_path in glob.glob(os.path.join(img_dir, ext)):
                        name = os.path.splitext(os.path.basename(img_path))[0]
                        mask_path = None
                        for mext in ['.png', '.jpg', '.bmp', '.tif']:
                            candidate = os.path.join(mask_dir, name + mext)
                            if os.path.exists(candidate):
                                mask_path = candidate
                                break
                        if mask_path:
                            pairs.append((img_path, mask_path, name))
                break  # 첫 번째 유효한 경로에서 중단

    # 중복 제거 (이름 기준)
    seen = set()
    unique_pairs = []
    for img, mask, name in pairs:
        basename = os.path.basename(img)
        if basename not in seen:
            seen.add(basename)
            unique_pairs.append((img, mask, name))

    return sorted(unique_pairs, key=lambda x: x[2])


def load_image_mask(img_path, mask_path):
    """이미지와 마스크 로드 (RGB normalized, binary mask)"""
    image = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')

    image = np.array(image).astype(np.float32) / 255.0  # (H, W, 3)
    mask = np.array(mask).astype(np.float32)

    # 마스크 이진화 (다양한 형식 지원)
    if mask.max() > 1:
        mask = mask / 255.0
    mask = (mask > 0.5).astype(np.uint8)

    # (H, W, 3) -> (3, H, W)
    image_tensor = image.transpose(2, 0, 1)

    return image_tensor, mask


def calculate_metric(pred, gt):
    """수정된 메트릭 계산 (fixed Dice)"""
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    pred_sum = pred.sum()
    gt_sum = gt.sum()

    if pred_sum > 0 and gt_sum > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, jc, hd95, asd
    elif pred_sum == 0 and gt_sum == 0:
        return 1.0, 1.0, 0.0, 0.0
    else:
        return 0.0, 0.0, 0.0, 0.0


def test_single(net, image, label, patch_size=256):
    """단일 이미지 테스트"""
    c, h, w = image.shape
    image_zoom = zoom(image, (1, patch_size / h, patch_size / w), order=0)
    input_tensor = torch.from_numpy(image_zoom).unsqueeze(0).float().cuda()

    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(input_tensor), dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()
        prediction = zoom(prediction, (h / patch_size, w / patch_size), order=0)

    return calculate_metric(prediction.copy(), label.copy())


def inference(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # 데이터셋 이미지 탐색
    use_all = not args.use_test_only
    pairs = find_all_images(args.dataset_path, use_all=use_all)

    if not pairs:
        print(f"[ERROR] No image-mask pairs found in {args.dataset_path}")
        print("Expected structure: images/ + masks/ or train/images/ + test/images/")
        return None

    print(f"\n{'='*60}")
    print(f"Cross-Dataset Test: {args.dataset_name}")
    print(f"{'='*60}")
    print(f"  Dataset path: {args.dataset_path}")
    print(f"  Total images: {len(pairs)}")
    print(f"  Model: {args.model_path}")
    print(f"  Mode: {'ALL images' if use_all else 'Test only'}")
    print(f"{'='*60}\n")

    # 모델 로드
    net = net_factory(net_type=args.model, in_chns=3, class_num=args.num_classes)
    net.load_state_dict(torch.load(args.model_path))
    net.eval()
    net.cuda()
    print(f"Model loaded: {os.path.basename(args.model_path)}")

    # 테스트 실행
    results = []
    for img_path, mask_path, name in tqdm(pairs, desc=f"Testing on {args.dataset_name}"):
        image, mask = load_image_mask(img_path, mask_path)
        dice, jc, hd95, asd = test_single(net, image, mask, args.patch_size)
        results.append({
            'case': name,
            'dice': dice,
            'jaccard': jc,
            'hd95': hd95,
            'asd': asd
        })

    # 결과 집계
    df = pd.DataFrame(results)
    mean_dice = df['dice'].mean()
    mean_jc = df['jaccard'].mean()
    mean_hd95 = df['hd95'].mean()
    mean_asd = df['asd'].mean()

    print(f"\n{'='*60}")
    print(f"Results on {args.dataset_name} ({len(pairs)} images)")
    print(f"{'='*60}")
    print(f"  Mean Dice:    {mean_dice:.4f} (std: {df['dice'].std():.4f})")
    print(f"  Mean Jaccard: {mean_jc:.4f} (std: {df['jaccard'].std():.4f})")
    print(f"  Mean HD95:    {mean_hd95:.2f} (std: {df['hd95'].std():.2f})")
    print(f"  Mean ASD:     {mean_asd:.2f} (std: {df['asd'].std():.2f})")
    print(f"{'='*60}")

    # CSV 저장
    if args.save_csv:
        model_dir = os.path.dirname(args.model_path)
        csv_path = os.path.join(model_dir, f'cross_test_{args.dataset_name}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    return {
        'dataset': args.dataset_name,
        'n_images': len(pairs),
        'mean_dice': mean_dice,
        'std_dice': df['dice'].std(),
        'mean_jaccard': mean_jc,
        'mean_hd95': mean_hd95,
        'mean_asd': mean_asd,
    }


if __name__ == "__main__":
    args = parser.parse_args()
    inference(args)
