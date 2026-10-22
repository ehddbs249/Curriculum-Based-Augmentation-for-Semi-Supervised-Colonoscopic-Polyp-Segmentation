# # # # # # # # # # # # # # # # # # # # # 
# # 시각화 헬퍼 함수들 (별도 파일)
# # # # # # # # # # # # # # # # # # # # # 

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def visualize_cutmix_process(img_before, img_after, mask_before, mask_after, 
                            bb_coords=None, save_path=None):
    """
    CutMix 적용 전후와 박스 영역 시각화
    
    Args:
        img_before: CutMix 적용 전 이미지 [C, H, W]
        img_after: CutMix 적용 후 이미지 [C, H, W]
        mask_before: CutMix 적용 전 마스크 [H, W]
        mask_after: CutMix 적용 후 마스크 [H, W]
        bb_coords: 박스 좌표 (bbx1, bby1, bbx2, bby2)
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # 이미지 정규화 함수
    def normalize_img(img):
        if img.dim() == 3 and img.shape[0] == 3:  # RGB
            img = img.permute(1, 2, 0)
        elif img.dim() == 3:  # 단일 채널
            img = img.mean(0)
        return img.detach().cpu().numpy()
    
    def normalize_mask(mask):
        return mask.detach().cpu().numpy()
    
    # Before 이미지와 마스크
    img_before_np = normalize_img(img_before)
    mask_before_np = normalize_mask(mask_before)
    
    axes[0, 0].imshow(img_before_np, cmap='gray' if len(img_before_np.shape) == 2 else None)
    axes[0, 0].set_title('Before CutMix\n(Image)', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask_before_np, cmap='viridis', vmin=0, vmax=3)
    axes[0, 1].set_title('Before CutMix\n(Mask)', fontsize=12)
    axes[0, 1].axis('off')
    
    # After 이미지와 마스크
    img_after_np = normalize_img(img_after)
    mask_after_np = normalize_mask(mask_after)
    
    axes[1, 0].imshow(img_after_np, cmap='gray' if len(img_after_np.shape) == 2 else None)
    axes[1, 0].set_title('After CutMix\n(Image)', fontsize=12, color='red')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mask_after_np, cmap='viridis', vmin=0, vmax=3)
    axes[1, 1].set_title('After CutMix\n(Mask)', fontsize=12, color='red')
    axes[1, 1].axis('off')
    
    # 박스 좌표가 있으면 표시
    if bb_coords is not None:
        bbx1, bby1, bbx2, bby2 = bb_coords
        for ax in axes.flatten():
            rect = patches.Rectangle((bbx1, bby1), bbx2-bbx1, bby2-bby1,
                                   linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def visualize_copypaste_process(img_before, img_after, mask_before, mask_after,
                               pasted_regions=None, save_path=None):
    """
    Copy-Paste 적용 전후와 붙여넣기된 영역 시각화
    
    Args:
        img_before: Copy-Paste 적용 전 이미지 [C, H, W]
        img_after: Copy-Paste 적용 후 이미지 [C, H, W]
        mask_before: Copy-Paste 적용 전 마스크 [H, W]
        mask_after: Copy-Paste 적용 후 마스크 [H, W]
        pasted_regions: 붙여넣기된 영역 마스크 [H, W] (optional)
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 이미지 정규화 함수
    def normalize_img(img):
        if img.dim() == 3 and img.shape[0] == 3:  # RGB
            img = img.permute(1, 2, 0)
        elif img.dim() == 3:  # 단일 채널
            img = img.mean(0)
        return img.detach().cpu().numpy()
    
    def normalize_mask(mask):
        return mask.detach().cpu().numpy()
    
    # Before 상태
    img_before_np = normalize_img(img_before)
    mask_before_np = normalize_mask(mask_before)
    
    axes[0, 0].imshow(img_before_np, cmap='gray' if len(img_before_np.shape) == 2 else None)
    axes[0, 0].set_title('Before Copy-Paste\n(Image)', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask_before_np, cmap='viridis', vmin=0, vmax=3)
    axes[0, 1].set_title('Before Copy-Paste\n(Mask)', fontsize=12)
    axes[0, 1].axis('off')
    
    # 붙여넣기 영역 표시
    if pasted_regions is not None:
        pasted_regions_np = normalize_mask(pasted_regions)
        axes[0, 2].imshow(pasted_regions_np, cmap='RdYlGn', vmin=0, vmax=1)
        axes[0, 2].set_title('Pasted Regions\n(Green Areas)', fontsize=12, color='green')
    else:
        axes[0, 2].text(0.5, 0.5, 'No Pasted\nRegions Info', ha='center', va='center',
                        transform=axes[0, 2].transAxes, fontsize=14)
        axes[0, 2].set_title('Pasted Regions', fontsize=12)
    axes[0, 2].axis('off')
    
    # After 상태
    img_after_np = normalize_img(img_after)
    mask_after_np = normalize_mask(mask_after)
    
    axes[1, 0].imshow(img_after_np, cmap='gray' if len(img_after_np.shape) == 2 else None)
    axes[1, 0].set_title('After Copy-Paste\n(Image)', fontsize=12, color='green')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mask_after_np, cmap='viridis', vmin=0, vmax=3)
    axes[1, 1].set_title('After Copy-Paste\n(Mask)', fontsize=12, color='green')
    axes[1, 1].axis('off')
    
    # 변화 분석
    if pasted_regions is not None:
        change_map = np.abs(mask_after_np - mask_before_np) * pasted_regions_np
        axes[1, 2].imshow(change_map, cmap='hot', vmin=0, vmax=3)
        axes[1, 2].set_title('Mask Changes\n(in Pasted Regions)', fontsize=12, color='red')
    else:
        change_map = np.abs(mask_after_np - mask_before_np)
        axes[1, 2].imshow(change_map, cmap='hot', vmin=0, vmax=3)
        axes[1, 2].set_title('Overall Mask Changes', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def extract_cutmix_bbox_coords(original_img, cutmixed_img, threshold=0.1):
    """
    CutMix된 이미지에서 박스 좌표를 추출하는 헬퍼 함수
    
    Args:
        original_img: 원본 이미지 [C, H, W]
        cutmixed_img: CutMix 적용된 이미지 [C, H, W]
        threshold: 차이 임계값
    
    Returns:
        (bbx1, bby1, bbx2, bby2) 또는 None
    """
    try:
        # 차이 계산
        if original_img.shape[0] == 3:  # RGB
            diff = torch.abs(original_img - cutmixed_img).mean(0)  # [H, W]
        else:
            diff = torch.abs(original_img.mean(0) - cutmixed_img.mean(0))  # [H, W]
        
        # 변화된 영역 찾기
        changed_mask = (diff > threshold).float()
        
        if changed_mask.sum() == 0:
            return None
        
        # 변화된 픽셀의 좌표 찾기
        changed_coords = torch.nonzero(changed_mask, as_tuple=False)
        if len(changed_coords) == 0:
            return None
        
        # 바운딩 박스 계산
        min_coords = changed_coords.min(0)[0]
        max_coords = changed_coords.max(0)[0]
        
        bby1, bbx1 = min_coords[0].item(), min_coords[1].item()
        bby2, bbx2 = max_coords[0].item() + 1, max_coords[1].item() + 1
        
        return (bbx1, bby1, bbx2, bby2)
    
    except Exception as e:
        print(f"[WARN] bbox extraction failed: {e}")
        return None


def extract_copypaste_regions(mask_before, mask_after, threshold=0.5):
    """
    Copy-Paste로 변경된 영역을 추출하는 헬퍼 함수
    
    Args:
        mask_before: Copy-Paste 전 마스크 [H, W]
        mask_after: Copy-Paste 후 마스크 [H, W]
        threshold: 변화 임계값
    
    Returns:
        pasted_regions: 변경된 영역 마스크 [H, W]
    """
    try:
        # 마스크 변화 계산
        changed_mask = (mask_before != mask_after).float()
        return changed_mask
    
    except Exception as e:
        print(f"[WARN] copypaste region extraction failed: {e}")
        return torch.zeros_like(mask_before)