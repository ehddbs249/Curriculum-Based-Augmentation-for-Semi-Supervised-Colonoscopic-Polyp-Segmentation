import os
import random
import h5py
import itertools
import numpy as np
from scipy import ndimage
from scipy.ndimage.interpolation import zoom

import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter


class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        sample["case"] = case  # 샘플 이름 (문자열) 추가
        return sample


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                          1. Samplers
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
class TwoStreamBatchSampler(Sampler): #
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        2. Generators
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        # Handle both RGB and grayscale images
        if len(image.shape) == 2:  # grayscale
            x, y = image.shape
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        else:  # RGB
            x, y, c = image.shape
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
            image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        
        x, y = label.shape
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample


class WeakStrongAugment(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image_org = sample["image"].copy()
        image, label = sample["image"], sample["label"]
        
        # geometry
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        # resize
        image_org = self.resize(image_org)
        image = self.resize(image)
        label = self.resize(label)
        
        # strong augmentation is color jitter
        image_strong = func_strong_augs(image, p_color=0.8, p_blur=0.2)

        # fix dimensions - handle both RGB (H,W,C) and grayscale (H,W)
        if len(image_org.shape) == 3:  # RGB image
            image_org = torch.from_numpy(image_org.astype(np.float32)).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
            image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        else:  # grayscale image
            image_org = torch.from_numpy(image_org.astype(np.float32)).unsqueeze(0)  # (H,W) -> (1,H,W)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # (H,W) -> (1,H,W)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image_org,
            "image_weak": image,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        if len(image.shape) == 2:  # grayscale image (H, W)
            x, y = image.shape
            return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        elif len(image.shape) == 3:  # RGB image (H, W, C)
            x, y, c = image.shape
            return zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
    

class WeakOnlyAugment(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image_org = sample["image"].copy()
        image, label = sample["image"], sample["label"]
        
        # geometry - weak augmentation only (기존 WeakStrongAugment에서 사용하던 것과 동일)
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        # resize
        image_org = self.resize(image_org)
        image = self.resize(image)
        label = self.resize(label)
        
        # strong augmentation 제거 - weak augmentation만 사용 (resize된 image를 그대로 사용)
        image_strong = image.copy()  # weak augmentation과 동일하게 설정

        # fix dimensions - handle both RGB (H,W,C) and grayscale (H,W)
        if len(image_org.shape) == 3:  # RGB image
            image_org = torch.from_numpy(image_org.astype(np.float32)).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
            image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
            image_strong = torch.from_numpy(image_strong.astype(np.float32)).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        else:  # grayscale image
            image_org = torch.from_numpy(image_org.astype(np.float32)).unsqueeze(0)  # (H,W) -> (1,H,W)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # (H,W) -> (1,H,W)
            image_strong = torch.from_numpy(image_strong.astype(np.float32)).unsqueeze(0)  # (H,W) -> (1,H,W)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image_org,
            "image_weak": image,
            "image_strong": image_strong,  # weak와 동일
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        if len(image.shape) == 2:  # grayscale image (H, W)
            x, y = image.shape
            return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        elif len(image.shape) == 3:  # RGB image (H, W, C)
            x, y, c = image.shape
            return zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")


class WeakStrongAugmentMore(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image_org = sample["image"].copy()
        image, label = sample["image"], sample["label"]
        
        # geometry
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        # elif random.random() > 0.5:
        #     image, label = random_rotate(image, label)
        
        # resize
        image_org = self.resize(image_org)
        image = self.resize(image)
        label = self.resize(label)
        
        # strong augmentation is color jitter
        image_strong = func_strong_augs(image, p_color=0.5, p_blur=0.2)
        image_strong_more = func_strong_augs(image, p_color=1.0, p_blur=0.2)

        # fix dimensions - handle both RGB (H,W,C) and grayscale (H,W)
        if len(image_org.shape) == 3:  # RGB image
            image_org = torch.from_numpy(image_org.astype(np.float32)).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
            image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        else:  # grayscale image
            image_org = torch.from_numpy(image_org.astype(np.float32)).unsqueeze(0)  # (H,W) -> (1,H,W)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # (H,W) -> (1,H,W)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image_org,
            "image_weak": image,
            "image_strong": image_strong,
            "image_strong_more": image_strong_more,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        if len(image.shape) == 2:  # grayscale image (H, W)
            x, y = image.shape
            return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        elif len(image.shape) == 3:  # RGB image (H, W, C)
            x, y, c = image.shape
            return zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                         3. Augmentations
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(image, p=1.0):
    # if not torch.is_tensor(image):
    #     np_to_tensor = transforms.ToTensor()
    #     image = np_to_tensor(image)
    # s is the strength of color distortion.
    # s = 1.0
    # jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    jitter = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)
    if np.random.random() < p:
        image = jitter(image)
    return image


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def func_strong_augs(image, p_color=0.8, p_blur=0.5):
    if len(image.shape) == 3:  # RGB image
        img = Image.fromarray((image * 255).astype(np.uint8))
        img = color_jitter(img, p_color)
        img = blur(img, p_blur)
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0  # (H,W,C) -> (C,H,W)
    else:  # grayscale image
        img = Image.fromarray((image * 255).astype(np.uint8))
        img = color_jitter(img, p_color)
        img = blur(img, p_blur)
        img = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0  # (H,W) -> (1,H,W)

    return img


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        Kvasir Dataset
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
class KvasirDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train.list", "r", encoding="utf-8") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/test.list", "r", encoding="utf-8") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "test":
            with open(self._base_dir + "/test.list", "r", encoding="utf-8") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        
        if self.split == "train":
            image_path = self._base_dir + "/train/images/{}.png".format(case)
            mask_path = self._base_dir + "/train/masks/{}.png".format(case) 
        elif self.split == "val":
            image_path = self._base_dir + "/test/images/{}.png".format(case)
            mask_path = self._base_dir + "/test/masks/{}.png".format(case)
        else:  # test
            image_path = self._base_dir + "/test/images/{}.png".format(case)
            mask_path = self._base_dir + "/test/masks/{}.png".format(case)
        
        # Load image and mask using PIL
        image = Image.open(image_path).convert('RGB')  # Keep RGB for color information
        mask = Image.open(mask_path).convert('L')  # grayscale
        
        # Convert to numpy arrays
        image = np.array(image).astype(np.float32) / 255.0  # normalize to [0,1], shape: (H, W, 3)
        mask = np.array(mask).astype(np.float32) / 255.0    # normalize to [0,1], shape: (H, W)
        
        # Convert mask to binary (0 or 1)
        mask = (mask > 0.5).astype(np.uint8)
        
        sample = {"image": image, "label": mask}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        else:
            # For validation/test, convert to tensor
            image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
            mask = torch.from_numpy(mask.astype(np.uint8))
            sample = {"image": image, "label": mask}
        sample["idx"] = idx
        sample["case"] = case  # 샘플 이름 (문자열) 추가
        return sample


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        CVC-ClinicDB Dataset
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
class CVCDataSets(KvasirDataSets):
    """CVC-ClinicDB Dataset - 동일 구조 (images/masks + train.list/test.list)"""
    pass
