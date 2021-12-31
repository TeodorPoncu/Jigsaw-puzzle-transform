import os
import random
from collections.abc import Callable
from functools import partial
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tvf
import torchvision.transforms.functional as TF

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', 'JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF'
]

def is_image_file(filename) -> bool:
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir: str) -> List[str]:
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images
  
  def patch_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize the input image by mean and standard deviation along the channel dimension.
    """
    if len(x.shape) == 2:
        x = (x - torch.mean(x, keepdim=True)) / torch.std(x, keepdim=True)
    elif len(x.shape) == 3:
        x = (x - torch.mean(x, dim=1, keepdim=True)) / torch.std(x, dim=1, keepdim=True)
    elif len(x.shape) == 4:
        x = (x - torch.mean(x, dim=(0, 1), keepdim=True)) / torch.std(x, dim=(0, 1), keepdim=True)
    else:
        raise ValueError("Input tensor must have 2, 3 or 4 dimensions, but has {}".format(len(x.shape)))
    return x

def create_jigsaw_grid(
        img_tensor: torch.Tensor,
        grid_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        crop_size: Tuple[int, int] = None,
        norm_patch: bool = False,
        permutation: torch.Tensor = None
)-> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a grid of patches from an image tensor.

    :param img_tensor: the image tensor to be split in patches
    :param grid_size:  a tuple containing the number of patches in each row and column
    :param patch_size: a tuple containing the size of the patches
    :param crop_size:  a tuple containing the size of the crop to be taken from each patch (default: None - no crop)
    :param norm_patch: if True, the patches are w.r.t. to patch statistics (default: False)
    :param permutation: a tensor containing the permutation of the patches (default: None - random permutation)
    :return: a tuple containing the image patches and  permutation indices
        ret[0] shape: [n_patches, n_channels, patch_height, patch_width], patch dimensions will be crop dimensions if specified
        ret[1] shape: [n_patches]
    """

    # get image height and width
    h, w = img_tensor.shape[-2:]
    # check that the grid size and patch size combination are valid
    assert patch_size[0] * grid_size[0] < h,\
        f"Final image grid size on axis 0 of {patch_size[0] * grid_size[0]} too large, given: \n" \
        f"grid_size {grid_size[0]} and patch_size {patch_size[0]} and input image with axis 0 size {h}"
    assert patch_size[1] * grid_size[1] < w,\
        f"Final image grid size on axis 1 of {patch_size[1] * grid_size[1]} too large, given: \n" \
        f"grid_size {grid_size[1]} and patch_size {patch_size[1]} and input image with axis 1 size {w}"

    patches = []
    # sample an anchor point to compute the jigsaw grid
    x_anchor_idx = random.randint(0, w - patch_size[1] * grid_size[1])
    y_anchor_idx = random.randint(0, h - patch_size[0] * grid_size[0])

    # check if we have crop_size and initialize crop function
    if crop_size is not None:
        crop_fn = tvf.RandomCrop(size=crop_size)
    # otherwise use identity function
    else:
        crop_fn = lambda x: x

    # iterate over the image and extract patches
    for xn in range(grid_size[1]):
        for yn in range(grid_size[0]):
            x_idx = x_anchor_idx + xn * patch_size[1]
            y_idx = y_anchor_idx + yn * patch_size[0]
            # extract patch
            patch = img_tensor[..., y_idx:y_idx + patch_size[0], x_idx:x_idx + patch_size[1]]
            patch = crop_fn(patch)
            if norm_patch:
                patch = patch_norm(patch)
            patches.append(patch)
    # create patch permutation
    if permutation is None:
        permutation = torch.randperm(len(patches))
    # permute the patches
    permuted_patches = [patches[i] for i in permutation]
    # stack patches across n_patches axis
    permuted_patches = torch.stack(permuted_patches, dim=0)
    return permuted_patches, permutation
  
class JigsawDataset(Dataset):
    def __init__(
            self,
            images_root: str,
            grid_size: Tuple[int, int],
            patch_size: Tuple[int, int],
            crop_size: Tuple[int, int] = None,
            norm_patch: bool = False,
            preprocess_transform: Callable = tvf.ToTensor(),
    ):
        super(JigsawDataset, self).__init__()
        self.transform = preprocess_transform
        self.jigsaw_transform = partial(create_jigsaw_grid, grid_size=grid_size, patch_size=patch_size, crop_size=crop_size, norm_patch=norm_patch)
        self.img_paths = make_dataset(images_root)
        self.size = len(self.img_paths)

    def __len__(self):
        return self.size

    def __getitem__(self, item) -> Tuple:
        sample = Image.open(self.img_paths[item])
        sample, permutation = self.jigsaw_transform(self.transform(sample), permutation=None)
        return sample, permutation
    
class FixedJigsawDataset(Dataset):
    def __init__(
            self,
            images_root: str,
            grid_size: Tuple[int, int],
            patch_size: Tuple[int, int],
            permutations: List[torch.Tensor],
            crop_size: Tuple[int, int] = None,
            norm_patch: bool = False,
            preprocess_transform: Callable = tvf.ToTensor(),
    ):
        super(JigsawDataset, self).__init__()
        self.transform = preprocess_transform
        self.jigsaw_transform = partial(create_jigsaw_grid, grid_size=grid_size, patch_size=patch_size, crop_size=crop_size, norm_patch=norm_patch)
        self.img_paths = make_dataset(images_root)
        self.size = len(self.img_paths)
        
        self.permutations = {i: permutations[i] for i in range(len(permutations))}

    def __len__(self):
        return self.size

    def __getitem__(self, item) -> Tuple:
        sample = Image.open(self.img_paths[item])
        puzzle_key, puzzle_value = random.choice(list(self.permutations.items()))
        sample, _ = self.jigsaw_transform(self.transform(sample), permutation=puzzle_value)
        return sample, puzzle_key
    
def generate_permutations(permutation_size: int, n_permutations: int) -> List[torch.Tensor]:
    """Function that generations n_permutations unique permutations of size permutation_size"""
    permutations = [torch.randperm(permutation_size) for _ in range(n_permutations)]
    while len(set(permutations)) != n_permutations:
        permutations = [torch.randperm(permutation_size) for _ in range(n_permutations)]
    return permutations
      
if __name__ == '__main__':
    dataset = JigsawDatasetGithub('trainA', preprocess_transform=tvf.ToTensor(), patch_size=(64, 64), grid_size=(4, 8))
    dataloader = DataLoader(dataset, batch_size=8, num_workers=8, pin_memory=True)
    jigsaw_stack, jigsaw_permutation = next(iter(dataloader))
    print(f'Jigsaw stack shape: {jigsaw_stack.shape}')
    print(f'Jigsaw permutation shape: {jigsaw_permutation.shape}')
