# Jigsaw-puzzle-transform
A repository that contains a simple function to get a patch permutation and its permutation indices. Used for jigsaw puzzle pretraining - [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/abs/1603.09246).
Contains a simple PyTorch dataloader wrapper as well. 

### Extracting a grid of N x M patches from an image
Setting `norm_patch=True` will normalize pixel values w.r.t. patch statistics as suggested in the paper.

```python3
import torch

img_tensor = torch.randn(size=(3, 360, 640))
jigsaw_stack, permutation = create_jigsaw_grid(
  img_tensor, grid_size=(4, 8), patch_size=(64, 64), crop_size=False, norm_patch=False
)

print(f'Jigsaw stack shape: {jigsaw_stack.shape}')
>>>Jigsaw stack shape: torch.Size([32, 3, 64, 64])

print(f'Jigsaw permutation shape: {permutation.shape}')
>>>Jigsaw permutation shape: torch.Size([32])
```

### Extracting a grid of N x M random crops taken from patches
If used on tensors that have more than 3 dimensions, all jigsaw stacks will be created using the same set of permutation indices.
```python3
import torch

img_tensor = torch.randn(size=(8, 3, 360, 640))
jigsaw_stack, permutation = create_jigsaw_grid(
  img_tensor, grid_size=(4, 8), patch_size=(64, 64), crop_size=(48, 48), norm_patch=False
)

print(f'Jigsaw stack shape: {jigsaw_stack.shape}')
>>>Jigsaw stack shape: torch.Size([32, 8, 3, 48, 48])

print(f'Jigsaw permutation shape: {permutation.shape}')
>>>Jigsaw permutation shape: torch.Size([32])
```

### Extracting a different permutation for all images from a batch
The recommended way to do this is to utilize the simple PyTorch Dataset wrapper that is provided
```python3
import torch
from torchvision import transforms as tvf
from torch.utils.data import DataLoader


image_folder_path = SOME_PATH
dataset = JigsawDataset(
  image_folder_path, preprocess_transform=tvf.Compose([tvf.Resize(size=(360, 640)), tvf.ToTensor()]),
  patch_size=(64, 64), grid_size=(4, 8), crop_size=(48, 48), norm_patch=False
)
dataloader = DataLoader(dataset, batch_size=8, num_workers=8)
jigsaw_stack, permutation = next(iter(dataloader))

print(f'Jigsaw stack shape: {jigsaw_stack.shape}')
>>>Jigsaw stack shape: torch.Size([8, 32, 3, 48, 48])

print(f'Jigsaw permutation shape: {permutation.shape}')
>>>Jigsaw permutation shape: torch.Size([8, 32])
```

### Using a fixed set of P permutations in the dataset
In [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/abs/1603.09246) only a subset of possible permutations is used. In order to achieve the same behaviour, `the class FixedJigsawDataset is provided`. The returned value will be the index of the used permutation to generate the stack.

```python3
import torch
from torchvision import transforms as tvf
from torch.utils.data import DataLoader

PATCH_SIZE = (64, 64)
CROP_SIZE = (48, 48)
GRID_SIZE = (4, 8)

permutations = generate_permutations(permutation_size=GRID_SIZE[0] * GRID_SIZE[1], n_permutations=4)
dataset = FixedJigsawDataset(
  image_folder_path, preprocess_transform=tvf.Compose([tvf.Resize(size=(360, 640)), tvf.ToTensor()]),
  patch_size=PATCH_SIZE, grid_size=GRID_SIZE, crop_size=CROP_SIZE, norm_patch=False, permutations=permutations
)

dataloader = DataLoader(dataset, batch_size=8, num_workers=8)
print(f'Dataset permutations{dataset.permutations}')
>>>{
>>>    0: tensor([27,  2,  4,  3,  9, 19,  1, 22, 26, 21,  5, 12, 14,  6, 10,  0,  8, 28, 23, 15, 17, 24, 16, 13, 30, 11, 29,  7, 18, 31, 25, 20]),
>>>    1: tensor([ 6, 31, 17, 14, 30,  5, 29,  7, 13, 28, 12, 23, 25, 20,  8, 11, 21, 10, 22, 18, 24,  9, 16, 19, 27, 0,   3, 26,  1,  4, 15,  2]),
>>>    2: tensor([ 5, 10, 16,  6,  9, 19,  0, 29, 24, 12, 25,  8, 17, 31, 11, 13,  3,  1, 27, 21, 30, 23, 28, 18, 22, 2,  26, 20, 14,  7,  4, 15]),
>>>    3: tensor([ 9, 25, 24, 30,  3, 22, 15,  7,  2,  8, 13,  6, 28,  0, 29, 10, 16, 27, 4, 12, 21, 23, 11, 18,   5, 17, 14, 31, 20, 19, 26,  1])
>>>}

jigsaw_stack, jigsaw_permutation = next(iter(dataloader))

print(f'Jigsaw stack shape: {jigsaw_stack.shape}')
>>>Jigsaw stack shape: torch.Size([8, 32, 3, 48, 48])

print(f'Jigsaw permutation shape: {jigsaw_permutation.shape}')
>>>Jigsaw permutation shape: torch.Size([8])

print(f'Jigsaw puzzle: {jigsaw_permutation[0]}')
>>>Jigsaw puzzle: 2

print(f'Jigsaw puzzle permutation: {dataset.permutations[jigsaw_permutation[0]]}')
>>>tensor([ 5, 10, 16,  6,  9, 19,  0, 29, 24, 12, 25,  8, 17, 31, 11, 13,  3,  1, 27, 21, 30, 23, 28, 18, 22, 2,  26, 20, 14,  7,  4, 15])
```

