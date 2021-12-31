# Jigsaw-puzzle-transform
A repository that contains a simple function to get a patch permutation and its permutation indices. Contains a simple PyTorch dataloader wrapper as well. 

### Extracting a grid of N x M patches from an image
```python3
import torch

img_tensor = torch.randn(size=(3, 360, 640))
jigsaw_stack, permutation = create_jigsaw_grid(
  img_tensor, grid_size=(4, 8), patch_size=(64, 64), crop_size=False, patch_norm=False
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
  img_tensor, grid_size=(4, 8), patch_size=(64, 64), crop_size=(48, 48), patch_norm=False
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
  patch_size=(64, 64), grid_size=(4, 8), crop_size=(48, 48)
)
dataloader = DataLoader(dataset, batch_size=8, num_workers=8)
jigsaw_stack, permutation = next(iter(dataloader))

print(f'Jigsaw stack shape: {jigsaw_stack.shape}')
>>>Jigsaw stack shape: torch.Size([8, 32, 3, 48, 48])

print(f'Jigsaw permutation shape: {permutation.shape}')
>>>Jigsaw permutation shape: torch.Size([8, 32])
```
