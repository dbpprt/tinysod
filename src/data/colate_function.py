from types import SimpleNamespace
from typing import Sequence, Tuple

import torch
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2


class CollateFunction:
    image_size: int
    transforms: v2.Transform | None = None

    def __init__(self, image_size: int, transforms: Sequence[v2.Transform]):
        self.image_size = image_size

        if transforms is not None and len(transforms) > 0:
            self.transforms = v2.Compose(transforms=transforms)

    def __call__(self, batch: Sequence[Tuple[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        result = SimpleNamespace(x=[], y=[])

        for i, sample in enumerate(batch):
            x, y = sample

            x = read_image(x, ImageReadMode.RGB)
            y = read_image(y, ImageReadMode.GRAY)

            assert x.shape[1:] == y.shape[1:], f"x and y must have the same size ({batch[i]})"

            if self.transforms is not None:
                x, y = self.transforms(x, tv_tensors.Mask(y))

            result.x.append(x)
            result.y.append(y)

        return torch.stack(result.x), torch.stack(result.y)
