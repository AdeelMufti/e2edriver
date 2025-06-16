# transforms.py
import torch
from torchvision.transforms import v2 as T

import config


class GaussianNoise(torch.nn.Module):
    def __init__(self, sigma=0.1):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, img):
        out = img + self.sigma * torch.randn_like(img).to(img.device)
        return out


class PreProcess(torch.nn.Module):
    def __init__(self):
        super(PreProcess, self).__init__()

    def forward(self, img):
        return img.to(config.DTYPE).permute(0, 3, 1, 2) / 255.0


def get_image_normalization_transforms():
    # Normalize using ImageNet mean and std dev as a common starting point
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transforms = T.Compose(
        [
            PreProcess(),
            normalize,
        ]
    )

    return transforms


def get_image_augmentation_transforms():
    # elastic_transform = T.ElasticTransform(alpha=25.0)
    # random_apply_elastic_transform = T.RandomApply(
    #     transforms=[elastic_transform], p=0.5
    # )

    random_erase = T.RandomErasing(p=0.5, scale=(0.02, 0.05))
    random_erasers = [random_erase for _ in range(5)]

    transforms = T.Compose(
        [
            # random_apply_elastic_transform,
            GaussianNoise(sigma=0.1),
            *random_erasers,
        ]
    )

    return transforms
