from typing import Tuple

import torch
import torchvision
from torchvision import transforms

CINIC_MEAN = [0.47889522, 0.47227842, 0.43047404]
CINIC_STD = [0.24205776, 0.23828046, 0.25874835]


class AddGaussianNoise(object):
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        tensor = transforms.ToTensor()(img)
        noise = torch.randn_like(tensor) * self.std + self.mean
        noisy_tensor = tensor + noise
        noisy_img = transforms.ToPILImage()(noisy_tensor)
        return noisy_img


def get_data(
    data_path: str = ".",
    batch_size: int = 256,
    augmentation: bool = False,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:

    transform_train = (
        transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.5, contrast=0.5),
                AddGaussianNoise(mean=0, std=0.005),
                transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=CINIC_MEAN, std=CINIC_STD),
            ]
        )
        if augmentation
        else transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=CINIC_MEAN, std=CINIC_STD),
            ]
        )
    )

    cinic_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            data_path + "/train", transform=transform_train
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    valid_and_test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CINIC_MEAN, std=CINIC_STD),
        ]
    )

    cinic_valid = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            data_path + "/valid", transform=valid_and_test_transform
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    cinic_test = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            data_path + "/test", transform=valid_and_test_transform
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return cinic_train, cinic_valid, cinic_test
