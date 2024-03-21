from typing import Tuple

import torch
import torchvision
from torchvision import transforms

CINIC_MEAN = [0.47889522, 0.47227842, 0.43047404]
CINIC_STD = [0.24205776, 0.23828046, 0.25874835]


def get_data(
    data_path: str = ".",
    batch_size: int = 256,
    augmentation: bool = False,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:

    if augmentation:
        # TODO: Not sure if we want all augmentations at once or if we want to be able to choose which ones to use + Resize to pretrained_model_input_size
        ...
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=CINIC_MEAN, std=CINIC_STD),
            ]
        )

    cinic_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(data_path + "/train", transform=transform),
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
        torchvision.datasets.ImageFolder(data_path + "/valid", transform=valid_and_test_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    cinic_test = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(data_path + "/test", transform=valid_and_test_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return cinic_train, cinic_valid, cinic_test
