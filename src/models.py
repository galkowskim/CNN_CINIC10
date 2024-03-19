import torch.nn as nn
import torchvision

# OUR MODELS


class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, xb):
        return self.network(xb)


# MODELS BASED ON PRETRAINED MODELS


class PretrainedResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained ResNet model and freeze its weights
        resnet = torchvision.models.resnet18(pretrained=True)
        for name, param in resnet.named_parameters():
            if (
                "fc" not in name
            ):  # Unfreeze all layers except the last layer (fully connected layer)
                param.requires_grad = False

        # Modify the last layer for CIFAR-10 (10 classes)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, 10)
        self.network = resnet

    def forward(self, xb):
        return self.network(xb)
