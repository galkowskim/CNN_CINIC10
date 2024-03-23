import torch
import torch.nn as nn
import torch.nn.functional as F
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


class LeNet5BasedModelFor32x32Images(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(16 * 6 * 6, 256)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class VGG16BasedModelFor32x32Images(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512 * 4, 512), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 512), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(512, 10))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# Blocks for Resnet model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBasedModelFor32x32Images(nn.Module):  # this model requires 224x224 images
    def __init__(self, block, layers, num_classes=10):
        super(ResNetBasedModelFor32x32Images, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer0 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class WideModel(nn.Module):
    def __init__(self):
        super(WideModel, self).__init__()
        self.branch1 = LeNet5BasedModelFor32x32Images()
        self.branch2 = VGG16BasedModelFor32x32Images()

        self.fc1 = nn.Linear(20, 10)  # Assuming the concatenated output size is 20

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)

        concatenated = torch.cat((out1, out2), dim=1)
        output = F.relu(self.fc1(concatenated))
        return output


# MODELS BASED ON PRETRAINED MODELS
class PretrainedResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained ResNet model and freeze its weights
        resnet = torchvision.models.resnet18(weights="DEFAULT")

        # Modify the last layer for CIFAR-10 (10 classes)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, 10)
        self.network = resnet
        self.transform = torchvision.models.ResNet18_Weights.DEFAULT.transforms(
            antialias=True
        )

    def forward(self, xb):
        xb = self.transform(xb)
        return self.network(xb)


class PretrainedVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained VGG16 model and freeze its weights
        vgg16 = torchvision.models.vgg16(weights="DEFAULT")

        num_ftrs = vgg16.classifier[6].in_features
        vgg16.classifier[6] = nn.Linear(num_ftrs, 10)
        self.network = vgg16
        self.transform = torchvision.models.VGG16_Weights.DEFAULT.transforms(
            antialias=True
        )

    def forward(self, xb):
        xb = self.transform(xb)
        return self.network(xb)


class PretrainedAlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained AlexNet model and freeze its weights
        alexnet = torchvision.models.alexnet(weights="DEFAULT")

        num_ftrs = alexnet.classifier[6].in_features
        alexnet.classifier[6] = nn.Linear(num_ftrs, 10)
        self.network = alexnet
        self.transform = torchvision.models.AlexNet_Weights.DEFAULT.transforms(
            antialias=True
        )

    def forward(self, xb):
        xb = self.transform(xb)
        return self.network(xb)
