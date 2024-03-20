import humanize

from models import (
    CustomCNN,
    LeNet5BasedModelFor32x32Images,
    PretrainedAlexNet,
    PretrainedResNet,
    PretrainedVGG16,
    ResidualBlock,
    ResNetBasedModelFor32x32Images,
    VGG16BasedModelFor32x32Images,
)

for i, model in enumerate(
    [
        CustomCNN,
        LeNet5BasedModelFor32x32Images,
        ResNetBasedModelFor32x32Images,
        VGG16BasedModelFor32x32Images,
        PretrainedAlexNet,
        PretrainedResNet,
        PretrainedVGG16,
    ]
):
    if i == 0:
        print("-----------------------------------------")
        print("Custom models - own implementations")
    if i == 4:
        print("-----------------------------------------")
        print("Pretrained models - torchvision.models")
    model = (
        model()
        if model != ResNetBasedModelFor32x32Images
        else model(ResidualBlock, [2, 2, 2, 2])
    )
    print(model.__class__.__name__)
    print(
        "Number of parameters:",
        humanize.intword(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    )
    print()
