import torch
import torch.nn as nn

from terratorch.models.backbones.unet import UNet


class UNetStudent(nn.Module):
    """Wraps terratorch's UNet backbone with a 1x1 conv head to produce class logits."""

    def __init__(self, in_channels: int, num_classes: int, out_channels: int = 32):
        super().__init__()
        self.backbone = UNet(in_channels=in_channels, out_channels=out_channels)
        self.head = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # UNet.forward returns decoder outputs from coarsest to finest resolution;
        # the last entry is at the input's full resolution.
        features = self.backbone(x)[-1]
        return self.head(features)


def create_student_model(
    in_channels: int = 7,
    num_classes: int = 2,
    pretrained: bool = False,
) -> nn.Module:
    """
    Creates a lightweight UNet student model using terratorch's UNet backbone.

    Args:
        in_channels (int): Number of input spectral bands (e.g., 7 for S2L1C).
        num_classes (int): Number of segmentation classes.
        pretrained (bool): Unused. terratorch's UNet backbone is trained from
                           scratch and has no pretrained weights available.
    """
    if pretrained:
        raise ValueError(
            "terratorch's UNet backbone has no pretrained weights available; "
            "call create_student_model with pretrained=False."
        )

    return UNetStudent(in_channels=in_channels, num_classes=num_classes)
