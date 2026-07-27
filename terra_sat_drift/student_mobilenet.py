import torch
import torch.nn as nn

from terratorch.models.backbones.unet import UNet


class UNetStudent(nn.Module):
    """Wraps terratorch's UNet backbone with a 1x1 conv head to produce class logits."""

    def __init__(self, in_channels: int, num_classes: int, out_channels: int = 32):
        super().__init__()
        self.backbone = UNet(in_channels=in_channels, out_channels=out_channels)
        self.head = nn.Conv2d(out_channels, num_classes, kernel_size=1)
        self.in_channels = in_channels
        self.decoder_out_channels = out_channels

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """Predicts class logits, optionally alongside intermediate features.

        Args:
            x: (B, C, H, W) input.
            return_features: when True, also return the representations the
                contrastive objectives in ``model_tasks.losses`` operate on.

        Returns:
            The logit map (B, num_classes, H, W) by default. With
            ``return_features=True``, a dict with keys:
              ``logits``     (B, num_classes, H, W)
              ``bottleneck`` (B, C_b, H/16, W/16) deepest encoder output, used
                             for the instance-level cross-sensor contrast
              ``decoder``    (B, out_channels, H, W) finest decoder output, used
                             for the pixel-level prototype contrast
        """
        # UNet.forward returns decoder outputs from coarsest to finest resolution:
        # index 0 is the encoder bottleneck, the last entry is at full resolution.
        stages = self.backbone(x)
        features = stages[-1]
        logits = self.head(features)
        if not return_features:
            return logits
        return {"logits": logits, "bottleneck": stages[0], "decoder": features}


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
