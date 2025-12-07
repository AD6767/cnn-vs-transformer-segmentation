import torch.nn as nn
import segmentation_models_pytorch as smp

class SegFormerB0Wrapper(nn.Module):
    """
    Wrap SMP's Segformer so it behaves like torchvision segmentation models:
    forward(x) -> {"out": logits} where logits = (B, num_classes, H, W).
    """
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        # SMP Segformer with MiT-B0 encoder.
        # encoder_name: "mit_b0" is the small SegFormer encoder (Mix Vision Transformer).
        # encoder_weights: "imagenet" for pretrained, or None for random init.
        encoder_weights = "imagenet" if pretrained else None
        self.model = smp.Segformer(encoder_name="mit_b0", encoder_weights=encoder_weights, in_channels=3, classes=num_classes)

    def forward(self, x):
        # SMP Segformer returns logits directly: (B, num_classes, H, W)
        logits = self.model(x)
        # Wrap to match your DeepLab API ("out" key)
        return {"out": logits}
    
def create_segformer_b0(num_classes: int, pretrained: bool = True) -> nn.Module:
    return SegFormerB0Wrapper(num_classes=num_classes, pretrained=pretrained)
