import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

def create_deeplab_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Create a DeepLabV3 model with a ResNet-50 backbone for semantic segmentation.
    Args:
        num_classes: number of segmentation classes in our dataset.
        pretrained:  if True, load torchvision's pretrained weights (good for transfer learning).
    Returns:
        model: a torch.nn.Module ready to train / evaluate.
    """
    # Load the base DeepLabV3-ResNet50 model from torchvision.
    # This comes with:
    # - backbone: ResNet-50 convolutional feature extractor
    # - classifier: DeepLab head that produces per-pixel logits
    model = deeplabv3_resnet50(pretrained=pretrained, progress=True)

    # model.classifier is the segmentation head that takes the backbone and produces per pixel class scores.
    # Model is usually defined as an nn.Sequential of several layers. Last layer is a Conv2d which maps from feature channels (eg: 256) to the number of classes used in pretraining (eg: 21)
    # print(model.backbone)
    # ResNet-50 is a really good feature extractor. All the conv layers up to layer4 learn rich visual features: edges, textures, shapes, object parts, etc.
    # The last part (avgpool + fc) is just: Global average pooling -> 2048-dim vector. Fully connected layer -> 1000 logits.
    # print(model.classifier)
    # For segmentation (or detection), keep the conv backbone (everything up to layer4). Discard avgpool and fc. `deeplabv3_resnet50` exactly does this!

    # We want to adapt the model to our own dataset with `num_classes` labels, so we replace the last convolution later.
    #   - 256 input channels  (must match the previous layer's output)
    #   - num_classes outputs (one logit per class, per pixel)
    #   - 1x1 kernel          (acts like a per-pixel fully connected layer)
    model.classifier[-1] = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)

    return model

'''
DeepLabV3(
  (backbone): IntermediateLayerGetter(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn1): BatchNorm2d(64, ...)
    (relu): ReLU(...)
    (maxpool): MaxPool2d(...)
    (layer1): Sequential(...)
    (layer2): Sequential(...)
    (layer3): Sequential(...)
    (layer4): Sequential(...)
  )
  (classifier): DeepLabHead(
    (0): ASPP(...)
    (1): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (2): BatchNorm2d(256, ...)
    (3): ReLU(...)
    (4): Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))   <-- final classification later, we need to replace.
  )
  (aux_classifier): ...
)
'''