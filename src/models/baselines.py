import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights
)

def build_resnet18(num_classes, pretrained=True):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights) # уже обучена на ImageNet, нужно дообучить только на наших классах
    model.fc = nn.Linear(model.fc.in_features, num_classes) # последний слой для нужного числа классов
    return model

def build_resnet50(num_classes, pretrained=True):
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def build_mobilenet_v3_large(num_classes, pretrained=True):
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model

def build_efficientnet_b0(num_classes, pretrained=True):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def build_efficientnet_b2(num_classes, pretrained=True):
    weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b2(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def build_convnext_tiny(num_classes, pretrained=True):
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = convnext_tiny(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model


MODEL_BUILDERS = {
    'resnet18': build_resnet18,
    'resnet50': build_resnet50,
    'mobilenet_v3_large': build_mobilenet_v3_large,
    'efficientnet_b0': build_efficientnet_b0,
    'efficientnet_b2': build_efficientnet_b2,
    'convnext_tiny': build_convnext_tiny
}