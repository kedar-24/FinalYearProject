import torch
import torch.nn as nn
from torchvision import models

class MultiClassFocalLoss(nn.Module):
    """
    Multi-class Focal Loss (gamma=2, alpha=0.25).
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = 'mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce_loss(logits, targets)
        pt = torch.exp(-ce_loss)
        
        # Calculate focal loss
        focal_term = (1 - pt) ** self.gamma
        loss = self.alpha * focal_term * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class FLEfficientNet(nn.Module):
    """
    Custom EfficientNet-B0 with a modified classifier head.
    Can be used with standard Cross-Entropy or Focal Loss logic.
    """
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(FLEfficientNet, self).__init__()
        
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Replace the final linear layer
        # EfficientNet uses 'classifier' as the head, which is Sequential with Dropout and Linear
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def get_efficientnet_model(num_classes, pretrained=True):
    return FLEfficientNet(num_classes, pretrained)
