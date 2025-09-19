import torch
import torch.nn as nn
from torchvision import models

class EfficientNetB0Model(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(EfficientNetB0Model, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(pretrained=pretrained)
        
        # Freeze early layers (optional for fine-tuning)
        for param in list(self.efficientnet.parameters())[:-15]:
            param.requires_grad = False
            
        # Replace the classifier
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.efficientnet(x)
    
    def unfreeze_all_layers(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.efficientnet.parameters():
            param.requires_grad = True