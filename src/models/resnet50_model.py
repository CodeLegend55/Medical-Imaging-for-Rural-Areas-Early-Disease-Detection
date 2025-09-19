import torch
import torch.nn as nn
from torchvision import models

class ResNet50Model(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(ResNet50Model, self).__init__()
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Freeze early layers (optional for fine-tuning)
        for param in list(self.resnet.parameters())[:-10]:
            param.requires_grad = False
            
        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)
    
    def unfreeze_all_layers(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.resnet.parameters():
            param.requires_grad = True