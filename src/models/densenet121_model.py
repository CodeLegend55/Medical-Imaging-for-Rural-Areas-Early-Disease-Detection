import torch
import torch.nn as nn
from torchvision import models

class DenseNet121Model(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(DenseNet121Model, self).__init__()
        
        # Load pre-trained DenseNet121
        self.densenet = models.densenet121(pretrained=pretrained)
        
        # Freeze early layers (optional for fine-tuning)
        for param in list(self.densenet.parameters())[:-20]:
            param.requires_grad = False
            
        # Replace the classifier
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.densenet(x)
    
    def unfreeze_all_layers(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.densenet.parameters():
            param.requires_grad = True