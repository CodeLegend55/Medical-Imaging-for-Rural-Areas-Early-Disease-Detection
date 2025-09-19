import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np

class ChestXRayDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        # Load images and labels
        self.images = []
        self.labels = []
        self.class_names = ['covid', 'normal', 'pneumonia', 'tb']
        
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(data_dir, split, class_name)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_path, img_name))
                        self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(image_size=(224, 224), is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def load_dataset(data_dir, batch_size=32, image_size=(224, 224), num_workers=0):
    # Create datasets
    train_dataset = ChestXRayDataset(
        data_dir, 
        split='train', 
        transform=get_transforms(image_size, is_training=True)
    )
    
    test_dataset = ChestXRayDataset(
        data_dir, 
        split='test', 
        transform=get_transforms(image_size, is_training=False)
    )
    
    # Create validation dataset from train dataset
    val_dataset = ChestXRayDataset(
        data_dir, 
        split='val', 
        transform=get_transforms(image_size, is_training=False)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader, train_dataset.class_names