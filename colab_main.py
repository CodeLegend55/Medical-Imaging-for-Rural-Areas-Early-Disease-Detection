"""
Colab-optimized version of the medical imaging project
This script is designed to run efficiently in Google Colab environment
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import sys
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Colab-specific configurations
class ColabConfig:
    # Paths (will be updated in Colab)
    DATA_DIR = "./chest_xray_merged"
    MODELS_DIR = "./models"
    
    # Training parameters optimized for Colab
    BATCH_SIZE = 32  # Can use larger batch size with Colab GPU
    IMAGE_SIZE = (224, 224)
    NUM_EPOCHS = 20  # Reduced for faster training in free tier
    LEARNING_RATE = 0.001
    NUM_CLASSES = 4
    NUM_WORKERS = 2  # Colab can handle multiple workers
    
    # Class names
    CLASS_NAMES = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']

def setup_device():
    """Setup and verify GPU availability"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  GPU not available. Training will be slow on CPU.")
        print("Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU")
    
    return device

def create_data_transforms():
    """Create data transforms optimized for medical images"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform

def load_colab_dataset(data_dir, batch_size=32, num_workers=2):
    """Load dataset optimized for Colab environment"""
    train_transform, val_test_transform = create_data_transforms()
    
    # Load datasets
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), 
        transform=train_transform
    )
    
    val_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'val'), 
        transform=val_test_transform
    )
    
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), 
        transform=val_test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    class_names = train_dataset.classes
    
    print(f"Dataset loaded successfully!")
    print(f"Classes: {class_names}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, class_names

class ColabModelTrainer:
    """Simplified model trainer for Colab environment"""
    
    def __init__(self, model, device, num_classes):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_model(self, train_loader, val_loader, num_epochs=20, learning_rate=0.001):
        """Train the model with progress tracking"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            for images, labels in train_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                train_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
            
            # Validation phase
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Update learning rate
            scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print('-' * 50)
        
        print("Training completed!")
        return self.model
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def evaluate_model(model, test_loader, class_names, device):
    """Evaluate model and show detailed results"""
    model.eval()
    y_true = []
    y_pred = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return y_true, y_pred, accuracy

def create_model(model_name, num_classes=4):
    """Create model based on name"""
    if model_name == 'ResNet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'DenseNet121':
        model = torchvision.models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'EfficientNetB0':
        model = torchvision.models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

def main_colab():
    """Main function optimized for Google Colab"""
    print("🚀 Starting Medical Imaging Project in Google Colab")
    print("=" * 60)
    
    # Setup
    device = setup_device()
    config = ColabConfig()
    
    # Load dataset
    try:
        train_loader, val_loader, test_loader, class_names = load_colab_dataset(
            config.DATA_DIR, 
            batch_size=config.BATCH_SIZE, 
            num_workers=config.NUM_WORKERS
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Please make sure your dataset is uploaded to Google Drive and paths are correct.")
        return None
    
    # Models to train
    model_names = ['ResNet50', 'DenseNet121', 'EfficientNetB0']
    results = {}
    
    # Create models directory
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # Train each model
    for model_name in model_names:
        print(f"\\n🏋️  Training {model_name}")
        print("=" * 50)
        
        try:
            # Create model
            model = create_model(model_name, config.NUM_CLASSES)
            
            # Create trainer
            trainer = ColabModelTrainer(model, device, config.NUM_CLASSES)
            
            # Train model
            trained_model = trainer.train_model(
                train_loader, 
                val_loader, 
                num_epochs=config.NUM_EPOCHS,
                learning_rate=config.LEARNING_RATE
            )
            
            # Plot training history
            trainer.plot_training_history()
            
            # Evaluate model
            print(f"\\n📊 Evaluating {model_name}...")
            y_true, y_pred, accuracy = evaluate_model(
                trained_model, test_loader, class_names, device
            )
            
            # Save model
            model_path = os.path.join(config.MODELS_DIR, f"{model_name}_colab.pth")
            torch.save(trained_model.state_dict(), model_path)
            print(f"💾 Model saved to {model_path}")
            
            results[model_name] = {
                'model': trained_model,
                'trainer': trainer,
                'accuracy': accuracy,
                'predictions': (y_true, y_pred)
            }
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            continue
    
    # Compare results
    if results:
        print("\\n🏆 FINAL RESULTS COMPARISON")
        print("=" * 60)
        
        model_names = []
        accuracies = []
        
        for model_name, result in results.items():
            print(f"{model_name}: {result['accuracy']:.4f}")
            model_names.append(model_name)
            accuracies.append(result['accuracy'])
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\\n🥇 Best model: {best_model[0]} with accuracy: {best_model[1]['accuracy']:.4f}")
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, 
                    f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    print("\\n✅ Training completed successfully in Google Colab!")
    return results

# For direct execution in Colab
if __name__ == "__main__":
    results = main_colab()