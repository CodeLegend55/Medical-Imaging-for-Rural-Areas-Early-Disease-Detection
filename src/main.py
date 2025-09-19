import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import load_dataset
from models.resnet50_model import ResNet50Model
from models.densenet121_model import DenseNet121Model
from models.efficientnet_model import EfficientNetB0Model
from training.train import ModelTrainer
from training.evaluate import ModelEvaluator

def main():
    # Configuration
    DATA_DIR = r"c:\Users\91701\Documents\ACADEMICS VITAP\4th year\Capstone\Code\chest_xray_merged"  # Update this path
    BATCH_SIZE = 16  # Reduced for memory efficiency
    IMAGE_SIZE = (224, 224)
    NUM_EPOCHS = 25
    LEARNING_RATE = 0.001
    NUM_CLASSES = 4
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    try:
        train_loader, val_loader, test_loader, class_names = load_dataset(
            DATA_DIR, 
            batch_size=BATCH_SIZE, 
            image_size=IMAGE_SIZE,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        print(f"Classes: {class_names}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please make sure the dataset is properly structured and the path is correct.")
        return
    
    # Models to train
    models_to_train = {
        'ResNet50': ResNet50Model(num_classes=NUM_CLASSES),
        'DenseNet121': DenseNet121Model(num_classes=NUM_CLASSES),
        'EfficientNetB0': EfficientNetB0Model(num_classes=NUM_CLASSES)
    }
    
    results = {}
    
    # Train each model
    for model_name, model in models_to_train.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        try:
            # Create trainer
            trainer = ModelTrainer(model, device, NUM_CLASSES)
            
            # Train model
            trained_model = trainer.train_model(
                train_loader, 
                val_loader, 
                num_epochs=NUM_EPOCHS,
                learning_rate=LEARNING_RATE
            )
            
            # Evaluate model
            print(f"\nEvaluating {model_name}...")
            evaluator = ModelEvaluator(trained_model, device)
            y_true, y_pred, accuracy = evaluator.evaluate_model(test_loader, class_names)
            
            # Plot training history
            trainer.plot_training_history()
            
            # Save model
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, f"{model_name}_chest_xray.pth")
            torch.save(trained_model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            
            results[model_name] = {
                'model': trained_model,
                'trainer': trainer,
                'accuracy': accuracy,
                'predictions': (y_true, y_pred)
            }
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue
    
    # Compare results
    if results:
        print("\n" + "="*50)
        print("FINAL RESULTS COMPARISON")
        print("="*50)
        for model_name, result in results.items():
            print(f"{model_name}: {result['accuracy']:.4f}")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest model: {best_model[0]} with accuracy: {best_model[1]['accuracy']:.4f}")
    
    print("\nTraining completed!")
    return results

if __name__ == "__main__":
    results = main()