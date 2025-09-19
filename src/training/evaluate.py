import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

class ModelEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def evaluate_model(self, test_loader, class_names):
        """Evaluate model on test dataset"""
        self.model.eval()
        y_pred = []
        y_true = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                y_pred.extend(preds.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Print overall metrics
        print(f"\nOverall Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Confusion matrix
        self.plot_confusion_matrix(y_true, y_pred, class_names)
        
        return y_true, y_pred, accuracy
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def predict_single_image(self, image_tensor):
        """Predict on a single image"""
        self.model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            return predicted.item(), probabilities.squeeze().cpu().numpy()

def evaluate_model(model, dataloader, device):
    """Legacy function for backwards compatibility"""
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = correct / total
    return accuracy, all_labels, all_predictions

def calculate_metrics(all_labels, all_predictions):
    """Legacy function for backwards compatibility"""
    report = classification_report(all_labels, all_predictions, 
                                 target_names=['COVID', 'Normal', 'Pneumonia', 'TB'])
    return report

def main_evaluation(model, val_dataloader, device):
    """Legacy function for backwards compatibility"""
    accuracy, all_labels, all_predictions = evaluate_model(model, val_dataloader, device)
    report = calculate_metrics(all_labels, all_predictions)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print('Classification Report:')
    print(report)