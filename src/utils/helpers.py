def save_model_checkpoint(model, filepath):
    """Saves the model checkpoint to the specified filepath."""
    torch.save(model.state_dict(), filepath)

def load_model_checkpoint(model, filepath):
    """Loads the model checkpoint from the specified filepath."""
    model.load_state_dict(torch.load(filepath))
    return model

def log_training_info(epoch, loss, accuracy):
    """Logs training information for each epoch."""
    print(f'Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

def calculate_class_distribution(labels):
    """Calculates the distribution of classes in the dataset."""
    class_counts = {}
    for label in labels:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    return class_counts

def visualize_sample_images(images, labels, num_samples=5):
    """Visualizes a few sample images from the dataset."""
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(labels[i])
        plt.axis('off')
    plt.show()