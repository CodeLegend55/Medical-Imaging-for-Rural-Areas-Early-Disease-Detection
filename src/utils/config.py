import os

class Config:
    def __init__(self):
        # Dataset paths
        self.dataset_path = os.path.join('data', 'chest_xray')
        self.train_data_path = os.path.join(self.dataset_path, 'train')
        self.val_data_path = os.path.join(self.dataset_path, 'val')
        self.test_data_path = os.path.join(self.dataset_path, 'test')

        # Model parameters
        self.input_shape = (224, 224, 3)  # Input shape for the models
        self.num_classes = 4  # COVID, Normal, Pneumonia, TB

        # Training parameters
        self.batch_size = 32
        self.epochs = 50
        self.learning_rate = 0.001

        # Checkpoint settings
        self.checkpoint_dir = 'checkpoints'
        self.model_save_path = os.path.join(self.checkpoint_dir, 'best_model.h5')

        # Logging settings
        self.log_dir = 'logs'