# Medical Imaging for Rural Areas - Early Disease Detection

A deep learning project for automated chest X-ray analysis to detect COVID-19, Normal, Pneumonia, and Tuberculosis cases using state-of-the-art pre-trained models.

## 🎯 Project Overview

This project aims to provide accessible medical imaging solutions for rural areas where specialist radiologists may not be available. The system uses fine-tuned deep learning models to automatically analyze chest X-rays and detect four conditions:

- **COVID-19**: SARS-CoV-2 related pneumonia
- **Normal**: Healthy chest X-rays
- **Pneumonia**: Bacterial/viral pneumonia (non-COVID)
- **Tuberculosis**: TB infections

## 👥 Team Members

- **Neeraj Tirumalasetty** - [https://github.com/neerajtirumalasetty](https://github.com/neerajtirumalasetty)
- **Jeevan Rushi** - [https://github.com/jeevanrushi07](https://github.com/jeevanrushi07)

## 🏗️ Architecture

The project implements three state-of-the-art CNN architectures:

1. **ResNet50**: Deep residual learning with skip connections
2. **DenseNet121**: Dense connectivity between layers
3. **EfficientNetB0**: Compound scaling for efficiency and accuracy

All models are pre-trained on ImageNet and fine-tuned on chest X-ray data for transfer learning.

## 📁 Project Structure

```
Medical-Imaging-for-Rural-Areas-Early-Disease-Detection/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py          # Data loading and preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resnet50_model.py       # ResNet50 implementation
│   │   ├── densenet121_model.py    # DenseNet121 implementation
│   │   └── efficientnet_model.py   # EfficientNet implementation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                # Training pipeline
│   │   └── evaluate.py             # Evaluation and metrics
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py               # Configuration utilities
│   │   └── helpers.py              # Helper functions
│   └── main.py                     # Main training script
├── notebooks/
│   ├── data_exploration.ipynb      # Data analysis
│   └── model_comparison.ipynb      # Model comparison
├── tests/
│   ├── __init__.py
│   └── test_models.py              # Unit tests
├── models/                         # Saved model weights
├── logs/                          # Training logs
├── chest_xray_merged/             # Dataset directory
│   ├── train/
│   ├── val/
│   └── test/
├── colab_setup.ipynb              # 🆕 Google Colab setup notebook
├── colab_main.py                  # 🆕 Colab-optimized training script
├── config.yaml                    # Configuration file
├── requirements.txt               # Dependencies
├── download_dataset.py           # Dataset download utility
└── README.md                     # This file
```

## 🚀 Quick Start

### Option 1: Google Colab (Recommended - Free GPU!) 🔥

For users without powerful hardware, we provide a complete Google Colab setup with **free GPU access**:

1. **Upload to Google Drive:**
   - Download/clone this repository
   - Upload the entire folder to your Google Drive

2. **Open Google Colab:**
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - Upload and open `colab_setup.ipynb`

3. **Enable GPU:**
   - Runtime → Change runtime type → Hardware accelerator → **GPU**
   - Select GPU type → **T4** (free tier)

4. **Run the notebook:**
   - Update the Google Drive path in the notebook
   - Run all cells sequentially
   - The notebook will automatically train all models!

**Colab Advantages:**
- ✅ Free Tesla T4 GPU (~16GB VRAM)
- ✅ Pre-installed PyTorch and dependencies
- ✅ No local setup required
- ✅ Automatic model saving to Drive
- ✅ ~12 hours of continuous GPU usage

### Option 2: Local Environment Setup

```bash
# Clone the repository
git clone <repository_url>
cd Medical-Imaging-for-Rural-Areas-Early-Disease-Detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Download

The project uses the [Chest X-Ray Images (Pneumonia/COVID19/Tuberculosis)](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis) dataset from Kaggle.

**Setup Kaggle API:**
1. Go to [Kaggle Account Settings](https://www.kaggle.com/account)
2. Create new API token (downloads `kaggle.json`)
3. Place `kaggle.json` in:
   - **Windows**: `C:\Users\{username}\.kaggle\`
   - **Linux/Mac**: `~/.kaggle/`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac)

**Download dataset:**
```bash
python download_dataset.py
```

### Training Models

#### Option 1: Google Colab (Easy!)
Simply run the `colab_setup.ipynb` notebook - it handles everything automatically!

#### Option 2: Local Training
```bash
# Train all models (ResNet50, DenseNet121, EfficientNet)
python src/main.py

# Or use the Colab-optimized script locally
python colab_main.py

# Or train individual models by modifying the main.py file
```

### Configuration

Modify `config.yaml` to adjust:
- Training parameters (batch size, learning rate, epochs)
- Model architectures
- Data augmentation settings
- Device settings

## 📊 Dataset Information

- **Total Classes**: 4 (COVID, Normal, Pneumonia, TB)
- **Image Format**: PNG/JPG chest X-rays
- **Image Size**: Resized to 224x224 pixels
- **Data Split**: Train/Validation/Test
- **Preprocessing**: Normalization with ImageNet statistics

### Data Augmentation
- Random horizontal flip
- Random rotation (±10°)
- Color jitter (brightness, contrast)
- Random crop and resize

## 🎯 Model Performance

The project evaluates models using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Class-wise precision scores
- **Recall**: Class-wise recall scores
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## 🔧 Technical Details

### Models Architecture
- **Input**: 224×224×3 RGB images
- **Pre-training**: ImageNet weights
- **Fine-tuning**: Last layers unfrozen
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout layers, weight decay

### Training Strategy
- **Transfer Learning**: Pre-trained backbone + custom classifier
- **Gradual Unfreezing**: Initial frozen layers, then gradual unfreezing
- **Learning Rate Scheduling**: StepLR with gamma=0.1, step_size=7
- **Early Stopping**: Based on validation accuracy

### Colab Optimizations 🚀
- **Batch Size**: Increased to 32 (leveraging GPU memory)
- **Data Loading**: Multi-worker data loading for efficiency
- **Memory Management**: Optimized for Colab's T4 GPU
- **Progress Tracking**: Real-time training progress bars
- **Auto-saving**: Models automatically saved to Google Drive

## 📈 Usage Examples

### Google Colab Usage (Easiest!)
1. Upload `colab_setup.ipynb` to Google Colab
2. Enable GPU runtime
3. Run all cells - everything is automated!

### Training Custom Model (Local/Colab)
```python
# Using the Colab-optimized trainer
from colab_main import ColabModelTrainer, create_model, setup_device

# Setup
device = setup_device()
model = create_model('ResNet50', num_classes=4)
trainer = ColabModelTrainer(model, device, 4)

# Train
trained_model = trainer.train_model(train_loader, val_loader, num_epochs=25)
```

### Original Implementation
```python
from src.models.resnet50_model import ResNet50Model
from src.training.train import ModelTrainer

# Initialize model
model = ResNet50Model(num_classes=4)
trainer = ModelTrainer(model, device='cuda')

# Train
trained_model = trainer.train_model(train_loader, val_loader, num_epochs=25)
```

### Making Predictions
```python
from src.training.evaluate import ModelEvaluator

evaluator = ModelEvaluator(trained_model, device='cuda')
prediction, probabilities = evaluator.predict_single_image(image_tensor)
```

## 🏥 Medical Applications

This system is designed for:
- **Rural Healthcare Centers**: Primary screening tool
- **Telemedicine**: Remote consultation support  
- **Medical Education**: Training and reference tool
- **Epidemiological Studies**: Large-scale screening

### Important Medical Disclaimer
⚠️ **This tool is for research and educational purposes only. It should not replace professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.**

## 📋 Requirements

### Hardware Requirements

#### For Google Colab Users 🎉
- ✅ **No special hardware needed!**
- ✅ Free Tesla T4 GPU provided by Google
- ✅ 12+ GB GPU memory available
- ✅ Works on any device with internet connection

#### For Local Training
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: 5GB+ for dataset and models
- **Internet**: For dataset download and Colab access

### Software
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.6+ (for GPU support)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis)
- Pre-trained models: PyTorch Model Zoo
- Inspiration: Medical AI research community

## 📞 Support

### Getting Started Issues?
- **Can't find GPU in Colab?** 
  - Runtime → Change runtime type → Hardware accelerator → GPU
- **Dataset loading errors?**
  - Verify Google Drive path in the Colab notebook
  - Check dataset folder structure matches expected format
- **Out of memory errors?**
  - Reduce batch size in config (try 16 or 8)
  - Use gradient checkpointing for larger models

### General Support
- Open an issue on GitHub
- Check the documentation in `/notebooks`
- Try the Google Colab setup first - it's the easiest!

### Troubleshooting Colab
- **Session timeout**: Colab sessions last ~12 hours of inactivity
- **Reconnect**: If disconnected, just reconnect and continue
- **Save progress**: Models are auto-saved to your Google Drive

---

**Made with ❤️ for rural healthcare accessibility**

---

## 🌟 New Features in Colab Version

- 🔥 **Free GPU Training**: No hardware requirements
- 📱 **Mobile Friendly**: Train models from phone/tablet
- ☁️ **Cloud Storage**: Auto-save to Google Drive
- 📊 **Real-time Visualization**: Live training progress
- 🎯 **One-click Setup**: Everything automated in notebook
- 💾 **Easy Model Download**: Direct download to local machine