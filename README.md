# Medical Imaging for Rural Areas - Early Disease Detection

A **Google Colab-ready** deep learning project for automated chest X-ray analysis to detect COVID-19, Normal, Pneumonia, and Tuberculosis cases using state-of-the-art pre-trained models.

## 🎯 Project Overview

This project provides accessible medical imaging solutions for rural areas where specialist radiologists may not be available. The system uses fine-tuned deep learning models to automatically analyze chest X-rays and detect four conditions:

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

## 📁 **Ultra-Minimal Project Structure** (Perfect for Colab)

```
Medical-Imaging-for-Rural-Areas-Early-Disease-Detection/
├── colab_setup_fixed.ipynb       # 🚀 THE ONLY FILE YOU NEED!
├── README.md                     # This documentation
└── chest_xray_merged/            # Dataset (upload this folder to Drive)
    ├── train/
    │   ├── covid/
    │   ├── normal/
    │   ├── pneumonia/
    │   └── tb/
    ├── val/
    │   ├── covid/
    │   ├── normal/
    │   ├── pneumonia/
    │   └── tb/
    └── test/
        ├── covid/
        ├── normal/
        ├── pneumonia/
        └── tb/
```

**🎯 That's it! Just TWO files + your dataset!**

## 🚀 **Quick Start - Google Colab (Recommended)**

### **⚡ One-File Solution!**

Everything you need is in **`colab_setup_fixed.ipynb`** - a single, self-contained notebook!

### **Steps:**

1. **📤 Upload to Google Drive:**
   - Download this repository
   - Upload the entire folder to your Google Drive

2. **🔓 Open Google Colab:**
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - Upload `colab_setup_fixed.ipynb`

3. **⚡ Enable GPU:**
   - Runtime → Change runtime type → Hardware accelerator → **GPU**
   - Select GPU type → **T4** (free tier)

4. **▶️ Run Everything:**
   - Run all cells sequentially
   - The notebook handles everything automatically!

### **🎯 What the Notebook Does:**

✅ **Auto-detects** your dataset location  
✅ **Verifies** dataset structure and image counts  
✅ **Installs** all required packages  
✅ **Creates** and trains all 3 models  
✅ **Evaluates** performance with detailed metrics  
✅ **Saves** trained models to your Drive  
✅ **Visualizes** results with plots and confusion matrices  

## 🎁 **Colab Benefits:**

- 🆓 **Free Tesla T4 GPU** (~16GB VRAM)
- 🚀 **No local setup required**
- 📱 **Works on any device** (phone, tablet, laptop)
- ☁️ **Auto-save to Google Drive**
- 🛡️ **Built-in error handling**
- 📊 **Real-time progress tracking**

## 🚀 Quick Start

### Dataset Download

The project uses the [Chest X-Ray Images (Pneumonia/COVID19/Tuberculosis)](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis) dataset from Kaggle.

**🎯 Easiest Method: Manual Upload**
1. Download the dataset from Kaggle
2. Extract and organize as shown in project structure above
3. Upload the `chest_xray_merged` folder to your Google Drive

**🔧 Alternative: Download in Colab**
The notebook includes **built-in instructions** for downloading directly in Colab using Kaggle API (no separate script needed!).

## 📊 Dataset Information

- **Total Classes**: 4 (COVID, Normal, Pneumonia, TB)
- **Image Format**: PNG/JPG chest X-rays
- **Image Size**: Resized to 224x224 pixels
- **Data Split**: Train/Validation/Test
- **Preprocessing**: Normalization with ImageNet statistics

### Data Augmentation (Built into Notebook)
- Random horizontal flip
- Random rotation (±10°)
- Color jitter (brightness, contrast)
- Random crop and resize

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

The notebook evaluates all models using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Class-wise precision scores
- **Recall**: Class-wise recall scores
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown
- **Training History**: Loss and accuracy plots

## 🔧 Technical Details

### Models Architecture (Built into Notebook)
- **Input**: 224×224×3 RGB images
- **Pre-training**: ImageNet weights (via torchvision.models)
- **Fine-tuning**: Last layers replaced for 4-class classification
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout layers, weight decay

### Training Strategy
- **Transfer Learning**: Pre-trained backbone + custom classifier
- **Learning Rate Scheduling**: StepLR with gamma=0.1, step_size=7
- **Batch Size**: 32 (optimized for Colab GPU)
- **Data Loading**: Multi-worker processing for efficiency

### Colab Optimizations 🚀
- **GPU Memory**: Optimized for Tesla T4 (16GB)
- **Error Handling**: Comprehensive dataset verification
- **Progress Tracking**: Real-time training progress bars
- **Auto-saving**: Models automatically saved to Google Drive
- **Visualization**: Built-in plots and confusion matrices

## 📈 Usage Examples

### **🚀 Primary Usage: Google Colab**
1. Upload `colab_setup_fixed.ipynb` to Google Colab
2. Enable GPU runtime
3. Run all cells - everything is automated!

### **📊 What You Get:**
- **Trained Models**: All 3 models (ResNet50, DenseNet121, EfficientNetB0)
- **Performance Metrics**: Detailed accuracy, precision, recall, F1-scores
- **Visualizations**: Training curves, confusion matrices, sample images
- **Model Files**: Downloadable .pth files saved to your Drive

### **🔧 Customization:**
The notebook includes easy-to-modify settings:
```python
# Configuration (in notebook)
BATCH_SIZE = 32          # Adjust based on GPU memory
NUM_EPOCHS = 20          # Training duration
LEARNING_RATE = 0.001    # Learning rate
NUM_CLASSES = 4          # Fixed for this dataset
```

### Making Predictions
```python
# The notebook includes built-in evaluation
# Results are automatically displayed with:
# - Accuracy scores for each model
# - Detailed classification reports
# - Confusion matrices
# - Performance comparison charts
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

#### For Google Colab Users 🎉
- ✅ **Nothing to install!** Everything is handled automatically
- ✅ Just a web browser and Google account
- ✅ All packages pre-installed or auto-installed by notebook
- ✅ No version conflicts or dependency issues

**💻 For Local Development (Not Recommended):**
- The notebook can technically run locally with Python 3.8+, PyTorch 1.12+
- But Google Colab is much easier and provides free GPU access!

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

**📱 For Google Colab (Recommended Path):**
- **Can't find GPU?** Runtime → Change runtime type → Hardware accelerator → GPU
- **Dataset not found?** Check Step 3 in notebook - it will guide you
- **Out of memory?** Reduce BATCH_SIZE in notebook (try 16 or 8)
- **Session timeout?** Colab sessions last ~12 hours - just reconnect

**💻 For Local Development:**
- Check `requirements.txt` for dependencies
- Ensure Python 3.8+ and PyTorch are installed
- Verify dataset folder structure matches project structure

### General Support
- 🐛 **Bug reports**: Open an issue on GitHub
- 📚 **Documentation**: Everything is in the Colab notebook
- 💡 **Questions**: Use GitHub discussions
- 🚀 **Best path**: Start with Google Colab - it's the easiest!

---

**Made with ❤️ for rural healthcare accessibility**

---

## ⭐ **Why This Project Rocks:**

### **🌟 Accessibility First**
- 🆓 **Completely free** to run (Google Colab)
- 📱 **Mobile friendly** - train on your phone!
- 🔧 **No setup required** - everything in one notebook
- 🌍 **Works anywhere** with internet connection

### **🚀 Technical Excellence**
- 🏆 **State-of-the-art models** (ResNet50, DenseNet121, EfficientNetB0)
- 📊 **Comprehensive evaluation** with detailed metrics
- 🛡️ **Robust error handling** and troubleshooting
- 💾 **Auto-save progress** to Google Drive

### **🏥 Real-World Impact**
- 🎯 **4-class medical detection** (COVID-19, Pneumonia, TB, Normal)
- 🏥 **Designed for rural healthcare** where specialists are scarce
- 📈 **High accuracy** through transfer learning
- ⚡ **Fast inference** suitable for real-time use

### **🎓 Educational Value**
- � **Complete learning resource** for medical AI
- 🔍 **Transparent methodology** - all code visible
- 📊 **Detailed visualizations** for understanding results
- 🎯 **Best practices** for medical image classification

---

**� Ready to detect diseases and save lives? Upload the notebook to Colab and get started in minutes!**