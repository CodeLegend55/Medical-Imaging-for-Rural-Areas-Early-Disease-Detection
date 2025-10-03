# Medical Imaging for Rural Areas - AI-Powered X-ray Analysis System

A deep learning solution for automated medical imaging analysis featuring chest X-ray analysis (COVID-19, Pneumonia, Tuberculosis detection), fracture detection, and osteoporosis detection using PyTorch and TensorFlow models.

## 🎯 Project Overview

This project provides accessible medical imaging solutions for rural areas where specialist radiologists may not be available. The system includes:

1. **Model Training**: Google Colab notebooks for training deep learning models
2. **Web Application**: Flask-based interface for real-time X-ray analysis
3. **Chest X-ray Analysis**: Detection of COVID-19, Pneumonia, TB, and Normal cases
4. **Fracture Detection**: Binary classification for bone fracture identification
5. **Osteoporosis Detection**: AI-powered bone density analysis using TensorFlow

## 👥 Team Members

- **Neeraj Tirumalasetty** - [https://github.com/neerajtirumalasetty](https://github.com/neerajtirumalasetty)
- **Jeevan Rushi** - [https://github.com/jeevanrushi07](https://github.com/jeevanrushi07)
- **RAHUL VARUN KOTAGIRI** - [https://github.com/RAHULVARUNKOTAGIRI](https://github.com/RAHULVARUNKOTAGIRI)

## 🏗️ Architecture

The project uses **9 deep learning models** (all PyTorch):

**Chest X-ray Models (4-class: COVID-19, Pneumonia, TB, Normal):**
- ResNet50
- DenseNet121  
- EfficientNetB0

**Fracture Detection Models (2-class: Fractured, Non-Fractured):**
- Fracture ResNet50
- Fracture DenseNet121
- Fracture EfficientNetB0

**Osteoporosis Detection Models (2-class: Normal, Osteoporosis):**
- Osteoporosis ResNet50
- Osteoporosis DenseNet121
- Osteoporosis EfficientNetB0

All models use **PyTorch** with transfer learning from ImageNet pre-trained weights and ensemble prediction for robust diagnosis.

## 📁 Project Structure

```
Medical-Imaging-for-Rural-Areas-Early-Disease-Detection/
├── .gitattributes
├── .gitignore
├── app.py                                        # Flask web application
├── config.py                                     # Configuration settings
├── Covid-pnemonia-tb-Chest_Xray_training.ipynb  # Chest X-ray training notebook
├── fracture_detection_training.ipynb            # Fracture detection training notebook
├── osteoporosis_detection_training.ipynb        # Osteoporosis detection training notebook
├── LICENSE
├── README.md
├── requirements.txt                              # Python dependencies
├── models/                                      # Trained AI model files (PyTorch)
│   ├── DenseNet121_colab.pth                    # Chest X-ray model
│   ├── EfficientNetB0_colab.pth                 # Chest X-ray model
│   ├── ResNet50_colab.pth                       # Chest X-ray model
│   ├── fracture_densenet121.pth                 # Fracture detection model
│   ├── fracture_efficientnetb0.pth              # Fracture detection model
│   ├── fracture_resnet50.pth                    # Fracture detection model
│   ├── osteoporosis_resnet50.pth                # Osteoporosis detection model
│   ├── osteoporosis_densenet121.pth             # Osteoporosis detection model
│   └── osteoporosis_efficientnetb0.pth          # Osteoporosis detection model
├── templates/
│   └── index.html                               # Web interface template
└── uploads/
    └── readme.txt
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/CodeLegend55/Medical-Imaging-for-Rural-Areas-Early-Disease-Detection.git
cd Medical-Imaging-for-Rural-Areas-Early-Disease-Detection
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv medical_imaging_env

# Activate environment
# Windows PowerShell:
.\medical_imaging_env\Scripts\Activate.ps1
# Windows Command Prompt:
medical_imaging_env\Scripts\activate.bat
# macOS/Linux:
source medical_imaging_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Web Application
```bash
python app.py
```

Open your browser and go to: http://localhost:5000

## 📊 Model Training

### Chest X-ray Models
Use `Covid-pnemonia-tb-Chest_Xray_training.ipynb` in Google Colab:
1. Upload notebook to Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → GPU)
3. Mount Google Drive for dataset access
4. Run all cells sequentially
5. Download trained models (.pth files)

### Fracture Detection Models  
Use `fracture_detection_training.ipynb` in Google Colab:
1. Upload notebook to Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → GPU)
3. Mount Google Drive for dataset access
4. Run all cells sequentially
5. Download trained models (.pth files)

### Osteoporosis Detection Models
Use `osteoporosis_detection_training.ipynb` in Google Colab:
1. Upload notebook to Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → GPU)
3. Mount Google Drive and upload dataset to `/content/drive/MyDrive/Capstone/Dataset/`
4. Verify dataset paths in the notebook
5. Run all cells sequentially
6. Download trained models (.pth files): osteoporosis_resnet50.pth, osteoporosis_densenet121.pth, osteoporosis_efficientnetb0.pth

## 💻 Web Application Features

- **Image Upload**: Support for PNG, JPG, JPEG X-ray images (max 16MB)
- **Multi-Model Ensemble**: Uses 3-model voting system for each detection type
- **Chest X-ray Detection**: COVID-19, Pneumonia, TB, Normal classification (4 classes)
- **Fracture Detection**: Binary fracture/no-fracture classification from bone X-rays
- **Osteoporosis Detection**: Binary normal/osteoporosis classification from knee X-rays
- **Problem Type Selection**: User-selectable analysis modes (Chest, Fracture, Osteoporosis)
- **Confidence Scoring**: Each model prediction includes confidence percentage
- **Best Model Selection**: Automatically identifies highest-confidence prediction
- **Professional Reports**: AI-generated medical analysis with specialized recommendations
- **OpenAI Integration**: Optional enhanced report generation using GPT models

## 📋 Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 2.4.1+ (primary deep learning framework)
- **Flask**: 3.0.0 (web framework)
- **TensorFlow**: 2.17.0+ (optional, for legacy compatibility)
- **GPU**: Recommended for training (CUDA-enabled)
- **RAM**: Minimum 8GB (16GB recommended for training)
- **Storage**: ~5GB for all models

See `requirements.txt` for the complete dependency list.

## 🔧 Configuration

Edit `config.py` to customize:
- **OpenAI API Key**: Set your API key for enhanced medical reports (optional)
- **Upload Folder**: Change upload directory path
- **Model Paths**: Update model file locations
- **Server Settings**: Modify host, port, and debug mode
- **Class Labels**: CLASSES, FRACTURE_CLASSES, OSTEOPOROSIS_CLASSES

**Default Settings:**
- Upload folder: `uploads/`
- Max file size: 16MB
- Server: `0.0.0.0:5000`
- Debug mode: Enabled

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/YourFeature`)
3. **Commit** your changes (`git commit -m 'Add YourFeature'`)
4. **Push** to the branch (`git push origin feature/YourFeature`)
5. **Submit** a pull request

### Development Guidelines:
- Follow PEP 8 coding standards
- Add docstrings to functions
- Test changes locally before submitting PR
- Update documentation for new features

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/CodeLegend55/Medical-Imaging-for-Rural-Areas-Early-Disease-Detection/issues)
- **Contact**: Reach out to any team member via their GitHub profiles

---

**⚠️ Medical Disclaimer**: This AI system is for research and educational purposes only. It should not replace professional medical diagnosis. Always consult qualified healthcare professionals for final diagnosis and treatment decisions.