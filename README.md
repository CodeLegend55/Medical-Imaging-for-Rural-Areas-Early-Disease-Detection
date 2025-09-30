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

The project uses 7 deep learning models:

**Chest X-ray Models (4-class: COVID-19, Pneumonia, TB, Normal):**
- ResNet50
- DenseNet121  
- EfficientNetB0

**Fracture Detection Models (2-class: Fractured, Non-Fractured):**
- Fracture ResNet50
- Fracture DenseNet121
- Fracture EfficientNetB0

**Osteoporosis Detection Model (2-class: Normal, Osteoporosis):**
- TensorFlow/Keras deep learning model

All PyTorch models use transfer learning from ImageNet pre-trained weights.

## 📁 Project Structure

```
Medical-Imaging-for-Rural-Areas-Early-Disease-Detection/
├── .gitattributes
├── .gitignore
├── app.py                                        # Flask web application
├── config.py                                     # Configuration settings
├── Covid-pnemonia-tb-Chest_Xray_training.ipynb  # Chest X-ray training notebook
├── fracture_detection_training.ipynb            # Fracture detection training notebook
├── LICENSE
├── README.md
├── requirements.txt                              # Python dependencies
├── models/                                      # Trained AI model files
│   ├── DenseNet121_colab.pth                    # Chest X-ray model
│   ├── EfficientNetB0_colab.pth                 # Chest X-ray model
│   ├── ResNet50_colab.pth                       # Chest X-ray model
│   ├── fracture_densenet121.pth                 # Fracture detection model
│   ├── fracture_efficientnetb0.pth              # Fracture detection model
│   ├── fracture_resnet50.pth                    # Fracture detection model
│   └── Osteoporosis_Model.h5                    # TensorFlow osteoporosis model
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
2. Enable GPU runtime
3. Run all cells
4. Download trained models

### Fracture Detection Models  
Use `fracture_detection_training.ipynb` in Google Colab:
1. Upload notebook to Google Colab
2. Enable GPU runtime
3. Run all cells
4. Download trained models

## 💻 Web Application Features

- **Image Upload**: Support for PNG, JPG, JPEG X-ray images
- **Multi-Model Analysis**: Uses ensemble of PyTorch and TensorFlow models
- **Chest X-ray Detection**: COVID-19, Pneumonia, TB, Normal classification
- **Fracture Detection**: Binary fracture/no-fracture classification
- **Osteoporosis Detection**: AI-powered bone density analysis with clinical recommendations
- **Problem Type Selection**: User-selectable analysis modes (Chest, Fracture, Osteoporosis)
- **Confidence Scoring**: Each prediction includes confidence percentage
- **Professional Reports**: Detailed medical analysis with specialized recommendations

## 📋 Requirements

- Python 3.8+
- PyTorch 2.4.1+
- TensorFlow 2.17.0+
- Flask 3.0.0
- See `requirements.txt` for complete list

## 🔧 Configuration

Edit `config.py` to customize:
- OpenAI API key (optional, for enhanced reports)
- Upload folder settings
- Model paths
- Server configuration

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Submit pull request

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/CodeLegend55/Medical-Imaging-for-Rural-Areas-Early-Disease-Detection/issues)
- **Contact**: Reach out to any team member via their GitHub profiles

---

**⚠️ Medical Disclaimer**: This AI system is for research and educational purposes only. It should not replace professional medical diagnosis. Always consult qualified healthcare professionals for final diagnosis and treatment decisions.