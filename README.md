# Medical Imaging for Rural Areas - User-Controlled X-ray Analysis with Smart Routing

A comprehensive deep learning solution for automated medical imaging analysis featuring **user-controlled X-ray type selection**, **smart model routing**, and **advanced multi-modal disease detection**. Users select whether their uploaded image is a chest X-ray or bone/joint X-ray, and the system routes it to the appropriate specialized models for optimal accuracy.

## 🎯 Project Overview

This project provides accessible medical imaging solutions for rural areas where specialist radiologists may not be available. The system includes:

1. **🤖 Model Training**: Google Colab-ready deep learning pipeline for chest X-ray models
2. **🌐 User-Controlled Multi-Modal Web Application**: Advanced interface with user X-ray type selection
3. **� User-Controlled X-ray Selection**: Medical professionals select chest vs bone/joint X-rays
4. **🦴 Specialized Detection**: Targeted models for fracture and osteoporosis detection
5. **📋 AI-Powered Reporting**: Comprehensive medical reports with user selection information

## 🚀 **User-Controlled X-ray Routing Technology**

### **� User X-ray Type Selection**
Medical professionals now have full control over X-ray routing:
- **Chest X-rays** → Routes to chest condition models (COVID-19, Pneumonia, TB detection)
- **Bone/Joint X-rays** → Routes to specialized bone models (fracture, osteoporosis detection)

### **🧠 Smart Model Routing Based on User Choice**
- **Chest X-rays** → PyTorch models (DenseNet121, EfficientNetB0, ResNet50)
- **Other X-rays** → TensorFlow models (Fracture Detection, Osteoporosis Detection)
- **Best Model Selection** → Automatically highlights the most confident model
- **Optimized Processing** → Only runs relevant models, saving time and resources

### **🎯 User Selection Benefits**
The user-controlled approach provides several advantages:
- **Medical Professional Control**: Doctors/technicians specify the correct X-ray type
- **Enhanced Accuracy**: No risk of misclassification from automatic detection
- **Faster Processing**: No time spent on image analysis for type detection
- **Simplified Workflow**: Clear, intuitive interface for medical professionals
- **Reliability**: Eliminates potential auto-detection errors

### Supported Medical Conditions
- **COVID-19**: SARS-CoV-2 related pneumonia
- **Normal**: Healthy chest X-rays
- **Pneumonia**: Bacterial/viral pneumonia (non-COVID)
- **Tuberculosis**: TB infections
- **Fractures**: Bone fracture detection
- **Osteoporosis**: Bone density abnormalities

## 👥 Team Members

- **Neeraj Tirumalasetty** - [https://github.com/neerajtirumalasetty](https://github.com/neerajtirumalasetty)
- **Jeevan Rushi** - [https://github.com/jeevanrushi07](https://github.com/jeevanrushi07)
- **RAHUL VARUN KOTAGIRI** - [https://github.com/RAHULVARUNKOTAGIRI](https://github.com/RAHULVARUNKOTAGIRI)

## 🏗️ Architecture

The project implements multiple state-of-the-art deep learning architectures:

**PyTorch Models (Chest Conditions):**
1. **ResNet50**: Deep residual learning with skip connections
2. **DenseNet121**: Dense connectivity between layers
3. **EfficientNetB0**: Compound scaling for efficiency and accuracy

**TensorFlow Models (Specialized Detection):**
4. **Fracture Detection Model**: Specialized model for bone fracture identification
5. **Osteoporosis Detection Model**: Advanced model for bone density analysis

All chest condition models are pre-trained on ImageNet and fine-tuned on chest X-ray data, while specialized models are trained for specific medical conditions.

## 📁 Project Structure

```
Medical-Imaging-for-Rural-Areas-Early-Disease-Detection/
├── Covid-pnemonia-tb-Chest_Xray_training.ipynb   # 🫁 Chest X-ray training (COVID, Pneumonia, TB)
├── fracture_detection_training.ipynb             # 🦴 Fracture detection training with Drive integration
├── colab_setup_fixed.ipynb                       # 🚀 Legacy complete training pipeline  
├── app.py                                        # 🌐 Flask web app with problem-type selection
├── config.py                                     # ⚙️ Configuration settings
├── demo_test.py                                  # 🧪 Demo testing script
├── test_setup.py                                 # 🔧 Setup verification script
├── templates/
│   └── index.html                               # Enhanced web interface with problem selection
├── models/                                      # Trained model files
│   ├── ResNet50_colab.pth                       # PyTorch chest X-ray model
│   ├── DenseNet121_colab.pth                    # PyTorch chest X-ray model
│   ├── EfficientNetB0_colab.pth                 # PyTorch chest X-ray model
│   ├── fracture_classification_model.h5         # TensorFlow fracture detection
│   └── Osteoporosis_Model.h5                    # TensorFlow osteoporosis detection
├── uploads/                                     # Temporary upload folder
├── requirements.txt                             # Complete Python dependencies (updated)
├── README.md                                    # This comprehensive documentation
├── LICENSE                                      # MIT License
├── start_app.bat                                # Windows batch file to start app
└── Chest_Xray_Dataset/                          # Training dataset (if downloaded locally)
    ├── train/
    │   ├── COVID19/, NORMAL/, PNEUMONIA/, TURBERCULOSIS/
    ├── val/
    │   ├── COVID19/, NORMAL/, PNEUMONIA/, TURBERCULOSIS/
    └── test/
        ├── COVID19/, NORMAL/, PNEUMONIA/, TURBERCULOSIS/
```

## 🚀 **Part 1: Model Training - Google Colab (Recommended)**

### **📚 Training Notebooks Available**

We provide **two comprehensive training notebooks** for different medical imaging tasks:

1. **`Covid-pnemonia-tb-Chest_Xray_training.ipynb`** - Chest X-ray analysis (4 conditions)
2. **`fracture_detection_training.ipynb`** - Fracture detection with Google Drive integration

### **⚡ Complete Training Solutions!**

Both notebooks are **self-contained** and handle everything automatically!

### **🔥 Quick Training Steps:**

#### **Option 1: Chest X-ray Models (COVID-19, Pneumonia, TB, Normal)**
1. **📤 Upload `Covid-pnemonia-tb-Chest_Xray_training.ipynb` to Google Colab**
2. **⚡ Enable GPU:** Runtime → Change runtime type → Hardware accelerator → **GPU**
3. **▶️ Run all cells** - handles dataset download, training, and model saving
4. **📱 Works with any device** - just need a web browser!

#### **Option 2: Fracture Detection Models (ResNet50, DenseNet121, EfficientNetB0)**
1. **📤 Upload `fracture_detection_training.ipynb` to Google Colab**
2. **⚡ Enable GPU:** Runtime → Change runtime type → Hardware accelerator → **GPU**  
3. **▶️ Run all cells** - includes Google Drive integration for persistent storage
4. **🔄 Persistent storage** - models and dataset saved to Google Drive

#### **Universal Steps:**
1. **🔓 Open Google Colab:** Go to [colab.research.google.com](https://colab.research.google.com)
2. **📂 Upload chosen notebook** from the repository
3. **⚡ Enable GPU:** Runtime → Change runtime type → Hardware accelerator → **GPU**
4. **▶️ Run Everything:** All cells sequentially - full automation!

### **🎯 What Each Training Notebook Does:**

#### **Chest X-ray Training Notebook (`Covid-pnemonia-tb-Chest_Xray_training.ipynb`)**
✅ **Auto-detects** dataset location or guides download setup  
✅ **Installs** PyTorch, scikit-learn, and visualization packages  
✅ **Creates** and trains 3 PyTorch models (ResNet50, DenseNet121, EfficientNetB0)  
✅ **Handles** 4-class classification (COVID-19, Pneumonia, TB, Normal)  
✅ **Evaluates** with accuracy, precision, recall, F1-score  
✅ **Saves** trained `.pth` models for your Flask app  
✅ **Visualizes** training curves and confusion matrices  

#### **Fracture Detection Notebook (`fracture_detection_training.ipynb`)**
✅ **Google Drive Integration** - persistent storage across sessions  
✅ **Kaggle API Setup** - automatic dataset download  
✅ **Advanced Augmentation** - Albumentations for better training  
✅ **Triple Architecture** - ResNet50, DenseNet121, EfficientNetB0 for fractures  
✅ **Comprehensive Evaluation** - detailed performance comparison  
✅ **Persistent Models** - all models saved to Google Drive  
✅ **Flask Integration Template** - ready-to-use code for web app  
✅ **Smart Data Management** - handles dataset exploration and organization  

## 🌐 **Part 2: Multi-Modal Web Application Deployment**

After training your models, deploy them in a comprehensive web application for real-time multi-modal medical imaging analysis!

### **Web App Features**

- 🖼️ **Smart Image Upload**: Drag & drop any X-ray type - system auto-detects and routes
- 🔍 **Intelligent X-ray Detection**: Automatically identifies chest vs bone/joint X-rays  
- 🤖 **Adaptive Model Selection**: Routes to appropriate models based on X-ray type
- 🎯 **Best Model Highlighting**: Shows the most confident model with star indicator
- 📊 **Comprehensive Analysis**: 
  - **Chest X-rays**: COVID-19, Pneumonia, TB, Normal detection
  - **Bone X-rays**: Fracture and Osteoporosis detection
- 📋 **Smart Medical Reports**: Type-specific recommendations and routing information
- 📱 **Mobile Friendly**: Works on desktop, tablet, and mobile devices
- ⚡ **Optimized Performance**: Only runs relevant models for faster analysis

### **🔄 How Smart Routing Works**

1. **Image Upload**: User uploads any X-ray image
2. **Auto-Detection**: System analyzes image characteristics to determine type
3. **Smart Routing**: 
   - **🫁 Chest X-ray detected** → Routes to PyTorch models (DenseNet, EfficientNet, ResNet)
   - **🦴 Bone X-ray detected** → Routes to TensorFlow models (Fracture, Osteoporosis)
4. **Best Model Selection**: Identifies the most confident prediction
5. **Enhanced Results**: Shows routing information and best model with clear indicators

### **Web App Quick Start**

#### 1. Set Up Python Environment (Recommended)

**Create a Virtual Environment:**
```bash
# On Windows (Command Prompt/PowerShell)
python -m venv medical_imaging_env
medical_imaging_env\Scripts\activate

# On macOS/Linux
python3 -m venv medical_imaging_env
source medical_imaging_env/bin/activate
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you don't need OpenAI API features, you can comment out the `openai==1.51.0` line in `requirements.txt` before installing.

#### 3. Configure the Application (Optional)

Edit `config.py` to add your OpenAI API key for enhanced reports:

```python
OPENAI_API_KEY = "your-actual-openai-api-key"
```

> **Note**: The app works perfectly without OpenAI - it will use built-in report generation.

#### 4. Run the Application

```bash
python app.py
```

#### 5. Open Your Browser

Navigate to: http://localhost:5000

### **Deactivating Virtual Environment**
When you're done working with the project:
```bash
deactivate
```

### **How to Use the Smart X-ray Analysis App**

1. **Upload Image**: Drag and drop any X-ray image (chest, bone, joint - PNG, JPG, JPEG)
2. **Auto-Detection**: System automatically detects X-ray type and shows routing decision
3. **Smart Analysis**: Appropriate models analyze the image based on detected type
4. **View Results**: 
   - See X-ray type indicator (🫁 Chest or 🦴 Bone/Joint)
   - Primary diagnosis with confidence level
   - Best performing model highlighted with ⭐
   - Type-specific recommendations
5. **Read Report**: Comprehensive medical report with routing information
6. **Repeat**: Upload another image for analysis

### **🧪 Testing the Smart Routing**

Test the new routing functionality:
```bash
python test_xray_routing.py
```

This script:
- Tests X-ray type detection with synthetic images
- Verifies model routing functionality  
- Shows routing decisions and best model selection  

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

## 📊 Dataset Information & Model Performance

### **Chest X-ray Dataset (PyTorch Models)**
- **Total Classes**: 4 (COVID-19, Normal, Pneumonia, Tuberculosis)
- **Image Format**: PNG/JPG chest X-rays
- **Image Size**: Resized to 224x224 pixels
- **Data Split**: Train/Validation/Test
- **Preprocessing**: Normalization with ImageNet statistics

### **Fracture & Osteoporosis Models (TensorFlow)**
- **Fracture Model**: Binary classification (Fracture/No Fracture)
- **Osteoporosis Model**: Binary classification (Osteoporosis/Normal)
- **Pre-trained**: Specialized models trained on bone imaging data
- **Input Processing**: Automatic preprocessing for TensorFlow compatibility

### **Multi-Modal Performance Features**
- **Ensemble Intelligence**: Combines predictions from all 5 models
- **Primary Diagnosis**: Chest condition analysis with confidence scoring
- **Secondary Findings**: High-confidence fracture and osteoporosis detection
- **Comprehensive Reporting**: Integrated analysis covering all conditions
- **Smart Thresholding**: Only shows secondary findings above 60% confidence

### Data Augmentation (Built into Notebook)
- Random horizontal flip
- Random rotation (±10°)
- Color jitter (brightness, contrast)
- Random crop and resize
- **Image Format**: PNG/JPG chest X-rays
- **Image Size**: Resized to 224x224 pixels
- **Data Split**: Train/Validation/Test
- **Preprocessing**: Normalization with ImageNet statistics

### Data Augmentation
- Random horizontal flip
- Random rotation (±10°)
- Color jitter (brightness, contrast)
- Random crop and resize

## 🎯 Multi-Modal Model Performance

### **PyTorch Models (Chest X-ray Analysis)**
The notebook evaluates chest condition models using:
- **Accuracy**: Overall classification accuracy for 4 chest conditions
- **Precision**: Class-wise precision scores (COVID-19, Pneumonia, TB, Normal)
- **Recall**: Class-wise recall scores
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown
- **Training History**: Loss and accuracy plots

### **TensorFlow Models (Specialized Detection)**
Fracture and osteoporosis models provide:
- **Binary Classification**: High-accuracy fracture detection
- **Confidence Scoring**: Precise probability outputs
- **Specialized Training**: Optimized for bone imaging analysis
- **Complementary Analysis**: Works alongside chest condition detection

### **Ensemble Performance**
- **Multi-Model Consensus**: Combines all 5 models for robust predictions
- **Intelligent Filtering**: Secondary findings only shown above 60% confidence
- **Comprehensive Coverage**: Simultaneous analysis of multiple conditions
- **Enhanced Accuracy**: Ensemble approach reduces false positives/negatives

## 🔧 Advanced Technical Details

### **PyTorch Models Architecture (Built into Notebook)**
- **Input**: 224×224×3 RGB images
- **Pre-training**: ImageNet weights (via torchvision.models)
- **Fine-tuning**: Last layers replaced for 4-class chest condition classification
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout layers, weight decay

### **TensorFlow Models Architecture**
- **Input**: 224×224×3 RGB images (auto-preprocessed)
- **Architecture**: Custom CNN architectures optimized for bone imaging
- **Output**: Binary classification with confidence scores
- **Integration**: Seamlessly integrated with PyTorch models in web app

### **Multi-Framework Integration**
- **Dual Framework Support**: PyTorch + TensorFlow in single application
- **Smart Preprocessing**: Automatic image preprocessing for each model type
- **Unified Interface**: Single web interface for all model types
- **Error Handling**: Graceful fallbacks if any model fails to load

### Training Strategy
- **Transfer Learning**: Pre-trained backbone + custom classifier (PyTorch)
- **Specialized Training**: Custom architectures for bone imaging (TensorFlow)
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

### **🚀 Primary Usage: Google Colab Training**
1. **Chest X-ray Models**: Upload `Covid-pnemonia-tb-Chest_Xray_training.ipynb` to Google Colab
2. **Fracture Models**: Upload `fracture_detection_training.ipynb` to Google Colab  
3. Enable GPU runtime for both notebooks
4. Run all cells - everything is automated!

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

## 🏥 Advanced Medical Applications & Use Cases

### **🌍 Expanded Clinical Applications**
- **Rural Healthcare Centers**: Comprehensive screening tool with multi-modal analysis
- **Telemedicine**: Remote consultation support with chest + bone imaging analysis
- **Emergency Medicine**: Rapid triage for chest conditions + fracture detection
- **Orthopedic Clinics**: Specialized fracture and osteoporosis screening
- **Geriatric Care**: Combined respiratory and bone health assessment
- **Medical Education**: Advanced training tool covering multiple specialties

### **💡 Enhanced Real-World Workflow**
1. **Training Phase**: Use Google Colab to train chest X-ray models
2. **Model Integration**: Add your specialized fracture/osteoporosis models
3. **Deployment Phase**: Set up comprehensive web application
4. **Multi-Modal Analysis**: Upload medical images for complete AI analysis
5. **Comprehensive Reporting**: Generate professional reports covering all findings
6. **Clinical Integration**: Use results to guide multi-specialty medical examination

### **🎯 Advanced Multi-Modal Benefits**
- **Comprehensive Analysis**: Simultaneous chest condition + bone health assessment
- **5-Model Ensemble**: Combines PyTorch (chest) + TensorFlow (bone) models
- **Intelligent Prioritization**: Primary diagnosis with secondary findings
- **Enhanced Accuracy**: Multi-framework approach reduces diagnostic errors
- **Specialist Support**: Assists multiple medical specialties in one platform
- **Complete Documentation**: Comprehensive reports covering all detected conditions

### **🦴 Specialized Detection Capabilities**
- **Fracture Detection**: Advanced AI model for bone fracture identification
- **Osteoporosis Screening**: Specialized model for bone density assessment
- **Multi-Condition Analysis**: Simultaneous screening for 6 different conditions
- **Confidence Thresholding**: Only displays high-confidence secondary findings (>60%)

### Important Medical Disclaimer
⚠️ **This comprehensive AI system is for research, screening, and educational purposes only. It should not replace professional medical diagnosis in any specialty. Always consult qualified healthcare professionals including radiologists, pulmonologists, and orthopedic specialists for final diagnosis and treatment decisions.**

## 📋 Requirements

### **🚀 For Model Training (Google Colab - Recommended)**
- ✅ **No special hardware needed!**
- ✅ Free Tesla T4 GPU provided by Google
- ✅ 12+ GB GPU memory available
- ✅ Works on any device with internet connection
- ✅ **Auto-installs dependencies!** Notebooks handle everything automatically
- ✅ Just a web browser and Google account

### **📚 Training Notebook Dependencies (Auto-installed)**

#### **Chest X-ray Training Notebook:**
- **Core Deep Learning:** PyTorch, torchvision, torch audio
- **Data Science:** NumPy, pandas, matplotlib, seaborn
- **ML Evaluation:** scikit-learn for metrics and evaluation
- **Progress Tracking:** tqdm for training progress bars
- **Utilities:** PyYAML, kaggle API for dataset management

#### **Fracture Detection Training Notebook:**
- **Deep Learning:** PyTorch + EfficientNet-PyTorch
- **Advanced Augmentation:** Albumentations for data augmentation
- **Computer Vision:** OpenCV (cv2) for image processing
- **Google Integration:** Google Colab API, Google Drive mounting
- **Data Management:** Kaggle API, pandas, PIL (Pillow)
- **Visualization:** matplotlib, seaborn for results visualization

### **🌐 For Problem-Type Selection Web Application**
- **Software**: Python 3.8+, see `requirements.txt` for complete list
- **Models**: Your trained `.pth` files from notebooks + specialized TensorFlow models
- **Computer Vision**: OpenCV for image processing and analysis
- **Optional**: OpenAI API key for enhanced reports
- **Storage**: Minimal space for temporary uploads

**Key Dependencies (Auto-installed via requirements.txt):**
- **Web Framework:** Flask 3.0.0, Werkzeug 3.0.1
- **Deep Learning:** PyTorch 2.4.1 + torchvision 0.19.1, TensorFlow 2.17.0 + Keras 3.5.0
- **Computer Vision:** OpenCV 4.10.0.84, Pillow 10.4.0
- **Data Processing:** NumPy 1.26.4, pandas 2.2.2, matplotlib 3.9.2, seaborn 0.13.2
- **ML Tools:** scikit-learn 1.5.2, albumentations 1.4.15, efficientnet-pytorch 0.7.1
- **Training Support:** jupyter 1.1.1, kaggle 1.6.17, tqdm 4.66.5, pyyaml 6.0.2
- **Optional AI:** OpenAI 1.51.0 (for enhanced reports)
- **Production:** gunicorn 23.0.0 (for deployment)

### **💻 For Local Development & Training (Advanced)**
- **Hardware**: NVIDIA GPU with CUDA support, 8GB+ RAM
- **Software**: Python 3.8+, see requirements.txt for complete environment
- **Storage**: 5GB+ for dataset and models
- **Setup**: Use `pip install -r requirements.txt` for complete environment
- **Note**: Google Colab is much easier and provides free GPU access!

### **📦 Using requirements.txt for Local Development**
```bash
# Create virtual environment
python -m venv medical_imaging_env

# Activate environment
# Windows:
medical_imaging_env\Scripts\activate
# macOS/Linux:
source medical_imaging_env/bin/activate

# Install all dependencies (training + web app)
pip install -r requirements.txt

# Now you can run training notebooks locally or start the web app
jupyter notebook  # For training
python app.py     # For web application
```

## 🌐 Smart Web Application API & Advanced Features

### **🔍 Smart Routing API Endpoints**
- `GET /` - Enhanced web interface with routing display
- `POST /upload` - Upload and analyze with automatic X-ray routing
- `GET /health` - Check server health and model status (now shows routing info)

### **🧠 X-ray Type Detection Function**
```python
from app import detect_xray_type

# Automatically detect X-ray type
xray_type = detect_xray_type("path/to/xray.jpg")
print(f"Detected type: {xray_type}")  # Returns 'chest' or 'other'
```

### **🎯 Smart Model Routing**
```python
from app import MedicalImagingModel

# Initialize model with smart routing
model = MedicalImagingModel()

# Get predictions with routing information
predictions, error, routing_info = model.predict_all("path/to/xray.jpg")

# Check routing decisions
print(f"X-ray type: {routing_info['xray_type']}")
print(f"Models used: {routing_info['models_used']}")
if 'best_model' in routing_info:
    best = routing_info['best_model']
    print(f"Best model: {best['name']} ({best['prediction']['confidence']:.1f}%)")
```

### **📊 Enhanced Results Structure**
```json
{
  "filename": "xray_image.jpg",
  "predictions": {
    "DenseNet121": {"class": "NORMAL", "confidence": 95.2},
    "EfficientNetB0": {"class": "NORMAL", "confidence": 92.8}
  },
  "ensemble": {
    "diagnosis": "NORMAL",
    "confidence": 94.0,
    "xray_type": "chest",
    "routing_info": {...}
  },
  "routing_info": {
    "xray_type": "chest",
    "models_used": ["DenseNet121", "EfficientNetB0", "ResNet50"],
    "best_model": {
      "name": "DenseNet121",
      "prediction": {"class": "NORMAL", "confidence": 95.2}
    }
  },
  "report": "Enhanced medical report with routing information..."
}
```

### Customization Options

#### Adding Your Own Models
1. Place your `.pth` files in the `models/` directory
2. Update the model loading code in `app.py` if needed
3. Modify `CLASSES` in `config.py` if you have different classes
4. **New**: Models are automatically routed based on X-ray type

#### Customizing X-ray Detection
The detection algorithm can be fine-tuned by modifying parameters in `detect_xray_type()`:
- Aspect ratio thresholds
- Intensity analysis parameters
- Edge detection sensitivity
- Symmetry scoring weights

### Security & Performance Notes
- Files are temporarily stored during processing
- No patient data is permanently stored
- **New**: X-ray type detection adds minimal processing overhead
- **Optimized**: Only relevant models run, improving performance
- Use HTTPS in production environments
- Consider adding authentication for production use
- Use GPU if available for faster inference
- Consider model quantization for mobile deployment

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

## 📞 Support & Troubleshooting

### **🚀 Model Training Issues (Google Colab)**

- **Can't find GPU?** Runtime → Change runtime type → Hardware accelerator → GPU
- **Dataset not found?** Check Step 3 in notebook - it will guide you
- **Out of memory?** Reduce BATCH_SIZE in notebook (try 16 or 8)
- **Session timeout?** Colab sessions last ~12 hours - just reconnect

### **🌐 Web Application Issues**

#### Installation & Environment Issues
- **Python version**: Ensure you're using Python 3.8 or higher
- **Virtual environment**: Always use a virtual environment to avoid conflicts
- **Dependencies**: Run `pip install -r requirements.txt` in your activated virtual environment
- **OpenAI issues**: Comment out `openai==0.28.1` in requirements.txt if not using AI reports

#### Models Not Loading
- Ensure your `.pth` model files are in the `models/` directory
- Check that the model files aren't corrupted
- Verify you have enough memory to load the models

#### Upload Issues
- Supported formats: PNG, JPG, JPEG
- Maximum file size: 16MB
- **New**: System accepts any X-ray type and automatically routes appropriately
- Ensure the image is a valid X-ray (chest, bone, joint, etc.)

#### Performance
- First prediction may take a few seconds (model loading)
- **Improved**: X-ray type detection is very fast (< 1 second)
- **Optimized**: Only appropriate models run, reducing total processing time
- Subsequent predictions are much faster
- Consider using GPU for better performance

### **💻 Local Development Support**
- Check dependencies are installed correctly
- Ensure Python 3.8+ and PyTorch are installed
- Verify dataset folder structure matches project structure

### General Support
- 🐛 **Bug reports**: Open an issue on GitHub
- 📚 **Documentation**: Everything is in this README
- 💡 **Questions**: Use GitHub discussions
- 🚀 **Best path**: Start with Google Colab for training, then deploy the web app

---

**Made with ❤️ for rural healthcare accessibility**

---

## ⭐ **Why This Smart X-ray Analysis Solution Rocks:**

### **🌟 Intelligent Medical AI Platform**
- 🔍 **Smart X-ray Detection** - Automatically identifies X-ray type and routes to best models
- 🆓 **Completely free** to train chest X-ray models (Google Colab)
- � **Adaptive Model Selection** - Only runs relevant models for optimal accuracy
- 🌐 **Advanced routing deployment** with intelligent analysis
- 🦴 **Comprehensive detection** for chest and bone conditions
- 📱 **Mobile friendly** - works across all devices
- 🔧 **Plug-and-play setup** - ready for clinical deployment
- 🌍 **Global accessibility** with internet connection only

### **🚀 Revolutionary Technical Excellence**
- 🏆 **Intelligent routing system** with automatic X-ray type detection
- 🔍 **Computer vision analysis** for smart image classification
- 🤖 **Dual framework integration** (PyTorch + TensorFlow)
- 📊 **Best model selection** with confidence-based highlighting
- 🛡️ **Production-grade reliability** with comprehensive error handling
- 💾 **Auto-save progress** and optimized processing
- 🌐 **Enterprise-ready interface** with smart routing displays

### **🏥 Revolutionary Healthcare Impact**
- 🎯 **Smart 6-condition detection** with automatic routing
- 🔄 **Adaptive analysis** - chest conditions OR bone conditions based on image type
- 🏥 **Multi-specialty support** with intelligent model selection
- 📈 **Enhanced diagnostic accuracy** through smart routing
- ⚡ **Rapid specialized screening** optimized for each X-ray type
- 📋 **Type-specific professional reports** with routing information
- 🦴 **Pioneering adaptive approach** setting new standards in medical AI

### **🎓 Advanced Educational & Research Value**
- 📚 **Complete intelligent AI learning platform** 
- 🔬 **Cutting-edge routing implementation** with computer vision
- 🔍 **Full transparency** - all detection algorithms visible
- 📊 **Smart visualizations** for understanding routing decisions
- 🎯 **Industry best practices** for adaptive medical AI
- 💼 **Real-world intelligent patterns** for clinical integration
- 🌟 **Pioneer smart routing approach** setting new standards in medical AI

---

## 🚀 **Quick Start Guide**

### **Step 1: Train Your Models** (30-45 minutes)

**Option A: Chest X-ray Models (COVID-19, Pneumonia, TB)**
1. Open Google Colab
2. Upload `Covid-pnemonia-tb-Chest_Xray_training.ipynb`
3. Enable GPU and run all cells
4. Download trained `.pth` models

**Option B: Fracture Detection Models** 
1. Open Google Colab  
2. Upload `fracture_detection_training.ipynb`
3. Enable GPU and run all cells
4. Models auto-saved to Google Drive (persistent storage!)

**Both options provide:** Complete training pipeline, automatic evaluation, visualization, and ready-to-use models

### **Step 2: Deploy Smart X-ray Analysis Web Application** (10 minutes)
1. Create virtual environment: `python -m venv medical_imaging_env`
2. Activate environment: 
   - **Windows CMD**: `medical_imaging_env\Scripts\activate.bat`
   - **Windows PowerShell**: `.\medical_imaging_env\Scripts\Activate.ps1`
   - **macOS/Linux**: `source medical_imaging_env/bin/activate`
3. Install dependencies: `pip install -r requirements.txt` (includes OpenCV for routing)
4. Add your fracture/osteoporosis models to `/models/` directory
5. Run the smart web app: `python app.py`
6. Open http://localhost:5000
7. Upload any X-ray image and watch the smart routing in action!

**🧪 Test the Smart Routing:**
```bash
python test_xray_routing.py
```

**🎉 Ready to revolutionize medical diagnosis with smart X-ray routing? Get started in just 40 minutes total!**

### **🏆 What You'll Have:**
- **Complete training pipeline** for chest X-ray models
- **Smart routing system** that auto-detects X-ray types
- **Adaptive model selection** with best model highlighting  
- **Type-specific AI analysis** (Chest OR Bone conditions)
- **Enhanced medical reports** with routing information
- **Production-ready deployment** with intelligent processing
- **Cutting-edge smart AI** system setting new medical AI standards