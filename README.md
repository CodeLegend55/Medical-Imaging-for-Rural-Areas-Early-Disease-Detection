# Medical Imaging for Rural Areas - Advanced Multi-Modal Disease Detection

A comprehensive deep learning solution for automated medical imaging analysis featuring both **model training** and **web application deployment**. Detect chest conditions (COVID-19, Pneumonia, Tuberculosis), fractures, and osteoporosis using state-of-the-art pre-trained models with an intuitive web interface.

## 🎯 Project Overview

This project provides accessible medical imaging solutions for rural areas where specialist radiologists may not be available. The system includes:

1. **🤖 Model Training**: Google Colab-ready deep learning pipeline for chest X-ray models
2. **🌐 Multi-Modal Web Application**: Advanced interface for comprehensive medical imaging analysis
3. **🦴 Specialized Detection**: Additional models for fracture and osteoporosis detection
4. **📋 Intelligent Reporting**: AI-powered medical reports with comprehensive findings

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
├── colab_setup_fixed.ipynb       # 🚀 Complete training pipeline
├── app.py                        # 🌐 Flask web application
├── config.py                     # ⚙️ Configuration settings
├── templates/
│   └── index.html               # Web interface
├── models/                      # Trained model files
│   ├── ResNet50_colab.pth       # PyTorch chest X-ray model
│   ├── DenseNet121_colab.pth    # PyTorch chest X-ray model
│   ├── EfficientNetB0_colab.pth # PyTorch chest X-ray model
│   ├── fracture_classification_model.h5    # TensorFlow fracture detection
│   └── Osteoporosis_Model.h5    # TensorFlow osteoporosis detection
├── uploads/                     # Temporary upload folder
├── requirements.txt             # Python dependencies
├── README.md                    # This documentation
└── Chest_Xray_Dataset/          # Training dataset
    ├── train/
    │   ├── COVID19/, NORMAL/, PNEUMONIA/, TURBERCULOSIS/
    ├── val/
    │   ├── COVID19/, NORMAL/, PNEUMONIA/, TURBERCULOSIS/
    └── test/
        ├── COVID19/, NORMAL/, PNEUMONIA/, TURBERCULOSIS/
```

## 🚀 **Part 1: Model Training - Google Colab (Recommended)**

### **⚡ One-File Training Solution!**

Everything you need for training is in **`colab_setup_fixed.ipynb`** - a single, self-contained notebook!

### **Training Steps:**

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

### **🎯 What the Training Notebook Does:**

✅ **Auto-detects** your dataset location  
✅ **Verifies** dataset structure and image counts  
✅ **Installs** all required packages  
✅ **Creates** and trains all 3 models  
✅ **Evaluates** performance with detailed metrics  
✅ **Saves** trained models to your Drive  
✅ **Visualizes** results with plots and confusion matrices  

## 🌐 **Part 2: Multi-Modal Web Application Deployment**

After training your models, deploy them in a comprehensive web application for real-time multi-modal medical imaging analysis!

### **Web App Features**

- 🖼️ **Easy Image Upload**: Drag & drop or click to upload chest X-rays and bone images
- 🤖 **Multi-Modal Analysis**: Uses PyTorch models for chest conditions + TensorFlow models for fractures/osteoporosis
- 📊 **Comprehensive Predictions**: Shows predictions from all 5 models with confidence scores
- 🎯 **Intelligent Ensemble**: Combines predictions for accurate primary diagnosis plus secondary findings
- 📋 **Advanced Medical Reports**: Generates professional reports covering all detected conditions
- 📱 **Mobile Friendly**: Works on desktop, tablet, and mobile devices
- ⚡ **Fast Analysis**: Multi-model processing with real-time results

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

### **How to Use the Multi-Modal Web App**

1. **Upload Image**: Drag and drop or click to select a medical image (chest X-ray, bone scan - PNG, JPG, JPEG)
2. **AI Analysis**: Click "Analyze Medical Image" button for comprehensive multi-modal analysis
3. **View Results**: See primary diagnosis + secondary findings with confidence scores from all 5 models
4. **Read Report**: Review comprehensive medical report covering all detected conditions
5. **Secondary Findings**: Check for additional detections like fractures or osteoporosis
6. **Repeat**: Upload another image for analysis  

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
- ✅ **Nothing to install!** Everything is handled automatically
- ✅ Just a web browser and Google account

### **🌐 For Web Application**
- **Software**: Python 3.8+, see `requirements.txt` for complete list
- **Models**: Your trained `.pth` files from Colab
- **Optional**: OpenAI API key for enhanced reports
- **Storage**: Minimal space for temporary uploads

**Key Dependencies (Auto-installed via requirements.txt):**
- Flask 3.0.0 (Web framework)
- PyTorch 2.4.1 + torchvision 0.19.1 (Deep learning - chest conditions)
- TensorFlow 2.17.0 + Keras 3.5.0 (Deep learning - fracture/osteoporosis)
- Pillow 10.4.0 (Image processing)
- NumPy, Pandas, Matplotlib (Data processing)
- OpenAI 1.51.0 (Optional, for enhanced reports)

### **💻 For Local Training (Not Recommended)**
- **Hardware**: NVIDIA GPU with CUDA support, 8GB+ RAM
- **Software**: Python 3.8+, PyTorch 1.12+
- **Storage**: 5GB+ for dataset and models
- **Note**: Google Colab is much easier and provides free GPU access!

## 🌐 Web Application API & Customization

### API Endpoints
- `GET /` - Main web interface
- `POST /upload` - Upload and analyze image
- `GET /health` - Check server health and model status

### Customization Options

#### Adding Your Own Models
1. Place your `.pth` files in the `models/` directory
2. Update the model loading code in `app.py` if needed
3. Modify `CLASSES` in `config.py` if you have different classes

#### Changing the Interface
- Edit `templates/index.html` to modify the web interface
- CSS styles are embedded in the HTML file for simplicity
- Add your logo or branding as needed

### Security & Performance Notes
- Files are temporarily stored during processing
- No patient data is permanently stored
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
- Ensure the image is a valid chest X-ray

#### Performance
- First prediction may take a few seconds (model loading)
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

## ⭐ **Why This Advanced Multi-Modal Solution Rocks:**

### **🌟 Comprehensive Medical AI Platform**
- 🆓 **Completely free** to train chest X-ray models (Google Colab)
- 🌐 **Advanced multi-modal deployment** with 5-model ensemble
- 🦴 **Specialized detection** for fractures and osteoporosis
- 📱 **Mobile friendly** - works across all devices
- 🔧 **Plug-and-play setup** - ready for clinical deployment
- 🌍 **Global accessibility** with internet connection only

### **🚀 Advanced Technical Excellence**
- 🏆 **Dual framework integration** (PyTorch + TensorFlow)
- 🤖 **5 specialized AI models** covering multiple medical conditions
- 📊 **Intelligent ensemble system** with primary + secondary findings
- 🛡️ **Production-grade reliability** with comprehensive error handling
- 💾 **Auto-save progress** and model management
- 🌐 **Enterprise-ready web interface** with professional medical reporting

### **🏥 Revolutionary Healthcare Impact**
- 🎯 **6-condition detection** (COVID-19, Pneumonia, TB, Normal, Fractures, Osteoporosis)
- 🏥 **Multi-specialty support** for rural and underserved areas
- 📈 **Enhanced diagnostic accuracy** through multi-modal analysis
- ⚡ **Rapid multi-condition screening** suitable for emergency and clinical settings
- 📋 **Comprehensive professional reports** with AI-generated insights for multiple specialties
- 🦴 **Pioneering bone health integration** with chest imaging analysis

### **🎓 Advanced Educational & Research Value**
- 📚 **Complete multi-modal AI learning platform** 
- 🔬 **Cutting-edge research implementation** combining multiple frameworks
- 🔍 **Full transparency** - all architectures and methodologies visible
- 📊 **Advanced visualizations** for understanding ensemble predictions
- 🎯 **Industry best practices** for medical AI deployment
- 💼 **Real-world production patterns** for clinical integration
- 🌟 **Pioneer multi-modal approach** setting new standards in medical AI

---

## 🚀 **Quick Start Guide**

### **Step 1: Train Your Models** (30 minutes)
1. Open Google Colab
2. Upload `colab_setup_fixed.ipynb`
3. Enable GPU and run all cells
4. Download trained models

### **Step 2: Deploy Advanced Multi-Modal Web Application** (10 minutes)
1. Create virtual environment: `python -m venv medical_imaging_env`
2. Activate environment: 
   - **Windows CMD**: `medical_imaging_env\Scripts\activate.bat`
   - **Windows PowerShell**: `.\medical_imaging_env\Scripts\Activate.ps1`
   - **macOS/Linux**: `source medical_imaging_env/bin/activate`
3. Install dependencies: `pip install -r requirements.txt` (includes TensorFlow + PyTorch)
4. Add your fracture/osteoporosis models to `/models/` directory
5. Run the comprehensive web app: `python app.py`
6. Open http://localhost:5000
7. Upload medical images and get multi-modal AI analysis covering 6 conditions!

**🎉 Ready to revolutionize medical diagnosis with advanced multi-modal AI? Get started in just 40 minutes total!**

### **🏆 What You'll Have:**
- **Complete training pipeline** for chest X-ray models
- **Advanced 5-model ensemble** web application  
- **Multi-specialty AI analysis** (Pulmonology + Orthopedics)
- **Professional medical reports** for clinical documentation
- **Production-ready deployment** suitable for healthcare facilities
- **Cutting-edge multi-modal AI** system setting new medical AI standards