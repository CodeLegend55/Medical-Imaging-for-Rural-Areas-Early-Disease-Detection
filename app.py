from flask import Flask, request, render_template, jsonify, redirect, url_for
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
import json
import config

# TensorFlow/Keras imports for the new models
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
    print("✓ TensorFlow available for fracture and osteoporosis models")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. Fracture and osteoporosis models will be disabled.")

# Try to import OpenAI (optional)
try:
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY != "your-openai-api-key-here" else None
    OPENAI_AVAILABLE = bool(client)
except ImportError:
    OPENAI_AVAILABLE = False
    client = None
    print("⚠️ OpenAI package not available. Using fallback report generation.")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CLASSES = config.CLASSES
FRACTURE_CLASSES = config.FRACTURE_CLASSES
OSTEOPOROSIS_CLASSES = config.OSTEOPOROSIS_CLASSES

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image preprocessing for PyTorch models
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Image preprocessing for TensorFlow models
def preprocess_for_tensorflow(image_path, target_size=(224, 224)):
    """Preprocess image for TensorFlow models"""
    if not TENSORFLOW_AVAILABLE:
        return None
    
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0  # Normalize to [0,1]
    return image_array

class MedicalImagingModel:
    def __init__(self):
        self.pytorch_models = {}
        self.tensorflow_models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_all_models()
    
    def create_pytorch_model(self, model_name):
        """Create PyTorch model architecture"""
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        elif model_name == 'densenet121':
            model = models.densenet121(pretrained=False)
            model.classifier = nn.Linear(model.classifier.in_features, len(CLASSES))
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
        else:
            raise ValueError(f"Unknown PyTorch model: {model_name}")
        
        return model
    
    def load_all_models(self):
        """Load all available models (PyTorch and TensorFlow)"""
        # PyTorch models for chest X-ray classification
        pytorch_model_files = {
            'ResNet50': 'models/ResNet50_colab.pth',
            'DenseNet121': 'models/DenseNet121_colab.pth',
            'EfficientNetB0': 'models/EfficientNetB0_colab.pth'
        }
        
        pytorch_model_names = {
            'ResNet50': 'resnet50',
            'DenseNet121': 'densenet121',
            'EfficientNetB0': 'efficientnet_b0'
        }
        
        # Load PyTorch models
        for model_key, model_path in pytorch_model_files.items():
            if os.path.exists(model_path):
                try:
                    model = self.create_pytorch_model(pytorch_model_names[model_key])
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    model.to(self.device)
                    model.eval()
                    self.pytorch_models[model_key] = model
                    print(f"✓ Loaded PyTorch {model_key}")
                except Exception as e:
                    print(f"✗ Error loading PyTorch {model_key}: {e}")
            else:
                print(f"✗ PyTorch model file not found: {model_path}")
        
        # TensorFlow models for fracture and osteoporosis detection
        if TENSORFLOW_AVAILABLE:
            tensorflow_model_files = {
                'Fracture_Model': 'models/fracture_classification_model.h5',
                'Osteoporosis_Model': 'models/Osteoporosis_Model.h5'
            }
            
            for model_key, model_path in tensorflow_model_files.items():
                if os.path.exists(model_path):
                    try:
                        model = keras.models.load_model(model_path)
                        self.tensorflow_models[model_key] = model
                        print(f"✓ Loaded TensorFlow {model_key}")
                    except Exception as e:
                        print(f"✗ Error loading TensorFlow {model_key}: {e}")
                else:
                    print(f"✗ TensorFlow model file not found: {model_path}")
        else:
            print("⚠️ TensorFlow not available, skipping fracture and osteoporosis models")
    
    def predict_pytorch_models(self, image_path):
        """Make predictions using PyTorch models for chest conditions"""
        if not self.pytorch_models:
            return {}, "No PyTorch models loaded"
        
        try:
            # Load and preprocess image for PyTorch
            image = Image.open(image_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(self.device)
            
            predictions = {}
            
            with torch.no_grad():
                for model_name, model in self.pytorch_models.items():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    
                    # Get prediction results
                    predicted_class_idx = torch.argmax(probabilities).item()
                    predicted_class = CLASSES[predicted_class_idx]
                    confidence = probabilities[predicted_class_idx].item() * 100
                    
                    predictions[model_name] = {
                        'class': predicted_class,
                        'confidence': round(confidence, 2),
                        'probabilities': {CLASSES[i]: round(probabilities[i].item() * 100, 2) for i in range(len(CLASSES))}
                    }
            
            return predictions, None
            
        except Exception as e:
            return {}, f"Error during PyTorch prediction: {str(e)}"
    
    def predict_tensorflow_models(self, image_path):
        """Make predictions using TensorFlow models for fracture and osteoporosis"""
        if not TENSORFLOW_AVAILABLE or not self.tensorflow_models:
            return {}, "No TensorFlow models available"
        
        try:
            predictions = {}
            
            # Fracture detection
            if 'Fracture_Model' in self.tensorflow_models:
                image_array = preprocess_for_tensorflow(image_path)
                if image_array is not None:
                    fracture_pred = self.tensorflow_models['Fracture_Model'].predict(image_array, verbose=0)
                    
                    # Assuming binary classification for fracture
                    fracture_prob = fracture_pred[0][0] if len(fracture_pred[0]) == 1 else np.max(fracture_pred[0])
                    fracture_class = "FRACTURE" if fracture_prob > 0.5 else "NO_FRACTURE"
                    fracture_confidence = fracture_prob * 100 if fracture_class == "FRACTURE" else (1 - fracture_prob) * 100
                    
                    predictions['Fracture_Model'] = {
                        'class': fracture_class,
                        'confidence': round(fracture_confidence, 2),
                        'probabilities': {
                            'FRACTURE': round(fracture_prob * 100, 2),
                            'NO_FRACTURE': round((1 - fracture_prob) * 100, 2)
                        }
                    }
            
            # Osteoporosis detection
            if 'Osteoporosis_Model' in self.tensorflow_models:
                image_array = preprocess_for_tensorflow(image_path)
                if image_array is not None:
                    osteo_pred = self.tensorflow_models['Osteoporosis_Model'].predict(image_array, verbose=0)
                    
                    # Assuming binary classification for osteoporosis
                    osteo_prob = osteo_pred[0][0] if len(osteo_pred[0]) == 1 else np.max(osteo_pred[0])
                    osteo_class = "OSTEOPOROSIS" if osteo_prob > 0.5 else "NORMAL"
                    osteo_confidence = osteo_prob * 100 if osteo_class == "OSTEOPOROSIS" else (1 - osteo_prob) * 100
                    
                    predictions['Osteoporosis_Model'] = {
                        'class': osteo_class,
                        'confidence': round(osteo_confidence, 2),
                        'probabilities': {
                            'OSTEOPOROSIS': round(osteo_prob * 100, 2),
                            'NORMAL': round((1 - osteo_prob) * 100, 2)
                        }
                    }
            
            return predictions, None
            
        except Exception as e:
            return {}, f"Error during TensorFlow prediction: {str(e)}"
    
    def predict_all(self, image_path):
        """Make predictions using all available models"""
        all_predictions = {}
        errors = []
        
        # PyTorch predictions (chest conditions)
        pytorch_preds, pytorch_error = self.predict_pytorch_models(image_path)
        if pytorch_error:
            errors.append(f"PyTorch: {pytorch_error}")
        else:
            all_predictions.update(pytorch_preds)
        
        # TensorFlow predictions (fracture and osteoporosis)
        tensorflow_preds, tensorflow_error = self.predict_tensorflow_models(image_path)
        if tensorflow_error:
            errors.append(f"TensorFlow: {tensorflow_error}")
        else:
            all_predictions.update(tensorflow_preds)
        
        if not all_predictions:
            return None, "; ".join(errors) if errors else "No models available"
        
        return all_predictions, None

def generate_medical_report(predictions, patient_info=None):
    """Generate medical report using OpenAI API or fallback"""
    
    # Get ensemble prediction (majority vote or highest confidence)
    ensemble_result = get_ensemble_prediction(predictions)
    
    # Try OpenAI first if available
    if OPENAI_AVAILABLE:
        try:
            return generate_openai_report(predictions, ensemble_result)
        except Exception as e:
            print(f"OpenAI API failed: {e}. Using fallback report.")
    
    # Use fallback report generation
    return generate_fallback_report(ensemble_result, predictions)

def generate_openai_report(predictions, ensemble_result):
    """Generate medical report using OpenAI API"""
    prompt = f"""
    As a medical AI assistant, generate a structured medical report based on chest X-ray analysis results.
    
    Analysis Results:
    - Primary Diagnosis: {ensemble_result['diagnosis']}
    - Confidence Level: {ensemble_result['confidence']:.1f}%
    
    Model Predictions:
    """
    
    for model_name, result in predictions.items():
        prompt += f"- {model_name}: {result['class']} ({result['confidence']:.1f}% confidence)\n"
    
    prompt += f"""
    
    Please provide a professional medical report with the following sections:
    1. CLINICAL FINDINGS
    2. DIAGNOSTIC IMPRESSION
    3. RECOMMENDATIONS
    4. IMPORTANT NOTES
    
    Keep the language clear and professional. Include appropriate medical disclaimers.
    Diagnosis: {ensemble_result['diagnosis']}
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a medical AI assistant helping to interpret chest X-ray analysis results. Provide professional, accurate, and helpful medical reports while emphasizing the need for professional medical consultation."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0.3
    )
    
    return response.choices[0].message.content.strip()

def get_ensemble_prediction(predictions):
    """Get ensemble prediction from multiple models"""
    if not predictions:
        return {'diagnosis': 'Unknown', 'confidence': 0.0, 'secondary_findings': []}
    
    # Separate chest condition predictions from other findings
    chest_predictions = {}
    other_findings = []
    
    for model_name, result in predictions.items():
        if model_name in ['ResNet50', 'DenseNet121', 'EfficientNetB0']:
            # These are chest condition models
            chest_predictions[model_name] = result
        else:
            # These are specialized models (fracture, osteoporosis)
            if result['confidence'] > 60:  # Only include high-confidence findings
                other_findings.append({
                    'type': model_name.replace('_Model', ''),
                    'finding': result['class'],
                    'confidence': result['confidence']
                })
    
    # Get primary diagnosis from chest condition models
    primary_diagnosis = 'Unknown'
    primary_confidence = 0.0
    
    if chest_predictions:
        # Simple majority vote with confidence weighting for chest conditions
        class_votes = {}
        total_confidence = 0
        
        for model_name, result in chest_predictions.items():
            diagnosis = result['class']
            confidence = result['confidence']
            
            if diagnosis not in class_votes:
                class_votes[diagnosis] = []
            class_votes[diagnosis].append(confidence)
            total_confidence += confidence
        
        # Calculate weighted average for each class
        final_scores = {}
        for diagnosis, confidences in class_votes.items():
            final_scores[diagnosis] = sum(confidences) / len(chest_predictions)
        
        # Get the highest scoring diagnosis
        primary_diagnosis = max(final_scores.keys(), key=lambda x: final_scores[x])
        primary_confidence = final_scores[primary_diagnosis]
    
    return {
        'diagnosis': primary_diagnosis,
        'confidence': primary_confidence,
        'secondary_findings': other_findings
    }

def generate_fallback_report(ensemble_result, predictions):
    """Generate a comprehensive report when OpenAI API is not available"""
    diagnosis = ensemble_result['diagnosis']
    confidence = ensemble_result['confidence']
    secondary_findings = ensemble_result.get('secondary_findings', [])
    
    report = f"""
AUTOMATED MEDICAL IMAGING ANALYSIS REPORT
=========================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CLINICAL FINDINGS:
Based on AI analysis using multiple deep learning models for comprehensive medical imaging evaluation.

PRIMARY DIAGNOSTIC IMPRESSION:
Main Finding: {diagnosis}
Confidence Level: {confidence:.1f}%

SECONDARY FINDINGS:"""
    
    if secondary_findings:
        for finding in secondary_findings:
            report += f"""
• {finding['type']}: {finding['finding']} (Confidence: {finding['confidence']:.1f}%)"""
    else:
        report += """
• No significant secondary findings detected"""
    
    report += f"""

DETAILED MODEL ANALYSIS:"""
    
    for model_name, result in predictions.items():
        report += f"""
• {model_name}: {result['class']} ({result['confidence']:.1f}%)"""
    
    # Primary diagnosis recommendations
    if diagnosis == 'COVID19':
        report += """

RECOMMENDATIONS FOR COVID-19 FINDING:
• Immediate medical consultation recommended
• Consider PCR/RT-PCR testing for COVID-19 confirmation
• Follow local COVID-19 protocols and isolation guidelines
• Monitor symptoms closely (fever, cough, shortness of breath)
• Contact tracing may be necessary"""
        
    elif diagnosis == 'PNEUMONIA':
        report += """

RECOMMENDATIONS FOR PNEUMONIA FINDING:
• Medical consultation recommended within 24 hours
• Clinical correlation with patient symptoms advised
• Consider sputum culture and blood tests
• Monitor respiratory symptoms and vital signs
• Antibiotic therapy may be indicated based on clinical assessment"""
        
    elif diagnosis == 'TURBERCULOSIS':
        report += """

RECOMMENDATIONS FOR TUBERCULOSIS FINDING:
• Urgent medical consultation required
• Sputum examination for AFB (Acid-Fast Bacilli) recommended
• Contact tracing and isolation precautions necessary
• Consider chest CT for better evaluation
• Follow TB treatment protocols if confirmed"""
        
    else:  # NORMAL
        report += """

RECOMMENDATIONS FOR NORMAL CHEST FINDINGS:
• No acute pulmonary findings detected on chest imaging
• Routine follow-up as clinically indicated
• Continue regular health monitoring"""
    
    # Secondary findings recommendations
    if secondary_findings:
        for finding in secondary_findings:
            if finding['type'] == 'Fracture' and finding['finding'] == 'FRACTURE':
                report += """

FRACTURE FINDING RECOMMENDATIONS:
• Orthopedic consultation recommended
• Immobilization may be required pending clinical evaluation
• Pain management as appropriate
• Follow-up imaging may be necessary to monitor healing"""
                
            elif finding['type'] == 'Osteoporosis' and finding['finding'] == 'OSTEOPOROSIS':
                report += """

OSTEOPOROSIS FINDING RECOMMENDATIONS:
• Endocrinology or rheumatology consultation advised
• Bone density (DEXA) scan recommended for confirmation
• Calcium and Vitamin D supplementation consideration
• Fall prevention measures important
• Lifestyle modifications (exercise, nutrition) recommended"""
    
    report += """

IMPORTANT MEDICAL DISCLAIMERS:
⚠️ This comprehensive AI analysis is for screening and research purposes only.
⚠️ Results should not replace professional medical diagnosis or clinical judgment.
⚠️ Always consult with qualified healthcare professionals for final diagnosis and treatment decisions.
⚠️ Clinical correlation with patient symptoms, history, and physical examination is essential.
⚠️ This multi-modal analysis tool is designed to assist healthcare providers but cannot replace expert medical interpretation.
⚠️ In case of emergency or acute symptoms, seek immediate medical attention regardless of AI analysis results.

TECHNICAL NOTES:
• Analysis performed using ensemble of PyTorch models (chest conditions) and TensorFlow models (fracture/osteoporosis detection)
• Multiple AI architectures used: ResNet50, DenseNet121, EfficientNetB0, and specialized fracture/osteoporosis models
• Confidence levels reflect model certainty and should be interpreted within clinical context

Disclaimer: This automated multi-modal analysis combines several AI models trained on medical imaging data. Results should be interpreted by qualified medical professionals in conjunction with clinical findings and patient presentation.
"""
    
    return report

# Initialize model
medical_model = MedicalImagingModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction using all models
            predictions, error = medical_model.predict_all(filepath)
            
            if error:
                return jsonify({'error': error}), 500
            
            if not predictions:
                return jsonify({'error': 'No models available for prediction'}), 500
            
            # Generate medical report
            report = generate_medical_report(predictions)
            ensemble_result = get_ensemble_prediction(predictions)
            
            result = {
                'filename': filename,
                'predictions': predictions,
                'ensemble': ensemble_result,
                'report': report,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Clean up uploaded file (optional)
            # os.remove(filepath)
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG images.'}), 400

@app.route('/health')
def health_check():
    total_models = len(medical_model.pytorch_models) + len(medical_model.tensorflow_models)
    all_models = list(medical_model.pytorch_models.keys()) + list(medical_model.tensorflow_models.keys())
    
    return jsonify({
        'status': 'healthy',
        'models_loaded': total_models,
        'available_models': all_models,
        'pytorch_models': list(medical_model.pytorch_models.keys()),
        'tensorflow_models': list(medical_model.tensorflow_models.keys())
    })

if __name__ == '__main__':
    print("🚀 Starting Medical Imaging Analysis Web Application...")
    print(f"📂 Upload folder: {app.config['UPLOAD_FOLDER']}")
    
    total_models = len(medical_model.pytorch_models) + len(medical_model.tensorflow_models)
    print(f"🤖 Total models loaded: {total_models}")
    print(f"🤖 PyTorch models: {list(medical_model.pytorch_models.keys())}")
    print(f"🤖 TensorFlow models: {list(medical_model.tensorflow_models.keys())}")
    print(f"🔗 OpenAI API: {'Available' if OPENAI_AVAILABLE else 'Not available (using fallback reports)'}")
    print("🌐 Access the application at: http://localhost:5000")
    app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT)