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
    
    def predict_all(self, image_path, user_xray_type='chest'):
        """Make predictions using appropriate models based on user selection"""
        # Use user-selected X-ray type instead of auto-detection
        xray_type = user_xray_type
        print(f"User selected X-ray type: {xray_type}")
        
        all_predictions = {}
        errors = []
        routing_info = {'xray_type': xray_type, 'models_used': [], 'selection_method': 'user'}
        
        if xray_type == 'chest':
            # For chest X-rays, use PyTorch models and return the best confident result
            pytorch_preds, pytorch_error = self.predict_pytorch_models(image_path)
            if pytorch_error:
                errors.append(f"Chest X-ray models: {pytorch_error}")
            else:
                all_predictions.update(pytorch_preds)
                routing_info['models_used'] = list(pytorch_preds.keys())
                
                # Find the most confident model for chest conditions
                best_model = None
                best_confidence = 0
                
                for model_name, prediction in pytorch_preds.items():
                    if prediction['confidence'] > best_confidence:
                        best_confidence = prediction['confidence']
                        best_model = model_name
                
                if best_model:
                    routing_info['best_model'] = {
                        'name': best_model,
                        'prediction': pytorch_preds[best_model]
                    }
        
        else:  # xray_type == 'other'
            # For other X-rays, use TensorFlow models (fracture and osteoporosis)
            tensorflow_preds, tensorflow_error = self.predict_tensorflow_models(image_path)
            if tensorflow_error:
                errors.append(f"Specialized X-ray models: {tensorflow_error}")
            else:
                all_predictions.update(tensorflow_preds)
                routing_info['models_used'] = list(tensorflow_preds.keys())
                
                # Find the most confident model for specialized conditions
                best_model = None
                best_confidence = 0
                
                for model_name, prediction in tensorflow_preds.items():
                    # Only consider positive findings with high confidence
                    if (prediction['class'] in ['FRACTURE', 'OSTEOPOROSIS'] and 
                        prediction['confidence'] > best_confidence and 
                        prediction['confidence'] > 70):  # High confidence threshold
                        best_confidence = prediction['confidence']
                        best_model = model_name
                
                if best_model:
                    routing_info['best_model'] = {
                        'name': best_model,
                        'prediction': tensorflow_preds[best_model]
                    }
        
        if not all_predictions:
            return None, "; ".join(errors) if errors else "No appropriate models available", None
        
        return all_predictions, None, routing_info

def generate_medical_report(predictions, patient_info=None, routing_info=None):
    """Generate medical report using OpenAI API or fallback with routing information"""
    
    # Get ensemble prediction with routing info
    ensemble_result = get_ensemble_prediction(predictions, routing_info)
    
    # Try OpenAI first if available
    if OPENAI_AVAILABLE:
        try:
            return generate_openai_report(predictions, ensemble_result, routing_info)
        except Exception as e:
            print(f"OpenAI API failed: {e}. Using fallback report.")
    
    # Use fallback report generation
    return generate_fallback_report(ensemble_result, predictions, routing_info)

def generate_openai_report(predictions, ensemble_result, routing_info=None):
    """Generate medical report using OpenAI API with routing information"""
    xray_type = ensemble_result.get('xray_type', 'unknown')
    
    prompt = f"""
    As a medical AI assistant, generate a structured medical report based on X-ray analysis results.
    
    X-ray Type: {xray_type.upper()} X-ray
    Analysis Approach: {"Chest condition models" if xray_type == 'chest' else "Specialized bone/joint models"}
    
    Analysis Results:
    - Primary Diagnosis: {ensemble_result['diagnosis']}
    - Confidence Level: {ensemble_result['confidence']:.1f}%
    
    Model Predictions:
    """
    
    for model_name, result in predictions.items():
        prompt += f"- {model_name}: {result['class']} ({result['confidence']:.1f}% confidence)\n"
    
    if routing_info and 'best_model' in routing_info:
        best_model = routing_info['best_model']
        prompt += f"\nHighest Confidence Model: {best_model['name']} - {best_model['prediction']['class']} ({best_model['prediction']['confidence']:.1f}%)\n"
    
    prompt += f"""
    
    Please provide a professional medical report with the following sections:
    1. CLINICAL FINDINGS
    2. DIAGNOSTIC IMPRESSION  
    3. RECOMMENDATIONS
    4. IMPORTANT NOTES
    
    Keep the language clear and professional. Include appropriate medical disclaimers.
    Consider the X-ray type in your interpretation and recommendations.
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are a medical AI assistant helping to interpret {'chest' if xray_type == 'chest' else 'bone/joint'} X-ray analysis results. Provide professional, accurate, and helpful medical reports while emphasizing the need for professional medical consultation."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0.3
    )
    
    return response.choices[0].message.content.strip()

def get_ensemble_prediction(predictions, routing_info=None):
    """Get ensemble prediction from multiple models with routing awareness"""
    if not predictions:
        return {'diagnosis': 'Unknown', 'confidence': 0.0, 'secondary_findings': [], 'xray_type': 'unknown'}
    
    xray_type = routing_info.get('xray_type', 'unknown') if routing_info else 'unknown'
    
    # If we have routing info with a best model, prioritize it
    if routing_info and 'best_model' in routing_info:
        best_model_info = routing_info['best_model']
        primary_diagnosis = best_model_info['prediction']['class']
        primary_confidence = best_model_info['prediction']['confidence']
    else:
        # Fallback to original ensemble logic
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
        
        # Get primary diagnosis from appropriate models
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
        
        elif xray_type == 'other':
            # For non-chest X-rays, look for the highest confidence finding
            max_confidence = 0
            for model_name, result in predictions.items():
                if result['confidence'] > max_confidence:
                    max_confidence = result['confidence']
                    primary_diagnosis = result['class']
                    primary_confidence = result['confidence']
    
    # Collect secondary findings
    secondary_findings = []
    for model_name, result in predictions.items():
        # Skip the model that provided the primary diagnosis
        if routing_info and 'best_model' in routing_info:
            if model_name == routing_info['best_model']['name']:
                continue
        
        if result['confidence'] > 60:  # Only include high-confidence findings
            model_type = model_name.replace('_Model', '')
            if model_type not in ['ResNet50', 'DenseNet121', 'EfficientNetB0']:
                secondary_findings.append({
                    'type': model_type,
                    'finding': result['class'],
                    'confidence': result['confidence']
                })
    
    return {
        'diagnosis': primary_diagnosis,
        'confidence': primary_confidence,
        'secondary_findings': secondary_findings,
        'xray_type': xray_type,
        'routing_info': routing_info
    }

def generate_fallback_report(ensemble_result, predictions, routing_info=None):
    """Generate a comprehensive report when OpenAI API is not available"""
    diagnosis = ensemble_result['diagnosis']
    confidence = ensemble_result['confidence']
    secondary_findings = ensemble_result.get('secondary_findings', [])
    xray_type = ensemble_result.get('xray_type', 'unknown')
    selection_method = routing_info.get('selection_method', 'auto') if routing_info else 'auto'
    
    report = f"""
AUTOMATED MEDICAL IMAGING ANALYSIS REPORT
=========================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

IMAGE ANALYSIS DETAILS:
X-ray Type: {xray_type.upper()} X-ray (User Selected)
Analysis Method: {"Chest condition-specific models" if xray_type == 'chest' else "Bone/joint specialized models"}"""

    if routing_info:
        report += f"""
Models Used: {', '.join(routing_info.get('models_used', []))}"""
        if 'best_model' in routing_info:
            best_model = routing_info['best_model']
            report += f"""
Best Performing Model: {best_model['name']} ({best_model['prediction']['confidence']:.1f}% confidence)"""

    report += f"""

CLINICAL FINDINGS:
Based on AI analysis using user-selected {'chest X-ray specialized' if xray_type == 'chest' else 'bone/joint specialized'} deep learning models.

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
    
    # Recommendations based on X-ray type and findings
    if xray_type == 'chest':
        # Chest X-ray specific recommendations
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
    
    else:  # Non-chest X-ray
        if diagnosis == 'FRACTURE':
            report += """

RECOMMENDATIONS FOR FRACTURE FINDING:
• Orthopedic consultation recommended immediately
• Immobilization may be required pending clinical evaluation
• Pain management as appropriate
• Follow-up imaging may be necessary to monitor healing
• Assess for complications and associated injuries"""
            
        elif diagnosis == 'OSTEOPOROSIS':
            report += """

RECOMMENDATIONS FOR OSTEOPOROSIS FINDING:
• Endocrinology or rheumatology consultation advised
• Bone density (DEXA) scan recommended for confirmation
• Calcium and Vitamin D supplementation consideration
• Fall prevention measures important
• Lifestyle modifications (exercise, nutrition) recommended
• Regular monitoring for fracture risk"""
            
        else:  # Normal or other findings
            report += """

RECOMMENDATIONS FOR BONE/JOINT ANALYSIS:
• No acute bone pathology detected on current imaging
• Clinical correlation with symptoms advised
• Follow-up as clinically indicated"""
    
    # Secondary findings recommendations
    if secondary_findings:
        for finding in secondary_findings:
            if finding['type'] == 'Fracture' and finding['finding'] == 'FRACTURE':
                report += """

ADDITIONAL FRACTURE FINDING:
• Consider comprehensive orthopedic evaluation
• Multiple imaging views may be beneficial
• Assess for associated soft tissue injury"""
                
            elif finding['type'] == 'Osteoporosis' and finding['finding'] == 'OSTEOPOROSIS':
                report += """

ADDITIONAL OSTEOPOROSIS CONCERN:
• Bone health assessment recommended
• Consider DEXA scan for quantitative evaluation
• Evaluate for metabolic bone disease"""
    
    report += f"""

IMPORTANT MEDICAL DISCLAIMERS:
⚠️ This AI analysis is for screening and research purposes only.
⚠️ Results should not replace professional medical diagnosis or clinical judgment.
⚠️ Always consult with qualified healthcare professionals for final diagnosis and treatment decisions.
⚠️ Clinical correlation with patient symptoms, history, and physical examination is essential.
⚠️ This {'chest X-ray' if xray_type == 'chest' else 'bone/joint imaging'} analysis tool assists healthcare providers but cannot replace expert medical interpretation.
⚠️ In case of emergency or acute symptoms, seek immediate medical attention regardless of AI analysis results.

TECHNICAL NOTES:
• X-ray type selected by user: {xray_type.upper()}
• Analysis performed using {'chest condition specialized models' if xray_type == 'chest' else 'bone/joint specialized models'}"""

    if xray_type == 'chest':
        report += """
• Chest models used: ResNet50, DenseNet121, EfficientNetB0 for respiratory condition detection"""
    else:
        report += """
• Specialized models used: Fracture detection and Osteoporosis assessment models"""

    report += f"""
• Confidence levels reflect model certainty and should be interpreted within clinical context
• Model selection based on user X-ray type specification

Disclaimer: This automated analysis uses specialized AI models trained on medical imaging data. The {'chest X-ray' if xray_type == 'chest' else 'bone/joint imaging'} models are optimized for their respective domains based on user selection. Results should be interpreted by qualified medical professionals in conjunction with clinical findings and patient presentation.
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
    
    # Get user-selected X-ray type
    user_xray_type = request.form.get('xray_type', 'chest')  # Default to chest if not specified
    if user_xray_type not in ['chest', 'other']:
        return jsonify({'error': 'Invalid X-ray type. Must be "chest" or "other"'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction using user-selected X-ray type for routing
            predictions, error, routing_info = medical_model.predict_all(filepath, user_xray_type)
            
            if error:
                return jsonify({'error': error}), 500
            
            if not predictions:
                return jsonify({'error': 'No appropriate models available for this X-ray type'}), 500
            
            # Generate medical report with routing information
            ensemble_result = get_ensemble_prediction(predictions, routing_info)
            report = generate_medical_report(predictions, routing_info=routing_info)
            
            result = {
                'filename': filename,
                'predictions': predictions,
                'ensemble': ensemble_result,
                'report': report,
                'routing_info': routing_info,
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