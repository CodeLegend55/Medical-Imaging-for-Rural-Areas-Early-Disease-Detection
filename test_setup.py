"""
Simple test script to verify the web application setup
"""
import os
import sys
import torch
from torchvision import models

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'flask', 'torch', 'torchvision', 'PIL', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    return missing_packages

def check_models():
    """Check if model files exist and can be loaded"""
    model_files = [
        'models/ResNet50_colab.pth',
        'models/DenseNet121_colab.pth', 
        'models/EfficientNetB0_colab.pth'
    ]
    
    existing_models = []
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                # Try to load the model state dict
                state_dict = torch.load(model_file, map_location='cpu')
                print(f"✅ {model_file} - Loadable")
                existing_models.append(model_file)
            except Exception as e:
                print(f"⚠️  {model_file} - Exists but has issues: {e}")
        else:
            print(f"❌ {model_file} - Not found")
    
    return existing_models

def check_directories():
    """Check if required directories exist"""
    directories = ['templates', 'models']
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ {directory}/ directory exists")
        else:
            print(f"❌ {directory}/ directory missing")
            return False
    
    # Check if template file exists
    if os.path.exists('templates/index.html'):
        print("✅ templates/index.html exists")
    else:
        print("❌ templates/index.html missing")
        return False
        
    return True

def main():
    print("🔍 Checking Chest X-ray Web Application Setup...\n")
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
    
    print("\n📦 Checking Required Packages:")
    missing_packages = check_requirements()
    
    print("\n📁 Checking Directories:")
    directories_ok = check_directories()
    
    print("\n🤖 Checking AI Models:")
    existing_models = check_models()
    
    print("\n" + "="*50)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    if not directories_ok:
        print("❌ Directory structure incomplete")
        return False
    
    if not existing_models:
        print("❌ No AI models found")
        print("Please ensure your trained .pth files are in the models/ directory")
        return False
    
    print("✅ Setup looks good!")
    print(f"✅ Found {len(existing_models)} AI models")
    print("\n🚀 You can now run: python app.py")
    print("🌐 Then visit: http://localhost:5000")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)