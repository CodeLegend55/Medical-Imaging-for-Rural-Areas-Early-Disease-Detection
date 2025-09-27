@echo off
echo 🚀 Starting Chest X-ray Analysis Web Application...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python is installed

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo 📚 Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install requirements
    pause
    exit /b 1
)

REM Check if models exist
echo 🤖 Checking for AI models...
if not exist "models\ResNet50_colab.pth" (
    echo ⚠️  Warning: ResNet50_colab.pth not found in models folder
)
if not exist "models\DenseNet121_colab.pth" (
    echo ⚠️  Warning: DenseNet121_colab.pth not found in models folder  
)
if not exist "models\EfficientNetB0_colab.pth" (
    echo ⚠️  Warning: EfficientNetB0_colab.pth not found in models folder
)

echo.
echo 🌐 Starting web application...
echo Access the app at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

REM Start the Flask app
python app.py

pause