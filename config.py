# Configuration file for the Chest X-ray Analysis Web Application

# OpenAI API Configuration (Optional)
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY = "your-openai-api-key-here"

# If you don't have an OpenAI API key, leave it as is
# The app will use a fallback report generation system

# Flask Configuration
UPLOAD_FOLDER = "uploads"
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# Model Configuration
MODEL_PATH = "models"
CLASSES = ["COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"]

# Fracture Model Classes
FRACTURE_CLASSES = ["NON_FRACTURED", "FRACTURED"]

# Server Configuration
DEBUG = True
HOST = "0.0.0.0"
PORT = 5000