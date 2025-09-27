"""
Demo script to test the web application API programmatically
"""
import requests
import json
import os

def test_web_app():
    """Test the web application endpoints"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Chest X-ray Web Application API...\n")
    
    # Test health endpoint
    try:
        health_response = requests.get(f"{base_url}/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print("✅ Health Check:")
            print(f"   Status: {health_data.get('status')}")
            print(f"   Models loaded: {health_data.get('models_loaded')}")
            print(f"   Available models: {health_data.get('available_models')}")
        else:
            print(f"❌ Health check failed: {health_response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to web application: {e}")
        print("   Make sure the app is running with: python app.py")
        return False
    
    print("\n" + "="*50)
    print("✅ Web Application is running successfully!")
    print(f"🌐 Access it at: {base_url}")
    print("\n📝 How to use:")
    print("1. Open your browser and go to the URL above")
    print("2. Upload a chest X-ray image (PNG, JPG, JPEG)")
    print("3. Click 'Analyze X-ray' to get AI predictions")
    print("4. Review the detailed medical report")
    
    return True

def demo_file_upload():
    """Demo file upload if test images are available"""
    base_url = "http://localhost:5000"
    
    # Look for test images in the dataset
    test_image_paths = [
        "Chest_Xray_Dataset/test/COVID19/COVID19(460).jpg",
        "Chest_Xray_Dataset/test/NORMAL", 
        "Chest_Xray_Dataset/test/PNEUMONIA",
        "Chest_Xray_Dataset/test/TURBERCULOSIS"
    ]
    
    for path in test_image_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                # Find first image in directory
                for file in os.listdir(path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        test_image = os.path.join(path, file)
                        break
                else:
                    continue
            else:
                test_image = path
            
            print(f"\n🖼️ Testing with image: {test_image}")
            
            try:
                with open(test_image, 'rb') as f:
                    files = {'file': f}
                    response = requests.post(f"{base_url}/upload", files=files, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    print("✅ Analysis successful!")
                    print(f"   Ensemble diagnosis: {result['ensemble']['diagnosis']}")
                    print(f"   Confidence: {result['ensemble']['confidence']:.1f}%")
                    print("   Individual models:")
                    for model, pred in result['predictions'].items():
                        print(f"   - {model}: {pred['class']} ({pred['confidence']:.1f}%)")
                else:
                    print(f"❌ Analysis failed: {response.status_code}")
                    print(f"   Error: {response.text}")
                
                return True
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ Request failed: {e}")
            except FileNotFoundError:
                print(f"❌ Test image not found: {test_image}")
            
            break
    else:
        print("\n💡 No test images found for automatic demo")
        print("   You can test manually by uploading images through the web interface")

if __name__ == "__main__":
    success = test_web_app()
    if success:
        demo_file_upload()