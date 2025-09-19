"""
Dataset download utility for Chest X-ray dataset from Kaggle.
Make sure you have kaggle API credentials set up before running this script.
"""

import os
import zipfile
import shutil
import requests
from pathlib import Path

def download_chest_xray_dataset():
    """Download the chest X-ray dataset from Kaggle"""
    
    print("Starting dataset download...")
    
    try:
        import kaggle
        print("Kaggle API found.")
    except ImportError:
        print("Kaggle API not installed. Please install it using: pip install kaggle")
        return False
    
    # Create data directory
    data_dir = Path("chest_xray_merged")
    data_dir.mkdir(exist_ok=True)
    
    try:
        # Download the dataset
        print("Downloading chest X-ray dataset from Kaggle...")
        kaggle.api.dataset_download_files(
            'jtiptj/chest-xray-pneumoniacovid19tuberculosis',
            path=str(data_dir),
            unzip=True
        )
        
        print("Dataset downloaded successfully!")
        
        # Show data structure
        print("\nDataset structure:")
        show_directory_structure(data_dir)
        
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nPlease make sure:")
        print("1. You have kaggle.json file in ~/.kaggle/ (Linux/Mac) or C:\\Users\\{username}\\.kaggle\\ (Windows)")
        print("2. Your Kaggle API credentials are valid")
        print("3. You have accepted the dataset's terms on Kaggle website")
        return False

def show_directory_structure(path, max_files=5):
    """Show the directory structure of the downloaded dataset"""
    path = Path(path)
    for root, dirs, files in os.walk(path):
        level = root.replace(str(path), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        
        # Show first few files only
        for i, file in enumerate(files):
            if i < max_files:
                print(f'{subindent}{file}')
        
        if len(files) > max_files:
            print(f'{subindent}... and {len(files) - max_files} more files')

def setup_kaggle_credentials():
    """Help user set up Kaggle credentials"""
    print("\nTo download the dataset, you need Kaggle API credentials:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Click 'Create New API Token' to download kaggle.json")
    print("3. Place kaggle.json in:")
    print("   - Windows: C:\\Users\\{username}\\.kaggle\\")
    print("   - Linux/Mac: ~/.kaggle/")
    print("4. Make sure the file permissions are 600 (chmod 600 ~/.kaggle/kaggle.json on Linux/Mac)")
    print("5. Accept the dataset terms at: https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis")

def check_dataset_structure():
    """Check if the dataset is properly structured"""
    expected_structure = {
        'chest_xray_merged': {
            'train': ['covid', 'normal', 'pneumonia', 'tb'],
            'test': ['covid', 'normal', 'pneumonia', 'tb'],
            'val': ['covid', 'normal', 'pneumonia', 'tb']
        }
    }
    
    base_path = Path("chest_xray_merged")
    if not base_path.exists():
        print("Dataset directory not found. Please download the dataset first.")
        return False
    
    print("Checking dataset structure...")
    all_good = True
    
    for split in ['train', 'test', 'val']:
        split_path = base_path / split
        if not split_path.exists():
            print(f"❌ Missing {split} directory")
            all_good = False
            continue
            
        for class_name in ['covid', 'normal', 'pneumonia', 'tb']:
            class_path = split_path / class_name
            if not class_path.exists():
                print(f"❌ Missing {split}/{class_name} directory")
                all_good = False
            else:
                image_count = len([f for f in class_path.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
                print(f"✅ {split}/{class_name}: {image_count} images")
    
    if all_good:
        print("\n✅ Dataset structure is correct!")
    else:
        print("\n❌ Dataset structure has issues. Please check the download.")
    
    return all_good

def main():
    """Main function to handle dataset download and setup"""
    print("="*50)
    print("Chest X-Ray Dataset Download Utility")
    print("="*50)
    
    # Check if dataset already exists
    if Path("chest_xray_merged").exists():
        print("Dataset directory already exists.")
        choice = input("Do you want to check the dataset structure? (y/n): ").lower()
        if choice == 'y':
            check_dataset_structure()
        return
    
    # Check for Kaggle API
    try:
        import kaggle
        print("Kaggle API is available.")
    except ImportError:
        print("Kaggle API not found. Installing...")
        os.system("pip install kaggle")
        try:
            import kaggle
        except ImportError:
            print("Failed to install Kaggle API. Please install manually: pip install kaggle")
            return
    
    # Check for credentials
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("Kaggle credentials not found.")
        setup_kaggle_credentials()
        return
    
    # Download dataset
    success = download_chest_xray_dataset()
    
    if success:
        print("\n" + "="*50)
        print("Dataset download completed!")
        print("="*50)
        print("You can now run the training script:")
        print("python src/main.py")
    else:
        print("\n" + "="*50)
        print("Dataset download failed!")
        print("="*50)
        setup_kaggle_credentials()

if __name__ == "__main__":
    main()