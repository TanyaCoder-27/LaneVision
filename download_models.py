#!/usr/bin/env python3
"""
Download YOLO models for deployment
This script downloads the required YOLO models during deployment
"""

import os
import urllib.request
from pathlib import Path

def download_file(url, filename):
    """Download a file from URL"""
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✅ {filename} downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def main():
    """Download required YOLO models"""
    print("Downloading YOLO models for deployment...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # YOLO model URLs (using official Ultralytics releases)
    models = {
        "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt", 
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    }
    
    success_count = 0
    for filename, url in models.items():
        if download_file(url, filename):
            success_count += 1
    
    print(f"\nDownloaded {success_count}/{len(models)} models successfully")
    
    if success_count == len(models):
        print("✅ All models downloaded successfully!")
        return True
    else:
        print("⚠️ Some models failed to download")
        return False

if __name__ == "__main__":
    main()
