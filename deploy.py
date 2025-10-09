#!/usr/bin/env python3
"""
LaneVision Deployment Helper Script
This script helps prepare your Django project for deployment
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"[INFO] {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"[SUCCESS] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("LaneVision Deployment Preparation")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('manage.py'):
        print("[ERROR] manage.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Collect static files
    if not run_command("python manage.py collectstatic --noinput", "Collecting static files"):
        print("[WARNING] Static files collection failed, but continuing...")
    
    # Run migrations
    if not run_command("python manage.py migrate", "Running database migrations"):
        print("[WARNING] Migrations failed, but continuing...")
    
    # Check for large files
    print("\n[INFO] Checking for large files...")
    large_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            filepath = os.path.join(root, file)
            if os.path.getsize(filepath) > 50 * 1024 * 1024:  # 50MB
                large_files.append(filepath)
    
    if large_files:
        print("[WARNING] Large files detected (>50MB):")
        for file in large_files:
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"   - {file} ({size_mb:.1f} MB)")
        print("\n[INFO] Consider using Git LFS for these files or hosting them separately.")
    
    # Check requirements
    print("\n[INFO] Checking requirements.txt...")
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
            if 'gunicorn' in requirements and 'whitenoise' in requirements:
                print("[SUCCESS] Production requirements found")
            else:
                print("[WARNING] Missing production requirements (gunicorn, whitenoise)")
    
    # Check Procfile
    if os.path.exists('Procfile'):
        print("[SUCCESS] Procfile found")
    else:
        print("[ERROR] Procfile not found - required for deployment")
    
    print("\n[SUCCESS] Deployment preparation complete!")
    print("\n[INFO] Next steps:")
    print("1. Choose a deployment platform (Railway, Render, or PythonAnywhere)")
    print("2. Follow the instructions in DEPLOYMENT.md")
    print("3. Set environment variables (DEBUG=False, SECRET_KEY)")
    print("4. Deploy your app!")
    
    print("\n[INFO] Quick links:")
    print("- Railway: https://railway.app")
    print("- Render: https://render.com")
    print("- PythonAnywhere: https://pythonanywhere.com")

if __name__ == "__main__":
    main()
