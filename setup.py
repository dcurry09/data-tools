#!/usr/bin/env python3
"""
Setup script for Data Science Toolkit
Handles environment setup and dependency installation
"""

import os
import sys
import subprocess
import platform

def create_virtual_env():
    """Create a virtual environment"""
    print("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    print("✓ Virtual environment created")

def get_activation_command():
    """Get the appropriate activation command for the OS"""
    system = platform.system()
    if system == "Windows":
        return os.path.join("venv", "Scripts", "activate")
    else:
        return os.path.join("venv", "bin", "activate")

def install_requirements():
    """Install required packages"""
    print("\nInstalling requirements...")
    
    # Determine pip executable
    if platform.system() == "Windows":
        pip_executable = os.path.join("venv", "Scripts", "pip")
    else:
        pip_executable = os.path.join("venv", "bin", "pip")
    
    # Upgrade pip first
    subprocess.run([pip_executable, "install", "--upgrade", "pip"], check=True)
    
    # Install requirements
    subprocess.run([pip_executable, "install", "-r", "requirements.txt"], check=True)
    print("✓ Requirements installed")

def create_directories():
    """Create project directory structure"""
    directories = ["data/raw", "data/processed", "notebooks", "src", "tests"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ Directory structure created")

def generate_test_data():
    """Generate test data if script exists"""
    if os.path.exists("generate_test_data.py"):
        print("\nGenerating test data...")
        subprocess.run([sys.executable, "generate_test_data.py"], check=True)
        print("✓ Test data generated")

def main():
    """Main setup function"""
    print("🚀 Setting up Data Science Toolkit")
    print("=" * 40)
    
    try:
        # Create virtual environment
        create_virtual_env()
        
        # Install requirements
        install_requirements()
        
        # Create directory structure
        create_directories()
        
        # Generate test data
        generate_test_data()
        
        print("\n✅ Setup completed successfully!")
        print("\nNext steps:")
        print(f"1. Activate virtual environment: source {get_activation_command()}")
        print("2. Launch Jupyter: jupyter notebook")
        print("3. Open notebooks/demo.ipynb to get started")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during setup: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
