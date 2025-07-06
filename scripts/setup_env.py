import sys
import subprocess
import os

def check_and_install_packages():
    print("Python Environment Information:")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    # List of required packages
    required_packages = ['pandas', 'numpy']
    
    print("\nChecking for required packages...")
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is not installed. Installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to install {package}. Error: {e}")

if __name__ == "__main__":
    check_and_install_packages()
    print("\nSetup complete. Try running process_matches.py again.")
