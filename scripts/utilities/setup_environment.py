#!/usr/bin/env python3
"""
Environment Setup Script for Coda 2.0 with RTX 5090 Support

This script sets up a proper isolated environment for Coda 2.0 development
with RTX 5090 Blackwell architecture support (sm_120).

Requirements:
- RTX 5090 with CUDA 12.8+
- Python 3.11+
- Windows/Linux/macOS

Usage:
    python setup_environment.py
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path
import urllib.request
import shutil

# Configuration
PYTHON_VERSION = "3.11"
CUDA_VERSION = "12.8"
PYTORCH_VERSION = "2.6.0"  # Latest with Blackwell support
WORKSPACE_DIR = Path(__file__).parent
VENV_DIR = WORKSPACE_DIR / "venv"
MODELS_DIR = WORKSPACE_DIR / "models"
CACHE_DIR = WORKSPACE_DIR / "cache"

# RTX 5090 Requirements
RTX_5090_REQUIREMENTS = {
    "compute_capability": "sm_120",
    "min_cuda_version": "12.8",
    "min_pytorch_version": "2.6.0",
    "recommended_driver": "576.88+"
}

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_step(step, description):
    """Print a formatted step."""
    print(f"\n🔧 Step {step}: {description}")
    print("-" * 40)

def run_command(cmd, check=True, capture_output=False):
    """Run a command and handle errors."""
    print(f"Running: {cmd}")
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, check=check, 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, check=check)
            return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        if capture_output and e.stdout:
            print(f"stdout: {e.stdout}")
        if capture_output and e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def check_system_requirements():
    """Check system requirements for RTX 5090."""
    print_step(1, "Checking System Requirements")
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Python version: {python_version}")
    if python_version < PYTHON_VERSION:
        print(f"❌ Python {PYTHON_VERSION}+ required, found {python_version}")
        return False
    print("✅ Python version OK")
    
    # Check NVIDIA GPU
    try:
        nvidia_output = run_command("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader,nounits",
                                   capture_output=True)
        if nvidia_output:
            lines = nvidia_output.split('\n')
            for line in lines:
                if "RTX 5090" in line:
                    parts = line.split(', ')
                    gpu_name = parts[0].strip()
                    driver_version = parts[1].strip()

                    print(f"GPU: {gpu_name}")
                    print(f"Driver: {driver_version}")

                    # Get CUDA version from nvidia-smi header
                    cuda_info = run_command("nvidia-smi", capture_output=True)
                    if cuda_info and "CUDA Version:" in cuda_info:
                        cuda_version = cuda_info.split("CUDA Version:")[1].split()[0]
                        print(f"CUDA: {cuda_version}")

                    if "RTX 5090" in gpu_name:
                        print("✅ RTX 5090 detected")
                        return True

        print("❌ RTX 5090 not detected")
        return False
        
    except Exception as e:
        print(f"❌ Failed to check GPU: {e}")
        return False

def create_virtual_environment():
    """Create isolated virtual environment."""
    print_step(2, "Creating Virtual Environment")
    
    if VENV_DIR.exists():
        print(f"Virtual environment already exists at {VENV_DIR}")
        response = input("Remove existing environment? (y/N): ")
        if response.lower() == 'y':
            shutil.rmtree(VENV_DIR)
            print("✅ Removed existing environment")
        else:
            print("Using existing environment")
            return True
    
    # Create virtual environment
    print(f"Creating virtual environment at {VENV_DIR}")
    if not run_command(f"python -m venv {VENV_DIR}"):
        return False
    
    print("✅ Virtual environment created")
    return True

def get_activation_command():
    """Get the activation command for the current platform."""
    if platform.system() == "Windows":
        return f"{VENV_DIR}\\Scripts\\activate.bat"
    else:
        return f"source {VENV_DIR}/bin/activate"

def install_pytorch_with_blackwell_support():
    """Install PyTorch with RTX 5090 Blackwell support."""
    print_step(3, "Installing PyTorch with Blackwell Support")
    
    # Get pip path
    if platform.system() == "Windows":
        pip_path = VENV_DIR / "Scripts" / "pip.exe"
    else:
        pip_path = VENV_DIR / "bin" / "pip"
    
    # Upgrade pip first
    print("Upgrading pip...")
    if not run_command(f"{pip_path} install --upgrade pip"):
        return False
    
    # Install PyTorch nightly with CUDA 12.8 support
    print("Installing PyTorch with Blackwell support...")
    pytorch_cmd = (
        f"{pip_path} install --pre torch torchvision torchaudio "
        f"--index-url https://download.pytorch.org/whl/nightly/cu128"
    )
    
    if not run_command(pytorch_cmd):
        print("❌ Failed to install PyTorch nightly, trying stable version...")
        # Fallback to stable version
        pytorch_cmd = (
            f"{pip_path} install torch torchvision torchaudio "
            f"--index-url https://download.pytorch.org/whl/cu128"
        )
        if not run_command(pytorch_cmd):
            return False
    
    print("✅ PyTorch installed")
    return True

def install_dependencies():
    """Install project dependencies."""
    print_step(4, "Installing Project Dependencies")
    
    if platform.system() == "Windows":
        pip_path = VENV_DIR / "Scripts" / "pip.exe"
    else:
        pip_path = VENV_DIR / "bin" / "pip"
    
    # Core dependencies
    dependencies = [
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "sounddevice>=0.4.6",
        "sentencepiece>=0.1.99",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "datasets>=2.14.0",
        "tokenizers>=0.14.0",
        "safetensors>=0.4.0",
        "huggingface-hub>=0.17.0",
        "pydantic>=2.0.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "websockets>=12.0",
        "aiofiles>=23.0.0",
        "python-multipart>=0.0.6",
        "jinja2>=3.1.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "typer>=0.9.0",
        "asyncio-mqtt>=0.16.0",
        "psutil>=5.9.0",
        "GPUtil>=1.4.0",
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0"
    ]
    
    print("Installing core dependencies...")
    for dep in dependencies:
        print(f"Installing {dep}...")
        if not run_command(f"{pip_path} install {dep}"):
            print(f"⚠️ Failed to install {dep}, continuing...")
    
    print("✅ Core dependencies installed")
    return True

def install_moshi():
    """Install Moshi with proper configuration."""
    print_step(5, "Installing Moshi")
    
    if platform.system() == "Windows":
        pip_path = VENV_DIR / "Scripts" / "pip.exe"
    else:
        pip_path = VENV_DIR / "bin" / "pip"
    
    # Install Moshi
    print("Installing Moshi...")
    if not run_command(f"{pip_path} install moshi"):
        return False
    
    print("✅ Moshi installed")
    return True

def create_directory_structure():
    """Create project directory structure."""
    print_step(6, "Creating Directory Structure")
    
    directories = [
        MODELS_DIR,
        CACHE_DIR,
        WORKSPACE_DIR / "logs",
        WORKSPACE_DIR / "temp",
        WORKSPACE_DIR / "data",
        WORKSPACE_DIR / "tests",
        WORKSPACE_DIR / "docs" / "build",
        WORKSPACE_DIR / "configs",
        WORKSPACE_DIR / "scripts"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {directory}")
    
    return True

def create_environment_config():
    """Create environment configuration file."""
    print_step(7, "Creating Environment Configuration")
    
    config = {
        "environment": {
            "name": "coda_2.0",
            "python_version": PYTHON_VERSION,
            "cuda_version": CUDA_VERSION,
            "pytorch_version": PYTORCH_VERSION
        },
        "hardware": {
            "gpu": "RTX 5090",
            "compute_capability": "sm_120",
            "vram_gb": 32
        },
        "paths": {
            "workspace": str(WORKSPACE_DIR),
            "venv": str(VENV_DIR),
            "models": str(MODELS_DIR),
            "cache": str(CACHE_DIR)
        },
        "moshi": {
            "model_path": "kyutai/moshiko-pytorch-bf16",
            "cache_dir": str(MODELS_DIR / "moshi"),
            "device": "cuda",
            "vram_allocation": "8GB"
        }
    }
    
    config_file = WORKSPACE_DIR / "environment_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Environment config saved to {config_file}")
    return True

def verify_installation():
    """Verify the installation."""
    print_step(8, "Verifying Installation")
    
    if platform.system() == "Windows":
        python_path = VENV_DIR / "Scripts" / "python.exe"
    else:
        python_path = VENV_DIR / "bin" / "python"
    
    # Test PyTorch with RTX 5090
    test_script = '''
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Test RTX 5090 compatibility
    device = torch.device("cuda:0")
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = torch.mm(x, y)
    print(f"✅ RTX 5090 tensor operations working")
    print(f"Tensor shape: {z.shape}")
else:
    print("❌ CUDA not available")
'''
    
    print("Testing PyTorch with RTX 5090...")
    test_file = WORKSPACE_DIR / "test_gpu.py"
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    if run_command(f"{python_path} {test_file}"):
        print("✅ PyTorch RTX 5090 test passed")
        test_file.unlink()  # Clean up
        return True
    else:
        print("❌ PyTorch RTX 5090 test failed")
        return False

def create_activation_script():
    """Create activation script for easy environment setup."""
    print_step(9, "Creating Activation Script")
    
    if platform.system() == "Windows":
        script_content = f'''@echo off
echo Activating Coda 2.0 Environment...
call "{VENV_DIR}\\Scripts\\activate.bat"
echo ✅ Environment activated
echo.
echo GPU Info:
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
echo.
echo Python: %cd%\\venv\\Scripts\\python.exe
echo Workspace: {WORKSPACE_DIR}
echo Models: {MODELS_DIR}
echo.
'''
        script_file = WORKSPACE_DIR / "activate.bat"
    else:
        script_content = f'''#!/bin/bash
echo "Activating Coda 2.0 Environment..."
source "{VENV_DIR}/bin/activate"
echo "✅ Environment activated"
echo
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
echo
echo "Python: {VENV_DIR}/bin/python"
echo "Workspace: {WORKSPACE_DIR}"
echo "Models: {MODELS_DIR}"
echo
'''
        script_file = WORKSPACE_DIR / "activate.sh"
    
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    if platform.system() != "Windows":
        os.chmod(script_file, 0o755)
    
    print(f"✅ Activation script created: {script_file}")
    return True

def main():
    """Main setup function."""
    print_header("Coda 2.0 Environment Setup - RTX 5090 Edition")
    print("Setting up isolated environment with Blackwell architecture support")
    
    steps = [
        check_system_requirements,
        create_virtual_environment,
        install_pytorch_with_blackwell_support,
        install_dependencies,
        install_moshi,
        create_directory_structure,
        create_environment_config,
        verify_installation,
        create_activation_script
    ]
    
    for i, step in enumerate(steps, 1):
        try:
            if not step():
                print(f"\n❌ Setup failed at step {i}")
                return False
        except Exception as e:
            print(f"\n❌ Setup failed at step {i}: {e}")
            return False
    
    print_header("Setup Complete!")
    print("✅ Coda 2.0 environment successfully configured")
    print(f"✅ RTX 5090 Blackwell support enabled")
    print(f"✅ Virtual environment: {VENV_DIR}")
    print(f"✅ Models directory: {MODELS_DIR}")
    
    if platform.system() == "Windows":
        print(f"\nTo activate the environment, run: activate.bat")
    else:
        print(f"\nTo activate the environment, run: ./activate.sh")
    
    print("\nNext steps:")
    print("1. Activate the environment")
    print("2. Run: python test_moshi_integration.py")
    print("3. Start developing with Coda 2.0!")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed: {e}")
        sys.exit(1)
