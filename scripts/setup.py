#!/usr/bin/env python3
"""
Setup script for Coda development environment.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        print(f"Error running command: {command}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    
    return result

def create_directories():
    """Create necessary directories."""
    directories = [
        "data/audio",
        "data/memory/long_term", 
        "data/temp",
        "logs",
        "cache",
        "models",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")

def check_gpu_availability():
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_gpu)
            print(f"GPU available: {gpu_name} ({gpu_count} devices)")
        else:
            print("GPU not available - using CPU only")
    except ImportError:
        print("PyTorch not installed - will install with dependencies")

def install_dependencies():
    """Install Python dependencies."""
    print("Installing dependencies...")
    
    # Install main dependencies
    run_command("pip install -e .")
    
    # Install development dependencies
    if "--dev" in sys.argv:
        run_command("pip install -e '.[dev]'")
        
        # Install pre-commit hooks
        run_command("pre-commit install")
        print("Pre-commit hooks installed")

def setup_ollama():
    """Check if Ollama is available and suggest installation."""
    result = run_command("ollama --version", check=False)
    
    if result.returncode == 0:
        print(f"Ollama is available: {result.stdout.strip()}")
        
        # Check if llama3 model is available
        result = run_command("ollama list", check=False)
        if "llama3" not in result.stdout:
            print("Downloading llama3 model (this may take a while)...")
            run_command("ollama pull llama3")
    else:
        print("Ollama not found. Please install from: https://ollama.com/")
        print("After installation, run: ollama pull llama3")

def create_sample_config():
    """Create a sample configuration file."""
    config_path = Path("configs/local.yaml")
    
    if not config_path.exists():
        print("Creating sample local configuration...")
        
        sample_config = """# Local Coda Configuration
# Copy from default.yaml and customize as needed

voice:
  stt:
    device: "cpu"  # Change to "cuda" if you have GPU
  tts:
    engine: "elevenlabs"
    # Add your ElevenLabs API key here
    # api_key: "your-api-key-here"

memory:
  long_term:
    device: "cpu"  # Change to "cuda" if you have GPU

logging:
  level: "DEBUG"  # More verbose logging for development

development:
  debug_mode: true
"""
        
        config_path.write_text(sample_config)
        print(f"Sample configuration created: {config_path}")

def main():
    """Main setup function."""
    print("🚀 Setting up Coda development environment...")
    
    # Check system requirements
    check_python_version()
    check_gpu_availability()
    
    # Create directories
    create_directories()
    
    # Install dependencies
    install_dependencies()
    
    # Setup external tools
    setup_ollama()
    
    # Create sample configuration
    create_sample_config()
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Configure your settings in configs/local.yaml")
    print("2. Add your ElevenLabs API key (if using)")
    print("3. Run: coda run --config configs/local.yaml")
    print("\nFor development:")
    print("- Run tests: pytest")
    print("- Format code: black src/ tests/")
    print("- Type check: mypy src/")

if __name__ == "__main__":
    main()
