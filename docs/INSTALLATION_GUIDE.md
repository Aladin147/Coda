# Coda Installation Guide

> **Complete setup guide for Coda - Core Operations & Digital Assistant**

## 📋 Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.10+ | 3.11+ |
| **RAM** | 8GB | 16GB+ |
| **Storage** | 10GB free | 20GB+ free |
| **GPU** | Optional | NVIDIA RTX 3060+ |
| **OS** | Windows 10/11, Linux, macOS | Windows 11, Ubuntu 22.04+ |

### Hardware Recommendations

#### For Voice Processing
- **GPU**: NVIDIA RTX 4060+ with 8GB+ VRAM
- **RAM**: 16GB+ system memory
- **CPU**: Modern multi-core processor (Intel i5/AMD Ryzen 5+)

#### For Text-Only Usage
- **RAM**: 8GB system memory
- **CPU**: Any modern processor
- **GPU**: Not required

## 🚀 Quick Installation

### Option 1: Standard Setup (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/your-repo/coda.git
cd coda

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install Coda
pip install -e .

# 4. Run initial setup
python scripts/setup.py

# 5. Start Coda
python coda_launcher.py
```

### Option 2: Development Setup

```bash
# 1. Clone and enter directory
git clone https://github.com/your-repo/coda.git
cd coda

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install with development dependencies
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Run tests to verify installation
pytest tests/

# 6. Start Coda
python coda_launcher.py --config configs/development.yaml
```

## 🔧 Detailed Installation Steps

### Step 1: Environment Setup

#### Create Isolated Environment
```bash
# Create virtual environment
python -m venv coda_env

# Activate environment
# On Windows:
coda_env\Scripts\activate
# On Linux/macOS:
source coda_env/bin/activate

# Verify Python version
python --version  # Should be 3.10+
```

#### Install Base Dependencies
```bash
# Update pip
pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt
```

### Step 2: GPU Setup (Optional but Recommended)

#### NVIDIA GPU Setup
```bash
# Check CUDA availability
nvidia-smi

# Install PyTorch with CUDA support
# For RTX 5090 (Blackwell architecture):
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# For other modern GPUs:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU setup
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

#### AMD GPU Setup (Experimental)
```bash
# Install ROCm support (Linux only)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
```

### Step 3: Voice Processing Setup

#### Install Voice Dependencies
```bash
# Install voice processing libraries
pip install librosa soundfile pyaudio

# Install Moshi dependencies (if using voice)
pip install moshi-pytorch  # Example - adjust based on actual library
```

#### Audio System Setup

**Windows:**
```bash
# Install Windows audio dependencies
pip install pyaudio-windows
```

**Linux:**
```bash
# Install system audio libraries
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio alsa-utils

# Install Python audio libraries
pip install pyaudio
```

**macOS:**
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install audio dependencies
brew install portaudio
pip install pyaudio
```

### Step 4: LLM Backend Setup

#### Ollama Setup (Recommended)
```bash
# Install Ollama
# Windows: Download from https://ollama.ai
# Linux/macOS:
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull recommended model
ollama pull qwen3:30b-a3b
```

#### Alternative LLM Providers
```bash
# For OpenAI API (optional fallback)
pip install openai
export OPENAI_API_KEY="your-api-key"

# For Anthropic Claude (optional)
pip install anthropic
export ANTHROPIC_API_KEY="your-api-key"
```

### Step 5: Database Setup

#### ChromaDB (Default)
```bash
# ChromaDB is installed automatically with requirements
# No additional setup required
```

#### PostgreSQL (Optional)
```bash
# Install PostgreSQL dependencies
pip install psycopg2-binary

# Setup database (example)
createdb coda_memory
```

### Step 6: Configuration

#### Create Configuration File
```bash
# Copy example configuration
cp configs/production.yaml.example configs/production.yaml

# Edit configuration
nano configs/production.yaml  # or your preferred editor
```

#### Basic Configuration Example
```yaml
# configs/production.yaml
llm:
  provider: "ollama"
  model: "qwen3:30b-a3b"
  base_url: "http://localhost:11434"

voice:
  enabled: true
  mode: "adaptive"
  audio:
    sample_rate: 24000
    channels: 1

memory:
  provider: "chromadb"
  storage_path: "data/memory"

dashboard:
  enabled: true
  host: "localhost"
  port: 8080

websocket:
  enabled: true
  host: "localhost"
  port: 8765
```

## ✅ Verification

### Test Installation
```bash
# Run system check
python scripts/system_check.py

# Run basic tests
python -m pytest tests/test_basic.py -v

# Test individual components
python -c "from coda.core.assistant import CodaAssistant; print('✅ Core import successful')"
python -c "from coda.llm.manager import LLMManager; print('✅ LLM import successful')"
python -c "from coda.memory.manager import MemoryManager; print('✅ Memory import successful')"
```

### Launch Coda
```bash
# Start with text-only mode (safe test)
python coda_launcher.py --no-voice

# Start with full features
python coda_launcher.py

# Start with dashboard
python coda_launcher.py --dashboard
```

### Verify Features
```bash
# Check system status
curl http://localhost:8080/api/status

# Test WebSocket connection
# Open browser to http://localhost:8080

# Test voice processing (if enabled)
# Use the dashboard interface or WebSocket client
```

## 🔧 Troubleshooting

### Common Issues

#### Import Errors
```bash
# If you get import errors:
pip install --upgrade pip setuptools wheel
pip install -e . --force-reinstall
```

#### GPU Issues
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch if needed
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Audio Issues
```bash
# Test audio system
python -c "import pyaudio; print('Audio system working')"

# Check audio devices
python scripts/test_audio.py
```

#### Port Conflicts
```bash
# Check if ports are in use
netstat -an | grep 8080  # Dashboard port
netstat -an | grep 8765  # WebSocket port

# Kill processes using ports
# Windows: taskkill /F /PID <pid>
# Linux/macOS: kill -9 <pid>
```

### Performance Issues

#### Memory Usage
```bash
# Monitor memory usage
python scripts/monitor_memory.py

# Reduce memory usage in config
# Set smaller model sizes, reduce cache sizes
```

#### Slow Response Times
```bash
# Check system resources
python scripts/system_monitor.py

# Optimize configuration for your hardware
python scripts/optimize_config.py
```

## 📚 Next Steps

After successful installation:

1. **Read the User Guide**: [docs/USER_GUIDE.md](USER_GUIDE.md)
2. **Explore Examples**: [examples/](../examples/)
3. **Configure for Your Needs**: [docs/CONFIGURATION.md](CONFIGURATION.md)
4. **Join the Community**: [GitHub Discussions](https://github.com/your-repo/coda/discussions)

## 🆘 Getting Help

- **Documentation**: [docs/](.)
- **Troubleshooting**: [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **GitHub Issues**: [Report bugs](https://github.com/your-repo/coda/issues)
- **Discussions**: [Ask questions](https://github.com/your-repo/coda/discussions)

---

**🎉 Welcome to Coda! You're ready to start your journey with next-generation AI assistance.**
