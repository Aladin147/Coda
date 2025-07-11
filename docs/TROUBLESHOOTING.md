# Coda Troubleshooting Guide

> **Comprehensive troubleshooting guide for common issues and solutions**

## 🚨 Quick Diagnostics

### System Health Check
```bash
# Run comprehensive system check
python scripts/system_check.py

# Check component status
python -c "
from coda.core.assistant import CodaAssistant
import asyncio

async def check():
    assistant = CodaAssistant()
    await assistant.initialize()
    status = await assistant.get_error_status()
    print('System Status:', status)

asyncio.run(check())
"
```

### Log Analysis
```bash
# View recent logs
tail -f logs/coda.log

# Search for errors
grep -i error logs/coda.log | tail -20

# Check specific component logs
grep -i "llm\|memory\|voice" logs/coda.log | tail -20
```

## 🔧 Installation Issues

### Python Environment Problems

#### Issue: Import Errors
```
ImportError: No module named 'coda'
```

**Solutions:**
```bash
# 1. Verify virtual environment is activated
which python  # Should point to your venv

# 2. Reinstall in development mode
pip install -e . --force-reinstall

# 3. Check Python path
python -c "import sys; print(sys.path)"

# 4. Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
```

#### Issue: Version Conflicts
```
ERROR: pip's dependency resolver does not currently consider all the packages that are installed
```

**Solutions:**
```bash
# 1. Create fresh environment
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate

# 2. Install with specific versions
pip install -r requirements-lock.txt

# 3. Use pip-tools for dependency resolution
pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt
```

### GPU Setup Issues

#### Issue: CUDA Not Available
```
RuntimeError: CUDA out of memory
RuntimeError: No CUDA GPUs are available
```

**Solutions:**
```bash
# 1. Check CUDA installation
nvidia-smi
nvcc --version

# 2. Verify PyTorch CUDA support
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"

# 3. Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. For RTX 5090 (Blackwell):
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### Issue: GPU Memory Errors
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**
```bash
# 1. Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# 2. Reduce model size in config
# Edit configs/production.yaml:
voice:
  model_size: "small"  # instead of "large"
  batch_size: 1        # reduce batch size

# 3. Enable memory optimization
voice:
  optimization:
    enable_memory_optimization: true
    max_vram_usage_gb: 6  # adjust for your GPU

# 4. Use CPU fallback for some components
llm:
  device: "cpu"  # Force CPU usage for LLM
```

## 🎤 Voice Processing Issues

### Audio System Problems

#### Issue: No Audio Input/Output
```
OSError: [Errno -9996] Invalid input device (no default output device)
```

**Solutions:**

**Windows:**
```bash
# 1. Check audio devices
python -c "
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f'{i}: {info[\"name\"]} - {info[\"maxInputChannels\"]} in, {info[\"maxOutputChannels\"]} out')
"

# 2. Install Windows audio drivers
# Update audio drivers through Device Manager

# 3. Set default audio device in Windows settings
```

**Linux:**
```bash
# 1. Install ALSA utilities
sudo apt-get install alsa-utils

# 2. Check audio devices
arecord -l  # List recording devices
aplay -l    # List playback devices

# 3. Test audio
arecord -d 5 test.wav  # Record 5 seconds
aplay test.wav         # Play back

# 4. Fix permissions
sudo usermod -a -G audio $USER
# Log out and back in
```

**macOS:**
```bash
# 1. Check audio permissions
# System Preferences > Security & Privacy > Microphone
# Ensure Terminal/Python has microphone access

# 2. Install audio dependencies
brew install portaudio
pip install pyaudio --force-reinstall
```

#### Issue: Poor Audio Quality
```
Voice recognition accuracy is low
Audio is choppy or distorted
```

**Solutions:**
```yaml
# Optimize audio settings in config
voice:
  audio:
    sample_rate: 16000      # Try different rates: 16000, 22050, 44100
    channels: 1             # Mono audio
    chunk_size: 1024        # Adjust chunk size
    enable_noise_reduction: true
    enable_echo_cancellation: true
    audio_format: "float32"
```

### Moshi Integration Issues

#### Issue: Moshi Model Loading Fails
```
FileNotFoundError: Moshi model not found
RuntimeError: Failed to load Moshi model
```

**Solutions:**
```bash
# 1. Download Moshi models manually
python scripts/download_models.py --model moshi

# 2. Check model path in config
voice:
  moshi:
    model_path: "models/moshi"  # Verify path exists
    
# 3. Use alternative model
voice:
  mode: "hybrid"  # Use hybrid mode instead of moshi-only
```

## 🧠 LLM Integration Issues

### Ollama Connection Problems

#### Issue: Cannot Connect to Ollama
```
ConnectionError: Failed to connect to Ollama server
```

**Solutions:**
```bash
# 1. Check if Ollama is running
curl http://localhost:11434/api/version

# 2. Start Ollama service
ollama serve

# 3. Check Ollama status
ollama list  # List installed models

# 4. Install required model
ollama pull qwen3:30b-a3b

# 5. Test Ollama directly
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:30b-a3b",
  "prompt": "Hello",
  "stream": false
}'
```

#### Issue: Model Not Found
```
Error: model 'qwen3:30b-a3b' not found
```

**Solutions:**
```bash
# 1. List available models
ollama list

# 2. Pull the required model
ollama pull qwen3:30b-a3b

# 3. Use alternative model
# Edit config to use available model:
llm:
  model: "llama3:8b"  # or another available model

# 4. Check model size requirements
ollama show qwen3:30b-a3b  # Check model details
```

### Performance Issues

#### Issue: Slow Response Times
```
LLM responses taking >10 seconds
High CPU/GPU usage
```

**Solutions:**
```yaml
# Optimize LLM settings
llm:
  optimization:
    max_tokens: 512        # Reduce max response length
    temperature: 0.7       # Lower temperature for faster generation
    batch_size: 1          # Reduce batch size
    enable_caching: true   # Enable response caching
    
performance:
  optimization_level: "aggressive"
  targets:
    max_response_time_ms: 2000
```

## 💾 Memory System Issues

### ChromaDB Problems

#### Issue: Database Connection Errors
```
chromadb.errors.ConnectionError: Could not connect to ChromaDB
```

**Solutions:**
```bash
# 1. Check ChromaDB installation
python -c "import chromadb; print('ChromaDB version:', chromadb.__version__)"

# 2. Clear ChromaDB data (if corrupted)
rm -rf data/memory/long_term/*

# 3. Reinitialize memory system
python scripts/reset_memory.py

# 4. Use alternative storage
# Edit config:
memory:
  provider: "sqlite"  # Fallback to SQLite
  storage_path: "data/memory/sqlite.db"
```

#### Issue: Memory Search Not Working
```
No relevant memories found
Memory retrieval is slow
```

**Solutions:**
```bash
# 1. Rebuild memory index
python scripts/rebuild_memory_index.py

# 2. Check embedding model
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print('Embedding model loaded successfully')
"

# 3. Optimize memory settings
memory:
  embedding:
    model: "all-MiniLM-L6-v2"
    device: "cuda"  # Use GPU if available
  search:
    max_results: 10
    similarity_threshold: 0.7
```

## 🌐 WebSocket & Dashboard Issues

### Connection Problems

#### Issue: WebSocket Connection Fails
```
WebSocket connection failed
Dashboard not loading
```

**Solutions:**
```bash
# 1. Check if ports are available
netstat -an | grep 8080  # Dashboard
netstat -an | grep 8765  # WebSocket

# 2. Kill processes using ports
# Find process ID:
lsof -i :8080
# Kill process:
kill -9 <PID>

# 3. Use different ports
# Edit config:
dashboard:
  port: 8081  # Use different port
websocket:
  port: 8766  # Use different port

# 4. Check firewall settings
# Windows: Allow Python through Windows Firewall
# Linux: sudo ufw allow 8080
```

#### Issue: Dashboard Not Responsive
```
Dashboard loads but doesn't update
WebSocket events not received
```

**Solutions:**
```bash
# 1. Check WebSocket server logs
grep -i websocket logs/coda.log

# 2. Test WebSocket connection manually
# Use browser console:
const ws = new WebSocket('ws://localhost:8765');
ws.onopen = () => console.log('Connected');
ws.onmessage = (e) => console.log('Message:', e.data);

# 3. Clear browser cache
# Hard refresh: Ctrl+F5 (Windows) / Cmd+Shift+R (Mac)

# 4. Check CORS settings
dashboard:
  cors:
    allow_origins: ["*"]  # Allow all origins for testing
```

## 🔄 Performance Issues

### High Resource Usage

#### Issue: High Memory Usage
```
System running out of memory
Python process using >8GB RAM
```

**Solutions:**
```yaml
# Optimize memory settings
memory:
  max_memories: 5000      # Reduce from default
  cleanup_interval: 300   # More frequent cleanup

performance:
  targets:
    max_memory_usage_percent: 70  # Lower threshold

# Enable memory optimization
voice:
  optimization:
    enable_memory_optimization: true
    
llm:
  optimization:
    enable_memory_optimization: true
```

#### Issue: High CPU Usage
```
CPU usage constantly >90%
System becomes unresponsive
```

**Solutions:**
```yaml
# Reduce processing load
voice:
  audio:
    chunk_size: 2048      # Larger chunks = less frequent processing
    
llm:
  optimization:
    max_concurrent_requests: 1  # Reduce concurrency
    
performance:
  optimization_level: "conservative"
```

## 🆘 Emergency Recovery

### Complete System Reset
```bash
# 1. Stop all Coda processes
pkill -f coda
pkill -f python.*coda

# 2. Clear all data (WARNING: This deletes all memories and sessions)
rm -rf data/
rm -rf logs/
rm -rf models/

# 3. Reinstall Coda
pip uninstall coda
pip install -e . --force-reinstall

# 4. Run setup again
python scripts/setup.py

# 5. Start with minimal config
python coda_launcher.py --config configs/minimal.yaml
```

### Safe Mode Start
```bash
# Start with minimal features
python coda_launcher.py \
  --no-voice \
  --no-dashboard \
  --config configs/safe_mode.yaml
```

## 📞 Getting Help

### Collect Debug Information
```bash
# Generate debug report
python scripts/generate_debug_report.py

# This creates debug_report.zip with:
# - System information
# - Configuration files
# - Recent logs
# - Component status
```

### Contact Support

1. **GitHub Issues**: [Report bugs](https://github.com/your-repo/coda/issues)
2. **Discussions**: [Ask questions](https://github.com/your-repo/coda/discussions)
3. **Documentation**: [Read docs](https://github.com/your-repo/coda/docs)

### Include in Bug Reports

- **System Information**: OS, Python version, GPU details
- **Error Messages**: Full error traceback
- **Configuration**: Relevant config sections (remove sensitive data)
- **Steps to Reproduce**: Exact steps that cause the issue
- **Debug Report**: Attach debug_report.zip if possible

---

**💡 Most issues can be resolved by following this guide. If you're still stuck, don't hesitate to ask for help!**
