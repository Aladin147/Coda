#!/bin/bash

echo ""
echo "========================================"
echo "  Coda 2.0 Development Environment"
echo "  RTX 5090 Blackwell Edition"
echo "========================================"
echo ""

# Activate virtual environment
source "venv/Scripts/activate"

echo "✅ Virtual environment activated"
echo ""

# Display system information
echo "🖥️  System Information:"
echo "  Workspace: $(pwd)"
echo "  Python: $(pwd)/venv/Scripts/python.exe"
echo "  Models: $(pwd)/models"
echo "  Cache: $(pwd)/cache"
echo ""

# Display GPU information
echo "🎮 GPU Information:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null || echo "  GPU info not available"
else
    echo "  nvidia-smi not found"
fi
echo ""

# Display PyTorch information
echo "🔥 PyTorch Information:"
venv/Scripts/python.exe -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'  Device: {torch.cuda.get_device_name(0)}')
else:
    print('  Device: CPU')
" 2>/dev/null || echo "  PyTorch info not available"

echo ""
echo "🚀 Environment ready for development!"
echo ""
echo "Available commands:"
echo "  python test_moshi_integration.py  - Test Moshi integration"
echo "  python -m pytest tests/          - Run tests"
echo "  python -m black src/              - Format code"
echo "  python -m mypy src/               - Type checking"
echo ""
