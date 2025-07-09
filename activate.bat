@echo off
echo.
echo ========================================
echo   Coda 2.0 Development Environment
echo   RTX 5090 Blackwell Edition
echo ========================================
echo.

REM Activate virtual environment
call "venv\Scripts\activate.bat"

echo ✅ Virtual environment activated
echo.

REM Display system information
echo 🖥️  System Information:
echo   Workspace: %cd%
echo   Python: %cd%\venv\Scripts\python.exe
echo   Models: %cd%\models
echo   Cache: %cd%\cache
echo.

REM Display GPU information
echo 🎮 GPU Information:
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>nul
if errorlevel 1 (
    echo   GPU info not available
) else (
    echo.
)

REM Display PyTorch information
echo 🔥 PyTorch Information:
venv\Scripts\python.exe -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.version.cuda}'); print(f'  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" 2>nul
if errorlevel 1 (
    echo   PyTorch info not available
)

echo.
echo 🚀 Environment ready for development!
echo.
echo Available commands:
echo   python test_moshi_integration.py  - Test Moshi integration
echo   python -m pytest tests/          - Run tests
echo   python -m black src/              - Format code
echo   python -m mypy src/               - Type checking
echo.
