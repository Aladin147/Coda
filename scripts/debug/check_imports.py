#!/usr/bin/env python3
"""
Check critical imports and identify missing dependencies.
"""

import importlib
import sys

def check_imports():
    """Check critical imports for the Coda project."""

    # Clear SSL environment variable that causes issues
    import os
    if 'SSL_CERT_FILE' in os.environ:
        del os.environ['SSL_CERT_FILE']

    critical_imports = [
        'torch', 'transformers', 'sentence_transformers',
        'fastapi', 'websockets', 'uvicorn', 'pydantic',
        'numpy', 'moshi', 'ollama', 'aiohttp', 'aiofiles'
    ]
    
    voice_specific = [
        'librosa', 'soundfile', 'pyaudio', 'webrtcvad', 
        'silero_vad', 'noisereduce'
    ]
    
    memory_specific = [
        'chromadb', 'sqlite3'
    ]
    
    performance_specific = [
        'pynvml', 'bitsandbytes', 'optimum', 'diskcache', 'redis'
    ]
    
    testing_specific = [
        'pytest', 'pytest_asyncio', 'pytest_cov', 'pytest_mock'
    ]
    
    all_packages = {
        'Critical Core': critical_imports,
        'Voice Processing': voice_specific,
        'Memory System': memory_specific,
        'Performance': performance_specific,
        'Testing': testing_specific
    }
    
    print(f"Python version: {sys.version}")
    print("=" * 60)
    
    total_missing = 0
    
    for category, packages in all_packages.items():
        print(f"\n{category}:")
        print("-" * 30)
        
        category_missing = 0
        for pkg in packages:
            try:
                importlib.import_module(pkg)
                print(f"✓ {pkg}")
            except ImportError as e:
                print(f"✗ {pkg}: {e}")
                category_missing += 1
                total_missing += 1
        
        print(f"Missing in {category}: {category_missing}/{len(packages)}")
    
    print("=" * 60)
    print(f"Total missing packages: {total_missing}")
    
    return total_missing

if __name__ == "__main__":
    missing_count = check_imports()
    sys.exit(missing_count)
