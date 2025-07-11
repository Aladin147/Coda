#!/usr/bin/env python3
"""
Validate core system functionality for Coda project.
"""

import os
import sys
import importlib

def clear_ssl_env():
    """Clear SSL environment variables that cause issues."""
    if 'SSL_CERT_FILE' in os.environ:
        del os.environ['SSL_CERT_FILE']

def test_gpu_detection():
    """Test RTX 5090 GPU detection."""
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            
            # Test basic CPU tensor operations (GPU not supported yet)
            x = torch.randn(10, 10)
            y = torch.mm(x, x.t())
            print("✅ PyTorch CPU operations working")
            return True
        else:
            print("❌ CUDA not available")
            return False
            
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_core_imports():
    """Test critical core imports."""
    core_modules = [
        'torch', 'transformers', 'sentence_transformers',
        'fastapi', 'websockets', 'uvicorn', 'pydantic',
        'numpy', 'moshi', 'ollama', 'aiohttp', 'aiofiles'
    ]
    
    failed = []
    for module in core_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed.append(module)
    
    return len(failed) == 0

def test_voice_components():
    """Test voice processing components."""
    voice_modules = ['silero_vad', 'noisereduce']
    
    failed = []
    for module in voice_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed.append(module)
    
    # Test webrtcvad (via webrtcvad-wheels)
    try:
        import webrtcvad
        vad = webrtcvad.Vad(1)
        print("✅ webrtcvad (via webrtcvad-wheels)")
    except ImportError:
        print("❌ webrtcvad not available")
        failed.append('webrtcvad')
    except Exception as e:
        print(f"❌ webrtcvad functionality failed: {e}")
        failed.append('webrtcvad')
    
    return len(failed) == 0

def test_memory_system():
    """Test memory system components."""
    try:
        import chromadb
        print("✅ chromadb")
        
        # Test basic ChromaDB functionality
        client = chromadb.PersistentClient(path="./temp/test_chroma")
        collection = client.get_or_create_collection("test")
        print("✅ ChromaDB basic operations working")
        return True
        
    except ImportError as e:
        print(f"❌ chromadb: {e}")
        return False
    except Exception as e:
        print(f"❌ ChromaDB operations failed: {e}")
        return False

def test_performance_tools():
    """Test performance monitoring tools."""
    perf_modules = ['pynvml', 'bitsandbytes', 'optimum', 'diskcache', 'redis', 'structlog']
    
    failed = []
    for module in perf_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed.append(module)
    
    return len(failed) == 0

def main():
    """Run all validation tests."""
    clear_ssl_env()
    
    print("🔍 Coda Core System Validation")
    print("=" * 50)
    
    tests = [
        ("GPU Detection", test_gpu_detection),
        ("Core Imports", test_core_imports),
        ("Voice Components", test_voice_components),
        ("Memory System", test_memory_system),
        ("Performance Tools", test_performance_tools),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("🏁 Validation Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core systems validated successfully!")
        return 0
    else:
        print("⚠️ Some systems need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
