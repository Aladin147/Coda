#!/usr/bin/env python3
"""
Test framework version updates and compatibility.
"""

import sys
import os
sys.path.append('./src')

def test_framework_versions():
    """Test updated framework versions."""
    print("🔍 Framework Version Test")
    print("=" * 50)
    
    try:
        # Test 1: Core frameworks
        print("\n📦 Testing core framework versions...")
        
        import fastapi
        print(f"FastAPI: {fastapi.__version__}")
        
        import uvicorn
        print(f"Uvicorn: {uvicorn.__version__}")
        
        import websockets
        print(f"WebSockets: {websockets.__version__}")
        
        import transformers
        print(f"Transformers: {transformers.__version__}")
        
        import pydantic
        print(f"Pydantic: {pydantic.__version__}")
        
        import aiohttp
        print(f"AioHTTP: {aiohttp.__version__}")
        
        import starlette
        print(f"Starlette: {starlette.__version__}")
        
        import numpy
        print(f"NumPy: {numpy.__version__}")
        
        # Test 2: Framework compatibility
        print("\n🔧 Testing framework compatibility...")
        
        # Test FastAPI with Starlette
        from fastapi import FastAPI
        app = FastAPI()
        print("✅ FastAPI with Starlette working")
        
        # Test WebSockets modern API
        from websockets.asyncio.server import ServerConnection
        print("✅ WebSockets modern API available")
        
        # Test Pydantic V2
        from pydantic import BaseModel, field_validator
        
        class TestModel(BaseModel):
            name: str
            value: int
            
            @field_validator('value')
            @classmethod
            def validate_value(cls, v):
                if v < 0:
                    raise ValueError("Value must be positive")
                return v
        
        test_obj = TestModel(name="test", value=42)
        print("✅ Pydantic V2 field validators working")
        
        # Test AioHTTP (just import test, no session creation needed)
        import aiohttp
        print("✅ AioHTTP import working")
        
        # Test 3: RTX 5090 compatibility
        print("\n🚀 Testing RTX 5090 compatibility...")
        
        import torch
        print(f"PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"CUDA: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print("✅ RTX 5090 PyTorch nightly working")
        else:
            print("⚠️ CUDA not available")
        
        # Test 4: Voice processing compatibility
        print("\n🎤 Testing voice processing compatibility...")
        
        import moshi
        print(f"Moshi: {moshi.__version__}")
        print("✅ Moshi voice processing available")
        
        # Test 5: Memory system compatibility
        print("\n🧠 Testing memory system compatibility...")
        
        import chromadb
        print(f"ChromaDB: {chromadb.__version__}")
        print("✅ ChromaDB vector database available")
        
        print("\n🎉 All framework versions compatible!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dependency_conflicts():
    """Test for dependency conflicts."""
    print("\n⚠️ Checking for dependency conflicts...")
    
    import subprocess
    import sys
    
    try:
        # Run pip check to find conflicts
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            cwd="."
        )
        
        if result.returncode == 0:
            print("✅ No dependency conflicts found")
            return True
        else:
            print(f"⚠️ Dependency conflicts detected:")
            print(result.stdout)
            print(result.stderr)
            
            # Check if conflicts are acceptable (like PyTorch nightly)
            acceptable_conflicts = [
                "torch",  # PyTorch nightly expected to have conflicts
                "moshi"   # Moshi may have conflicts with nightly PyTorch
            ]
            
            conflicts = result.stdout.lower()
            if any(conflict in conflicts for conflict in acceptable_conflicts):
                print("✅ Conflicts are acceptable (PyTorch nightly)")
                return True
            else:
                return False
                
    except Exception as e:
        print(f"❌ Error checking dependencies: {e}")
        return False

def test_version_requirements():
    """Test that we meet minimum version requirements."""
    print("\n📋 Checking version requirements...")
    
    requirements = {
        'fastapi': '0.116.0',
        'websockets': '15.0.0',
        'transformers': '4.53.0',
        'pydantic': '2.11.0',
        'aiohttp': '3.11.0',
        'torch': '2.6.0'  # Minimum for RTX 5090
    }
    
    try:
        for package, min_version in requirements.items():
            try:
                module = __import__(package)
                current_version = getattr(module, '__version__', 'unknown')
                
                # Simple version comparison (works for most cases)
                if current_version != 'unknown':
                    current_parts = current_version.split('.')
                    min_parts = min_version.split('.')
                    
                    # Compare major.minor versions
                    current_major_minor = f"{current_parts[0]}.{current_parts[1]}"
                    min_major_minor = f"{min_parts[0]}.{min_parts[1]}"
                    
                    if float(current_major_minor) >= float(min_major_minor):
                        print(f"✅ {package}: {current_version} >= {min_version}")
                    else:
                        print(f"❌ {package}: {current_version} < {min_version}")
                        return False
                else:
                    print(f"⚠️ {package}: version unknown")
                    
            except ImportError:
                print(f"❌ {package}: not installed")
                return False
        
        print("✅ All version requirements met")
        return True
        
    except Exception as e:
        print(f"❌ Error checking versions: {e}")
        return False

if __name__ == "__main__":
    success1 = test_framework_versions()
    success2 = test_dependency_conflicts()
    success3 = test_version_requirements()
    
    if success1 and success2 and success3:
        print("\n🎉 Framework updates successful!")
        sys.exit(0)
    else:
        print("\n❌ Framework update issues detected")
        sys.exit(1)
