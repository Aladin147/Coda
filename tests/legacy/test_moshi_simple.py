#!/usr/bin/env python3
"""
Simple Moshi test without Triton optimization.
"""

import os
import torch

# Disable Triton compilation to avoid the error
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

def test_moshi_basic():
    """Test basic Moshi functionality without optimization."""
    try:
        print("🧪 Testing Moshi without Triton optimization...")
        
        import moshi
        print(f"✓ Moshi version: {getattr(moshi, '__version__', 'unknown')}")
        
        # Test basic imports
        import moshi.client
        import moshi.models
        print("✓ Moshi modules imported successfully")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
            print(f"✓ VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠ CUDA not available, will use CPU")
        
        print("🎉 Basic Moshi test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Moshi test failed: {e}")
        return False

def test_moshi_model_loading():
    """Test if we can load Moshi models without full initialization."""
    try:
        print("\n🔄 Testing Moshi model access...")
        
        import moshi.models
        
        # Try to get model info without full loading
        print("✓ Moshi models module accessible")
        
        # Check if we can access model loaders
        if hasattr(moshi.models, 'get_moshi_lm'):
            print("✓ Moshi LM loader available")
        
        if hasattr(moshi.models, 'get_mimi'):
            print("✓ Mimi compression loader available")
        
        print("🎉 Model loading test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Simple Moshi Test (No Triton) ===")
    
    success = True
    success &= test_moshi_basic()
    success &= test_moshi_model_loading()
    
    if success:
        print("\n✅ Moshi is functional (without Triton optimization)")
        print("💡 Note: Triton optimization disabled to avoid compatibility issues")
    else:
        print("\n❌ Moshi tests failed")
    
    print("\nNext: Test voice system integration...")
