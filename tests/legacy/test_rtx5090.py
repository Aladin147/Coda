#!/usr/bin/env python3
"""
Test RTX 5090 compatibility with PyTorch CUDA 12.8 nightly.
"""

import torch
import sys

def test_rtx5090():
    """Test RTX 5090 GPU functionality."""
    print("🚀 RTX 5090 Compatibility Test")
    print("=" * 50)
    
    # Basic PyTorch info
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    # GPU information
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Check for SM_120 support
    compute_capability = torch.cuda.get_device_capability(0)
    if compute_capability >= (12, 0):
        print("✅ SM_120 compute capability detected!")
    else:
        print(f"⚠️ Compute capability {compute_capability} may not be optimal")
    
    try:
        print("\n🧪 Testing GPU Operations...")
        
        # Test basic tensor operations
        print("Testing basic tensor operations...")
        x = torch.randn(100, 100, device='cuda')
        y = torch.mm(x, x.t())
        print("✅ Basic GPU tensor operations successful")
        
        # Test larger operations
        print("Testing larger tensor operations...")
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.mm(x, x.t())
        print("✅ Large GPU tensor operations successful")
        
        # Memory usage
        memory_used = torch.cuda.memory_allocated(0) / 1024**3
        memory_cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"GPU Memory Used: {memory_used:.3f} GB")
        print(f"GPU Memory Cached: {memory_cached:.3f} GB")
        
        # Enable RTX 5090 optimizations
        print("\n⚡ Enabling RTX 5090 Optimizations...")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✅ TF32 optimizations enabled")
        
        # Test with optimizations
        print("Testing with TF32 optimizations...")
        x = torch.randn(2000, 2000, device='cuda', dtype=torch.float32)
        y = torch.mm(x, x.t())
        print("✅ TF32 optimized operations successful")
        
        # Clear memory
        torch.cuda.empty_cache()
        print("✅ GPU memory cleared")
        
        print("\n🎉 RTX 5090 fully functional with PyTorch!")
        return True
        
    except RuntimeError as e:
        if "CUDA capability sm_120 is not compatible" in str(e):
            print(f"❌ SM_120 compatibility issue: {e}")
            print("💡 This may require a newer PyTorch nightly build")
            return False
        else:
            print(f"❌ GPU operation failed: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_rtx5090()
    sys.exit(0 if success else 1)
