#!/usr/bin/env python3
"""
Simple test script for Phase 1 components.
"""

import sys
sys.path.append('.')

def test_imports():
    """Test that all Phase 1 components can be imported."""
    try:
        # Test voice models
        from src.coda.components.voice.models import VoiceConfig, AudioConfig
        print("✓ Voice models imported")

        # Test configuration
        from src.coda.components.voice.config import ConfigurationTemplate
        config = ConfigurationTemplate.development()
        print(f"✓ Configuration created: {config.mode}")

        # Test audio processor
        from src.coda.components.voice.audio_processor import AudioProcessor
        print("✓ AudioProcessor imported")

        # Test pipeline
        from src.coda.components.voice.pipeline import AudioPipeline
        print("✓ AudioPipeline imported")

        # Test VRAM manager
        from src.coda.components.voice.vram_manager import DynamicVRAMManager
        print("✓ VRAMManager imported")

        # Test utilities
        from src.coda.components.voice.utils import PerformanceMonitor, LatencyTracker
        print("✓ Utilities imported")

        print("🎉 All Phase 1 components imported successfully!")
        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of Phase 1 components."""
    try:
        from src.coda.components.voice.config import ConfigurationTemplate
        from src.coda.components.voice.utils import LatencyTracker, AudioUtils
        
        # Test configuration
        config = ConfigurationTemplate.development()
        assert config.mode.value == "moshi_only"
        print("✓ Configuration validation passed")
        
        # Test latency tracker
        tracker = LatencyTracker("test")
        tracker.start()
        import time
        time.sleep(0.01)
        latency = tracker.stop()
        assert latency > 0
        print(f"✓ Latency tracker working: {latency:.1f}ms")
        
        # Test audio utils
        assert AudioUtils.validate_audio_format("wav")
        assert not AudioUtils.validate_audio_format("invalid")
        print("✓ Audio utilities working")
        
        print("🎉 Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def test_pytorch_integration():
    """Test PyTorch integration."""
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
        else:
            print("⚠ CUDA not available (CPU mode)")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Phase 1 Core Infrastructure Test ===\n")
    
    success = True
    
    print("1. Testing imports...")
    success &= test_imports()
    print()
    
    print("2. Testing basic functionality...")
    success &= test_basic_functionality()
    print()
    
    print("3. Testing PyTorch integration...")
    success &= test_pytorch_integration()
    print()
    
    if success:
        print("✅ Phase 1 Core Infrastructure: ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("❌ Phase 1 Core Infrastructure: SOME TESTS FAILED!")
        sys.exit(1)
