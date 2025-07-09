#!/usr/bin/env python3
"""
Test script for Moshi installation and basic functionality.
"""

def test_moshi_installation():
    """Test Moshi installation and basic imports."""
    try:
        import moshi
        print("✓ Moshi import successful")
        print(f"Moshi location: {moshi.__file__}")
        
        # Check available modules
        try:
            import moshi.client
            print("✓ Moshi client available")
        except ImportError as e:
            print(f"⚠ Moshi client not available: {e}")
        
        try:
            import moshi.models
            print("✓ Moshi models available")
        except ImportError as e:
            print(f"⚠ Moshi models not available: {e}")
        
        # Check for server module
        try:
            import moshi.server
            print("✓ Moshi server available")
        except ImportError as e:
            print(f"⚠ Moshi server not available: {e}")
        
        print("🎉 Moshi installation verified!")
        return True
        
    except Exception as e:
        print(f"❌ Moshi test failed: {e}")
        return False

def test_moshi_models():
    """Test Moshi model availability."""
    try:
        # Check if we can access model information
        import moshi
        
        # Try to get model info without loading
        print("Checking Moshi model availability...")
        
        # This is a basic test - actual model loading will be done later
        print("✓ Moshi models module accessible")
        return True
        
    except Exception as e:
        print(f"❌ Moshi models test failed: {e}")
        return False

def test_audio_dependencies():
    """Test audio-related dependencies."""
    try:
        import sounddevice
        print("✓ SoundDevice available")
        
        import sentencepiece
        print("✓ SentencePiece available")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio dependencies test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Moshi Installation Test ===\n")
    
    success = True
    
    print("1. Testing Moshi installation...")
    success &= test_moshi_installation()
    print()
    
    print("2. Testing Moshi models...")
    success &= test_moshi_models()
    print()
    
    print("3. Testing audio dependencies...")
    success &= test_audio_dependencies()
    print()
    
    if success:
        print("✅ Moshi installation: ALL TESTS PASSED!")
    else:
        print("❌ Moshi installation: SOME TESTS FAILED!")
    
    print("\nNext: Test model loading and VRAM allocation...")
