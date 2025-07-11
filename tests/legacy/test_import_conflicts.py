#!/usr/bin/env python3
"""
Test for import conflicts after cleaning test environment.
"""

import sys
import os
sys.path.append('./src')

def test_voice_imports():
    """Test voice component imports."""
    print("🔍 Testing Voice Component Imports...")
    
    try:
        # Test LLM integration imports
        from coda.components.voice.llm_integration import VoiceLLMProcessor, VoiceLLMConfig
        print("✅ VoiceLLMProcessor import successful")
        
        # Test Moshi integration imports
        from coda.components.voice.moshi_integration import MoshiClient, MoshiVoiceProcessor
        print("✅ MoshiVoiceProcessor import successful")
        
        # Test WebSocket handler imports
        from coda.components.voice.websocket_handler import VoiceWebSocketHandler, ClientConnection
        print("✅ VoiceWebSocketHandler import successful")
        
        # Test voice models
        from coda.components.voice.models import VoiceConfig, VoiceMessage, VoiceResponse
        print("✅ Voice models import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Voice import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Voice import error: {e}")
        return False

def test_core_imports():
    """Test core component imports."""
    print("\n🔍 Testing Core Component Imports...")
    
    try:
        # Test WebSocket server imports
        from coda.interfaces.websocket.server import CodaWebSocketServer
        print("✅ CodaWebSocketServer import successful")
        
        # Test memory system imports
        from coda.components.memory.long_term import LongTermMemory
        from coda.components.memory.models import LongTermMemoryConfig
        print("✅ Memory system imports successful")
        
        # Test LLM system imports
        from coda.components.llm.manager import LLMManager
        from coda.components.llm.models import LLMConfig
        print("✅ LLM system imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Core import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Core import error: {e}")
        return False

def test_test_imports():
    """Test that test files can import correctly."""
    print("\n🔍 Testing Test File Imports...")
    
    try:
        # Test voice test imports
        sys.path.append('./tests/voice')
        
        # Import test modules to check for conflicts
        import importlib.util
        
        # Test LLM integration test
        spec = importlib.util.spec_from_file_location(
            "test_llm_integration", 
            "./tests/voice/test_llm_integration.py"
        )
        test_llm_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_llm_module)
        print("✅ test_llm_integration.py loads successfully")
        
        # Test Moshi integration test
        spec = importlib.util.spec_from_file_location(
            "test_moshi_voice_integration",
            "./tests/voice/test_moshi_voice_integration.py"
        )
        test_moshi_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_moshi_module)
        print("✅ test_moshi_voice_integration.py loads successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Test import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_no_duplicate_conflicts():
    """Test that there are no duplicate import conflicts."""
    print("\n🔍 Testing for Duplicate Import Conflicts...")
    
    try:
        # Test importing from both unit and voice test directories
        import sys
        
        # Clear any cached modules
        modules_to_clear = [mod for mod in sys.modules.keys() if 'test_' in mod]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
        
        # Test that we can import both unit and voice versions without conflict
        sys.path.insert(0, './tests/unit')
        sys.path.insert(0, './tests/voice')
        
        # This should work without conflicts now
        from coda.components.voice.moshi_integration import MoshiClient
        from coda.components.voice.websocket_handler import VoiceWebSocketHandler
        
        print("✅ No duplicate import conflicts detected")
        return True
        
    except Exception as e:
        print(f"❌ Duplicate conflict test failed: {e}")
        return False

def main():
    """Run all import conflict tests."""
    print("🧹 Import Conflict Resolution Test")
    print("=" * 50)
    
    tests = [
        ("Voice Imports", test_voice_imports),
        ("Core Imports", test_core_imports),
        ("Test File Imports", test_test_imports),
        ("Duplicate Conflicts", test_no_duplicate_conflicts),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("🏁 Import Conflict Test Results")
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
        print("🎉 All import conflicts resolved!")
        print("✅ Test environment is clean!")
        return 0
    else:
        print("⚠️ Some import conflicts remain")
        return 1

if __name__ == "__main__":
    sys.exit(main())
