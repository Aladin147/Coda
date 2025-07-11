#!/usr/bin/env python3
"""
Test component integration fixes.
"""

import sys
import os
import time
sys.path.append('./src')

def test_exception_imports():
    """Test that all exception classes can be imported correctly."""
    print("🔍 Testing Exception Imports...")
    
    try:
        from coda.components.voice.exceptions import (
            VoiceProcessingError, WebSocketError, ResourceExhaustionError,
            ComponentNotInitializedError, ComponentFailureError,
            ErrorCodes, create_error, wrap_exception
        )
        print("✅ All exception classes imported successfully")
        
        # Test exception creation
        error = WebSocketError("Test WebSocket error")
        assert isinstance(error, VoiceProcessingError)
        print("✅ WebSocketError creation working")
        
        resource_error = ResourceExhaustionError("Test resource error")
        assert isinstance(resource_error, VoiceProcessingError)
        print("✅ ResourceExhaustionError creation working")
        
        return True
        
    except ImportError as e:
        print(f"❌ Exception import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Exception test failed: {e}")
        return False

def test_voice_stream_chunk_model():
    """Test VoiceStreamChunk model with correct field names."""
    print("\n🔍 Testing VoiceStreamChunk Model...")
    
    try:
        from coda.components.voice.models import VoiceStreamChunk
        
        # Test with new field names
        chunk = VoiceStreamChunk(
            conversation_id="test-conv-123",
            chunk_index=0,
            text_content="Hello world",
            audio_data=b"fake_audio_data",
            timestamp=time.time(),
            is_complete=False,
            chunk_type="audio"
        )
        
        # Verify field names
        assert hasattr(chunk, 'chunk_index'), "chunk_index field missing"
        assert hasattr(chunk, 'text_content'), "text_content field missing"
        assert hasattr(chunk, 'is_complete'), "is_complete field missing"
        assert hasattr(chunk, 'timestamp'), "timestamp field missing"
        
        # Verify field types
        assert isinstance(chunk.chunk_index, int), "chunk_index should be int"
        assert isinstance(chunk.text_content, str), "text_content should be str"
        assert isinstance(chunk.is_complete, bool), "is_complete should be bool"
        assert isinstance(chunk.timestamp, float), "timestamp should be float"
        
        print("✅ VoiceStreamChunk model working with correct field names")
        
        # Test that old field names don't exist
        assert not hasattr(chunk, 'sequence_number'), "Old sequence_number field still exists"
        assert not hasattr(chunk, 'text_delta'), "Old text_delta field still exists"
        assert not hasattr(chunk, 'is_final'), "Old is_final field still exists"
        
        print("✅ Old field names properly removed")
        
        return True
        
    except ImportError as e:
        print(f"❌ VoiceStreamChunk import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ VoiceStreamChunk test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_component_imports():
    """Test that all components can be imported without errors."""
    print("\n🔍 Testing Component Imports...")
    
    try:
        # Test voice manager import
        from coda.components.voice.manager import VoiceManager
        print("✅ VoiceManager import successful")
        
        # Test moshi integration import
        from coda.components.voice.moshi_integration import MoshiVoiceProcessor, MoshiClient
        print("✅ Moshi integration imports successful")
        
        # Test WebSocket handler import
        from coda.components.voice.websocket_handler import VoiceWebSocketHandler
        print("✅ WebSocket handler import successful")
        
        # Test resource management import
        from coda.components.voice.resource_management import ResourcePool, with_timeout, with_retry
        print("✅ Resource management imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Component import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Component import test failed: {e}")
        return False

def test_error_handling_integration():
    """Test that error handling works correctly across components."""
    print("\n🔍 Testing Error Handling Integration...")
    
    try:
        from coda.components.voice.exceptions import (
            WebSocketError, ResourceExhaustionError, ErrorCodes, create_error
        )
        
        # Test error creation with codes
        websocket_error = create_error(
            WebSocketError,
            "WebSocket connection failed",
            ErrorCodes.NETWORK_CONNECTION_FAILED,
            host="localhost",
            port=8765
        )
        
        assert websocket_error.error_code == ErrorCodes.NETWORK_CONNECTION_FAILED
        assert "localhost" in str(websocket_error.context)
        print("✅ WebSocket error creation with context working")
        
        # Test resource exhaustion error
        resource_error = create_error(
            ResourceExhaustionError,
            "Memory pool exhausted",
            ErrorCodes.RESOURCE_EXHAUSTED,
            pool_size=10,
            requested=15
        )
        
        assert resource_error.error_code == ErrorCodes.RESOURCE_EXHAUSTED
        print("✅ Resource exhaustion error creation working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_model_field_consistency():
    """Test that model fields are consistent across the codebase."""
    print("\n🔍 Testing Model Field Consistency...")
    
    try:
        from coda.components.voice.models import VoiceStreamChunk
        
        # Create a chunk using the expected field names
        chunk = VoiceStreamChunk(
            conversation_id="test-conv",
            chunk_index=42,
            text_content="Test content",
            is_complete=True,
            timestamp=time.time()
        )
        
        # Test that the chunk can be serialized/deserialized
        chunk_dict = chunk.model_dump()
        
        expected_fields = {
            'conversation_id', 'chunk_index', 'text_content', 
            'is_complete', 'timestamp', 'audio_data', 'chunk_type',
            'processing_latency_ms', 'confidence_score'
        }
        
        actual_fields = set(chunk_dict.keys())
        
        if expected_fields <= actual_fields:  # Check if expected is subset of actual
            print("✅ All expected fields present in VoiceStreamChunk")
        else:
            missing = expected_fields - actual_fields
            print(f"❌ Missing fields in VoiceStreamChunk: {missing}")
            return False
        
        # Test reconstruction from dict
        reconstructed = VoiceStreamChunk(**chunk_dict)
        assert reconstructed.conversation_id == chunk.conversation_id
        assert reconstructed.chunk_index == chunk.chunk_index
        assert reconstructed.text_content == chunk.text_content
        assert reconstructed.is_complete == chunk.is_complete
        
        print("✅ VoiceStreamChunk serialization/deserialization working")
        
        return True
        
    except Exception as e:
        print(f"❌ Model field consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_compatibility():
    """Test that components can work together without field mismatches."""
    print("\n🔍 Testing Integration Compatibility...")
    
    try:
        from coda.components.voice.models import VoiceStreamChunk, VoiceMessage, VoiceResponse, VoiceProcessingMode
        from coda.components.voice.exceptions import VoiceProcessingError

        # Test that we can create all voice models
        message = VoiceMessage(
            message_id="test-msg-123",
            conversation_id="test-conv",
            text_content="Hello world",
            audio_data=b"fake_audio",
            processing_mode=VoiceProcessingMode.MOSHI_ONLY
        )
        print("✅ VoiceMessage creation working")
        
        response = VoiceResponse(
            response_id="test-resp-123",
            conversation_id="test-conv",
            message_id="test-msg-123",
            text_content="Hello back!",
            processing_mode=VoiceProcessingMode.MOSHI_ONLY,
            total_latency_ms=100.0
        )
        print("✅ VoiceResponse creation working")
        
        chunk = VoiceStreamChunk(
            conversation_id="test-conv",
            chunk_index=0,
            text_content="Hello",
            is_complete=False
        )
        print("✅ VoiceStreamChunk creation working")
        
        # Test that all models use consistent field naming
        assert hasattr(message, 'conversation_id')
        assert hasattr(response, 'conversation_id')
        assert hasattr(chunk, 'conversation_id')
        print("✅ Consistent conversation_id field across models")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration compatibility test failed: {e}")
        return False

def main():
    """Run all component integration tests."""
    print("🔧 Component Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Exception Imports", test_exception_imports),
        ("VoiceStreamChunk Model", test_voice_stream_chunk_model),
        ("Component Imports", test_component_imports),
        ("Error Handling Integration", test_error_handling_integration),
        ("Model Field Consistency", test_model_field_consistency),
        ("Integration Compatibility", test_integration_compatibility),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("🏁 Component Integration Test Results")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All component integration issues resolved!")
        print("✅ Components are properly integrated!")
        return 0
    else:
        print("⚠️ Some integration issues remain")
        return 1

if __name__ == "__main__":
    sys.exit(main())
