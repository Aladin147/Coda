#!/usr/bin/env python3
"""
System validation script to verify all components are working correctly.

This script performs basic validation of all voice processing components
to ensure they can be imported and initialized without errors.
"""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all components can be imported."""
    print("🔍 Testing imports...")
    
    try:
        # Core models
        from coda.components.voice.models import (
            VoiceConfig, AudioConfig, MoshiConfig, 
            VoiceMessage, VoiceResponse, VoiceStreamChunk,
            VoiceProcessingMode, ConversationState
        )
        print("✅ Core models imported successfully")
        
        # Exceptions
        from coda.components.voice.exceptions import (
            VoiceProcessingError, ValidationError, ComponentFailureError,
            WebSocketError, ResourceExhaustionError, ComponentNotInitializedError
        )
        print("✅ Exception classes imported successfully")
        
        # Voice manager
        from coda.components.voice.manager import VoiceManager
        print("✅ Voice manager imported successfully")
        
        # WebSocket components
        from coda.components.voice.websocket_handler import VoiceWebSocketHandler
        from coda.components.voice.websocket_events import VoiceEventBroadcaster
        from coda.components.voice.websocket_audio_streaming import AudioStreamProcessor
        from coda.components.voice.websocket_monitoring import WebSocketMonitor
        from coda.components.voice.websocket_server import VoiceWebSocketServer
        print("✅ WebSocket components imported successfully")
        
        # Performance components
        from coda.components.voice.audio_buffer_pool import AudioBufferPool
        from coda.components.voice.optimized_cache import OptimizedLRUCache
        from coda.components.voice.optimized_vram_manager import OptimizedVRAMManager
        from coda.components.voice.performance_profiler import PerformanceProfiler
        print("✅ Performance components imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_model_creation():
    """Test that models can be created and validated."""
    print("\n🔍 Testing model creation...")
    
    try:
        from coda.components.voice.models import (
            VoiceConfig, AudioConfig, MoshiConfig,
            VoiceMessage, VoiceResponse, VoiceStreamChunk
        )
        
        # Test configuration creation
        audio_config = AudioConfig(
            sample_rate=16000,
            channels=1,
            format="wav"
        )
        
        moshi_config = MoshiConfig(
            device="cpu",
            vram_allocation="1GB"
        )
        
        voice_config = VoiceConfig(
            audio=audio_config,
            moshi=moshi_config
        )
        print("✅ Configuration models created successfully")
        
        # Test message creation
        from coda.components.voice.models import VoiceProcessingMode
        voice_message = VoiceMessage(
            message_id="test-msg-123",
            conversation_id="test-conv",
            audio_data=b"test-audio",
            text_content="Hello world",
            processing_mode=VoiceProcessingMode.MOSHI_ONLY
        )
        
        voice_response = VoiceResponse(
            response_id="test-resp-123",
            conversation_id="test-conv",
            message_id="test-msg-123",
            text_content="Hello back!",
            processing_mode=VoiceProcessingMode.MOSHI_ONLY,
            total_latency_ms=100.0
        )
        
        stream_chunk = VoiceStreamChunk(
            conversation_id="test-conv",
            chunk_index=0,
            text_content="Hello",
            is_complete=False
        )
        print("✅ Message models created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        traceback.print_exc()
        return False


def test_component_initialization():
    """Test that components can be initialized."""
    print("\n🔍 Testing component initialization...")
    
    try:
        from coda.components.voice.models import VoiceConfig, AudioConfig, MoshiConfig
        from coda.components.voice.manager import VoiceManager
        from coda.components.voice.audio_buffer_pool import AudioBufferPool
        from coda.components.voice.optimized_cache import OptimizedLRUCache
        from coda.components.voice.optimized_vram_manager import OptimizedVRAMManager
        from coda.components.voice.performance_profiler import PerformanceProfiler
        
        # Test voice manager initialization
        voice_config = VoiceConfig(
            audio=AudioConfig(),
            moshi=MoshiConfig(device="cpu")
        )
        voice_manager = VoiceManager(voice_config)
        print("✅ Voice manager initialized successfully")
        
        # Test performance components
        buffer_pool = AudioBufferPool(max_buffers=10)
        cache = OptimizedLRUCache(max_size=100)
        vram_manager = OptimizedVRAMManager(total_vram_gb=1.0)
        profiler = PerformanceProfiler()
        print("✅ Performance components initialized successfully")
        
        # Cleanup
        buffer_pool.clear_pool()
        cache.clear()
        vram_manager.cleanup()
        profiler.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        traceback.print_exc()
        return False


def test_websocket_components():
    """Test WebSocket component initialization."""
    print("\n🔍 Testing WebSocket components...")
    
    try:
        from unittest.mock import Mock
        from coda.components.voice.websocket_handler import VoiceWebSocketHandler
        from coda.components.voice.websocket_events import VoiceEventBroadcaster
        from coda.components.voice.websocket_audio_streaming import AudioStreamProcessor
        from coda.components.voice.websocket_monitoring import WebSocketMonitor
        from coda.components.voice.websocket_server import VoiceWebSocketServer
        
        # Mock voice manager for testing
        mock_voice_manager = Mock()
        
        # Test WebSocket handler
        handler = VoiceWebSocketHandler(
            voice_manager=mock_voice_manager,
            host="localhost",
            port=0,  # Random port
            auth_required=False
        )
        print("✅ WebSocket handler initialized successfully")
        
        # Test event broadcaster
        broadcaster = VoiceEventBroadcaster(handler)
        print("✅ Event broadcaster initialized successfully")
        
        # Test audio processor
        audio_processor = AudioStreamProcessor(handler, mock_voice_manager)
        print("✅ Audio stream processor initialized successfully")
        
        # Test monitor
        monitor = WebSocketMonitor(handler, broadcaster)
        print("✅ WebSocket monitor initialized successfully")
        
        # Test server
        server = VoiceWebSocketServer(
            host="localhost",
            port=0,
            auth_required=False
        )
        print("✅ WebSocket server initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ WebSocket component initialization failed: {e}")
        traceback.print_exc()
        return False


def test_exception_handling():
    """Test exception handling."""
    print("\n🔍 Testing exception handling...")
    
    try:
        from coda.components.voice.exceptions import (
            VoiceProcessingError, ValidationError, WebSocketError,
            ErrorCodes, wrap_exception
        )
        
        # Test exception creation
        error = VoiceProcessingError(
            "Test error",
            error_code=ErrorCodes.PROCESSING_FAILED,
            context={"test": "data"}
        )
        assert error.message == "Test error"
        assert error.error_code == ErrorCodes.PROCESSING_FAILED
        print("✅ Exception creation works correctly")
        
        # Test exception wrapping
        original_error = ValueError("Original error")
        wrapped = wrap_exception(
            original_error,
            ValidationError,
            "Wrapped error",
            ErrorCodes.CONFIG_INVALID
        )
        assert isinstance(wrapped, ValidationError)
        print("✅ Exception wrapping works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Exception handling test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("🚀 Starting system validation...\n")
    
    tests = [
        test_imports,
        test_model_creation,
        test_component_initialization,
        test_websocket_components,
        test_exception_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"\n❌ Test {test.__name__} failed!")
    
    print(f"\n📊 Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation tests passed! System is ready for testing.")
        return 0
    else:
        print("⚠️  Some validation tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
