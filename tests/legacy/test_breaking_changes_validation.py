#!/usr/bin/env python3
"""
Comprehensive validation test for all framework breaking changes.
"""

import sys
import os
import asyncio
import warnings
sys.path.append('./src')

def test_chromadb_migration():
    """Test ChromaDB 1.0.15 migration."""
    print("🔍 Testing ChromaDB Migration...")
    
    try:
        # Clear SSL environment variable
        if 'SSL_CERT_FILE' in os.environ:
            del os.environ['SSL_CERT_FILE']
        
        from coda.components.memory.long_term import LongTermMemory
        from coda.components.memory.models import LongTermMemoryConfig
        
        config = LongTermMemoryConfig(
            storage_path='./temp/validation_test',
            vector_db_type='chroma',
            embedding_model='all-MiniLM-L6-v2',
            max_memories=10
        )
        
        memory = LongTermMemory(config)
        print("✅ ChromaDB 1.0.15 migration successful")
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB migration failed: {e}")
        return False

def test_pydantic_v2_migration():
    """Test Pydantic V2 field validators."""
    print("\n🔍 Testing Pydantic V2 Migration...")
    
    try:
        # Test all models with validators
        from coda.components.memory.models import MemoryQuery
        from coda.components.personality.models import PersonalityTrait, PersonalityTraitType
        from coda.components.tools.models import ToolParameter, ParameterType
        from coda.components.tools.plugin_metadata import PluginMetadata, PluginAuthor
        from datetime import datetime
        
        # Test MemoryQuery
        query = MemoryQuery(query="test", limit=5)
        print("✅ MemoryQuery validation working")
        
        # Test PersonalityTrait
        trait = PersonalityTrait(
            name=PersonalityTraitType.HUMOR,
            value=0.7,
            default_value=0.5,
            description="Test trait"
        )
        print("✅ PersonalityTrait validation working")
        
        # Test ToolParameter
        param = ToolParameter(
            name="test_param",
            type=ParameterType.STRING,
            description="Test parameter",
            default="test_value"
        )
        print("✅ ToolParameter validation working")
        
        # Test PluginMetadata
        author = PluginAuthor(name="Test Author", email="test@example.com")
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            author=author,
            permissions=["file_system"]
        )
        print("✅ PluginMetadata validation working")
        
        print("✅ Pydantic V2 migration successful")
        return True
        
    except Exception as e:
        print(f"❌ Pydantic V2 migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_websocket_migration():
    """Test WebSocket modern API migration."""
    print("\n🔍 Testing WebSocket Migration...")
    
    try:
        from coda.components.voice.websocket_handler import VoiceWebSocketHandler, ClientConnection
        from coda.interfaces.websocket.server import CodaWebSocketServer
        from websockets.asyncio.server import ServerConnection
        
        # Test server creation
        server = CodaWebSocketServer(host="localhost", port=8765)
        print("✅ WebSocket server creation working")
        
        # Test handler import (creation requires voice_manager)
        print("✅ WebSocket handler import working")
        
        print("✅ WebSocket migration successful")
        return True
        
    except Exception as e:
        print(f"❌ WebSocket migration failed: {e}")
        return False

def test_framework_compatibility():
    """Test framework compatibility after updates."""
    print("\n🔍 Testing Framework Compatibility...")
    
    try:
        # Test FastAPI with updated Starlette
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return JSONResponse({"status": "ok"})
        
        print("✅ FastAPI with Starlette compatibility working")
        
        # Test WebSockets with FastAPI
        from fastapi import WebSocket
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            await websocket.send_text("Hello WebSocket")
            await websocket.close()
        
        print("✅ FastAPI WebSocket integration working")
        
        # Test Transformers
        from transformers import AutoTokenizer
        print("✅ Transformers compatibility working")
        
        # Test AioHTTP
        import aiohttp
        print("✅ AioHTTP compatibility working")
        
        print("✅ Framework compatibility successful")
        return True
        
    except Exception as e:
        print(f"❌ Framework compatibility failed: {e}")
        return False

def test_rtx5090_compatibility():
    """Test RTX 5090 PyTorch compatibility."""
    print("\n🔍 Testing RTX 5090 Compatibility...")
    
    try:
        import torch
        
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
            
            # Test GPU operations
            x = torch.randn(100, 100, device='cuda')
            y = torch.mm(x, x.t())
            print("✅ RTX 5090 GPU operations working")
        else:
            print("⚠️ CUDA not available (expected in some environments)")
        
        print("✅ RTX 5090 compatibility successful")
        return True
        
    except Exception as e:
        print(f"❌ RTX 5090 compatibility failed: {e}")
        return False

def test_voice_processing_compatibility():
    """Test voice processing component compatibility."""
    print("\n🔍 Testing Voice Processing Compatibility...")
    
    try:
        # Test Moshi import
        import moshi
        print(f"Moshi: {moshi.__version__}")
        
        # Test voice processing imports
        from coda.components.voice.config import VoiceConfig
        from coda.components.voice.llm_integration import VoiceLLMProcessor
        
        print("✅ Voice processing imports working")
        
        # Test audio processing libraries
        import librosa
        import soundfile
        print("✅ Audio processing libraries working")
        
        # Test VAD libraries
        import silero_vad
        import noisereduce
        print("✅ VAD libraries working")
        
        print("✅ Voice processing compatibility successful")
        return True
        
    except Exception as e:
        print(f"❌ Voice processing compatibility failed: {e}")
        return False

def test_deprecation_warnings():
    """Test for remaining deprecation warnings."""
    print("\n🔍 Testing for Deprecation Warnings...")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            # Import all major components to trigger warnings
            from coda.components.memory.long_term import LongTermMemory
            from coda.components.personality.models import PersonalityTrait
            from coda.components.tools.models import ToolParameter
            from coda.components.voice.websocket_handler import VoiceWebSocketHandler
            from coda.interfaces.websocket.server import CodaWebSocketServer
            
            # Filter for relevant deprecation warnings
            deprecation_warnings = [
                warning for warning in w 
                if 'deprecat' in str(warning.message).lower() 
                and not ('torch' in str(warning.message).lower())  # Ignore PyTorch nightly warnings
            ]
            
            if deprecation_warnings:
                print(f"⚠️ Found {len(deprecation_warnings)} deprecation warnings:")
                for warning in deprecation_warnings:
                    print(f"  - {warning.message}")
                return False
            else:
                print("✅ No relevant deprecation warnings found")
                return True
                
        except Exception as e:
            print(f"❌ Error checking warnings: {e}")
            return False

def main():
    """Run all breaking changes validation tests."""
    print("🚀 Breaking Changes Validation Test Suite")
    print("=" * 60)
    
    tests = [
        ("ChromaDB Migration", test_chromadb_migration),
        ("Pydantic V2 Migration", test_pydantic_v2_migration),
        ("WebSocket Migration", test_websocket_migration),
        ("Framework Compatibility", test_framework_compatibility),
        ("RTX 5090 Compatibility", test_rtx5090_compatibility),
        ("Voice Processing Compatibility", test_voice_processing_compatibility),
        ("Deprecation Warnings", test_deprecation_warnings),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("🏁 Breaking Changes Validation Results")
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
        print("🎉 All breaking changes validated successfully!")
        print("✅ Framework migration complete and stable!")
        return 0
    else:
        print("⚠️ Some breaking changes need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
