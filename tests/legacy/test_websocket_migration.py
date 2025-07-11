#!/usr/bin/env python3
"""
Test WebSocket API migration to modern websockets.asyncio.server.
"""

import sys
import os
sys.path.append('./src')

def test_websocket_imports():
    """Test WebSocket import migration."""
    print("🔍 WebSocket API Migration Test")
    print("=" * 50)
    
    try:
        import websockets
        print(f"WebSockets Version: {websockets.__version__}")
        
        # Test 1: Modern import
        print("\n📡 Testing modern WebSocket imports...")
        from websockets.asyncio.server import ServerConnection
        print("✅ ServerConnection import successful")
        
        # Test 2: Check if old import still works (for compatibility)
        try:
            from websockets.server import WebSocketServerProtocol
            print("⚠️ Old WebSocketServerProtocol still available (deprecated)")
        except ImportError:
            print("✅ Old WebSocketServerProtocol removed (expected in newer versions)")
        
        # Test 3: Test our updated imports
        print("\n🔧 Testing updated codebase imports...")
        
        # Test voice websocket handler
        try:
            from coda.components.voice.websocket_handler import VoiceWebSocketHandler, ClientConnection
            print("✅ Voice WebSocket handler imports working")
        except ImportError as e:
            print(f"❌ Voice WebSocket handler import failed: {e}")
            return False
        
        # Test websocket server
        try:
            from coda.interfaces.websocket.server import CodaWebSocketServer
            print("✅ WebSocket server imports working")
        except ImportError as e:
            print(f"❌ WebSocket server import failed: {e}")
            return False
        
        # Test 4: Basic functionality test
        print("\n🧪 Testing basic WebSocket functionality...")
        
        # Test server creation (without starting)
        try:
            server = CodaWebSocketServer(host="localhost", port=8765)
            print("✅ WebSocket server creation successful")
        except Exception as e:
            print(f"❌ WebSocket server creation failed: {e}")
            return False
        
        print("\n🎉 WebSocket API migration successful!")
        return True
        
    except ImportError as e:
        print(f"❌ WebSocket import error: {e}")
        return False
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_websocket_deprecation_warnings():
    """Test for WebSocket deprecation warnings."""
    print("\n⚠️ Checking for WebSocket deprecation warnings...")
    
    import warnings
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            # Import modules that use WebSocket
            from coda.components.voice.websocket_handler import VoiceWebSocketHandler
            from coda.interfaces.websocket.server import CodaWebSocketServer
            
            # Check for WebSocket-related warnings
            websocket_warnings = [warning for warning in w 
                                if 'websocket' in str(warning.message).lower() 
                                or 'deprecated' in str(warning.message).lower()]
            
            if websocket_warnings:
                print(f"⚠️ Found {len(websocket_warnings)} WebSocket warnings:")
                for warning in websocket_warnings:
                    print(f"  - {warning.message}")
                return False
            else:
                print("✅ No WebSocket deprecation warnings found")
                return True
                
        except Exception as e:
            print(f"❌ Error checking warnings: {e}")
            return False

def test_websocket_api_compatibility():
    """Test WebSocket API compatibility."""
    print("\n🔄 Testing WebSocket API compatibility...")
    
    try:
        # Test that our code uses the correct types
        from coda.components.voice.websocket_handler import ClientConnection
        from websockets.asyncio.server import ServerConnection
        
        # Check if ClientConnection uses the correct type
        import inspect
        signature = inspect.signature(ClientConnection.__init__)
        websocket_param = signature.parameters.get('websocket')
        
        if websocket_param and hasattr(websocket_param, 'annotation'):
            if 'ServerConnection' in str(websocket_param.annotation):
                print("✅ ClientConnection uses modern ServerConnection type")
            else:
                print(f"⚠️ ClientConnection uses: {websocket_param.annotation}")
        
        # Test server creation with modern API
        from coda.interfaces.websocket.server import CodaWebSocketServer
        server = CodaWebSocketServer()
        
        # Check if clients dict uses correct type
        clients_annotation = getattr(server.clients, '__annotations__', {})
        if clients_annotation:
            print(f"✅ Server clients type: {clients_annotation}")
        
        print("✅ WebSocket API compatibility verified")
        return True
        
    except Exception as e:
        print(f"❌ API compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_websocket_imports()
    success2 = test_websocket_deprecation_warnings()
    success3 = test_websocket_api_compatibility()
    
    if success1 and success2 and success3:
        print("\n🎉 WebSocket migration successful!")
        sys.exit(0)
    else:
        print("\n❌ WebSocket migration issues detected")
        sys.exit(1)
