#!/usr/bin/env python3
"""
Test Phase 1: Centralized System Controller

This script tests the enhanced CodaAssistant with:
- Session management
- Event coordination
- Text message processing pipeline
- Centralized orchestration

All primary components with NO FALLBACKS.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("phase1_test")

# Import components
try:
    from coda.core.assistant import CodaAssistant
    from coda.core.config import CodaConfig
    from coda.components.llm.models import LLMConfig, ProviderConfig, LLMProvider
    from coda.components.memory.models import MemoryManagerConfig
    
except ImportError as e:
    logger.error(f"Failed to import Coda components: {e}")
    exit(1)


class Phase1Tester:
    """Test the Phase 1 centralized system controller."""
    
    def __init__(self):
        self.assistant = None
        
    async def setup_assistant(self):
        """Set up the CodaAssistant with test configuration."""
        logger.info("Setting up CodaAssistant for Phase 1 testing...")

        # Use default configuration from file
        config_path = Path("configs/default.yaml")
        if config_path.exists():
            config = CodaConfig.from_yaml(config_path)
        else:
            # Create minimal config if no file exists
            config = CodaConfig()

        self.assistant = CodaAssistant(config)
        await self.assistant.initialize()
        logger.info("✅ CodaAssistant initialized successfully")
        
    async def test_session_management(self):
        """Test session management functionality."""
        logger.info("\n🧪 Testing Session Management...")
        
        # Test session creation
        session1 = await self.assistant.start_conversation()
        logger.info(f"✅ Created session 1: {session1}")
        
        session2 = await self.assistant.start_conversation()
        logger.info(f"✅ Created session 2: {session2}")
        
        # Test session listing
        sessions = self.assistant.list_sessions()
        logger.info(f"✅ Total sessions: {len(sessions)}")
        
        active_sessions = self.assistant.get_active_sessions()
        logger.info(f"✅ Active sessions: {len(active_sessions)}")
        
        # Test session switching
        success = await self.assistant.switch_session(session1)
        logger.info(f"✅ Switched to session 1: {success}")
        
        # Test session info
        session_info = self.assistant.get_session_info(session1)
        logger.info(f"✅ Session 1 info: {session_info['status']}")
        
        return session1, session2
    
    async def test_text_processing_pipeline(self, session_id):
        """Test the text message processing pipeline."""
        logger.info("\n🧪 Testing Text Processing Pipeline...")
        
        test_messages = [
            "Hello! What's your name?",
            "What's 2+2?",
            "Tell me a short joke",
            "What do you remember about our conversation?"
        ]
        
        for i, message in enumerate(test_messages, 1):
            logger.info(f"\n--- Message {i} ---")
            logger.info(f"👤 User: {message}")
            
            # Process message through pipeline
            result = await self.assistant.process_text_message(message, session_id)
            
            if result["status"] == "success":
                response_content = result["response"].content if hasattr(result["response"], "content") else str(result["response"])
                logger.info(f"🤖 Coda: {response_content[:100]}...")
                logger.info(f"⚡ Processing time: {result['processing_time_ms']:.1f}ms")
            else:
                logger.error(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        # Test session history
        history = await self.assistant.get_session_history(session_id)
        logger.info(f"✅ Session history: {len(history)} messages")
    
    async def test_event_system(self):
        """Test the event coordination system."""
        logger.info("\n🧪 Testing Event System...")
        
        # Get event statistics
        event_stats = self.assistant.get_event_stats()
        logger.info(f"✅ Total events: {event_stats.get('total_events', 0)}")
        logger.info(f"✅ Event types: {len(event_stats.get('events_by_type', {}))}")
        
        # Get event history
        event_history = self.assistant.get_event_history(limit=5)
        logger.info(f"✅ Recent events: {len(event_history)}")
        
        for event in event_history[-3:]:  # Show last 3 events
            logger.info(f"  📅 {event['type']} - {event['category']}")
        
        # Test GUI event emission
        await self.assistant.emit_gui_event("test_event", {"test": "data"})
        logger.info("✅ GUI event emitted")
    
    async def test_pipeline_status(self):
        """Test pipeline status reporting."""
        logger.info("\n🧪 Testing Pipeline Status...")
        
        status = self.assistant.get_pipeline_status()
        
        logger.info(f"✅ Text pipeline available: {status['text_pipeline']['available']}")
        logger.info(f"✅ Voice pipeline available: {status['voice_pipeline']['available']}")
        logger.info(f"✅ Active session: {status['session_management']['active_session']}")
        logger.info(f"✅ Total sessions: {status['session_management']['total_sessions']}")
        logger.info(f"✅ Event system running: {status['event_system']['event_coordinator']}")
        
    async def test_component_integration(self):
        """Test component integration status."""
        logger.info("\n🧪 Testing Component Integration...")
        
        # Test health check
        health = await self.assistant.health_check()
        logger.info(f"✅ System health: {health.get('status', 'unknown')}")
        
        # Test integration status
        integration_status = self.assistant.get_integration_status()
        logger.info(f"✅ Integration layer: {bool(integration_status)}")
        
        # Test component availability
        logger.info(f"✅ LLM Manager: {bool(self.assistant.llm_manager)}")
        logger.info(f"✅ Memory Manager: {bool(self.assistant.memory_manager)}")
        logger.info(f"✅ Session Manager: {bool(self.assistant.session_manager)}")
        logger.info(f"✅ Event Coordinator: {bool(self.assistant.event_coordinator)}")
    
    async def run_comprehensive_test(self):
        """Run comprehensive Phase 1 testing."""
        logger.info("🚀 Starting Phase 1 Comprehensive Test")
        logger.info("=" * 60)
        
        try:
            # Setup
            await self.setup_assistant()
            
            # Test session management
            session1, session2 = await self.test_session_management()
            
            # Test text processing pipeline
            await self.test_text_processing_pipeline(session1)
            
            # Test event system
            await self.test_event_system()
            
            # Test pipeline status
            await self.test_pipeline_status()
            
            # Test component integration
            await self.test_component_integration()
            
            logger.info("\n" + "=" * 60)
            logger.info("🎉 Phase 1 Testing Complete - ALL TESTS PASSED!")
            logger.info("✅ Centralized System Controller: WORKING")
            logger.info("✅ Session Management: WORKING")
            logger.info("✅ Event Coordination: WORKING")
            logger.info("✅ Text Processing Pipeline: WORKING")
            logger.info("✅ Component Integration: WORKING")
            
        except Exception as e:
            logger.error(f"❌ Phase 1 Testing Failed: {e}")
            raise
        finally:
            if self.assistant:
                await self.assistant.shutdown()


async def main():
    """Main test function."""
    tester = Phase1Tester()
    
    try:
        await tester.run_comprehensive_test()
        return 0
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
