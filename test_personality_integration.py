#!/usr/bin/env python3
"""
Test Voice-Personality Integration

This script tests the integration between voice processing and personality systems.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from src.coda.components.voice.personality_integration import (
        VoicePersonalityIntegration, VoicePersonalityConfig, VoicePersonalityManager
    )
    from src.coda.components.voice.models import VoiceMessage, VoiceResponse, VoiceProcessingMode
    logger.info("✓ Successfully imported voice-personality integration components")
except ImportError as e:
    logger.error(f"❌ Failed to import components: {e}")
    exit(1)


class MockPersonalityManager:
    """Mock personality manager for testing."""
    
    def __init__(self):
        self.trait_adjustments = []
        
    async def adjust_trait(self, trait_name: str, adjustment: float, source: str):
        """Adjust a personality trait."""
        self.trait_adjustments.append({
            "trait": trait_name,
            "adjustment": adjustment,
            "source": source,
            "timestamp": datetime.now()
        })


def create_test_voice_message(conversation_id: str, text_content: str) -> VoiceMessage:
    """Create a test voice message."""
    audio_data = np.random.randint(-32768, 32767, 24000, dtype=np.int16).tobytes()
    
    return VoiceMessage(
        message_id=f"test_msg_{datetime.now().strftime('%H%M%S%f')}",
        conversation_id=conversation_id,
        audio_data=audio_data,
        text_content=text_content,
        processing_mode=VoiceProcessingMode.HYBRID,
        timestamp=datetime.now()
    )


def create_test_voice_response(message_id: str, conversation_id: str, text_content: str) -> VoiceResponse:
    """Create a test voice response."""
    return VoiceResponse(
        response_id=f"response_{message_id}",
        conversation_id=conversation_id,
        message_id=message_id,
        text_content=text_content,
        audio_data=b"",
        processing_mode=VoiceProcessingMode.HYBRID,
        total_latency_ms=150.0,
        response_relevance=0.8
    )


async def test_voice_personality_config():
    """Test voice personality configuration."""
    logger.info("=== Testing Voice Personality Configuration ===")
    
    try:
        config = VoicePersonalityConfig()
        
        logger.info("✓ Default configuration created")
        logger.info(f"  Personality injection enabled: {config.enable_personality_injection}")
        logger.info(f"  Response adaptation enabled: {config.enable_response_adaptation}")
        logger.info(f"  Personality learning enabled: {config.enable_personality_learning}")
        logger.info(f"  Voice traits enabled: {config.enable_voice_traits}")
        logger.info(f"  Adjustment sensitivity: {config.adjustment_sensitivity}")
        
        # Test custom configuration
        custom_config = VoicePersonalityConfig(
            enable_personality_injection=False,
            adjustment_sensitivity=0.2,
            voice_confidence_factor=1.5
        )
        
        logger.info("✓ Custom configuration created")
        logger.info(f"  Personality injection enabled: {custom_config.enable_personality_injection}")
        logger.info(f"  Voice confidence factor: {custom_config.voice_confidence_factor}")
        
        logger.info("✓ Voice personality configuration test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


async def test_personality_context_enhancement():
    """Test personality context enhancement."""
    logger.info("=== Testing Personality Context Enhancement ===")
    
    try:
        personality_manager = MockPersonalityManager()
        config = VoicePersonalityConfig()
        integration = VoicePersonalityIntegration(personality_manager, config)
        
        logger.info("✓ Voice personality integration created")
        
        # Test personality context enhancement
        voice_message = create_test_voice_message(
            "test_conversation",
            "Can you help me understand machine learning concepts?"
        )
        
        personality_context = await integration.enhance_voice_context(voice_message)
        
        logger.info("✓ Personality context enhanced")
        logger.info(f"  Context keys: {list(personality_context.keys())}")
        
        # Check personality traits
        traits = personality_context.get("traits", {})
        logger.info(f"  Personality traits: {list(traits.keys())}")
        logger.info(f"  Helpfulness: {traits.get('helpfulness', 'N/A')}")
        logger.info(f"  Enthusiasm: {traits.get('enthusiasm', 'N/A')}")
        logger.info(f"  Technical expertise: {traits.get('technical_expertise', 'N/A')}")
        
        # Check speaking style
        speaking_style = personality_context.get("speaking_style", {})
        logger.info(f"  Speaking style: {list(speaking_style.keys())}")
        
        # Check personality description
        description = personality_context.get("description", "")
        logger.info(f"  Personality description: {description}")
        
        await integration.cleanup()
        logger.info("✓ Personality context enhancement test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Context enhancement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_response_adaptation():
    """Test personality-based response adaptation."""
    logger.info("=== Testing Response Adaptation ===")
    
    try:
        personality_manager = MockPersonalityManager()
        config = VoicePersonalityConfig()
        integration = VoicePersonalityIntegration(personality_manager, config)
        
        # Test different response adaptations
        test_cases = [
            {
                "original": "I would recommend that you consider using machine learning algorithms.",
                "description": "formal response"
            },
            {
                "original": "That's a great question. Let me explain how neural networks work.",
                "description": "enthusiastic response"
            },
            {
                "original": "The implementation might be challenging but it's possible.",
                "description": "confident response"
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            voice_message = create_test_voice_message(
                "test_conversation",
                f"Test question {i+1}"
            )
            
            voice_response = create_test_voice_response(
                voice_message.message_id,
                voice_message.conversation_id,
                test_case["original"]
            )
            
            adapted_response = await integration.adapt_voice_response(
                voice_response, voice_message
            )
            
            logger.info(f"✓ Response adaptation {i+1} ({test_case['description']}):")
            logger.info(f"  Original: {test_case['original']}")
            logger.info(f"  Adapted: {adapted_response.text_content}")
        
        await integration.cleanup()
        logger.info("✓ Response adaptation test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Response adaptation test failed: {e}")
        return False


async def test_personality_learning():
    """Test personality learning from voice interactions."""
    logger.info("=== Testing Personality Learning ===")
    
    try:
        personality_manager = MockPersonalityManager()
        config = VoicePersonalityConfig(adjustment_sensitivity=0.2)
        integration = VoicePersonalityIntegration(personality_manager, config)
        
        # Test learning from user feedback
        voice_message = create_test_voice_message(
            "test_conversation",
            "Can you be less formal in your responses?"
        )
        
        voice_response = create_test_voice_response(
            voice_message.message_id,
            voice_message.conversation_id,
            "I would be happy to assist you with that request."
        )
        
        # Simulate user feedback
        user_feedback = {
            "type": "too_formal",
            "score": 0.3,
            "text": "Please be more casual"
        }
        
        await integration.learn_from_voice_interaction(
            voice_message, voice_response, user_feedback=user_feedback
        )
        
        logger.info("✓ Learning from user feedback completed")
        
        # Test learning from conversation flow
        for i in range(5):
            msg = create_test_voice_message(
                "test_conversation",
                f"Test message {i+1} for conversation flow learning"
            )
            
            resp = create_test_voice_response(
                msg.message_id,
                msg.conversation_id,
                f"Response {i+1} with varying relevance"
            )
            
            # Vary response relevance to trigger learning
            resp.response_relevance = 0.9 - (i * 0.1)
            
            await integration.learn_from_voice_interaction(msg, resp)
        
        # Check statistics
        stats = integration.get_integration_stats()
        logger.info(f"✓ Learning statistics:")
        logger.info(f"  Voice interactions: {stats['voice_personality_stats']['voice_interactions']}")
        logger.info(f"  Personality adjustments: {stats['voice_personality_stats']['personality_adjustments']}")
        logger.info(f"  Active conversations: {stats['active_conversations']}")
        
        await integration.cleanup()
        logger.info("✓ Personality learning test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Learning test failed: {e}")
        return False


async def test_voice_trait_adjustments():
    """Test voice-specific trait adjustments."""
    logger.info("=== Testing Voice Trait Adjustments ===")
    
    try:
        personality_manager = MockPersonalityManager()
        config = VoicePersonalityConfig(
            enable_voice_traits=True,
            voice_confidence_factor=1.3,
            voice_engagement_factor=1.2
        )
        integration = VoicePersonalityIntegration(personality_manager, config)
        
        # Test voice trait adjustments
        voice_message = create_test_voice_message(
            "test_conversation",
            "This is a voice interaction test"
        )
        
        # High relevance response should boost enthusiasm
        voice_response = create_test_voice_response(
            voice_message.message_id,
            voice_message.conversation_id,
            "Great! I'm excited to help you with this voice interaction."
        )
        voice_response.response_relevance = 0.9
        
        await integration.learn_from_voice_interaction(voice_message, voice_response)
        
        # Check voice adaptations in context
        personality_context = await integration.enhance_voice_context(voice_message)
        voice_adaptations = personality_context.get("voice_adaptations", {})
        
        logger.info(f"✓ Voice trait adjustments:")
        logger.info(f"  Voice confidence boost: {voice_adaptations.get('voice_confidence_boost', 'N/A')}")
        logger.info(f"  Voice engagement boost: {voice_adaptations.get('voice_engagement_boost', 'N/A')}")
        logger.info(f"  Prefers voice interaction: {voice_adaptations.get('prefers_voice_interaction', 'N/A')}")
        
        await integration.cleanup()
        logger.info("✓ Voice trait adjustments test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Voice trait adjustments test failed: {e}")
        return False


async def test_personality_caching():
    """Test personality context caching."""
    logger.info("=== Testing Personality Caching ===")
    
    try:
        personality_manager = MockPersonalityManager()
        config = VoicePersonalityConfig(personality_cache_ttl_minutes=1)
        integration = VoicePersonalityIntegration(personality_manager, config)
        
        # Test caching with same conversation
        voice_message = create_test_voice_message(
            "test_conversation",
            "Test message for caching"
        )
        
        # First call (should cache)
        context1 = await integration.enhance_voice_context(voice_message)
        
        # Second call (should use cache)
        context2 = await integration.enhance_voice_context(voice_message)
        
        # Check cache statistics
        stats = integration.get_integration_stats()
        cache_hits = stats['voice_personality_stats']['cache_hits']
        
        logger.info(f"✓ Personality caching test:")
        logger.info(f"  Cache hits: {cache_hits}")
        logger.info(f"  Cache size: {stats['cache_size']}")
        
        # Verify contexts are similar (from cache)
        assert context1.get('description') == context2.get('description')
        logger.info("✓ Cached context matches original")
        
        await integration.cleanup()
        logger.info("✓ Personality caching test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Caching test failed: {e}")
        return False


async def test_voice_personality_manager():
    """Test high-level voice personality manager."""
    logger.info("=== Testing Voice Personality Manager ===")
    
    try:
        personality_manager = MockPersonalityManager()
        config = VoicePersonalityConfig()
        voice_personality_manager = VoicePersonalityManager(personality_manager, config)
        
        logger.info("✓ Voice personality manager created")
        
        # Test context enhancement
        voice_message = create_test_voice_message(
            "test_conversation",
            "How can I improve my programming skills?"
        )
        
        enhanced_context = await voice_personality_manager.enhance_voice_context(voice_message)
        
        logger.info("✓ Voice context enhanced with personality")
        logger.info(f"  Context keys: {list(enhanced_context.keys())}")
        
        # Test response adaptation
        voice_response = create_test_voice_response(
            voice_message.message_id,
            voice_message.conversation_id,
            "I would suggest that you practice coding regularly and work on projects."
        )
        
        adapted_response = await voice_personality_manager.adapt_voice_response(
            voice_response, voice_message
        )
        
        logger.info("✓ Voice response adapted with personality")
        logger.info(f"  Original: {voice_response.text_content}")
        logger.info(f"  Adapted: {adapted_response.text_content}")
        
        # Test learning from interaction
        await voice_personality_manager.learn_from_interaction(voice_message, adapted_response)
        
        logger.info("✓ Learning from interaction completed")
        
        # Test statistics
        stats = voice_personality_manager.get_personality_stats()
        logger.info(f"✓ Personality statistics:")
        logger.info(f"  Personality injections: {stats['voice_personality_stats']['personality_injections']}")
        logger.info(f"  Response adaptations: {stats['voice_personality_stats']['response_adaptations']}")
        
        await voice_personality_manager.cleanup()
        logger.info("✓ Voice personality manager test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Personality manager test failed: {e}")
        return False


async def main():
    """Run all voice-personality integration tests."""
    logger.info("🚀 Starting Voice-Personality Integration Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Voice Personality Configuration", test_voice_personality_config),
        ("Personality Context Enhancement", test_personality_context_enhancement),
        ("Response Adaptation", test_response_adaptation),
        ("Personality Learning", test_personality_learning),
        ("Voice Trait Adjustments", test_voice_trait_adjustments),
        ("Personality Caching", test_personality_caching),
        ("Voice Personality Manager", test_voice_personality_manager),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = "✅ PASSED" if result else "❌ FAILED"
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = "❌ CRASHED"
    
    # Print results summary
    logger.info("=" * 50)
    logger.info("🏁 Voice-Personality Integration Test Results")
    logger.info("=" * 50)
    
    for test_name, result in results.items():
        logger.info(f"{test_name:<35}: {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    
    logger.info("=" * 50)
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All voice-personality integration tests passed!")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the logs above.")


if __name__ == "__main__":
    asyncio.run(main())
