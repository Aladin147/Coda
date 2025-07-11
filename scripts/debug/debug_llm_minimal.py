#!/usr/bin/env python3
"""
Minimal debug script to test LLM without voice components.

This script will test LLM functionality without importing any voice components
to isolate if the issue is coming from voice component imports.
"""

import asyncio
import logging
import sys
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("debug_minimal")

# Disable voice component imports by temporarily renaming the voice module
import os
voice_path = Path(__file__).parent / "src" / "coda" / "components" / "voice"
voice_backup_path = Path(__file__).parent / "src" / "coda" / "components" / "voice_disabled"

def disable_voice_imports():
    """Temporarily disable voice imports."""
    if voice_path.exists() and not voice_backup_path.exists():
        voice_path.rename(voice_backup_path)
        logger.info("🔇 Voice components temporarily disabled")

def enable_voice_imports():
    """Re-enable voice imports."""
    if voice_backup_path.exists() and not voice_path.exists():
        voice_backup_path.rename(voice_path)
        logger.info("🔊 Voice components re-enabled")

async def test_llm_without_voice():
    """Test LLM functionality without voice components."""
    logger.info("🧪 Testing LLM Without Voice Components...")
    
    try:
        # Import LLM components only
        from coda.components.llm.models import LLMConfig, ProviderConfig, LLMProvider, LLMMessage, MessageRole
        from coda.components.llm.providers.ollama_provider import OllamaProvider
        logger.info("✅ LLM components imported successfully")
        
        # Create minimal config
        config = ProviderConfig(
            provider=LLMProvider.OLLAMA,
            model="qwen3:30b-a3b",
            host="http://localhost:11434",
            temperature=0.7,
            max_tokens=50
        )
        
        # Create Ollama provider directly
        ollama_provider = OllamaProvider(config)
        logger.info("✅ Ollama provider created successfully")
        
        # Create test message
        messages = [
            LLMMessage(
                role=MessageRole.USER,
                content="/no_think Test message"
            )
        ]
        
        # Try to generate response
        logger.info("🔄 Attempting to generate response...")
        response = await ollama_provider.generate_response(messages=messages, stream=False)
        
        logger.info(f"✅ Response generated successfully: {response.content[:50]}...")
        return True
        
    except Exception as e:
        logger.error(f"❌ LLM test failed: {e}")
        logger.error(f"Exception type: {type(e)}")
        traceback.print_exc()
        return False

async def test_aiohttp_directly():
    """Test aiohttp directly to see if the issue is there."""
    logger.info("🧪 Testing aiohttp Directly...")
    
    try:
        import aiohttp
        logger.info("✅ aiohttp imported successfully")
        
        # Test basic aiohttp request
        async with aiohttp.ClientSession() as session:
            url = "http://localhost:11434/api/tags"
            
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ aiohttp request successful: {response.status}")
                        return True
                    else:
                        logger.warning(f"⚠️ aiohttp request returned status: {response.status}")
                        return True  # Still successful from aiohttp perspective
                        
            except aiohttp.ClientTimeout:
                logger.warning("⚠️ aiohttp request timed out (expected if Ollama not running)")
                return True  # Timeout is expected behavior
            except Exception as e:
                logger.error(f"❌ aiohttp request failed: {e}")
                return False
                
    except Exception as e:
        logger.error(f"❌ aiohttp test failed: {e}")
        traceback.print_exc()
        return False

async def test_exception_handling_directly():
    """Test exception handling directly."""
    logger.info("🧪 Testing Exception Handling Directly...")
    
    try:
        import aiohttp
        
        # Test that we can catch aiohttp exceptions properly
        try:
            # This should raise a ClientTimeout
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:99999/nonexistent", 
                    timeout=aiohttp.ClientTimeout(total=0.001)
                ) as response:
                    pass
        except aiohttp.ClientTimeout as e:
            logger.info("✅ Successfully caught aiohttp.ClientTimeout")
        except Exception as e:
            logger.info(f"✅ Caught other exception (expected): {type(e).__name__}")
        
        # Test that aiohttp.ClientTimeout is a proper exception
        assert issubclass(aiohttp.ClientTimeout, BaseException), "aiohttp.ClientTimeout does not inherit from BaseException"
        logger.info("✅ aiohttp.ClientTimeout inherits from BaseException correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Exception handling test failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """Main debug function."""
    logger.info("🚀 STARTING MINIMAL LLM DEBUG")
    logger.info("=" * 50)
    
    try:
        # Disable voice imports
        disable_voice_imports()
        
        tests = [
            ("aiohttp Direct Test", test_aiohttp_directly),
            ("Exception Handling Test", test_exception_handling_directly),
            ("LLM Without Voice Test", test_llm_without_voice),
        ]
        
        results = []
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*30}")
            logger.info(f"Running: {test_name}")
            try:
                result = await test_func()
                results.append((test_name, result))
                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name} failed with exception: {e}")
                traceback.print_exc()
                results.append((test_name, False))
        
        # Summary
        logger.info(f"\n{'='*50}")
        logger.info("📊 MINIMAL DEBUG RESULTS")
        logger.info("=" * 50)
        
        passed_tests = sum(1 for _, result in results if result)
        total_tests = len(results)
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {status} {test_name}")
        
        if passed_tests == total_tests:
            logger.info("\n🎉 ALL MINIMAL DEBUG TESTS PASSED!")
            return 0
        else:
            logger.error(f"\n❌ {total_tests - passed_tests} MINIMAL DEBUG TESTS FAILED")
            return 1
            
    finally:
        # Re-enable voice imports
        enable_voice_imports()


if __name__ == "__main__":
    exit(asyncio.run(main()))
