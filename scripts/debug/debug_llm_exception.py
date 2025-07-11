#!/usr/bin/env python3
"""
Debug script to isolate the LLM exception handling issue.

This script will help us identify exactly where the 
"catching classes that do not inherit from BaseException is not allowed" error is coming from.
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
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("debug_llm")

# Import components one by one to isolate the issue
try:
    logger.info("Importing LLM models...")
    from coda.components.llm.models import LLMConfig, ProviderConfig, LLMProvider
    logger.info("✅ LLM models imported successfully")
    
    logger.info("Importing LLM manager...")
    from coda.components.llm.manager import LLMManager
    logger.info("✅ LLM manager imported successfully")
    
except ImportError as e:
    logger.error(f"Failed to import LLM components: {e}")
    traceback.print_exc()
    exit(1)


async def test_minimal_llm():
    """Test minimal LLM functionality to isolate the exception issue."""
    logger.info("🧪 Testing Minimal LLM Functionality...")
    
    try:
        # Create minimal config
        config = LLMConfig(
            providers={
                "ollama": ProviderConfig(
                    provider=LLMProvider.OLLAMA,
                    model="qwen3:30b-a3b",
                    host="http://localhost:11434",
                    temperature=0.7,
                    max_tokens=50,
                    system_message="/no_think Respond briefly."
                )
            },
            default_provider="ollama"
        )
        logger.info("✅ LLM config created successfully")
        
        # Create LLM manager
        llm_manager = LLMManager(config)
        logger.info("✅ LLM manager created successfully")
        
        # Try to generate a response
        logger.info("🔄 Attempting to generate response...")
        response = await llm_manager.generate_response(
            prompt="/no_think Test message",
            provider="ollama"
        )
        
        logger.info(f"✅ Response generated successfully: {response.content[:30]}...")
        return True
        
    except Exception as e:
        logger.error(f"❌ LLM test failed: {e}")
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Exception args: {e.args}")
        
        # Print full traceback
        logger.error("Full traceback:")
        traceback.print_exc()
        
        return False


async def test_exception_classes():
    """Test that all exception classes are properly defined."""
    logger.info("🧪 Testing Exception Classes...")
    
    try:
        # Import and test LLM exceptions
        from coda.components.llm.base_provider import LLMError, LLMTimeoutError, LLMRateLimitError
        logger.info("✅ LLM exceptions imported successfully")
        
        # Test that they inherit from BaseException
        assert issubclass(LLMError, BaseException), "LLMError does not inherit from BaseException"
        assert issubclass(LLMTimeoutError, BaseException), "LLMTimeoutError does not inherit from BaseException"
        assert issubclass(LLMRateLimitError, BaseException), "LLMRateLimitError does not inherit from BaseException"
        logger.info("✅ LLM exceptions inherit from BaseException correctly")
        
        # Import and test voice exceptions
        from coda.components.voice.exceptions import VoiceProcessingError, VoiceTimeoutError
        logger.info("✅ Voice exceptions imported successfully")
        
        # Test that they inherit from BaseException
        assert issubclass(VoiceProcessingError, BaseException), "VoiceProcessingError does not inherit from BaseException"
        assert issubclass(VoiceTimeoutError, BaseException), "VoiceTimeoutError does not inherit from BaseException"
        logger.info("✅ Voice exceptions inherit from BaseException correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Exception class test failed: {e}")
        traceback.print_exc()
        return False


async def test_imports_step_by_step():
    """Test imports step by step to find the problematic import."""
    logger.info("🧪 Testing Imports Step by Step...")
    
    try:
        logger.info("Step 1: Importing base provider...")
        from coda.components.llm.base_provider import BaseLLMProvider
        logger.info("✅ Base provider imported")
        
        logger.info("Step 2: Importing Ollama provider...")
        from coda.components.llm.providers.ollama_provider import OllamaProvider
        logger.info("✅ Ollama provider imported")
        
        logger.info("Step 3: Importing conversation manager...")
        from coda.components.llm.conversation_manager import ConversationManager
        logger.info("✅ Conversation manager imported")
        
        logger.info("Step 4: Importing prompt enhancer...")
        from coda.components.llm.prompt_enhancer import PromptEnhancer
        logger.info("✅ Prompt enhancer imported")
        
        logger.info("Step 5: Importing function calling...")
        from coda.components.llm.function_calling_orchestrator import FunctionCallingOrchestrator
        logger.info("✅ Function calling imported")
        
        logger.info("Step 6: Testing Ollama provider creation...")
        config = ProviderConfig(
            provider=LLMProvider.OLLAMA,
            model="qwen3:30b-a3b",
            host="http://localhost:11434",
            temperature=0.7,
            max_tokens=50
        )
        
        ollama_provider = OllamaProvider(config)
        logger.info("✅ Ollama provider created")
        
        logger.info("Step 7: Testing simple request...")
        from coda.components.llm.models import LLMMessage, MessageRole
        
        messages = [
            LLMMessage(
                role=MessageRole.USER,
                content="/no_think Test message"
            )
        ]
        
        response = await ollama_provider.generate_response(messages=messages, stream=False)
        logger.info(f"✅ Simple request successful: {response.content[:30]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Step-by-step import test failed at: {e}")
        traceback.print_exc()
        return False


async def main():
    """Main debug function."""
    logger.info("🚀 STARTING LLM EXCEPTION DEBUG")
    logger.info("=" * 50)
    
    tests = [
        ("Exception Classes Test", test_exception_classes),
        ("Step-by-Step Imports Test", test_imports_step_by_step),
        ("Minimal LLM Test", test_minimal_llm),
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
    logger.info("📊 DEBUG RESULTS SUMMARY")
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
        logger.info("\n🎉 ALL DEBUG TESTS PASSED!")
        return 0
    else:
        logger.error(f"\n❌ {total_tests - passed_tests} DEBUG TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
