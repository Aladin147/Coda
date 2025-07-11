#!/usr/bin/env python3
"""
Test Ollama integration with the updated configuration.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from coda.components.llm.models import (
    LLMConfig,
    ProviderConfig,
    LLMProvider,
    MessageRole,
    LLMMessage,
)
from coda.components.llm.manager import LLMManager
from coda.components.llm.providers.ollama_provider import OllamaProvider

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ollama_test")


async def test_ollama_integration():
    """Test Ollama integration with qwen3:30b-a3b model."""
    logger.info("🚀 Testing Ollama integration with qwen3:30b-a3b...")
    
    try:
        # Create Ollama configuration
        config = LLMConfig(
            providers={
                "ollama": ProviderConfig(
                    provider=LLMProvider.OLLAMA,
                    model="qwen3:30b-a3b",
                    host="http://localhost:11434",
                    temperature=0.7,
                    max_tokens=256
                )
            },
            default_provider="ollama"
        )
        
        # Create LLM manager
        llm_manager = LLMManager(config)
        
        # Test basic response with /no_think
        logger.info("💬 Testing basic response generation with /no_think...")

        prompt = "/no_think Hello! Can you tell me what model you are and confirm you're working correctly? Keep it brief."

        response = await llm_manager.generate_response(
            prompt=prompt,
            provider="ollama"
        )
        
        if response:
            logger.info(f"✅ Response received: {response.content[:100]}...")
            logger.info(f"📊 Tokens: {response.total_tokens}, Time: {response.response_time_ms}ms")
            return True
        else:
            logger.error("❌ No response received")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing Ollama: {e}")
        return False


async def main():
    """Main test function."""
    logger.info("🧪 Starting Ollama Integration Test")
    
    success = await test_ollama_integration()
    
    if success:
        logger.info("🎉 Ollama integration test PASSED!")
        return 0
    else:
        logger.error("💥 Ollama integration test FAILED!")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
