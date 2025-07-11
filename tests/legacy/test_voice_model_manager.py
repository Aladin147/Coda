#!/usr/bin/env python3
"""
Test Voice Model Manager

This script tests the dynamic model loading and VRAM optimization capabilities.
"""

import asyncio
import logging
import torch
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from src.coda.components.voice.model_manager import VoiceModelManager, ModelSize, ModelPriority
    logger.info("✓ Successfully imported voice model manager components")
except ImportError as e:
    logger.error(f"❌ Failed to import components: {e}")
    exit(1)


async def test_model_manager_initialization():
    """Test model manager initialization."""
    logger.info("=== Testing Model Manager Initialization ===")
    
    try:
        # Simulate Moshi VRAM usage
        moshi_vram = 15.0
        manager = VoiceModelManager(moshi_vram_usage=moshi_vram)
        
        logger.info(f"✓ Model manager created")
        logger.info(f"  Total VRAM: {manager.total_vram_gb:.1f}GB")
        logger.info(f"  Available for LLMs: {manager.available_vram_gb:.1f}GB")
        logger.info(f"  Available models: {len(manager.available_models)}")
        
        # List available models
        for name, model in manager.available_models.items():
            logger.info(f"    {name}: {model.size_category.value}, {model.estimated_vram_gb:.1f}GB, {model.priority.value}")
        
        await manager.cleanup()
        logger.info("✓ Model manager initialization test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_optimal_model_selection():
    """Test optimal model selection based on context."""
    logger.info("=== Testing Optimal Model Selection ===")
    
    try:
        manager = VoiceModelManager(moshi_vram_usage=15.0)
        
        # Test different contexts
        contexts = [
            {"use_case": "general_conversation", "complexity": "normal"},
            {"use_case": "quick_responses", "complexity": "low"},
            {"use_case": "reasoning", "complexity": "high"},
            {"use_case": "coding", "complexity": "high"},
            {}  # Default context
        ]
        
        for i, context in enumerate(contexts):
            optimal_model = manager.get_optimal_model(context)
            recommendations = manager.get_model_recommendations(context)
            
            logger.info(f"✓ Context {i+1}: {context}")
            logger.info(f"  Optimal model: {optimal_model}")
            logger.info(f"  Top 3 recommendations:")
            for j, (name, score) in enumerate(recommendations[:3]):
                logger.info(f"    {j+1}. {name} (score: {score:.3f})")
        
        await manager.cleanup()
        logger.info("✓ Optimal model selection test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model selection test failed: {e}")
        return False


async def test_vram_management():
    """Test VRAM management and model loading."""
    logger.info("=== Testing VRAM Management ===")
    
    try:
        manager = VoiceModelManager(moshi_vram_usage=15.0)
        
        # Check initial VRAM status
        vram_status = manager.get_vram_status()
        logger.info(f"✓ Initial VRAM status:")
        logger.info(f"  Total: {vram_status['total_vram_gb']:.1f}GB")
        logger.info(f"  Available: {vram_status['available_gb']:.1f}GB")
        logger.info(f"  Loaded models: {vram_status['loaded_models']}")
        
        # Test loading a small model
        small_model = "gemma3:1b"
        logger.info(f"Loading small model: {small_model}")
        
        try:
            processor = await manager.load_model(small_model)
            logger.info(f"✓ Successfully loaded {small_model}")
            
            # Check VRAM status after loading
            vram_status = manager.get_vram_status()
            logger.info(f"  VRAM after loading: {vram_status['available_gb']:.1f}GB available")
            logger.info(f"  Current model: {vram_status['current_model']}")
            
            # Test model switching
            logger.info("Testing model switching...")
            if "gemma3:4b" in manager.available_models:
                processor2 = await manager.switch_model("gemma3:4b")
                logger.info("✓ Successfully switched to gemma3:4b")
                
                vram_status = manager.get_vram_status()
                logger.info(f"  VRAM after switch: {vram_status['available_gb']:.1f}GB available")
                logger.info(f"  Loaded models: {vram_status['loaded_models']}")
            
        except Exception as e:
            logger.warning(f"Model loading failed (expected if models not available): {e}")
        
        await manager.cleanup()
        logger.info("✓ VRAM management test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ VRAM management test failed: {e}")
        return False


async def test_model_performance_tracking():
    """Test model performance tracking."""
    logger.info("=== Testing Model Performance Tracking ===")
    
    try:
        manager = VoiceModelManager(moshi_vram_usage=15.0)
        
        # Test efficiency calculations
        for name, model in manager.available_models.items():
            efficiency = model.efficiency_score
            logger.info(f"  {name}: efficiency={efficiency:.3f} (quality={model.quality_score:.2f}, vram={model.estimated_vram_gb:.1f}GB)")
        
        # Find most efficient model
        most_efficient = max(manager.available_models.items(), key=lambda x: x[1].efficiency_score)
        logger.info(f"✓ Most efficient model: {most_efficient[0]} (efficiency: {most_efficient[1].efficiency_score:.3f})")
        
        await manager.cleanup()
        logger.info("✓ Performance tracking test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance tracking test failed: {e}")
        return False


async def test_fallback_scenarios():
    """Test fallback scenarios and error handling."""
    logger.info("=== Testing Fallback Scenarios ===")
    
    try:
        manager = VoiceModelManager(moshi_vram_usage=30.0)  # High VRAM usage to test fallbacks
        
        # Test with limited VRAM
        vram_status = manager.get_vram_status()
        logger.info(f"✓ Limited VRAM scenario:")
        logger.info(f"  Available: {vram_status['available_gb']:.1f}GB")
        
        # Try to get optimal model with limited VRAM
        optimal_model = manager.get_optimal_model({"use_case": "coding", "complexity": "high"})
        logger.info(f"  Optimal model with limited VRAM: {optimal_model}")
        
        # Test recommendations with limited VRAM
        recommendations = manager.get_model_recommendations({"use_case": "coding"})
        logger.info(f"  Available models with limited VRAM: {len(recommendations)}")
        for name, score in recommendations[:3]:
            model_info = manager.available_models[name]
            logger.info(f"    {name}: {model_info.estimated_vram_gb:.1f}GB, score={score:.3f}")
        
        await manager.cleanup()
        logger.info("✓ Fallback scenarios test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Fallback scenarios test failed: {e}")
        return False


async def test_context_aware_selection():
    """Test context-aware model selection."""
    logger.info("=== Testing Context-Aware Selection ===")
    
    try:
        manager = VoiceModelManager(moshi_vram_usage=15.0)
        
        # Test different use cases
        use_cases = [
            "quick_responses",
            "general_conversation", 
            "reasoning",
            "coding",
            "complex_reasoning"
        ]
        
        for use_case in use_cases:
            context = {"use_case": use_case, "complexity": "normal"}
            optimal = manager.get_optimal_model(context)
            model_info = manager.available_models[optimal]
            
            logger.info(f"✓ Use case '{use_case}': {optimal}")
            logger.info(f"    Size: {model_info.size_category.value}, Quality: {model_info.quality_score:.2f}")
            logger.info(f"    Suitable: {use_case in model_info.use_cases}")
        
        await manager.cleanup()
        logger.info("✓ Context-aware selection test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Context-aware selection test failed: {e}")
        return False


async def main():
    """Run all voice model manager tests."""
    logger.info("🚀 Starting Voice Model Manager Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Initialization", test_model_manager_initialization),
        ("Model Selection", test_optimal_model_selection),
        ("VRAM Management", test_vram_management),
        ("Performance Tracking", test_model_performance_tracking),
        ("Fallback Scenarios", test_fallback_scenarios),
        ("Context-Aware Selection", test_context_aware_selection),
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
    logger.info("🏁 Voice Model Manager Test Results")
    logger.info("=" * 50)
    
    for test_name, result in results.items():
        logger.info(f"{test_name:<25}: {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    
    logger.info("=" * 50)
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All voice model manager tests passed!")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the logs above.")


if __name__ == "__main__":
    asyncio.run(main())
