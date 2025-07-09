"""
Voice-LLM Model Management

This module provides dynamic model loading and VRAM optimization for voice processing
with multiple LLM models alongside Moshi.
"""

import asyncio
import logging
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import psutil
import time

from ..llm.models import LLMProvider
from .llm_integration import VoiceLLMProcessor, VoiceLLMConfig

logger = logging.getLogger("coda.voice.model_manager")


class ModelSize(str, Enum):
    """Model size categories for VRAM optimization."""
    TINY = "tiny"      # < 1GB (gemma3:1b)
    SMALL = "small"    # 1-4GB (gemma3:4b, llama3:8b-q4)
    MEDIUM = "medium"  # 4-10GB (deepseek-r1:8b, qwen2.5-coder:14b)
    LARGE = "large"    # 10-20GB (qwen2.5-coder:32b, qwen3:30b)
    XLARGE = "xlarge"  # > 20GB (deepseek-r1:70b)


class ModelPriority(str, Enum):
    """Model loading priority."""
    CRITICAL = "critical"  # Always keep loaded
    HIGH = "high"         # Load when needed, keep if possible
    NORMAL = "normal"     # Load on demand
    LOW = "low"          # Load only if VRAM available


@dataclass
class ModelInfo:
    """Information about an available model."""
    name: str
    provider: LLMProvider
    size_category: ModelSize
    estimated_vram_gb: float
    priority: ModelPriority
    use_cases: List[str]
    performance_score: float  # 0-1, higher is better
    quality_score: float      # 0-1, higher is better
    
    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score (quality/vram ratio)."""
        return self.quality_score / max(self.estimated_vram_gb, 0.1)


class VoiceModelManager:
    """
    Manages dynamic loading and optimization of LLM models for voice processing.
    
    Features:
    - VRAM-aware model loading
    - Dynamic model switching based on context
    - Performance optimization
    - Fallback model selection
    """
    
    def __init__(self, moshi_vram_usage: float = 15.0):
        """
        Initialize the voice model manager.
        
        Args:
            moshi_vram_usage: VRAM used by Moshi in GB
        """
        self.moshi_vram_usage = moshi_vram_usage
        self.available_models: Dict[str, ModelInfo] = {}
        self.loaded_models: Dict[str, VoiceLLMProcessor] = {}
        self.current_model: Optional[str] = None
        
        # VRAM tracking
        self.total_vram_gb = self._get_total_vram()
        self.available_vram_gb = self.total_vram_gb - moshi_vram_usage
        
        # Performance tracking
        self.model_performance: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"VoiceModelManager initialized - Total VRAM: {self.total_vram_gb:.1f}GB, Available: {self.available_vram_gb:.1f}GB")
        
        # Initialize available models
        self._initialize_available_models()
    
    def _get_total_vram(self) -> float:
        """Get total VRAM available."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return 0.0
    
    def _get_current_vram_usage(self) -> float:
        """Get current VRAM usage."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) / (1024**3)
        return 0.0
    
    def _initialize_available_models(self) -> None:
        """Initialize the catalog of available models."""
        models = [
            ModelInfo(
                name="gemma3:1b",
                provider=LLMProvider.OLLAMA,
                size_category=ModelSize.TINY,
                estimated_vram_gb=0.8,
                priority=ModelPriority.HIGH,
                use_cases=["quick_responses", "fallback", "testing"],
                performance_score=0.6,
                quality_score=0.7
            ),
            ModelInfo(
                name="gemma3:4b",
                provider=LLMProvider.OLLAMA,
                size_category=ModelSize.SMALL,
                estimated_vram_gb=3.3,
                priority=ModelPriority.NORMAL,
                use_cases=["general_conversation", "quick_reasoning"],
                performance_score=0.7,
                quality_score=0.8
            ),
            ModelInfo(
                name="llama3:8b-instruct-q4_0",
                provider=LLMProvider.OLLAMA,
                size_category=ModelSize.SMALL,
                estimated_vram_gb=4.7,
                priority=ModelPriority.HIGH,
                use_cases=["conversation", "instruction_following", "reasoning"],
                performance_score=0.8,
                quality_score=0.9
            ),
            ModelInfo(
                name="deepseek-r1:8b",
                provider=LLMProvider.OLLAMA,
                size_category=ModelSize.SMALL,
                estimated_vram_gb=4.9,
                priority=ModelPriority.NORMAL,
                use_cases=["reasoning", "problem_solving", "analysis"],
                performance_score=0.7,
                quality_score=0.9
            ),
            ModelInfo(
                name="qwen2.5-coder:14b",
                provider=LLMProvider.OLLAMA,
                size_category=ModelSize.MEDIUM,
                estimated_vram_gb=9.0,
                priority=ModelPriority.NORMAL,
                use_cases=["coding", "technical_discussion", "complex_reasoning"],
                performance_score=0.6,
                quality_score=0.95
            ),
            ModelInfo(
                name="qwen2.5-coder:32b",
                provider=LLMProvider.OLLAMA,
                size_category=ModelSize.LARGE,
                estimated_vram_gb=19.0,
                priority=ModelPriority.LOW,
                use_cases=["complex_coding", "advanced_reasoning", "research"],
                performance_score=0.4,
                quality_score=0.98
            )
        ]
        
        for model in models:
            self.available_models[model.name] = model
        
        logger.info(f"Initialized {len(models)} available models")
    
    def get_optimal_model(self, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Select the optimal model based on context and VRAM availability.
        
        Args:
            context: Context information (use_case, complexity, etc.)
            
        Returns:
            Model name to use
        """
        use_case = context.get("use_case", "general_conversation") if context else "general_conversation"
        complexity = context.get("complexity", "normal") if context else "normal"
        
        # Filter models that fit in available VRAM
        suitable_models = []
        current_vram_usage = self._get_current_vram_usage()
        available_vram = self.total_vram_gb - current_vram_usage
        
        for name, model in self.available_models.items():
            if model.estimated_vram_gb <= available_vram:
                # Check if model is suitable for use case
                if use_case in model.use_cases or "general_conversation" in model.use_cases:
                    suitable_models.append((name, model))
        
        if not suitable_models:
            # Fallback to smallest model
            smallest = min(self.available_models.items(), key=lambda x: x[1].estimated_vram_gb)
            logger.warning(f"No suitable models found, falling back to {smallest[0]}")
            return smallest[0]
        
        # Score models based on context
        scored_models = []
        for name, model in suitable_models:
            score = self._calculate_model_score(model, use_case, complexity)
            scored_models.append((name, model, score))
        
        # Sort by score (highest first)
        scored_models.sort(key=lambda x: x[2], reverse=True)
        
        best_model = scored_models[0][0]
        logger.info(f"Selected optimal model: {best_model} (score: {scored_models[0][2]:.2f})")
        
        return best_model
    
    def _calculate_model_score(self, model: ModelInfo, use_case: str, complexity: str) -> float:
        """Calculate a score for model selection."""
        score = 0.0
        
        # Base quality score
        score += model.quality_score * 0.4
        
        # Performance score (speed)
        score += model.performance_score * 0.3
        
        # Efficiency score (quality/vram ratio)
        score += model.efficiency_score * 0.2
        
        # Use case match bonus
        if use_case in model.use_cases:
            score += 0.1
        
        # Complexity adjustment
        if complexity == "high" and model.size_category in [ModelSize.MEDIUM, ModelSize.LARGE]:
            score += 0.1
        elif complexity == "low" and model.size_category in [ModelSize.TINY, ModelSize.SMALL]:
            score += 0.1
        
        return score
    
    async def load_model(self, model_name: str) -> VoiceLLMProcessor:
        """
        Load a specific model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded VoiceLLMProcessor
        """
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} already loaded")
            return self.loaded_models[model_name]
        
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available")
        
        model_info = self.available_models[model_name]
        
        # Check VRAM availability
        current_usage = self._get_current_vram_usage()
        required_vram = model_info.estimated_vram_gb
        available_vram = self.total_vram_gb - current_usage
        
        if required_vram > available_vram:
            # Try to free up VRAM by unloading low-priority models
            await self._free_vram(required_vram - available_vram)
            
            # Check again
            current_usage = self._get_current_vram_usage()
            available_vram = self.total_vram_gb - current_usage
            
            if required_vram > available_vram:
                raise RuntimeError(f"Insufficient VRAM for {model_name}: need {required_vram:.1f}GB, have {available_vram:.1f}GB")
        
        # Create configuration for the model
        config = VoiceLLMConfig(
            llm_provider=model_info.provider,
            llm_model=model_name,
            llm_temperature=0.7,
            llm_max_tokens=512,
            enable_streaming=True,
            llm_timeout_seconds=10.0
        )
        
        # Load the model
        logger.info(f"Loading model {model_name} ({model_info.estimated_vram_gb:.1f}GB)")
        start_time = time.time()
        
        processor = VoiceLLMProcessor(config)
        await processor.initialize()
        
        load_time = time.time() - start_time
        
        # Store the loaded model
        self.loaded_models[model_name] = processor
        self.current_model = model_name
        
        # Track performance
        if model_name not in self.model_performance:
            self.model_performance[model_name] = {}
        self.model_performance[model_name]["load_time"] = load_time
        
        logger.info(f"Model {model_name} loaded successfully in {load_time:.2f}s")
        
        return processor
    
    async def _free_vram(self, required_gb: float) -> None:
        """Free up VRAM by unloading models."""
        logger.info(f"Attempting to free {required_gb:.1f}GB of VRAM")
        
        # Sort loaded models by priority (lowest first)
        models_by_priority = []
        for name, processor in self.loaded_models.items():
            model_info = self.available_models[name]
            priority_order = {
                ModelPriority.LOW: 0,
                ModelPriority.NORMAL: 1,
                ModelPriority.HIGH: 2,
                ModelPriority.CRITICAL: 3
            }
            models_by_priority.append((priority_order[model_info.priority], name, model_info.estimated_vram_gb))
        
        models_by_priority.sort()  # Sort by priority (lowest first)
        
        freed_vram = 0.0
        for priority, name, vram_usage in models_by_priority:
            if freed_vram >= required_gb:
                break
            
            if priority < 3:  # Don't unload CRITICAL models
                logger.info(f"Unloading model {name} to free {vram_usage:.1f}GB")
                await self.unload_model(name)
                freed_vram += vram_usage
        
        logger.info(f"Freed {freed_vram:.1f}GB of VRAM")
    
    async def unload_model(self, model_name: str) -> None:
        """Unload a specific model."""
        if model_name not in self.loaded_models:
            logger.warning(f"Model {model_name} not loaded")
            return
        
        processor = self.loaded_models[model_name]
        await processor.cleanup()
        
        del self.loaded_models[model_name]
        
        if self.current_model == model_name:
            self.current_model = None
        
        # Force garbage collection
        torch.cuda.empty_cache()
        
        logger.info(f"Model {model_name} unloaded")
    
    async def switch_model(self, model_name: str) -> VoiceLLMProcessor:
        """Switch to a different model."""
        if self.current_model == model_name:
            return self.loaded_models[model_name]
        
        # Load new model (this will handle VRAM management)
        processor = await self.load_model(model_name)
        
        # Optionally unload previous model if it's not high priority
        if self.current_model and self.current_model != model_name:
            prev_model_info = self.available_models[self.current_model]
            if prev_model_info.priority in [ModelPriority.LOW, ModelPriority.NORMAL]:
                await self.unload_model(self.current_model)
        
        self.current_model = model_name
        return processor
    
    def get_vram_status(self) -> Dict[str, Any]:
        """Get current VRAM status."""
        current_usage = self._get_current_vram_usage()
        
        return {
            "total_vram_gb": self.total_vram_gb,
            "moshi_usage_gb": self.moshi_vram_usage,
            "current_usage_gb": current_usage,
            "available_gb": self.total_vram_gb - current_usage,
            "loaded_models": list(self.loaded_models.keys()),
            "current_model": self.current_model
        }
    
    def get_model_recommendations(self, context: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """Get model recommendations with scores."""
        use_case = context.get("use_case", "general_conversation") if context else "general_conversation"
        complexity = context.get("complexity", "normal") if context else "normal"
        
        recommendations = []
        current_vram_usage = self._get_current_vram_usage()
        available_vram = self.total_vram_gb - current_vram_usage
        
        for name, model in self.available_models.items():
            if model.estimated_vram_gb <= available_vram:
                score = self._calculate_model_score(model, use_case, complexity)
                recommendations.append((name, score))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations
    
    async def cleanup(self) -> None:
        """Clean up all loaded models."""
        logger.info("Cleaning up voice model manager")
        
        for model_name in list(self.loaded_models.keys()):
            await self.unload_model(model_name)
        
        self.current_model = None
        logger.info("Voice model manager cleanup completed")
