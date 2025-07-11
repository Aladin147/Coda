"""
Model validation and health checking utilities for Coda.

This module provides robust model validation, availability checking,
and fallback mechanisms for all model types used in the system.
"""

import logging
import os
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from transformers import AutoTokenizer

logger = logging.getLogger("coda.model_validator")


class ModelValidator:
    """Comprehensive model validation and health checking."""
    
    @staticmethod
    def validate_ollama_model(model_name: str, api_base: str = "http://localhost:11434") -> Tuple[bool, str]:
        """
        Validate Ollama model availability and health.
        
        Args:
            model_name: Name of the Ollama model
            api_base: Ollama API base URL
            
        Returns:
            Tuple of (is_healthy, status_message)
        """
        try:
            # Check if Ollama service is running
            response = requests.get(f"{api_base}/api/tags", timeout=5)
            if response.status_code != 200:
                return False, f"Ollama service not responding (status: {response.status_code})"
            
            # Check if specific model is available
            models = response.json().get("models", [])
            available_models = [m["name"] for m in models]
            
            if model_name not in available_models:
                return False, f"Model '{model_name}' not found. Available: {available_models[:3]}"
            
            # Test model generation
            test_payload = {
                "model": model_name,
                "prompt": "Test",
                "stream": False
            }
            
            gen_response = requests.post(f"{api_base}/api/generate", 
                                       json=test_payload, timeout=15)
            
            if gen_response.status_code == 200:
                return True, f"Model '{model_name}' is healthy and responsive"
            else:
                return False, f"Model generation failed (status: {gen_response.status_code})"
                
        except requests.exceptions.ConnectionError:
            return False, "Ollama service not running or unreachable"
        except requests.exceptions.Timeout:
            return False, "Ollama service timeout"
        except Exception as e:
            return False, f"Ollama validation error: {str(e)}"
    
    @staticmethod
    def validate_huggingface_model(model_path: str, device: str = "cuda") -> Tuple[bool, str]:
        """
        Validate Hugging Face model availability and health.
        
        Args:
            model_path: Model path or ID
            device: Target device (cuda/cpu)
            
        Returns:
            Tuple of (is_healthy, status_message)
        """
        try:
            # Check if it's a local path or HF model ID
            if os.path.exists(model_path):
                # Local model path
                if not os.path.isdir(model_path):
                    return False, f"Local model path '{model_path}' is not a directory"
                
                # Check for essential model files
                required_files = ["config.json"]
                missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
                
                if missing_files:
                    return False, f"Missing model files: {missing_files}"
                
                return True, f"Local model '{model_path}' is available"
            
            else:
                # Hugging Face model ID
                try:
                    # Try to load tokenizer as a lightweight test
                    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                    return True, f"HuggingFace model '{model_path}' is accessible"
                except Exception as e:
                    # Check if model exists in cache
                    cache_path = os.path.expanduser("~/.cache/huggingface/hub")
                    model_cache_name = f"models--{model_path.replace('/', '--')}"
                    cached_model_path = os.path.join(cache_path, model_cache_name)
                    
                    if os.path.exists(cached_model_path):
                        return True, f"HuggingFace model '{model_path}' is cached locally"
                    else:
                        return False, f"HuggingFace model '{model_path}' not accessible: {str(e)}"
                        
        except Exception as e:
            return False, f"Model validation error: {str(e)}"
    
    @staticmethod
    def validate_gpu_availability(required_vram_gb: float = 8.0) -> Tuple[bool, str]:
        """
        Validate GPU availability and VRAM.
        
        Args:
            required_vram_gb: Required VRAM in GB
            
        Returns:
            Tuple of (is_available, status_message)
        """
        try:
            if not torch.cuda.is_available():
                return False, "CUDA not available"
            
            device_count = torch.cuda.device_count()
            if device_count == 0:
                return False, "No CUDA devices found"
            
            # Check primary GPU
            device = torch.cuda.get_device_properties(0)
            total_memory_gb = device.total_memory / (1024**3)
            
            if total_memory_gb < required_vram_gb:
                return False, f"Insufficient VRAM: {total_memory_gb:.1f}GB < {required_vram_gb}GB required"
            
            # Check compute capability for RTX 5090 compatibility
            major, minor = torch.cuda.get_device_capability(0)
            compute_capability = f"SM_{major}{minor}"
            
            return True, f"GPU available: {device.name}, {total_memory_gb:.1f}GB VRAM, {compute_capability}"
            
        except Exception as e:
            return False, f"GPU validation error: {str(e)}"
    
    @staticmethod
    def get_model_recommendations(failed_model: str, model_type: str) -> List[str]:
        """
        Get fallback model recommendations.
        
        Args:
            failed_model: The model that failed validation
            model_type: Type of model (ollama, huggingface, etc.)
            
        Returns:
            List of recommended fallback models
        """
        recommendations = {
            "ollama": [
                "qwen2.5-coder:32b",
                "llama3:8b-instruct-q4_0", 
                "gemma3:4b",
                "qwen2.5-coder:14b"
            ],
            "huggingface_voice": [
                "kyutai/moshiko-pytorch-bf16",
                "facebook/mms-tts-eng",
                "microsoft/speecht5_tts"
            ],
            "huggingface_embedding": [
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "sentence-transformers/all-MiniLM-L12-v2"
            ]
        }
        
        return recommendations.get(model_type, [])


class ModelHealthMonitor:
    """Continuous model health monitoring."""
    
    def __init__(self):
        self.health_cache: Dict[str, Tuple[bool, str, float]] = {}
        self.cache_ttl = 300  # 5 minutes
    
    def check_all_models(self, config: Any) -> Dict[str, Tuple[bool, str]]:
        """
        Check health of all configured models.
        
        Args:
            config: System configuration
            
        Returns:
            Dictionary of model_name -> (is_healthy, status_message)
        """
        results = {}
        
        # Check Ollama models
        if hasattr(config, 'llm') and hasattr(config.llm, 'model'):
            ollama_health = ModelValidator.validate_ollama_model(
                config.llm.model, 
                getattr(config.llm, 'base_url', 'http://localhost:11434')
            )
            results['ollama_llm'] = ollama_health
        
        # Check Voice models
        if hasattr(config, 'voice') and hasattr(config.voice, 'moshi'):
            if isinstance(config.voice.moshi, dict):
                model_path = config.voice.moshi.get('model_path', 'kyutai/moshiko-pytorch-bf16')
            else:
                model_path = getattr(config.voice.moshi, 'model_path', 'kyutai/moshiko-pytorch-bf16')
            
            voice_health = ModelValidator.validate_huggingface_model(model_path)
            results['voice_moshi'] = voice_health
        
        # Check GPU availability
        gpu_health = ModelValidator.validate_gpu_availability()
        results['gpu'] = gpu_health
        
        return results
    
    def get_health_summary(self, config: Any) -> str:
        """Get a human-readable health summary."""
        results = self.check_all_models(config)
        
        healthy_models = sum(1 for is_healthy, _ in results.values() if is_healthy)
        total_models = len(results)
        
        summary = f"Model Health: {healthy_models}/{total_models} healthy\n"
        
        for model_name, (is_healthy, message) in results.items():
            status = "✅" if is_healthy else "❌"
            summary += f"  {status} {model_name}: {message}\n"
        
        return summary
