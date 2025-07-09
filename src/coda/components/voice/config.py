"""
Configuration management for Coda 2.0 voice system.

This module provides comprehensive configuration management including:
- Environment-specific configurations
- Dynamic configuration loading and validation
- Configuration templates and presets
- Runtime configuration updates
"""

import os
import yaml
import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field, validator
# Temporarily disable BaseSettings to avoid dependency issues
# try:
#     from pydantic_settings import BaseSettings
# except ImportError:
#     # Fallback for older pydantic versions
#     from pydantic import BaseSettings

from .models import (
    VoiceConfig,
    AudioConfig,
    MoshiConfig,
    ExternalLLMConfig,
    VoiceProcessingMode,
    ConversationMode,
    AudioFormat,
    VoiceProvider
)

logger = logging.getLogger(__name__)


class VoiceSettings(BaseModel):
    """Voice system settings from environment variables."""
    
    # Environment settings
    voice_log_level: str = Field(default="INFO", env="VOICE_LOG_LEVEL")
    voice_cache_dir: str = Field(default="./cache/voice", env="VOICE_CACHE_DIR")
    voice_models_dir: str = Field(default="./models/voice", env="VOICE_MODELS_DIR")
    
    # Hardware settings
    cuda_device: int = Field(default=0, env="CUDA_DEVICE")
    total_vram_gb: int = Field(default=32, env="TOTAL_VRAM_GB")
    reserved_vram_gb: int = Field(default=4, env="RESERVED_VRAM_GB")
    
    # Moshi settings
    moshi_model_path: str = Field(default="kyutai/moshika-pytorch-bf16", env="MOSHI_MODEL_PATH")
    moshi_vram_gb: int = Field(default=8, env="MOSHI_VRAM_GB")
    
    # External LLM settings
    ollama_host: str = Field(default="http://localhost:11434", env="OLLAMA_HOST")
    ollama_model: str = Field(default="llama3.1:70b-instruct-q4_K_M", env="OLLAMA_MODEL")
    ollama_vram_gb: int = Field(default=20, env="OLLAMA_VRAM_GB")
    
    # Performance settings
    max_concurrent_conversations: int = Field(default=5, env="MAX_CONCURRENT_CONVERSATIONS")
    enable_performance_monitoring: bool = Field(default=True, env="ENABLE_PERFORMANCE_MONITORING")
    
    # Temporarily disable env file loading
    # class Config:
    #     env_file = ".env"
    #     env_file_encoding = "utf-8"


class ConfigurationTemplate:
    """Configuration templates for different use cases."""
    
    @staticmethod
    def development() -> VoiceConfig:
        """Development configuration with debugging enabled."""
        return VoiceConfig(
            mode=VoiceProcessingMode.MOSHI_ONLY,
            conversation_mode=ConversationMode.TURN_BASED,
            
            audio=AudioConfig(
                sample_rate=24000,
                channels=1,
                vad_enabled=True,
                vad_threshold=0.3,  # Lower threshold for development
                noise_reduction=False,  # Disable for debugging
                echo_cancellation=False,
                auto_gain_control=False
            ),
            
            moshi=MoshiConfig(
                model_path="kyutai/moshika-pytorch-bf16",
                device="cuda",
                optimization="bf16",
                target_latency_ms=500,  # Relaxed for development
                vram_allocation="8GB",
                enable_streaming=True,
                external_llm_enabled=False,  # Start simple
                inner_monologue_enabled=True
            ),
            
            external_llm=ExternalLLMConfig(
                provider="ollama",
                model="llama3.1:8b-instruct-q4_K_M",  # Smaller model for dev
                vram_allocation="8GB",
                reasoning_mode="basic",
                parallel_processing=False,
                fallback_enabled=True
            ),
            
            memory_integration_enabled=False,  # Disable integrations for dev
            personality_integration_enabled=False,
            tools_integration_enabled=False,
            websocket_events_enabled=True,
            
            total_vram="32GB",
            reserved_system="8GB",  # More conservative for dev
            dynamic_allocation=True
        )
    
    @staticmethod
    def production() -> VoiceConfig:
        """Production configuration optimized for performance."""
        return VoiceConfig(
            mode=VoiceProcessingMode.HYBRID,
            conversation_mode=ConversationMode.FULL_DUPLEX,
            
            audio=AudioConfig(
                sample_rate=24000,
                channels=1,
                vad_enabled=True,
                vad_threshold=0.5,
                silence_duration_ms=800,  # Shorter for responsiveness
                noise_reduction=True,
                echo_cancellation=True,
                auto_gain_control=True
            ),
            
            moshi=MoshiConfig(
                model_path="kyutai/moshika-pytorch-bf16",
                device="cuda",
                optimization="bf16",
                target_latency_ms=200,
                vram_allocation="8GB",
                enable_streaming=True,
                external_llm_enabled=True,
                inner_monologue_enabled=True
            ),
            
            external_llm=ExternalLLMConfig(
                provider="ollama",
                model="llama3.1:70b-instruct-q4_K_M",
                vram_allocation="20GB",
                reasoning_mode="enhanced",
                context_window=8192,
                temperature=0.7,
                parallel_processing=True,
                fallback_enabled=True
            ),
            
            memory_integration_enabled=True,
            personality_integration_enabled=True,
            tools_integration_enabled=True,
            websocket_events_enabled=True,
            
            total_vram="32GB",
            reserved_system="4GB",
            dynamic_allocation=True,
            
            enable_traditional_pipeline=False,
            fallback_whisper_model="large-v3",
            fallback_tts_model="xtts_v2"
        )
    
    @staticmethod
    def lightweight() -> VoiceConfig:
        """Lightweight configuration for resource-constrained environments."""
        return VoiceConfig(
            mode=VoiceProcessingMode.TRADITIONAL,
            conversation_mode=ConversationMode.TURN_BASED,
            
            audio=AudioConfig(
                sample_rate=16000,  # Lower sample rate
                channels=1,
                chunk_size=512,  # Smaller chunks
                vad_enabled=True,
                vad_threshold=0.6,
                noise_reduction=False,
                echo_cancellation=False,
                auto_gain_control=True
            ),
            
            moshi=MoshiConfig(
                model_path="kyutai/moshika-pytorch-bf16",
                device="cpu",  # CPU fallback
                optimization="int8",
                target_latency_ms=1000,
                vram_allocation="2GB",
                enable_streaming=False,
                external_llm_enabled=False
            ),
            
            external_llm=ExternalLLMConfig(
                provider="ollama",
                model="llama3.1:8b-instruct-q4_K_M",
                vram_allocation="4GB",
                reasoning_mode="basic",
                parallel_processing=False,
                fallback_enabled=True
            ),
            
            memory_integration_enabled=False,
            personality_integration_enabled=False,
            tools_integration_enabled=False,
            websocket_events_enabled=False,
            
            total_vram="16GB",
            reserved_system="2GB",
            dynamic_allocation=False,
            
            enable_traditional_pipeline=True
        )
    
    @staticmethod
    def testing() -> VoiceConfig:
        """Testing configuration with mocked components."""
        return VoiceConfig(
            mode=VoiceProcessingMode.MOSHI_ONLY,
            conversation_mode=ConversationMode.TURN_BASED,
            
            audio=AudioConfig(
                sample_rate=16000,
                channels=1,
                chunk_size=256,
                vad_enabled=False,  # Disable for consistent testing
                noise_reduction=False,
                echo_cancellation=False,
                auto_gain_control=False
            ),
            
            moshi=MoshiConfig(
                model_path="mock://moshi",  # Mock model
                device="cpu",
                optimization="none",
                target_latency_ms=100,
                vram_allocation="1GB",
                enable_streaming=False,
                external_llm_enabled=False
            ),
            
            external_llm=ExternalLLMConfig(
                provider="mock",
                model="mock://llm",
                vram_allocation="1GB",
                reasoning_mode="basic",
                parallel_processing=False,
                fallback_enabled=False
            ),
            
            memory_integration_enabled=False,
            personality_integration_enabled=False,
            tools_integration_enabled=False,
            websocket_events_enabled=False,
            
            total_vram="8GB",
            reserved_system="1GB",
            dynamic_allocation=False
        )


class ConfigurationManager:
    """Manages voice system configuration."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_dir = Path(config_dir or "./configs/voice")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.settings = VoiceSettings()
        self.current_config: Optional[VoiceConfig] = None
        
        # Create default configuration files if they don't exist
        self._create_default_configs()
    
    def load_config(self, config_name: str = "default") -> VoiceConfig:
        """Load configuration by name."""
        try:
            config_file = self.config_dir / f"{config_name}.yaml"
            
            if not config_file.exists():
                logger.warning(f"Config file {config_file} not found, using template")
                return self._get_template_config(config_name)
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Merge with environment settings
            config_data = self._merge_with_env_settings(config_data)
            
            # Validate and create config
            config = VoiceConfig(**config_data)
            self.current_config = config
            
            logger.info(f"Loaded voice configuration: {config_name}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config {config_name}: {e}")
            logger.info("Falling back to development template")
            return ConfigurationTemplate.development()
    
    def save_config(self, config: VoiceConfig, config_name: str = "custom") -> None:
        """Save configuration to file."""
        try:
            config_file = self.config_dir / f"{config_name}.yaml"
            
            # Convert to dict and save
            config_dict = config.model_dump()
            
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved voice configuration: {config_name}")
            
        except Exception as e:
            logger.error(f"Failed to save config {config_name}: {e}")
            raise
    
    def get_template_config(self, template_name: str) -> VoiceConfig:
        """Get configuration template by name."""
        return self._get_template_config(template_name)
    
    def list_configs(self) -> List[str]:
        """List available configuration files."""
        config_files = list(self.config_dir.glob("*.yaml"))
        return [f.stem for f in config_files]
    
    def validate_config(self, config: VoiceConfig) -> Dict[str, Any]:
        """Validate configuration and return validation results."""
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        try:
            # Check VRAM allocation
            total_vram_gb = int(config.total_vram.replace('GB', ''))
            moshi_vram_gb = int(config.moshi.vram_allocation.replace('GB', ''))
            llm_vram_gb = int(config.external_llm.vram_allocation.replace('GB', ''))
            reserved_gb = int(config.reserved_system.replace('GB', ''))
            
            total_allocated = moshi_vram_gb + llm_vram_gb + reserved_gb
            
            if total_allocated > total_vram_gb:
                validation_results['errors'].append(
                    f"VRAM over-allocation: {total_allocated}GB > {total_vram_gb}GB"
                )
                validation_results['valid'] = False
            elif total_allocated > total_vram_gb * 0.9:
                validation_results['warnings'].append(
                    f"High VRAM usage: {total_allocated}GB / {total_vram_gb}GB"
                )
            
            # Check model compatibility
            if config.mode == VoiceProcessingMode.HYBRID and not config.moshi.external_llm_enabled:
                validation_results['warnings'].append(
                    "Hybrid mode enabled but external LLM integration disabled"
                )
            
            # Check performance settings
            if config.moshi.target_latency_ms < 100:
                validation_results['warnings'].append(
                    "Very low latency target may cause instability"
                )
            
            # Check audio settings
            if config.audio.sample_rate < 16000:
                validation_results['warnings'].append(
                    "Low sample rate may affect audio quality"
                )
            
            # Recommendations
            if config.mode == VoiceProcessingMode.MOSHI_ONLY:
                validation_results['recommendations'].append(
                    "Consider hybrid mode for enhanced reasoning capabilities"
                )
            
            if not config.audio.vad_enabled:
                validation_results['recommendations'].append(
                    "Enable VAD for better conversation flow"
                )
            
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {e}")
            validation_results['valid'] = False
        
        return validation_results
    
    def optimize_config_for_hardware(self, config: VoiceConfig) -> VoiceConfig:
        """Optimize configuration for current hardware."""
        try:
            import torch
            
            # Get GPU info
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_name = torch.cuda.get_device_name(0)
                
                logger.info(f"Optimizing for GPU: {gpu_name} ({gpu_memory_gb:.1f}GB)")
                
                # Adjust VRAM allocations based on available memory
                if gpu_memory_gb < 16:
                    # Low VRAM - use lightweight config
                    config.moshi.vram_allocation = "4GB"
                    config.external_llm.vram_allocation = "8GB"
                    config.external_llm.model = "llama3.1:8b-instruct-q4_K_M"
                elif gpu_memory_gb < 24:
                    # Medium VRAM
                    config.moshi.vram_allocation = "6GB"
                    config.external_llm.vram_allocation = "12GB"
                    config.external_llm.model = "llama3.1:32b-instruct-q4_K_M"
                else:
                    # High VRAM - use full config
                    config.moshi.vram_allocation = "8GB"
                    config.external_llm.vram_allocation = "20GB"
                    config.external_llm.model = "llama3.1:70b-instruct-q4_K_M"
                
                config.total_vram = f"{int(gpu_memory_gb)}GB"
                
            else:
                # CPU fallback
                logger.warning("CUDA not available, using CPU configuration")
                config.moshi.device = "cpu"
                config.moshi.vram_allocation = "0GB"
                config.external_llm.vram_allocation = "0GB"
                config.mode = VoiceProcessingMode.TRADITIONAL
            
            return config
            
        except Exception as e:
            logger.error(f"Hardware optimization failed: {e}")
            return config
    
    def _create_default_configs(self) -> None:
        """Create default configuration files."""
        configs = {
            'development': ConfigurationTemplate.development(),
            'production': ConfigurationTemplate.production(),
            'lightweight': ConfigurationTemplate.lightweight(),
            'testing': ConfigurationTemplate.testing()
        }
        
        for name, config in configs.items():
            config_file = self.config_dir / f"{name}.yaml"
            if not config_file.exists():
                self.save_config(config, name)
    
    def _get_template_config(self, template_name: str) -> VoiceConfig:
        """Get configuration template by name."""
        templates = {
            'development': ConfigurationTemplate.development,
            'production': ConfigurationTemplate.production,
            'lightweight': ConfigurationTemplate.lightweight,
            'testing': ConfigurationTemplate.testing,
            'default': ConfigurationTemplate.development
        }
        
        template_func = templates.get(template_name, ConfigurationTemplate.development)
        return template_func()
    
    def _merge_with_env_settings(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration with environment settings."""
        # Update with environment variables
        if hasattr(self.settings, 'moshi_model_path'):
            config_data.setdefault('moshi', {})['model_path'] = self.settings.moshi_model_path
        
        if hasattr(self.settings, 'ollama_model'):
            config_data.setdefault('external_llm', {})['model'] = self.settings.ollama_model
        
        if hasattr(self.settings, 'total_vram_gb'):
            config_data['total_vram'] = f"{self.settings.total_vram_gb}GB"
        
        return config_data


# Global configuration manager instance
config_manager = ConfigurationManager()


def load_voice_config(config_name: str = "default") -> VoiceConfig:
    """Load voice configuration by name."""
    return config_manager.load_config(config_name)


def get_current_config() -> Optional[VoiceConfig]:
    """Get current voice configuration."""
    return config_manager.current_config


def validate_voice_config(config: VoiceConfig) -> Dict[str, Any]:
    """Validate voice configuration."""
    return config_manager.validate_config(config)
