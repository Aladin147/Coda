"""
Configuration validation and enhancement utilities for Coda.

This module provides robust configuration validation, fallback mechanisms,
and automatic configuration healing for the Coda voice assistant system.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import yaml

from .model_validator import ModelHealthMonitor, ModelValidator
from .config import CodaConfig, load_config

logger = logging.getLogger("coda.config_validator")


class ConfigValidator:
    """Comprehensive configuration validation and enhancement."""
    
    def __init__(self):
        self.model_monitor = ModelHealthMonitor()
        self.validation_cache: Dict[str, Any] = {}
    
    def validate_configuration(self, config: CodaConfig) -> Tuple[bool, List[str], List[str]]:
        """
        Validate entire configuration and provide recommendations.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        
        # Validate LLM configuration
        llm_errors, llm_warnings = self._validate_llm_config(config.llm)
        errors.extend(llm_errors)
        warnings.extend(llm_warnings)
        
        # Validate Voice configuration
        voice_errors, voice_warnings = self._validate_voice_config(config.voice)
        errors.extend(voice_errors)
        warnings.extend(voice_warnings)
        
        # Validate Memory configuration
        memory_errors, memory_warnings = self._validate_memory_config(config.memory)
        errors.extend(memory_errors)
        warnings.extend(memory_warnings)
        
        # Validate system resources
        resource_errors, resource_warnings = self._validate_system_resources(config)
        errors.extend(resource_errors)
        warnings.extend(resource_warnings)
        
        is_valid = len(errors) == 0
        return is_valid, errors, warnings
    
    def _validate_llm_config(self, llm_config: Any) -> Tuple[List[str], List[str]]:
        """Validate LLM configuration."""
        errors = []
        warnings = []
        
        # Check if Ollama model is available
        if hasattr(llm_config, 'model') and hasattr(llm_config, 'base_url'):
            is_healthy, message = ModelValidator.validate_ollama_model(
                llm_config.model, 
                llm_config.base_url or "http://localhost:11434"
            )
            
            if not is_healthy:
                errors.append(f"LLM Model validation failed: {message}")
                
                # Suggest fallbacks
                recommendations = ModelValidator.get_model_recommendations(
                    llm_config.model, "ollama"
                )
                if recommendations:
                    warnings.append(f"Consider fallback models: {', '.join(recommendations[:3])}")
        
        # Check temperature and token limits
        if hasattr(llm_config, 'temperature'):
            if not 0.0 <= llm_config.temperature <= 2.0:
                warnings.append(f"Temperature {llm_config.temperature} outside recommended range [0.0, 2.0]")
        
        if hasattr(llm_config, 'max_tokens'):
            if llm_config.max_tokens > 8192:
                warnings.append(f"Max tokens {llm_config.max_tokens} may cause memory issues")
        
        return errors, warnings
    
    def _validate_voice_config(self, voice_config: Any) -> Tuple[List[str], List[str]]:
        """Validate Voice configuration."""
        errors = []
        warnings = []
        
        # Check Moshi model availability
        if hasattr(voice_config, 'moshi'):
            if isinstance(voice_config.moshi, dict):
                model_path = voice_config.moshi.get('model_path', 'kyutai/moshiko-pytorch-bf16')
            else:
                model_path = getattr(voice_config.moshi, 'model_path', 'kyutai/moshiko-pytorch-bf16')
            
            is_healthy, message = ModelValidator.validate_huggingface_model(model_path)
            
            if not is_healthy:
                errors.append(f"Voice Model validation failed: {message}")
                
                # Suggest fallbacks
                recommendations = ModelValidator.get_model_recommendations(
                    model_path, "huggingface_voice"
                )
                if recommendations:
                    warnings.append(f"Consider fallback voice models: {', '.join(recommendations[:2])}")
        
        # Check VRAM allocation
        if hasattr(voice_config, 'total_vram'):
            try:
                vram_gb = float(voice_config.total_vram.replace('GB', ''))
                is_available, gpu_message = ModelValidator.validate_gpu_availability(vram_gb * 0.8)
                
                if not is_available:
                    warnings.append(f"GPU validation: {gpu_message}")
            except:
                warnings.append(f"Invalid VRAM specification: {voice_config.total_vram}")
        
        return errors, warnings
    
    def _validate_memory_config(self, memory_config: Any) -> Tuple[List[str], List[str]]:
        """Validate Memory configuration."""
        errors = []
        warnings = []
        
        # Check vector database configuration
        if hasattr(memory_config, 'long_term'):
            if hasattr(memory_config.long_term, 'vector_db_type'):
                db_type = memory_config.long_term.vector_db_type
                if db_type not in ['chroma', 'sqlite', 'in_memory']:
                    errors.append(f"Unsupported vector database type: {db_type}")
            
            # Check storage path
            if hasattr(memory_config.long_term, 'storage_path'):
                storage_path = Path(memory_config.long_term.storage_path)
                try:
                    storage_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create memory storage path: {e}")
        
        # Check embedding model
        if hasattr(memory_config, 'long_term') and hasattr(memory_config.long_term, 'embedding_model'):
            embedding_model = memory_config.long_term.embedding_model
            is_healthy, message = ModelValidator.validate_huggingface_model(embedding_model)
            
            if not is_healthy:
                warnings.append(f"Embedding model validation: {message}")
                
                recommendations = ModelValidator.get_model_recommendations(
                    embedding_model, "huggingface_embedding"
                )
                if recommendations:
                    warnings.append(f"Consider fallback embedding models: {', '.join(recommendations[:2])}")
        
        return errors, warnings
    
    def _validate_system_resources(self, config: CodaConfig) -> Tuple[List[str], List[str]]:
        """Validate system resource requirements."""
        errors = []
        warnings = []
        
        # Check GPU availability
        is_available, gpu_message = ModelValidator.validate_gpu_availability()
        
        if not is_available:
            warnings.append(f"GPU not available: {gpu_message}")
            warnings.append("System will fall back to CPU processing (slower performance)")
        
        # Check disk space for models and data
        try:
            import shutil
            free_space_gb = shutil.disk_usage('.').free / (1024**3)
            
            if free_space_gb < 50:  # 50GB minimum recommended
                warnings.append(f"Low disk space: {free_space_gb:.1f}GB free (50GB+ recommended)")
        except:
            warnings.append("Could not check disk space")
        
        return errors, warnings
    
    def auto_heal_configuration(self, config: CodaConfig) -> Tuple[CodaConfig, List[str]]:
        """
        Automatically heal configuration issues where possible.
        
        Args:
            config: Configuration to heal
            
        Returns:
            Tuple of (healed_config, applied_fixes)
        """
        applied_fixes = []
        healed_config = config.model_copy(deep=True)
        
        # Auto-heal LLM configuration
        if hasattr(healed_config.llm, 'model'):
            is_healthy, _ = ModelValidator.validate_ollama_model(healed_config.llm.model)
            
            if not is_healthy:
                # Try fallback models
                recommendations = ModelValidator.get_model_recommendations(
                    healed_config.llm.model, "ollama"
                )
                
                for fallback_model in recommendations:
                    is_healthy, _ = ModelValidator.validate_ollama_model(fallback_model)
                    if is_healthy:
                        healed_config.llm.model = fallback_model
                        applied_fixes.append(f"Switched LLM model to {fallback_model}")
                        break
        
        # Auto-heal Voice configuration
        if hasattr(healed_config.voice, 'moshi'):
            if isinstance(healed_config.voice.moshi, dict):
                model_path = healed_config.voice.moshi.get('model_path')
            else:
                model_path = getattr(healed_config.voice.moshi, 'model_path', None)
            
            if model_path:
                is_healthy, _ = ModelValidator.validate_huggingface_model(model_path)
                
                if not is_healthy:
                    # Try fallback voice models
                    recommendations = ModelValidator.get_model_recommendations(
                        model_path, "huggingface_voice"
                    )
                    
                    for fallback_model in recommendations:
                        is_healthy, _ = ModelValidator.validate_huggingface_model(fallback_model)
                        if is_healthy:
                            if isinstance(healed_config.voice.moshi, dict):
                                healed_config.voice.moshi['model_path'] = fallback_model
                            else:
                                healed_config.voice.moshi.model_path = fallback_model
                            applied_fixes.append(f"Switched voice model to {fallback_model}")
                            break
        
        # Auto-heal GPU configuration
        is_gpu_available, _ = ModelValidator.validate_gpu_availability()
        
        if not is_gpu_available:
            # Switch to CPU mode
            if hasattr(healed_config.voice, 'moshi'):
                if isinstance(healed_config.voice.moshi, dict):
                    healed_config.voice.moshi['device'] = 'cpu'
                else:
                    healed_config.voice.moshi.device = 'cpu'
                applied_fixes.append("Switched voice processing to CPU mode")
            
            if hasattr(healed_config.memory, 'long_term'):
                healed_config.memory.long_term.device = 'cpu'
                applied_fixes.append("Switched memory processing to CPU mode")
        
        return healed_config, applied_fixes
    
    def generate_health_report(self, config: CodaConfig) -> str:
        """Generate a comprehensive health report."""
        is_valid, errors, warnings = self.validate_configuration(config)
        model_health = self.model_monitor.get_health_summary(config)
        
        report = "=== CODA CONFIGURATION HEALTH REPORT ===\n\n"
        
        # Overall status
        status = "✅ HEALTHY" if is_valid else "❌ ISSUES DETECTED"
        report += f"Overall Status: {status}\n\n"
        
        # Model health
        report += model_health + "\n"
        
        # Errors
        if errors:
            report += "🚨 CRITICAL ERRORS:\n"
            for error in errors:
                report += f"  ❌ {error}\n"
            report += "\n"
        
        # Warnings
        if warnings:
            report += "⚠️  WARNINGS:\n"
            for warning in warnings:
                report += f"  ⚠️  {warning}\n"
            report += "\n"
        
        # Recommendations
        if not is_valid:
            report += "💡 RECOMMENDATIONS:\n"
            report += "  1. Run auto-heal to fix common issues\n"
            report += "  2. Check model availability and download if needed\n"
            report += "  3. Verify system resources (GPU, disk space)\n"
            report += "  4. Review configuration file for typos\n"
        
        return report


def validate_and_heal_config(config_path: Path) -> Tuple[CodaConfig, str]:
    """
    Convenience function to validate and auto-heal a configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (healed_config, health_report)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Validate and heal
    validator = ConfigValidator()
    healed_config, applied_fixes = validator.auto_heal_configuration(config)
    
    # Generate report
    health_report = validator.generate_health_report(healed_config)
    
    if applied_fixes:
        health_report += "\n🔧 AUTO-APPLIED FIXES:\n"
        for fix in applied_fixes:
            health_report += f"  ✅ {fix}\n"
    
    return healed_config, health_report
