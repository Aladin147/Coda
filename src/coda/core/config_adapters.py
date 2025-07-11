"""
Configuration adapters to bridge core config and component configs.

This module provides adapters that convert between the simple core configuration
structure and the more complex component-specific configuration structures.
"""

from typing import Any

# Component config imports
from ..components.llm.models import (
    ConversationConfig,
    FunctionCallingConfig,
)
from ..components.llm.models import LLMConfig as ComponentLLMConfig
from ..components.llm.models import (
    LLMProvider,
    ProviderConfig,
)
from ..components.personality.models import (
    BehavioralConditioningConfig,
)
from ..components.personality.models import (
    PersonalityConfig as ComponentPersonalityConfig,
)
from ..components.personality.models import (
    PersonalityParameterConfig,
    PersonalLoreConfig,
    SessionManagerConfig,
    TopicAwarenessConfig,
)
from ..components.tools.models import ToolConfig as ComponentToolConfig
from ..components.tools.models import (
    ToolExecutorConfig,
    ToolRegistryConfig,
)
from ..components.voice.models import VoiceConfig as ComponentVoiceConfig
from .config import (
    CodaConfig,
)
from .config import LLMConfig as CoreLLMConfig
from .config import PersonalityConfig as CorePersonalityConfig
from .config import ToolsConfig as CoreToolsConfig
from .config import VoiceConfig as CoreVoiceConfig


class LLMConfigAdapter:
    """Adapter for LLM configuration."""

    @staticmethod
    def core_to_component(core_config: CoreLLMConfig) -> ComponentLLMConfig:
        """Convert core LLM config to component LLM config."""

        # Map provider string to enum
        provider_map = {
            "openai": LLMProvider.OPENAI,
            "anthropic": LLMProvider.ANTHROPIC,
            "ollama": LLMProvider.OLLAMA,
            "local": LLMProvider.LOCAL,
        }

        provider_enum = provider_map.get(core_config.provider, LLMProvider.OPENAI)

        # Create provider config
        provider_config = ProviderConfig(
            provider=provider_enum,
            model=core_config.model,
            api_key=core_config.api_key,
            api_base=core_config.base_url,
            temperature=core_config.temperature,
            max_tokens=core_config.max_tokens,
        )

        # Create component config with defaults for missing sections
        return ComponentLLMConfig(
            providers={"default": provider_config},
            default_provider="default",
            conversation=ConversationConfig(),  # Use defaults
            function_calling=FunctionCallingConfig(),  # Use defaults
            websocket_events_enabled=True,
            stream_responses=True,
            analytics_enabled=True,
            performance_monitoring=True,
        )


class ToolsConfigAdapter:
    """Adapter for Tools configuration."""

    @staticmethod
    def core_to_component(core_config: CoreToolsConfig) -> ComponentToolConfig:
        """Convert core tools config to component tools config."""

        # Create registry config
        registry_config = ToolRegistryConfig(
            auto_discover_plugins=False,  # Disable auto-discovery for now
            plugin_directories=[],
            max_tools=100,
            allow_dangerous_tools=False,
            require_auth_for_dangerous=True,
        )

        # Create executor config
        executor_config = ToolExecutorConfig(
            default_timeout_seconds=30.0,
            max_concurrent_executions=10,
            enable_retries=True,
            max_retry_attempts=3,
            execution_logging=True,
            performance_monitoring=True,
        )

        return ComponentToolConfig(
            registry=registry_config,
            executor=executor_config,
            websocket_events_enabled=True,
            analytics_enabled=True,
            memory_integration_enabled=True,
            personality_integration_enabled=True,
        )


class PersonalityConfigAdapter:
    """Adapter for Personality configuration."""

    @staticmethod
    def core_to_component(core_config: CorePersonalityConfig) -> ComponentPersonalityConfig:
        """Convert core personality config to component personality config."""

        # Create all required sub-configs with defaults
        parameters_config = PersonalityParameterConfig()

        behavioral_config = BehavioralConditioningConfig(
            enabled=core_config.adaptation_enabled,
            learning_rate=0.1,
            confidence_threshold=0.7,
            max_recent_interactions=20,
        )

        topic_awareness_config = TopicAwarenessConfig()
        personal_lore_config = PersonalLoreConfig()
        session_manager_config = SessionManagerConfig()

        return ComponentPersonalityConfig(
            parameters=parameters_config,
            behavioral_conditioning=behavioral_config,
            topic_awareness=topic_awareness_config,
            personal_lore=personal_lore_config,
            session_manager=session_manager_config,
            websocket_events_enabled=True,
            analytics_enabled=True,
        )


class VoiceConfigAdapter:
    """Adapter for Voice configuration."""

    @staticmethod
    def core_to_component(core_config: CoreVoiceConfig) -> ComponentVoiceConfig:
        """Convert core voice config to component voice config."""

        # Import here to avoid circular imports
        from coda.components.voice.models import (
            VoiceConfig as ComponentVoiceConfig,
            AudioConfig,
            MoshiConfig,
            ExternalLLMConfig,
            VoiceProcessingMode
        )

        # Map mode string to enum
        mode_mapping = {
            "moshi_only": VoiceProcessingMode.MOSHI_ONLY,
            "hybrid": VoiceProcessingMode.HYBRID,
            "traditional": VoiceProcessingMode.TRADITIONAL
        }

        # Extract audio config from core config
        audio_dict = core_config.audio or {}
        audio_config = AudioConfig(
            sample_rate=audio_dict.get('sample_rate', 16000),
            channels=audio_dict.get('channels', 1),
            bit_depth=audio_dict.get('bit_depth', 16),
            chunk_size=audio_dict.get('chunk_size', 1024),
            format=audio_dict.get('format', 'wav'),
            vad_enabled=audio_dict.get('vad_enabled', True),
            vad_threshold=audio_dict.get('vad_threshold', 0.5),
            silence_duration_ms=audio_dict.get('silence_duration_ms', 1000),
            noise_reduction=audio_dict.get('noise_reduction', True),
            echo_cancellation=audio_dict.get('echo_cancellation', True),
            auto_gain_control=audio_dict.get('auto_gain_control', True)
        )

        # Extract moshi config from core config
        moshi_dict = core_config.moshi or {}
        moshi_config = MoshiConfig(
            model_path=moshi_dict.get('model_path', 'kyutai/moshiko-pytorch-bf16'),
            device=moshi_dict.get('device', 'cuda'),
            optimization=moshi_dict.get('optimization', 'bf16'),
            max_conversation_length=moshi_dict.get('max_conversation_length', 300),
            target_latency_ms=moshi_dict.get('target_latency_ms', 200),
            vram_allocation=moshi_dict.get('vram_allocation', '8GB'),
            enable_streaming=moshi_dict.get('enable_streaming', True),
            external_llm_enabled=moshi_dict.get('external_llm_enabled', True),
            inner_monologue_enabled=moshi_dict.get('inner_monologue_enabled', True)
        )

        # Extract external LLM config from core config
        external_llm_dict = core_config.external_llm or {}
        external_llm_config = ExternalLLMConfig(
            provider=external_llm_dict.get('provider', 'ollama'),
            model=external_llm_dict.get('model', 'qwen3:30b-a3b'),
            vram_allocation=external_llm_dict.get('vram_allocation', '20GB'),
            reasoning_mode=external_llm_dict.get('reasoning_mode', 'enhanced'),
            context_window=external_llm_dict.get('context_window', 8192),
            temperature=external_llm_dict.get('temperature', 0.7),
            parallel_processing=external_llm_dict.get('parallel_processing', True),
            fallback_enabled=external_llm_dict.get('fallback_enabled', True)
        )

        # Create proper voice config from core config
        return ComponentVoiceConfig(
            mode=mode_mapping.get(core_config.mode, VoiceProcessingMode.MOSHI_ONLY),
            conversation_mode=core_config.conversation_mode,
            audio=audio_config,
            moshi=moshi_config,
            external_llm=external_llm_config,
            total_vram=core_config.total_vram,
            reserved_system=core_config.reserved_system,
            dynamic_allocation=core_config.dynamic_allocation,
            enable_traditional_pipeline=core_config.enable_traditional_pipeline,
            fallback_whisper_model=core_config.fallback_whisper_model,
            fallback_tts_model=core_config.fallback_tts_model,
            memory_integration_enabled=core_config.memory_integration_enabled,
            personality_integration_enabled=core_config.personality_integration_enabled,
            tools_integration_enabled=core_config.tools_integration_enabled,
            websocket_events_enabled=core_config.websocket_events_enabled
        )


class ConfigAdapter:
    """Main configuration adapter."""

    @staticmethod
    def adapt_config_for_component(coda_config: CodaConfig, component: str) -> Any:
        """
        Adapt core config for a specific component.

        Args:
            coda_config: The main Coda configuration
            component: Component name ('llm', 'tools', 'personality', 'voice')

        Returns:
            Component-specific configuration object
        """

        if component == "llm":
            return LLMConfigAdapter.core_to_component(coda_config.llm)
        elif component == "tools":
            return ToolsConfigAdapter.core_to_component(coda_config.tools)
        elif component == "personality":
            return PersonalityConfigAdapter.core_to_component(coda_config.personality)
        elif component == "voice":
            return VoiceConfigAdapter.core_to_component(coda_config.voice)
        else:
            raise ValueError(f"Unknown component: {component}")

    @staticmethod
    def get_component_config(coda_config: CodaConfig, component: str) -> Any:
        """
        Get component configuration, using adapter if needed.

        This method first tries to use the core config directly,
        and falls back to the adapter if there are compatibility issues.
        """

        try:
            # Try to use core config directly first
            if component == "llm":
                return coda_config.llm
            elif component == "tools":
                return coda_config.tools
            elif component == "personality":
                return coda_config.personality
            elif component == "voice":
                return coda_config.voice
            else:
                raise ValueError(f"Unknown component: {component}")
        except Exception:
            # Fall back to adapter if direct usage fails
            return ConfigAdapter.adapt_config_for_component(coda_config, component)
