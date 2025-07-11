"""
Configuration management for Coda.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

# Import RTX 5090 optimizer for automatic optimization
try:
    from .rtx5090_optimizer import apply_rtx5090_optimizations
    RTX5090_AVAILABLE = True
except ImportError:
    RTX5090_AVAILABLE = False


class STTConfig(BaseModel):
    """STT configuration."""

    engine: str = "whisper"
    model: str = "base"
    device: str = "cuda"
    compute_type: str = "float16"
    language: Optional[str] = "en"
    vad_filter: bool = True


class TTSConfig(BaseModel):
    """TTS configuration."""

    model_config = {"protected_namespaces": ()}

    engine: str = "elevenlabs"
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    model_id: str = "eleven_multilingual_v2"
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True


class VoiceConfig(BaseModel):
    """Voice system configuration."""

    # Basic voice settings
    mode: str = "moshi_only"
    conversation_mode: str = "turn_based"

    # Audio configuration
    audio: Optional[Dict[str, Any]] = None

    # Moshi configuration
    moshi: Optional[Dict[str, Any]] = None

    # External LLM configuration
    external_llm: Optional[Dict[str, Any]] = None

    # System resource configuration
    total_vram: str = "32GB"
    reserved_system: str = "4GB"
    dynamic_allocation: bool = True

    # Integration settings
    memory_integration_enabled: bool = True
    personality_integration_enabled: bool = True
    tools_integration_enabled: bool = True
    websocket_events_enabled: bool = True

    # Fallback settings
    enable_traditional_pipeline: bool = False
    fallback_whisper_model: str = "large-v3"
    fallback_tts_model: str = "xtts_v2"

    # Legacy STT/TTS configs for backward compatibility
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)


class LLMConfig(BaseModel):
    """LLM system configuration."""

    provider: str = "openai"
    model: str = "gpt-4"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048


class ShortTermMemoryConfig(BaseModel):
    """Short-term memory configuration."""

    max_turns: int = 20
    max_tokens: int = 800


class LongTermMemoryConfig(BaseModel):
    """Long-term memory configuration."""

    enabled: bool = True
    storage_path: str = "data/memory/long_term"
    vector_db_type: str = "chroma"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_memories: int = 1000
    device: str = "cpu"
    chunk_size: int = 200
    chunk_overlap: int = 50
    min_chunk_length: int = 50
    time_decay_days: float = 30.0
    auto_persist: bool = True
    persist_interval: int = 5


class MemoryConfig(BaseModel):
    """Memory system configuration."""

    short_term: ShortTermMemoryConfig = Field(default_factory=ShortTermMemoryConfig)
    long_term: LongTermMemoryConfig = Field(default_factory=LongTermMemoryConfig)


class ToolsConfig(BaseModel):
    """Tools system configuration."""

    enabled: bool = True
    available_tools: List[str] = Field(
        default_factory=lambda: [
            "get_time",
            "get_date",
            "tell_joke",
            "get_weather",
            "search_memory",
            "add_memory",
            "list_tools",
            "show_capabilities",
        ]
    )
    enabled_tools: List[str] = Field(
        default_factory=lambda: [
            "get_time",
            "get_date",
            "tell_joke",
            "get_weather",
            "search_memory",
            "add_memory",
            "list_tools",
            "show_capabilities",
        ]
    )


class PersonalityConfig(BaseModel):
    """Personality system configuration."""

    enabled: bool = True
    base_personality: str = "helpful_assistant"
    adaptation_enabled: bool = True


class WebSocketConfig(BaseModel):
    """WebSocket configuration."""

    host: str = "localhost"
    port: int = 8765
    enabled: bool = False


class DashboardConfig(BaseModel):
    """Dashboard configuration."""

    host: str = "localhost"
    port: int = 8080
    enabled: bool = True


class CodaConfig(BaseModel):
    """Main Coda configuration."""

    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    personality: PersonalityConfig = Field(default_factory=PersonalityConfig)
    websocket: WebSocketConfig = Field(default_factory=WebSocketConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)

    # Global settings
    debug: bool = False
    data_directory: str = "data"
    log_level: str = "INFO"


def _configure_ssl_for_development():
    """Configure SSL settings for development environment."""
    # Remove problematic SSL_CERT_FILE environment variable for local development
    if "SSL_CERT_FILE" in os.environ:
        ssl_cert_file = os.environ["SSL_CERT_FILE"]
        try:
            # Test if the SSL cert file is accessible
            with open(ssl_cert_file, 'r'):
                pass
        except (PermissionError, FileNotFoundError):
            # Remove the problematic SSL_CERT_FILE to allow local connections
            os.environ.pop("SSL_CERT_FILE", None)


def load_config(config_path: Optional[Path] = None) -> CodaConfig:
    """Load configuration from file or environment variables."""
    # Configure SSL for development
    _configure_ssl_for_development()

    if config_path is None:
        config_path = Path("configs/default.yaml")

    config_data = {}

    # Load from YAML file if it exists
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

    # Override with environment variables
    config_data = _apply_env_overrides(config_data)

    # Apply RTX 5090 optimizations if available
    if RTX5090_AVAILABLE:
        try:
            apply_rtx5090_optimizations()
        except Exception:
            # Don't fail config loading if GPU optimization fails
            pass

    # Create data directory if it doesn't exist
    data_dir = Path(config_data.get("data_directory", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    return CodaConfig(**config_data)


def _apply_env_overrides(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration."""
    env_mappings = {
        "CODA_LLM_API_KEY": ["llm", "api_key"],
        "CODA_LLM_PROVIDER": ["llm", "provider"],
        "CODA_LLM_MODEL": ["llm", "model"],
        "CODA_DEBUG": ["debug"],
        "CODA_LOG_LEVEL": ["log_level"],
        "CODA_WEBSOCKET_HOST": ["websocket", "host"],
        "CODA_WEBSOCKET_PORT": ["websocket", "port"],
        "CODA_DATA_DIR": ["data_directory"],
    }

    for env_var, config_path in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            # Navigate to the nested config location
            current = config_data
            for key in config_path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Convert value to appropriate type
            if config_path[-1] in ["debug"]:
                value = value.lower() in ("true", "1", "yes", "on")
            elif config_path[-1] in ["port"]:
                value = int(value)
            elif config_path[-1] in ["temperature"]:
                value = float(value)

            current[config_path[-1]] = value

    return config_data
