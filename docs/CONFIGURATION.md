# Coda Configuration Guide

> **Complete guide to configuring Coda for your specific needs and environment**

## 📋 Configuration Overview

Coda uses YAML configuration files to customize behavior across all components. Configuration files are located in the `configs/` directory and can be specified when starting Coda.

### Configuration Hierarchy

1. **Default Configuration**: Built-in defaults for all components
2. **Base Configuration**: `configs/base.yaml` - Common settings
3. **Environment Configuration**: `configs/production.yaml`, `configs/development.yaml`
4. **Custom Configuration**: Your specific configuration file
5. **Command Line Arguments**: Override any setting via CLI

### Quick Start Configurations

```bash
# Use production configuration
python coda_launcher.py --config configs/production.yaml

# Use development configuration  
python coda_launcher.py --config configs/development.yaml

# Use custom configuration
python coda_launcher.py --config configs/my_config.yaml
```

## 🏗️ Configuration Structure

### Complete Configuration Template

```yaml
# configs/complete_example.yaml

# System-wide settings
system:
  name: "Coda Assistant"
  version: "2.0.0"
  environment: "production"  # development, staging, production
  debug: false
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Large Language Model Configuration
llm:
  provider: "ollama"  # ollama, openai, anthropic, huggingface
  model: "qwen3:30b-a3b"
  base_url: "http://localhost:11434"
  api_key: null  # For external providers
  
  # Generation settings
  generation:
    max_tokens: 2048
    temperature: 0.8
    top_p: 0.9
    top_k: 40
    repetition_penalty: 1.1
    stop_sequences: ["Human:", "Assistant:"]
  
  # Performance optimization
  optimization:
    enable_caching: true
    cache_ttl: 3600  # seconds
    max_concurrent_requests: 3
    timeout: 30  # seconds
    enable_memory_optimization: true

# Voice Processing Configuration
voice:
  enabled: true
  mode: "adaptive"  # moshi-only, hybrid, adaptive
  
  # Audio settings
  audio:
    sample_rate: 24000
    channels: 1
    chunk_size: 1024
    audio_format: "float32"
    enable_noise_reduction: true
    enable_echo_cancellation: true
    input_device: null  # null for default
    output_device: null  # null for default
  
  # Moshi configuration
  moshi:
    model_path: "models/moshi"
    device: "cuda"  # cuda, cpu, auto
    precision: "float16"  # float32, float16, bfloat16
    batch_size: 1
    max_sequence_length: 2048
  
  # Voice optimization
  optimization:
    enable_vram_management: true
    max_vram_usage_gb: 8
    enable_audio_buffer_pooling: true
    enable_response_caching: true

# Memory System Configuration
memory:
  provider: "chromadb"  # chromadb, sqlite, postgresql
  storage_path: "data/memory"
  
  # Short-term memory
  short_term:
    max_turns: 20
    cleanup_interval: 300  # seconds
  
  # Long-term memory
  long_term:
    max_memories: 10000
    enable_auto_learning: true
    similarity_threshold: 0.7
    max_search_results: 10
  
  # Embedding configuration
  embedding:
    model: "all-MiniLM-L6-v2"
    device: "cpu"  # cpu, cuda
    batch_size: 32
    normalize_embeddings: true
  
  # Memory optimization
  optimization:
    enable_compression: true
    cleanup_interval: 3600  # seconds
    max_memory_usage_mb: 1000

# Personality System Configuration
personality:
  enabled: true
  
  # Core personality traits
  traits:
    friendliness: 0.8      # 0.0 - 1.0
    formality: 0.3         # 0.0 - 1.0
    enthusiasm: 0.7        # 0.0 - 1.0
    helpfulness: 0.9       # 0.0 - 1.0
    creativity: 0.6        # 0.0 - 1.0
    assertiveness: 0.5     # 0.0 - 1.0
    empathy: 0.8          # 0.0 - 1.0
    humor: 0.4            # 0.0 - 1.0
    curiosity: 0.7        # 0.0 - 1.0
    patience: 0.8         # 0.0 - 1.0
  
  # Adaptation settings
  adaptation:
    enable_learning: true
    learning_rate: 0.1
    feedback_weight: 0.3
    context_weight: 0.7
    max_adaptation_per_session: 0.2
  
  # Behavioral conditioning
  conditioning:
    enable_positive_reinforcement: true
    enable_negative_feedback: true
    reinforcement_strength: 0.1

# Tools System Configuration
tools:
  enabled: true
  
  # Tool categories to enable
  categories:
    - "utility"      # time, date, calculator, etc.
    - "memory"       # memory operations
    - "system"       # system information
    # - "web"        # web search, browsing
    # - "file"       # file operations
    # - "code"       # code execution
  
  # Safety settings
  safety:
    enable_dangerous_tools: false
    require_confirmation: true
    sandbox_execution: true
    max_execution_time: 30  # seconds
  
  # Tool optimization
  optimization:
    enable_caching: true
    cache_ttl: 1800  # seconds
    max_concurrent_executions: 5

# WebSocket Server Configuration
websocket:
  enabled: true
  host: "localhost"
  port: 8765
  
  # Connection settings
  connection:
    max_connections: 100
    ping_interval: 30  # seconds
    ping_timeout: 10   # seconds
    close_timeout: 10  # seconds
  
  # Message settings
  message:
    max_size: 1048576  # 1MB
    enable_compression: true
    compression_threshold: 1024  # bytes
  
  # Security settings
  security:
    enable_cors: true
    allowed_origins: ["*"]  # Restrict in production
    enable_auth: false
    auth_token: null

# Dashboard Configuration
dashboard:
  enabled: true
  host: "localhost"
  port: 8080
  
  # UI settings
  ui:
    title: "Coda Dashboard"
    theme: "dark"  # light, dark, auto
    enable_voice_interface: true
    enable_text_interface: true
    show_performance_metrics: true
  
  # Static file serving
  static:
    enable_caching: true
    cache_ttl: 3600  # seconds
    compression: true
  
  # Security settings
  security:
    enable_cors: true
    allowed_origins: ["*"]  # Restrict in production
    enable_auth: false
    auth_token: null

# Performance Optimization Configuration
performance:
  optimization_level: "balanced"  # conservative, balanced, aggressive
  
  # Performance targets
  targets:
    max_response_time_ms: 2000
    max_memory_usage_percent: 80
    max_cpu_usage_percent: 70
    max_gpu_memory_percent: 85
    min_cache_hit_rate: 0.7
    target_throughput_rps: 50
  
  # Connection pooling
  connection_pooling:
    max_connections: 100
    min_connections: 10
    connection_timeout: 30
    idle_timeout: 300
    health_check_interval: 60
  
  # Caching configuration
  caching:
    max_memory_mb: 500
    default_ttl: 300
    cleanup_interval: 60
    max_entries: 10000
    enable_compression: true

# Error Handling & Recovery Configuration
error_handling:
  enabled: true
  
  # Error classification
  classification:
    enable_auto_classification: true
    enable_pattern_matching: true
    enable_severity_assessment: true
  
  # Recovery settings
  recovery:
    enable_auto_recovery: true
    max_retry_attempts: 3
    retry_delay: 1.0  # seconds
    enable_component_restart: true
    restart_cooldown: 30  # seconds
  
  # User interface
  user_interface:
    enable_friendly_messages: true
    show_technical_details: false
    enable_suggestions: true

# Logging Configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  # File logging
  file:
    enabled: true
    path: "logs/coda.log"
    max_size_mb: 100
    backup_count: 5
    rotation: "time"  # size, time
  
  # Console logging
  console:
    enabled: true
    level: "INFO"
    colored: true
  
  # Component-specific logging
  components:
    voice: "INFO"
    llm: "INFO"
    memory: "INFO"
    personality: "INFO"
    tools: "INFO"
    websocket: "INFO"
    dashboard: "INFO"

# Session Management Configuration
sessions:
  storage_path: "data/sessions"
  
  # Session settings
  settings:
    max_sessions: 1000
    session_timeout: 3600  # seconds
    cleanup_interval: 300  # seconds
    enable_persistence: true
  
  # Session data
  data:
    max_history_length: 100
    enable_compression: true
    auto_save_interval: 60  # seconds

# Integration Configuration
integrations:
  # Component integration settings
  components:
    enable_voice_memory: true
    enable_voice_personality: true
    enable_voice_tools: true
    enable_voice_llm: true
    enable_memory_personality: true
  
  # Event coordination
  events:
    enable_event_bus: true
    max_event_queue_size: 1000
    event_timeout: 5  # seconds
  
  # WebSocket integration
  websocket_integration:
    enable_component_events: true
    enable_performance_events: true
    enable_error_events: true
    event_batch_size: 10
    event_batch_timeout: 1  # seconds
```

## 🎯 Environment-Specific Configurations

### Development Configuration

```yaml
# configs/development.yaml
system:
  environment: "development"
  debug: true
  log_level: "DEBUG"

llm:
  model: "qwen3:7b"  # Smaller model for development
  generation:
    max_tokens: 512

voice:
  enabled: false  # Disable voice for faster development

memory:
  long_term:
    max_memories: 1000  # Smaller for development

performance:
  optimization_level: "conservative"

logging:
  console:
    level: "DEBUG"
  components:
    voice: "DEBUG"
    llm: "DEBUG"
```

### Production Configuration

```yaml
# configs/production.yaml
system:
  environment: "production"
  debug: false
  log_level: "INFO"

llm:
  model: "qwen3:30b-a3b"  # Full model for production
  optimization:
    enable_caching: true
    max_concurrent_requests: 5

voice:
  enabled: true
  mode: "adaptive"
  optimization:
    enable_vram_management: true

memory:
  long_term:
    max_memories: 50000  # Large capacity for production

performance:
  optimization_level: "aggressive"
  targets:
    max_response_time_ms: 1000

dashboard:
  security:
    allowed_origins: ["https://yourdomain.com"]
    enable_auth: true

websocket:
  security:
    allowed_origins: ["https://yourdomain.com"]
    enable_auth: true
```

### Minimal Configuration

```yaml
# configs/minimal.yaml
system:
  environment: "minimal"
  log_level: "WARNING"

llm:
  provider: "ollama"
  model: "qwen3:7b"

voice:
  enabled: false

memory:
  provider: "sqlite"
  storage_path: "data/memory.db"

tools:
  categories: ["utility"]

dashboard:
  enabled: false

websocket:
  enabled: false

performance:
  optimization_level: "conservative"
```

## ⚙️ Hardware-Specific Optimizations

### High-End GPU Configuration (RTX 4090/5090)

```yaml
voice:
  moshi:
    device: "cuda"
    precision: "float16"
    batch_size: 4

llm:
  optimization:
    enable_memory_optimization: false  # Plenty of VRAM

performance:
  targets:
    max_gpu_memory_percent: 90
  optimization_level: "aggressive"
```

### Mid-Range GPU Configuration (RTX 3060/4060)

```yaml
voice:
  moshi:
    device: "cuda"
    precision: "float16"
    batch_size: 1

llm:
  optimization:
    enable_memory_optimization: true

performance:
  targets:
    max_gpu_memory_percent: 80
  optimization_level: "balanced"
```

### CPU-Only Configuration

```yaml
voice:
  enabled: false  # Voice processing requires GPU

llm:
  model: "qwen3:7b"  # Smaller model for CPU

memory:
  embedding:
    device: "cpu"

performance:
  optimization_level: "conservative"
  targets:
    max_cpu_usage_percent: 80
```

## 🔧 Advanced Configuration

### Custom Tool Configuration

```yaml
tools:
  custom_tools:
    - name: "my_custom_tool"
      module: "my_tools.custom"
      enabled: true
      config:
        api_key: "${MY_API_KEY}"
        timeout: 30
```

### Multi-Model LLM Configuration

```yaml
llm:
  providers:
    primary:
      provider: "ollama"
      model: "qwen3:30b-a3b"
    fallback:
      provider: "openai"
      model: "gpt-4"
      api_key: "${OPENAI_API_KEY}"
```

### Advanced Memory Configuration

```yaml
memory:
  providers:
    short_term:
      provider: "redis"
      host: "localhost"
      port: 6379
    long_term:
      provider: "postgresql"
      host: "localhost"
      database: "coda_memory"
      username: "${DB_USER}"
      password: "${DB_PASSWORD}"
```

## 🔒 Security Configuration

### Production Security Settings

```yaml
dashboard:
  security:
    enable_auth: true
    auth_token: "${DASHBOARD_TOKEN}"
    allowed_origins: ["https://yourdomain.com"]
    enable_https: true
    ssl_cert: "/path/to/cert.pem"
    ssl_key: "/path/to/key.pem"

websocket:
  security:
    enable_auth: true
    auth_token: "${WEBSOCKET_TOKEN}"
    allowed_origins: ["https://yourdomain.com"]
    enable_wss: true

tools:
  safety:
    enable_dangerous_tools: false
    require_confirmation: true
    sandbox_execution: true
```

## 📊 Monitoring Configuration

### Performance Monitoring

```yaml
monitoring:
  enabled: true
  
  metrics:
    enable_system_metrics: true
    enable_component_metrics: true
    enable_performance_metrics: true
    collection_interval: 30  # seconds
  
  alerts:
    enable_alerts: true
    thresholds:
      high_memory_usage: 85
      high_cpu_usage: 80
      slow_response_time: 5000  # ms
  
  export:
    enable_prometheus: false
    prometheus_port: 9090
    enable_grafana: false
```

## 🔄 Configuration Validation

### Validate Configuration

```bash
# Validate configuration file
python scripts/validate_config.py configs/my_config.yaml

# Test configuration
python scripts/test_config.py configs/my_config.yaml

# Generate configuration template
python scripts/generate_config_template.py > configs/template.yaml
```

---

**💡 Pro Tip**: Start with a base configuration and gradually customize it for your specific needs. Use environment variables for sensitive data like API keys.**
