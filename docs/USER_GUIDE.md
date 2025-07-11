# Coda User Guide

Welcome to Coda - your next-generation, local-first voice assistant! This guide will help you get started and make the most of Coda's powerful features.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Voice Conversations](#voice-conversations)
5. [Memory System](#memory-system)
6. [Personality Adaptation](#personality-adaptation)
7. [Tools and Functions](#tools-and-functions)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Features](#advanced-features)

## Quick Start

### Prerequisites

- **Python 3.10 or higher**
- **8GB+ RAM** (16GB recommended for voice features)
- **CUDA-capable GPU** (optional but recommended for voice processing)
- **Internet connection** (for initial model downloads)

### 5-Minute Setup

```bash
# 1. Clone and install
git clone <your-repo-url>
cd coda
pip install -e .

# 2. Run setup
python scripts/setup.py

# 3. Start Coda
python -m coda

# 4. Try it out!
# Coda will start in text mode by default
```

## Installation

### Standard Installation

```bash
# Install Coda with basic features
pip install -e .

# Install with voice processing support
pip install -e ".[voice]"

# Install with all features (development)
pip install -e ".[dev]"
```

### GPU Setup (Recommended)

For optimal voice processing performance:

```bash
# Install CUDA support (if you have an NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Environment Setup

```bash
# Create isolated environment
python -m venv coda_env
source coda_env/bin/activate  # On Windows: coda_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### Starting Coda

```bash
# Start with default settings
python -m coda

# Start with custom configuration
python -m coda --config configs/custom.yaml

# Start with WebSocket dashboard
python -m coda --dashboard

# Start in voice mode (requires GPU)
python -m coda --voice
```

### First Conversation

Once Coda starts, you can begin chatting immediately:

```
You: Hello Coda, how are you today?
Coda: Hello! I'm doing well, thank you for asking. I'm ready to help you with whatever you need. What would you like to talk about or work on today?

You: Can you remember that I prefer morning meetings?
Coda: Absolutely! I've noted that you prefer morning meetings. I'll keep this in mind for any scheduling-related discussions we have in the future.

You: What's 15 multiplied by 23?
Coda: Let me calculate that for you. 15 × 23 = 345.
```

## Voice Conversations

### Enabling Voice Mode

Voice processing requires additional setup:

```bash
# Check if voice is available
python -c "from src.coda.components.voice import VoiceManager; print('Voice available!')"

# Start with voice enabled
python -m coda --voice --mode adaptive
```

### Voice Processing Modes

Coda offers multiple voice processing modes:

- **Adaptive** (Default): Automatically selects the best mode based on context
- **Moshi-only**: Fastest response (100-200ms), natural speech patterns
- **LLM-enhanced**: Highest quality (500-1500ms), advanced reasoning
- **Hybrid**: Balanced approach, uses both systems in parallel

```bash
# Start with specific voice mode
python -m coda --voice --mode moshi-only    # Fastest
python -m coda --voice --mode llm-enhanced  # Highest quality
python -m coda --voice --mode hybrid        # Balanced
```

### Voice Commands

When in voice mode, you can:

- **Speak naturally**: Just talk to Coda as you would to a person
- **Use wake words**: "Hey Coda" or "Coda" to get attention
- **Control conversation**: "Stop", "Pause", "Continue"
- **Switch modes**: "Use fast mode" or "Use high quality mode"

## Memory System

Coda's memory system learns from your interactions and remembers important information.

### What Coda Remembers

- **Preferences**: Your likes, dislikes, and preferred ways of working
- **Facts**: Important information you share about yourself or your work
- **Conversations**: Context from previous discussions
- **Patterns**: How you like to communicate and what topics interest you

### Memory Examples

```
You: I'm a software engineer working on machine learning projects
Coda: That's interesting! I'll remember that you're a software engineer focused on ML. 

[Later conversation]
You: Can you help me with a coding problem?
Coda: Of course! Given your background in software engineering and machine learning, I'd be happy to help. What specific coding challenge are you working on?
```

### Managing Memory

```bash
# View memory statistics
python scripts/memory_demo.py --stats

# Clear old memories
python scripts/memory_demo.py --cleanup

# Export memories
python scripts/memory_demo.py --export memories.json
```

## Personality Adaptation

Coda's personality adapts to your communication style and preferences over time.

### How Personality Works

- **Initial State**: Coda starts with a balanced, helpful personality
- **Learning**: Adapts based on your feedback and interaction patterns
- **Traits**: Adjusts traits like formality, enthusiasm, technical depth
- **Context**: Different personality aspects for different topics

### Personality Examples

```
# Technical Discussion
You: Explain neural networks
Coda: Neural networks are computational models inspired by biological neural networks...

# Casual Chat
You: How's your day going?
Coda: Pretty good! I've been helping folks with various projects today. How about you?

# After feedback that you prefer concise answers
You: What's machine learning?
Coda: ML is algorithms that learn patterns from data to make predictions or decisions.
```

### Providing Feedback

Help Coda learn your preferences:

```
You: That explanation was too technical
Coda: Got it! I'll aim for simpler explanations in the future.

You: I love detailed examples
Coda: Perfect! I'll include more detailed examples when explaining concepts.

You: Be more casual in our conversations
Coda: Sure thing! I'll keep things more relaxed and conversational.
```

## Tools and Functions

Coda can use various tools to help you accomplish tasks.

### Available Tools

- **Calculator**: Math calculations and conversions
- **Web Search**: Find information online
- **File Operations**: Read, write, and manage files
- **System Info**: Check system status and resources
- **Time/Date**: Current time, scheduling, reminders
- **Code Execution**: Run code snippets safely

### Using Tools

Tools are automatically suggested and used based on your requests:

```
You: What time is it?
Coda: [Using time tool] It's currently 2:30 PM on Tuesday, March 15th.

You: Calculate the square root of 144
Coda: [Using calculator] The square root of 144 is 12.

You: Search for Python tutorials
Coda: [Using web search] I found several great Python tutorials for you...
```

### Custom Tools

You can add your own tools:

```python
# Create custom tool
from coda.components.tools import Tool

@Tool.register("my_custom_tool")
async def my_tool(parameter: str) -> str:
    """My custom tool description."""
    return f"Processed: {parameter}"
```

## Configuration

### Configuration Files

Coda uses YAML configuration files:

```yaml
# configs/my_config.yaml
voice:
  enabled: true
  mode: "adaptive"
  audio:
    sample_rate: 24000
    enable_noise_reduction: true

memory:
  max_memories: 10000
  enable_auto_learning: true

personality:
  adaptation_rate: 0.1
  enable_learning: true

llm:
  provider: "ollama"
  model: "gemma2:2b"
  temperature: 0.8
```

### Environment Variables

```bash
# Set configuration via environment
export CODA_CONFIG_PATH="configs/my_config.yaml"
export CODA_LOG_LEVEL="DEBUG"
export CODA_VOICE_ENABLED="true"
export CODA_GPU_MEMORY_LIMIT="8GB"
```

### Runtime Configuration

```python
# Programmatic configuration
from coda import Coda, CodaConfig

config = CodaConfig(
    voice_enabled=True,
    memory_max_size=10000,
    personality_adaptation_rate=0.1
)

coda = Coda(config)
await coda.start()
```

## Troubleshooting

### Common Issues

**Voice not working:**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Check voice dependencies
pip install -e ".[voice]"

# Start without voice
python -m coda --no-voice
```

**Memory issues:**
```bash
# Clear memory cache
python scripts/memory_demo.py --clear-cache

# Reduce memory limit
export CODA_MEMORY_LIMIT="5000"
```

**Performance issues:**
```bash
# Check system resources
python scripts/validate_system.py

# Reduce concurrent conversations
export CODA_MAX_CONVERSATIONS="5"
```

### Debug Mode

```bash
# Enable debug logging
python -m coda --debug

# Verbose output
python -m coda --verbose

# Profile performance
python -m coda --profile
```

### Getting Help

```bash
# Show help
python -m coda --help

# Check system status
python scripts/validate_system.py

# Run diagnostics
python scripts/run_comprehensive_tests.py
```

## Advanced Features

### WebSocket API

Access Coda programmatically:

```javascript
// Connect to Coda WebSocket
const ws = new WebSocket('ws://localhost:8765');

// Send message
ws.send(JSON.stringify({
    type: 'chat',
    content: 'Hello Coda!'
}));

// Receive response
ws.onmessage = (event) => {
    const response = JSON.parse(event.data);
    console.log(response.content);
};
```

### Batch Processing

Process multiple inputs:

```python
from coda import Coda

coda = Coda()
await coda.start()

# Batch process
inputs = ["Hello", "How are you?", "Goodbye"]
responses = await coda.process_batch(inputs)

for response in responses:
    print(response.content)
```

### Custom Integrations

Integrate with your applications:

```python
from coda.components.memory import MemoryManager
from coda.components.personality import PersonalityManager

# Use components independently
memory = MemoryManager()
await memory.store_memory("Important information")

personality = PersonalityManager()
adapted_response = await personality.adapt_response(
    "Hello", user_preferences
)
```

### Performance Optimization

```yaml
# High-performance configuration
performance:
  enable_caching: true
  batch_size: 100
  max_concurrent: 20
  gpu_memory_fraction: 0.8
  
voice:
  enable_streaming: true
  chunk_size: 1024
  
memory:
  enable_compression: true
  cache_size: 1000
```

## Next Steps

- **Explore Examples**: Check out `examples/` directory for more use cases
- **Read API Docs**: See `docs/api/` for detailed API reference
- **Join Community**: Participate in discussions and contribute
- **Customize**: Create your own tools and integrations

## Support

- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **Issues**: [GitHub Issues](https://github.com/your-repo/coda/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/coda/discussions)

---

**Welcome to the future of personal AI assistance with Coda!** 🚀
