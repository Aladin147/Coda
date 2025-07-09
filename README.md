# Coda - Core Operations & Digital Assistant

> **Next-generation, local-first voice assistant built for real-time interaction and extensibility**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Vision

Coda is evolving from a voice assistant into a **comprehensive digital operator** that can:

- 🗣️ **Real-time voice interaction** with sub-second latency
- 🧠 **Advanced memory systems** with long-term learning
- 🎭 **Adaptive personality** that learns from user feedback  
- 🔧 **Extensible tool ecosystem** for complex task execution
- 🔒 **Privacy-first design** with local processing
- 🌐 **Multi-modal capabilities** (voice, text, vision planned)

## ⚡ Key Features

### **Real-Time Voice Processing**
- **Streaming STT/TTS** with Kyutai integration (planned)
- **Semantic Voice Activity Detection** for natural conversation flow
- **Interruption handling** with precise resume capabilities
- **Voice cloning** from short audio samples

### **Intelligent Memory System**
- **Short-term memory** for conversation context
- **Long-term memory** with vector embeddings and semantic search
- **Memory consolidation** across sessions
- **User preference learning** and adaptation

### **Advanced Personality Engine**
- **Behavioral conditioning** based on user feedback
- **Context-aware responses** with topic tracking
- **Personal lore system** for consistent character development
- **Dynamic personality adjustment** over time

### **Extensible Architecture**
- **WebSocket-based** event-driven system
- **Plugin-ready tool system** for easy extension
- **Real-time dashboard** for monitoring and interaction
- **Production-ready** with batching and scaling support

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd coda

# Install dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"

# Run initial setup
python scripts/setup.py
```

### Basic Usage

```bash
# Start Coda with default configuration
python -m coda

# Start with WebSocket dashboard
python -m coda --dashboard

# Start with custom configuration
python -m coda --config configs/custom.yaml
```

## 📁 Project Structure

```
coda/
├── src/coda/                 # Main package
│   ├── core/                 # Core assistant logic
│   ├── components/           # Modular components
│   │   ├── memory/           # Memory management
│   │   ├── personality/      # Personality engine
│   │   ├── voice/            # STT/TTS processing
│   │   └── tools/            # Tool ecosystem
│   └── interfaces/           # External interfaces
│       ├── websocket/        # WebSocket server
│       └── dashboard/        # Web dashboard
├── tests/                    # Test suite
├── docs/                     # Documentation
├── configs/                  # Configuration files
└── scripts/                  # Utility scripts
```

## 🛣️ Roadmap

### **Phase 1: Foundation (Current)**
- [x] Core architecture migration from Coda Lite
- [x] Modern project structure and tooling
- [ ] Critical bug fixes and cleanup
- [ ] Comprehensive documentation

### **Phase 2: Kyutai Integration (Next)**
- [ ] Real-time STT with Kyutai models
- [ ] Streaming TTS with voice cloning
- [ ] Sub-second latency optimization
- [ ] Multi-user support with batching

### **Phase 3: Advanced Features (Future)**
- [ ] Multi-modal input (vision, documents)
- [ ] Advanced tool orchestration
- [ ] Proactive assistance capabilities
- [ ] Cross-platform deployment

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/development/contributing.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black src/ tests/
```

## 📚 Documentation

- [Architecture Overview](docs/architecture/overview.md)
- [Component Guide](docs/components/)
- [API Reference](docs/api/)
- [Deployment Guide](docs/deployment/)

## 🙏 Acknowledgments

Built on the foundation of Coda Lite and inspired by the open-source community. Special thanks to:

- [Kyutai Labs](https://kyutai.org/) for revolutionary voice processing models
- [Ollama](https://ollama.com/) for local LLM infrastructure
- The broader AI and voice assistant research community

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Coda** - Where voice meets intelligence, locally and privately.
