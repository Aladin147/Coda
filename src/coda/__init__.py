"""
Coda - Core Operations & Digital Assistant

A next-generation, local-first voice assistant built for real-time interaction and extensibility.
"""

__version__ = "2.0.0-alpha"
__author__ = "Coda Development Team"
__email__ = "dev@coda-assistant.com"
__description__ = (
    "Core Operations & Digital Assistant - Next-generation local-first voice assistant"
)

# Core imports (optional - may not exist yet)
try:
    from .core.assistant import CodaAssistant
except ImportError:
    CodaAssistant = None

try:
    from .core.config import Config, load_config
except ImportError:
    Config = None
    load_config = None

try:
    from .core.events import Event, EventBus
except ImportError:
    EventBus = None
    Event = None

# Component imports
try:
    from .components.memory import MemoryManager
except ImportError:
    MemoryManager = None

try:
    from .components.personality import PersonalityEngine
except ImportError:
    PersonalityEngine = None

try:
    from .components.voice import VoiceManager
except ImportError:
    VoiceManager = None

# Temporarily disable tools import
# try:
#     from .components.tools import ToolManager
# except ImportError:
ToolManager = None

__all__ = [
    # Core
    "CodaAssistant",
    "Config",
    "load_config",
    "EventBus",
    "Event",
    # Components
    "MemoryManager",
    "PersonalityEngine",
    "VoiceManager",
    "ToolManager",
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__description__",
]
