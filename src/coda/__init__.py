"""
Coda - Core Operations & Digital Assistant

A next-generation, local-first voice assistant built for real-time interaction and extensibility.
"""

__version__ = "2.0.0-alpha"
__author__ = "Coda Development Team"
__email__ = "dev@coda-assistant.com"
__description__ = "Core Operations & Digital Assistant - Next-generation local-first voice assistant"

# Core imports
from .core.assistant import CodaAssistant
from .core.config import Config, load_config
from .core.events import EventBus, Event

# Component imports
from .components.memory import MemoryManager
from .components.personality import PersonalityEngine
from .components.voice import VoiceProcessor
from .components.tools import ToolManager

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
    "VoiceProcessor",
    "ToolManager",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__description__",
]
