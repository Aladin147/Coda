# Coda Personality System

> **Advanced adaptive personality engine with behavioral conditioning, topic awareness, and personal lore management**

The personality system provides sophisticated personality management for Coda, featuring dynamic trait adjustment, behavioral learning, topic-aware adaptation, and personal lore injection for authentic, evolving interactions.

## Features

- 🧠 **Dynamic Personality Traits** - 10 configurable traits with context-aware adjustments
- 🎓 **Behavioral Learning** - Learns user preferences and adapts personality accordingly
- 🎯 **Topic Awareness** - Detects conversation topics and adjusts personality per context
- 📚 **Personal Lore** - Rich backstory, quirks, and memories for authentic personality
- ⏰ **Session Management** - Intelligent session tracking and closure detection
- ⚡ **Real-time WebSocket Events** - Live personality monitoring and analytics
- 📊 **Comprehensive Analytics** - Deep insights into personality evolution
- 🔧 **Prompt Enhancement** - LLM prompt enhancement with personality context

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Personality    │───▶│  Personality     │───▶│  Behavioral     │
│  Parameters     │    │  Manager         │    │  Conditioner    │
│  (Traits)       │    │  (Orchestrator)  │    │  (Learning)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Topic          │    │  Personal        │    │  Session        │
│  Awareness      │    │  Lore            │    │  Manager        │
│  (Context)      │    │  (Backstory)     │    │  (State)        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### Basic Usage

```python
from coda.components.personality import PersonalityManager, PersonalityConfig

# Create personality manager
config = PersonalityConfig()
personality = PersonalityManager(config)

# Process user input with personality adaptation
result = await personality.process_user_input("I'm learning Python programming")
print(f"Topic detected: {result['topic_context']['current_topic']}")
print(f"Personality adjustments: {len(result['personality_adjustments'])}")

# Enhance prompts with personality
enhanced_prompt = await personality.enhance_prompt(
    "You are a helpful assistant.",
    "technical programming discussion"
)

# Process user feedback for learning
feedback_result = await personality.process_feedback("Please be more detailed")
print(f"Learning applied: {feedback_result['learning_updated']}")

# Get personality state
state = personality.get_personality_state()
print(f"Current verbosity: {state['parameters']['verbosity']['value']}")
```

### WebSocket Integration

```python
from coda.components.personality import WebSocketPersonalityManager
from coda.interfaces.websocket import CodaWebSocketServer, CodaWebSocketIntegration

# Set up WebSocket server
server = CodaWebSocketServer()
integration = CodaWebSocketIntegration(server)

# Create WebSocket-enabled personality manager
personality = WebSocketPersonalityManager(config)
await personality.set_websocket_integration(integration)

# All personality operations now broadcast real-time events
await personality.process_user_input("Hello!")  # Broadcasts topic_detected, trait_adjustment events
```

## Personality Traits

The system manages 10 core personality traits:

### Core Traits

| Trait | Description | Context Adjustments |
|-------|-------------|-------------------|
| **Verbosity** | Controls response length and detail | Technical: +0.2, Casual: -0.1 |
| **Assertiveness** | Controls confidence and directness | Emergency: +0.3, Personal: -0.1 |
| **Humor** | Controls humor and playfulness | Entertainment: +0.3, Technical: -0.2 |
| **Formality** | Controls language formality | Professional: +0.3, Casual: -0.3 |
| **Proactivity** | Controls tendency to offer additional help | Educational: +0.2, Emergency: +0.4 |
| **Confidence** | Controls confidence in responses | Technical: +0.2, Creative: -0.1 |
| **Empathy** | Controls emotional awareness | Personal: +0.2, Technical: -0.2 |
| **Creativity** | Controls creative and imaginative responses | Creative: +0.3, Formal: -0.2 |
| **Analytical** | Controls logical approach | Technical: +0.3, Entertainment: -0.1 |
| **Enthusiasm** | Controls energy and excitement | Entertainment: +0.3, Formal: -0.2 |

### Trait Management

```python
from coda.components.personality import PersonalityParameterManager

# Create parameter manager
params = PersonalityParameterManager()

# Adjust traits manually
adjustment = params.adjust_trait(
    PersonalityTraitType.HUMOR,
    0.2,  # Increase humor by 0.2
    "User requested more humor",
    confidence=0.9
)

# Apply context adjustments
adjustments = params.apply_context_adjustments("technical")
print(f"Technical context adjustments: {adjustments}")

# Reset traits
params.reset_trait(PersonalityTraitType.VERBOSITY)
params.reset_all_traits()  # Reset everything
```

## Behavioral Learning

The system learns user preferences through interaction patterns and explicit feedback:

### Learning Mechanisms

```python
from coda.components.personality import BehavioralConditioner

conditioner = BehavioralConditioner()

# Process user input for implicit learning
result = await conditioner.process_user_input("Please be more concise")
print(f"Detected preferences: {result['explicit_patterns']}")

# Process explicit feedback
feedback_result = await conditioner.process_user_feedback(
    "I love your detailed explanations!",
    feedback_type="style"
)

# Analyze patterns
patterns = await conditioner.analyze_interaction_patterns()
print(f"Detected {len(patterns)} behavior patterns")

# Get learned preferences
profile = conditioner.get_behavior_profile()
print(f"Learned preferences: {profile.user_preferences}")
```

### Learning Features

- **Explicit Preference Detection** - Recognizes direct requests ("be more brief")
- **Implicit Pattern Recognition** - Learns from interaction styles and engagement
- **Confidence Scoring** - Weights learning based on signal strength
- **Feedback Integration** - Incorporates user feedback with high confidence
- **Pattern Analysis** - Detects engagement, length preferences, style preferences

## Topic Awareness

Automatically detects conversation topics and adjusts personality accordingly:

### Topic Categories

- **Technical** - Programming, algorithms, technical discussions
- **Creative** - Art, writing, design, imagination
- **Educational** - Learning, teaching, explanations
- **Entertainment** - Games, fun, humor, leisure
- **Personal** - Emotions, relationships, personal experiences
- **Professional** - Work, business, career, formal contexts
- **Casual** - Daily life, weather, general conversation
- **Emergency** - Urgent help, problems, critical issues

### Topic Detection

```python
from coda.components.personality import TopicAwareness

topic_awareness = TopicAwareness()

# Detect topic in text
context = await topic_awareness.detect_topic("How do I write Python code?")
print(f"Topic: {context.current_topic}")
print(f"Category: {context.category}")
print(f"Confidence: {context.confidence}")

# Get personality adjustments for topic
adjustments = await topic_awareness.get_topic_personality_adjustments(context)
print(f"Topic adjustments: {adjustments}")
```

## Personal Lore

Rich backstory and personality elements for authentic interactions:

### Lore Components

- **Backstory** - Origin, purpose, development, values, aspirations
- **Traits** - Core personality characteristics
- **Quirks** - Contextual personality expressions
- **Memories** - Formative experiences and learnings
- **Preferences** - Communication and interaction styles
- **Anchor Phrases** - Context-specific expressions

### Lore Management

```python
from coda.components.personality import PersonalLoreManager

lore_manager = PersonalLoreManager()

# Get relevant lore for context
relevant_lore = lore_manager.get_relevant_lore("technical", ["programming", "code"])
print(f"Found {len(relevant_lore)} relevant lore elements")

# Add new lore element
lore_manager.add_lore_element(
    "quirk",
    "I get excited about elegant algorithms",
    ["algorithm", "elegant", "code"],
    importance=0.7
)

# Enhance prompts with lore
enhanced = lore_manager.enhance_prompt(
    "You are a helpful assistant.",
    "technical",
    ["programming"]
)
```

## Session Management

Intelligent session tracking and closure detection:

### Session Features

```python
from coda.components.personality import SessionManager

session_manager = SessionManager()

# Process interactions
session_manager.process_interaction("user", "Hello")
session_manager.process_interaction("assistant", "Hi there!")

# Check session state
state = session_manager.get_session_state()
print(f"Session duration: {state.duration_seconds}s")
print(f"Turn count: {state.turn_count}")

# Check for closure
should_close, reason = session_manager.should_enter_closure_mode()
if should_close:
    adjustments = session_manager.enter_closure_mode(reason)
    print(f"Entering closure mode: {reason}")

# Generate session summary
summary = session_manager.generate_session_summary()
print(f"Session quality: {summary['session_quality']}")
```

## WebSocket Events

When using `WebSocketPersonalityManager`, the following events are broadcast:

- `personality_topic_detected` - Topic detection and changes
- `personality_trait_adjustment` - Personality trait adjustments
- `personality_behavior_patterns_detected` - Behavioral pattern detection
- `personality_feedback_processed` - User feedback processing
- `personality_session_closure` - Session closure events
- `personality_state_update` - Complete personality state updates
- `personality_analytics_update` - Analytics and insights
- `personality_lore_updated` - Personal lore updates

## Configuration

### Complete Configuration

```python
from coda.components.personality import (
    PersonalityConfig,
    PersonalityParameterConfig,
    BehavioralConditioningConfig,
    TopicAwarenessConfig,
    PersonalLoreConfig,
    SessionManagerConfig
)

config = PersonalityConfig(
    parameters=PersonalityParameterConfig(
        # Custom trait defaults and context adjustments
    ),
    behavioral_conditioning=BehavioralConditioningConfig(
        learning_rate=0.1,
        confidence_threshold=0.7,
        max_recent_interactions=20
    ),
    topic_awareness=TopicAwarenessConfig(
        confidence_threshold=0.5,
        max_topic_history=10
    ),
    personal_lore=PersonalLoreConfig(
        lore_injection_probability=0.3,
        max_lore_elements_per_response=2
    ),
    session_manager=SessionManagerConfig(
        long_session_threshold_minutes=30,
        idle_threshold_minutes=10
    ),
    websocket_events_enabled=True,
    analytics_enabled=True
)
```

## Testing

### Run Unit Tests
```bash
pytest tests/unit/test_personality_system.py -v
```

### Run Demo
```bash
python scripts/personality_demo.py
```

## Advanced Features

### Personality Analytics

```python
# Get comprehensive analytics
analytics = personality.get_analytics()

# Trait evolution over time
for trait, info in analytics['parameters'].items():
    if info['adjustments'] > 0:
        print(f"{trait}: {info['value']:.2f} ({info['adjustments']} adjustments)")

# Behavioral insights
behavior_stats = analytics['behavior_learning']
print(f"Learning confidence: {behavior_stats['average_confidence']:.2f}")

# Session patterns
session_analytics = analytics['session_analytics']
print(f"Conversation momentum: {session_analytics['session_flow']['conversation_momentum']:.2f}")
```

### State Persistence

```python
# Export personality state
exported_state = personality.export_personality_state()

# Import to new personality instance
new_personality = PersonalityManager()
success = new_personality.import_personality_state(exported_state)
```

### Prompt Enhancement

```python
from coda.components.personality import PersonalityPromptEnhancer

enhancer = PersonalityPromptEnhancer()

# Different enhancement levels
enhanced_minimal = enhancer.enhance_system_prompt(
    base_prompt, personality_params, lore, topic_context, session_state,
    enhancement_level="minimal"
)

enhanced_full = enhancer.enhance_system_prompt(
    base_prompt, personality_params, lore, topic_context, session_state,
    enhancement_level="full"
)
```

## Migration from Coda Lite

Key improvements over the original implementation:

✅ **Type-safe models** with Pydantic validation  
✅ **Modern async/await** architecture  
✅ **Comprehensive behavioral learning** with confidence scoring  
✅ **Advanced topic detection** with category-based adjustments  
✅ **Rich personal lore system** with context-aware injection  
✅ **Intelligent session management** with closure detection  
✅ **WebSocket integration** for real-time monitoring  
✅ **Comprehensive analytics** and explainability  
✅ **Prompt enhancement** utilities for LLM integration  
✅ **Extensive test coverage** and documentation  

## Performance Considerations

- **Learning Rate**: Adjust `learning_rate` for faster/slower adaptation
- **Confidence Thresholds**: Higher thresholds = more conservative learning
- **Context Sensitivity**: Balance topic detection sensitivity vs. stability
- **Lore Injection**: Control frequency to avoid overwhelming responses
- **Session Tracking**: Monitor session analytics for optimal thresholds

## Next Steps

- [ ] Advanced emotion detection and empathy modeling
- [ ] Multi-user personality profiles and adaptation
- [ ] Long-term personality evolution tracking
- [ ] Integration with external personality frameworks
- [ ] Advanced natural language understanding for better topic detection
- [ ] Personality A/B testing and optimization

## Dependencies

- `pydantic` - Data validation and settings
- `asyncio` - Asynchronous operations
- `datetime` - Time-based operations
- `collections` - Data structures
- `re` - Regular expressions for pattern matching
