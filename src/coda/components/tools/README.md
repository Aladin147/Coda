# Coda Tool System

> **Advanced extensible tool system with function calling, plugin architecture, and real-time monitoring**

The tool system provides comprehensive tool management for Coda, featuring dynamic tool registration, LLM function calling integration, plugin architecture, and real-time WebSocket monitoring.

## Features

- 🔧 **Dynamic Tool Registration** - Register and manage tools at runtime
- ⚡ **Function Calling Integration** - OpenAI-compatible function calling for LLMs
- 🔌 **Plugin Architecture** - Load tools dynamically from Python files
- 📊 **Performance Monitoring** - Real-time execution metrics and analytics
- 🌐 **WebSocket Integration** - Live tool event broadcasting
- 🛡️ **Security & Validation** - Parameter validation and safe execution
- 🎯 **Category Organization** - Tools organized by functional categories
- 🔄 **Error Handling** - Comprehensive error handling and retry logic

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Tool           │───▶│  Tool            │───▶│  Function       │
│  Registry       │    │  Manager         │    │  Calling        │
│  (Discovery)    │    │  (Orchestrator)  │    │  (LLM Bridge)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Tool           │    │  Plugin          │    │  WebSocket      │
│  Executor       │    │  Loader          │    │  Integration    │
│  (Execution)    │    │  (Dynamic)       │    │  (Events)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### Basic Usage

```python
from coda.components.tools import ToolManager, ToolConfig

# Create tool manager
config = ToolConfig()
tool_manager = ToolManager(config)

# Get available tools
tools = tool_manager.get_available_tools()
print(f"Available tools: {len(tools)}")

# Process function call from LLM
result = await tool_manager.process_function_call({
    "name": "calculator",
    "arguments": '{"expression": "2 + 3 * 4", "precision": 2}'
})
print(f"Result: {result.result}")

# Get function schemas for LLM
schemas = tool_manager.get_function_schemas()
```

### WebSocket Integration

```python
from coda.components.tools import WebSocketToolManager
from coda.interfaces.websocket import CodaWebSocketServer, CodaWebSocketIntegration

# Set up WebSocket server
server = CodaWebSocketServer()
integration = CodaWebSocketIntegration(server)

# Create WebSocket-enabled tool manager
tool_manager = WebSocketToolManager(config)
await tool_manager.set_websocket_integration(integration)

# All tool operations now broadcast real-time events
await tool_manager.process_function_call(function_call)  # Broadcasts events
```

## Built-in Tools

The system includes comprehensive built-in tools across multiple categories:

### Utility Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| **get_time** | Get current time in various formats | format, timezone |
| **get_date** | Get current date in various formats | format, custom_format |
| **random_number** | Generate random numbers | type, min_value, max_value, count |
| **generate_uuid** | Generate UUIDs | version, count, uppercase |
| **timestamp** | Convert between timestamps and dates | operation, datetime_string, timestamp |

### Calculation Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| **calculator** | Perform mathematical calculations | expression, precision |

### Memory Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| **store_memory** | Store information in memory | content, category, tags, importance |
| **search_memory** | Search for relevant memories | query, tags, category, limit |
| **get_memory_stats** | Get memory system statistics | include_categories, include_tags |
| **delete_memory** | Delete specific memories | memory_id, confirm |
| **update_memory** | Update existing memories | memory_id, content, tags, importance |

### System Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| **get_system_info** | Get system information | info_type |
| **get_environment** | Get environment variables | variable_name, include_sensitive |
| **execute_command** | Execute safe system commands | command, args, timeout |
| **get_process_info** | Get running process information | filter_name, limit, include_details |

## Creating Custom Tools

### Basic Tool Creation

```python
from coda.components.tools import BaseTool, ToolDefinition, ToolCategory

class MyCustomTool(BaseTool):
    """A custom tool example."""
    
    def _create_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="my_custom_tool",
            description="Does something useful",
            category=ToolCategory.UTILITY,
            parameters=[
                self.create_string_parameter(
                    "input",
                    "Input text to process",
                    required=True,
                    examples=["hello world"]
                ),
                self.create_integer_parameter(
                    "count",
                    "Number of times to repeat",
                    required=False,
                    default=1,
                    minimum=1,
                    maximum=10
                )
            ],
            examples=[
                {"input": "hello", "count": 3}
            ],
            tags=["custom", "example"],
            timeout_seconds=10.0
        )
    
    async def _execute_impl(self, parameters, context=None):
        input_text = parameters["input"]
        count = parameters.get("count", 1)
        
        return " ".join([input_text] * count)

# Register the tool
tool_manager = ToolManager()
await tool_manager.register_tool(MyCustomTool())
```

### Advanced Tool Features

```python
class AdvancedTool(BaseTool):
    """Advanced tool with context integration."""
    
    def _create_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="advanced_tool",
            description="Advanced tool with memory integration",
            category=ToolCategory.CUSTOM,
            parameters=[
                self.create_string_parameter(
                    "query",
                    "Search query",
                    required=True
                )
            ],
            requires_auth=True,  # Requires authentication
            is_dangerous=False,  # Safe tool
            timeout_seconds=30.0
        )
    
    async def _execute_impl(self, parameters, context=None):
        query = parameters["query"]
        
        # Access memory manager from context
        memory_manager = self.get_context_value(context, "memory_manager")
        if memory_manager:
            # Use memory system
            memories = await memory_manager.search_memories(query=query, limit=5)
            return f"Found {len(memories)} relevant memories"
        
        return f"Processed query: {query}"
```

## Plugin System

### Creating Plugins

Create a Python file with tools:

```python
# my_plugin.py
from coda.components.tools import BaseTool, ToolDefinition, ToolCategory

class PluginTool(BaseTool):
    """A tool from a plugin."""
    
    def _create_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="plugin_tool",
            description="Tool loaded from plugin",
            category=ToolCategory.CUSTOM,
            parameters=[
                self.create_string_parameter(
                    "message",
                    "Message to process",
                    required=True
                )
            ]
        )
    
    async def _execute_impl(self, parameters, context=None):
        return f"Plugin processed: {parameters['message']}"

def get_tools():
    """Return tools provided by this plugin."""
    return [PluginTool()]
```

### Loading Plugins

```python
from coda.components.tools import PluginLoader

plugin_loader = PluginLoader()

# Discover plugins in directory
plugin_files = await plugin_loader.discover_plugins("./plugins")

# Load specific plugin
tools = await plugin_loader.load_plugin("my_plugin.py")

# Register loaded tools
for tool in tools:
    await tool_manager.register_tool(tool)
```

## Function Calling Integration

### OpenAI Function Calling

```python
# Get function schemas for LLM
schemas = tool_manager.get_function_schemas()

# Send to OpenAI
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What time is it?"}],
    functions=schemas,
    function_call="auto"
)

# Process function call
if response.choices[0].message.get("function_call"):
    function_call = response.choices[0].message["function_call"]
    result = await tool_manager.process_function_call(function_call)
    
    # Format result for LLM
    formatted_result = tool_manager.function_calling.format_tool_result(result)
```

### Multiple Function Calls

```python
# Process multiple function calls
function_calls = [
    {"name": "get_time", "arguments": "{}"},
    {"name": "calculator", "arguments": '{"expression": "2+2"}'},
    {"name": "random_number", "arguments": '{"count": 1}'}
]

results = await tool_manager.process_function_calls(function_calls)

for result in results:
    print(f"{result.tool_name}: {result.result}")
```

## WebSocket Events

When using `WebSocketToolManager`, the following events are broadcast:

- `function_call_received` - Function call received from LLM
- `function_call_completed` - Function call execution completed
- `tool_registered` - New tool registered
- `tool_unregistered` - Tool unregistered
- `tool_execution_completed` - Tool execution finished
- `tool_error` - Tool execution error
- `tool_analytics_update` - Analytics data update
- `tool_performance_metrics` - Performance metrics update
- `tool_system_snapshot` - Complete system snapshot

## Configuration

### Complete Configuration

```python
from coda.components.tools import (
    ToolConfig,
    ToolRegistryConfig,
    ToolExecutorConfig
)

config = ToolConfig(
    registry=ToolRegistryConfig(
        auto_discover_plugins=True,
        plugin_directories=["./plugins", "./custom_tools"],
        max_tools=100,
        allow_dangerous_tools=False,
        require_auth_for_dangerous=True
    ),
    executor=ToolExecutorConfig(
        default_timeout_seconds=30.0,
        max_concurrent_executions=10,
        enable_retries=True,
        max_retry_attempts=3,
        execution_logging=True,
        performance_monitoring=True
    ),
    websocket_events_enabled=True,
    analytics_enabled=True,
    memory_integration_enabled=True,
    personality_integration_enabled=True
)
```

## Testing

### Run Unit Tests
```bash
pytest tests/unit/test_tool_system.py -v
```

### Run Demo
```bash
python scripts/tools_demo.py
```

## Performance & Analytics

### Tool Analytics

```python
# Get comprehensive analytics
analytics = tool_manager.get_tool_analytics()

# Registry statistics
registry_stats = analytics["registry"]
print(f"Total tools: {registry_stats['total_tools']}")
print(f"Available tools: {registry_stats['available_tools']}")

# Execution statistics
execution_stats = analytics["execution"]
print(f"Total executions: {execution_stats['total_executions']}")
print(f"Success rate: {execution_stats['success_rate']:.1f}%")
print(f"Average execution time: {execution_stats['average_execution_time_ms']:.2f}ms")

# Performance metrics
performance = analytics["performance"]
for tool_name, metrics in performance.items():
    print(f"{tool_name}: {metrics['average_time_ms']:.2f}ms avg")
```

### Real-time Monitoring

```python
# With WebSocket integration
websocket_tool_manager = WebSocketToolManager(config)

# Broadcast analytics periodically
await websocket_tool_manager.broadcast_tool_analytics()
await websocket_tool_manager.broadcast_performance_metrics()

# Trigger system snapshot
snapshot = await websocket_tool_manager.trigger_tool_snapshot()
```

## Security Considerations

### Safe Tool Execution

- **Parameter Validation**: All parameters are validated against schemas
- **Timeout Protection**: Tools have configurable execution timeouts
- **Dangerous Tool Marking**: Tools can be marked as dangerous
- **Command Restrictions**: System commands are restricted to safe operations
- **Authentication**: Tools can require authentication

### Best Practices

```python
# Mark dangerous tools
ToolDefinition(
    name="dangerous_tool",
    description="A potentially dangerous operation",
    is_dangerous=True,
    requires_auth=True
)

# Validate parameters thoroughly
def _execute_impl(self, parameters, context=None):
    # Always validate input
    input_text = self.validate_string_not_empty(parameters["input"], "input")
    count = self.validate_positive_number(parameters.get("count", 1), "count")
    
    # Sanitize output
    result = process_input(input_text, count)
    return sanitize_tool_result(result)
```

## Migration from Coda Lite

Key improvements over the original implementation:

✅ **Type-safe models** with Pydantic validation  
✅ **Modern async/await** architecture  
✅ **Plugin system** for dynamic tool loading  
✅ **Function calling integration** with OpenAI compatibility  
✅ **WebSocket integration** for real-time monitoring  
✅ **Comprehensive error handling** and retry logic  
✅ **Performance monitoring** and analytics  
✅ **Security features** and parameter validation  
✅ **Memory system integration** for persistent tool data  
✅ **Extensive test coverage** and documentation  

## Advanced Features

### Tool Chaining

```python
# Tools can call other tools through the manager
class ChainedTool(BaseTool):
    async def _execute_impl(self, parameters, context=None):
        tool_manager = self.get_context_value(context, "tool_manager")
        
        # Call another tool
        time_result = await tool_manager.process_function_call({
            "name": "get_time",
            "arguments": "{}"
        })
        
        return f"Current time: {time_result.result}"
```

### Context Integration

```python
# Tools can access memory and personality systems
class ContextAwareTool(BaseTool):
    async def _execute_impl(self, parameters, context=None):
        # Access memory system
        memory_manager = self.get_context_value(context, "memory_manager")
        
        # Access personality system
        personality_manager = self.get_context_value(context, "personality_manager")
        
        # Use context for enhanced functionality
        if memory_manager:
            await memory_manager.store_memory(
                content=f"Tool executed: {self.get_definition().name}",
                category="tool_usage"
            )
        
        return "Context-aware execution completed"
```

## Next Steps

- [ ] Advanced tool composition and chaining
- [ ] Tool marketplace and sharing
- [ ] AI-powered tool discovery and recommendation
- [ ] Integration with external APIs and services
- [ ] Advanced security and sandboxing
- [ ] Tool performance optimization
- [ ] Multi-language tool support

## Dependencies

- `pydantic` - Data validation and settings
- `asyncio` - Asynchronous operations
- `uuid` - Unique identifier generation
- `json` - JSON parsing and formatting
- `logging` - Comprehensive logging
- `importlib` - Dynamic module loading
