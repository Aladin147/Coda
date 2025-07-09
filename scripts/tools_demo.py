#!/usr/bin/env python3
"""
Tool system demonstration script.

This script demonstrates the tool system functionality by:
1. Creating a tool manager with all built-in tools
2. Demonstrating function calling and tool execution
3. Showing WebSocket integration capabilities
4. Testing plugin loading and management
5. Displaying tool analytics and performance metrics
"""

import asyncio
import logging
import sys
import json
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coda.components.tools.models import ToolConfig, ToolCategory
from coda.components.tools.manager import ToolManager
from coda.components.tools.websocket_integration import WebSocketToolManager
from coda.components.tools.plugin_loader import PluginLoader
from coda.interfaces.websocket.server import CodaWebSocketServer
from coda.interfaces.websocket.integration import CodaWebSocketIntegration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tools_demo")


async def demonstrate_basic_tools():
    """Demonstrate basic tool functionality."""
    logger.info("🔧 Starting basic tool demonstration...")
    
    # Create tool manager
    config = ToolConfig()
    tool_manager = ToolManager(config)
    
    # Get available tools
    tools = tool_manager.get_available_tools()
    logger.info(f"📋 Available tools: {len(tools)}")
    
    for tool in tools[:5]:  # Show first 5 tools
        logger.info(f"  - {tool.name} ({tool.category.value}): {tool.description}")
    
    # Demonstrate function calling
    logger.info("⚡ Demonstrating function calls...")
    
    # Time tool
    logger.info("🕐 Getting current time...")
    result = await tool_manager.process_function_call({
        "name": "get_time",
        "arguments": '{"format": "24h"}'
    })
    logger.info(f"Time result: {result.result}")
    
    # Date tool
    logger.info("📅 Getting current date...")
    result = await tool_manager.process_function_call({
        "name": "get_date",
        "arguments": '{"format": "long"}'
    })
    logger.info(f"Date result: {result.result}")
    
    # Calculator tool
    logger.info("🧮 Performing calculations...")
    calculations = [
        "2 + 3 * 4",
        "sqrt(16)",
        "sin(pi/2)",
        "10 / 3"
    ]
    
    for expr in calculations:
        result = await tool_manager.process_function_call({
            "name": "calculator",
            "arguments": json.dumps({"expression": expr, "precision": 3})
        })
        logger.info(f"  {expr} = {result.result}")
    
    # Random number tool
    logger.info("🎲 Generating random numbers...")
    result = await tool_manager.process_function_call({
        "name": "random_number",
        "arguments": '{"type": "integer", "min_value": 1, "max_value": 100, "count": 5}'
    })
    logger.info(f"Random numbers: {result.result}")
    
    # UUID generator
    logger.info("🆔 Generating UUIDs...")
    result = await tool_manager.process_function_call({
        "name": "generate_uuid",
        "arguments": '{"count": 3}'
    })
    logger.info(f"UUIDs: {result.result}")
    
    # System info tool
    logger.info("💻 Getting system information...")
    result = await tool_manager.process_function_call({
        "name": "get_system_info",
        "arguments": '{"info_type": "basic"}'
    })
    if result.success:
        system_info = result.result
        logger.info(f"OS: {system_info['basic']['operating_system']}")
        logger.info(f"Architecture: {system_info['basic']['architecture']}")
    
    logger.info("✅ Basic tool demonstration completed!")


async def demonstrate_tool_categories():
    """Demonstrate tools by category."""
    logger.info("📂 Starting tool category demonstration...")
    
    tool_manager = ToolManager()
    
    # Group tools by category
    category_tools = {}
    for tool in tool_manager.get_available_tools():
        category = tool.category
        if category not in category_tools:
            category_tools[category] = []
        category_tools[category].append(tool)
    
    # Show tools by category
    for category, tools in category_tools.items():
        logger.info(f"📁 {category.value.upper()} tools ({len(tools)}):")
        for tool in tools:
            logger.info(f"  - {tool.name}: {tool.description[:60]}...")
    
    # Demonstrate category-specific functionality
    logger.info("🧪 Testing category-specific tools...")
    
    # Utility tools
    logger.info("🔧 Utility tools:")
    result = await tool_manager.process_function_call({
        "name": "timestamp",
        "arguments": '{"operation": "current"}'
    })
    logger.info(f"  Current timestamp: {result.result}")
    
    # Calculation tools
    logger.info("🧮 Calculation tools:")
    result = await tool_manager.process_function_call({
        "name": "calculator",
        "arguments": '{"expression": "log(100)", "precision": 4}'
    })
    logger.info(f"  log(100) = {result.result}")
    
    # System tools
    logger.info("💻 System tools:")
    result = await tool_manager.process_function_call({
        "name": "get_environment",
        "arguments": '{"filter_pattern": "PATH"}'
    })
    if result.success:
        env_vars = result.result.get("environment_variables", {})
        logger.info(f"  Found {len(env_vars)} PATH-related environment variables")
    
    logger.info("✅ Tool category demonstration completed!")


async def demonstrate_error_handling():
    """Demonstrate error handling and validation."""
    logger.info("⚠️ Starting error handling demonstration...")
    
    tool_manager = ToolManager()
    
    # Test invalid tool name
    logger.info("🚫 Testing invalid tool name...")
    result = await tool_manager.process_function_call({
        "name": "nonexistent_tool",
        "arguments": "{}"
    })
    logger.info(f"Result: {result.success}, Error: {result.error}")
    
    # Test invalid parameters
    logger.info("🚫 Testing invalid parameters...")
    result = await tool_manager.process_function_call({
        "name": "calculator",
        "arguments": '{"expression": ""}'  # Empty expression
    })
    logger.info(f"Result: {result.success}, Error: {result.error}")
    
    # Test malformed JSON
    logger.info("🚫 Testing malformed JSON...")
    result = await tool_manager.process_function_call({
        "name": "get_time",
        "arguments": '{"format": "invalid_json"'  # Malformed JSON
    })
    logger.info(f"Result: {result.success}, Error: {result.error}")
    
    # Test parameter validation
    logger.info("🚫 Testing parameter validation...")
    result = await tool_manager.process_function_call({
        "name": "random_number",
        "arguments": '{"min_value": 100, "max_value": 10}'  # Invalid range
    })
    logger.info(f"Result: {result.success}, Error: {result.error}")
    
    logger.info("✅ Error handling demonstration completed!")


async def demonstrate_websocket_integration():
    """Demonstrate WebSocket integration."""
    logger.info("🌐 Starting WebSocket tool demonstration...")
    
    try:
        # Set up WebSocket server
        server = CodaWebSocketServer(host="localhost", port=8768)
        integration = CodaWebSocketIntegration(server)
        
        # Create WebSocket-enabled tool manager
        config = ToolConfig()
        tool_manager = WebSocketToolManager(config)
        await tool_manager.set_websocket_integration(integration)
        
        # Start WebSocket server
        await server.start()
        logger.info(f"🌐 WebSocket server running at ws://{server.host}:{server.port}")
        logger.info("💡 Connect with: wscat -c ws://localhost:8768")
        
        # Wait for potential clients
        logger.info("⏳ Waiting 3 seconds for WebSocket clients...")
        await asyncio.sleep(3)
        
        # Demonstrate tool operations with WebSocket events
        logger.info("🎬 Demonstrating tool operations with WebSocket events...")
        
        # Function calls with events
        await tool_manager.process_function_call({
            "name": "get_time",
            "arguments": '{"format": "iso"}'
        })
        await asyncio.sleep(0.5)
        
        await tool_manager.process_function_call({
            "name": "calculator",
            "arguments": '{"expression": "2^10"}'
        })
        await asyncio.sleep(0.5)
        
        await tool_manager.process_function_call({
            "name": "random_number",
            "arguments": '{"count": 3}'
        })
        await asyncio.sleep(0.5)
        
        # Broadcast analytics
        await tool_manager.broadcast_tool_analytics()
        await tool_manager.broadcast_performance_metrics()
        await tool_manager.broadcast_registry_stats()
        
        # Trigger comprehensive snapshot
        snapshot = await tool_manager.trigger_tool_snapshot()
        logger.info(f"📸 Tool system snapshot captured with {len(snapshot)} data points")
        
        # Show server stats
        server_stats = server.get_stats()
        logger.info(f"📊 WebSocket server stats: {server_stats}")
        
        logger.info("✅ WebSocket tool demonstration completed!")
        logger.info("⏳ Server will stop in 3 seconds...")
        await asyncio.sleep(3)
        
        await server.stop()
        
    except Exception as e:
        logger.error(f"❌ Error in WebSocket demonstration: {e}")


async def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring and analytics."""
    logger.info("📊 Starting performance monitoring demonstration...")
    
    tool_manager = ToolManager()
    
    # Execute multiple tools to generate performance data
    logger.info("🏃 Executing tools to generate performance data...")
    
    test_calls = [
        {"name": "get_time", "arguments": "{}"},
        {"name": "calculator", "arguments": '{"expression": "sqrt(2)"}'},
        {"name": "random_number", "arguments": '{"count": 1}'},
        {"name": "generate_uuid", "arguments": "{}"},
        {"name": "get_date", "arguments": "{}"},
    ]
    
    # Execute each tool multiple times
    for i in range(10):
        for call in test_calls:
            await tool_manager.process_function_call(call)
        
        if i % 3 == 0:
            logger.info(f"  Completed {(i + 1) * len(test_calls)} tool executions...")
    
    # Get comprehensive analytics
    logger.info("📈 Analyzing performance metrics...")
    analytics = tool_manager.get_tool_analytics()
    
    # Registry stats
    registry_stats = analytics["registry"]
    logger.info("📋 Registry Statistics:")
    logger.info(f"  Total tools: {registry_stats['total_tools']}")
    logger.info(f"  Available tools: {registry_stats['available_tools']}")
    logger.info(f"  Category distribution: {registry_stats['category_distribution']}")
    
    # Execution stats
    execution_stats = analytics["execution"]
    logger.info("⚡ Execution Statistics:")
    logger.info(f"  Total executions: {execution_stats['total_executions']}")
    logger.info(f"  Successful executions: {execution_stats['successful_executions']}")
    logger.info(f"  Success rate: {execution_stats.get('success_rate', 0):.1f}%")
    logger.info(f"  Average execution time: {execution_stats['average_execution_time_ms']:.2f}ms")
    logger.info(f"  Most used tool: {execution_stats.get('most_used_tool', 'N/A')}")
    
    # Performance metrics
    performance_metrics = analytics["performance"]
    logger.info("🚀 Performance Metrics:")
    for tool_name, metrics in performance_metrics.items():
        if isinstance(metrics, dict):
            logger.info(f"  {tool_name}:")
            logger.info(f"    Executions: {metrics.get('execution_count', 0)}")
            logger.info(f"    Avg time: {metrics.get('average_time_ms', 0):.2f}ms")
            logger.info(f"    Min time: {metrics.get('min_time_ms', 0):.2f}ms")
            logger.info(f"    Max time: {metrics.get('max_time_ms', 0):.2f}ms")
    
    logger.info("✅ Performance monitoring demonstration completed!")


async def demonstrate_plugin_system():
    """Demonstrate plugin loading system."""
    logger.info("🔌 Starting plugin system demonstration...")
    
    plugin_loader = PluginLoader()
    
    # Create a simple test plugin
    test_plugin_content = '''
"""Test plugin for demonstration."""

from src.coda.components.tools.base_tool import BaseTool, create_simple_tool_definition
from src.coda.components.tools.models import ToolCategory

class HelloWorldTool(BaseTool):
    """A simple hello world tool."""
    
    def _create_definition(self):
        return create_simple_tool_definition(
            name="hello_world",
            description="Say hello to the world",
            category=ToolCategory.UTILITY,
            parameters=[
                self.create_string_parameter(
                    "name",
                    "Name to greet",
                    required=False,
                    default="World"
                )
            ]
        )
    
    async def _execute_impl(self, parameters, context=None):
        name = parameters.get("name", "World")
        return f"Hello, {name}!"

def get_tools():
    """Return tools provided by this plugin."""
    return [HelloWorldTool()]
'''
    
    # Write test plugin to temporary file
    test_plugin_path = "test_plugin.py"
    try:
        with open(test_plugin_path, 'w') as f:
            f.write(test_plugin_content)
        
        # Validate plugin
        logger.info("🔍 Validating test plugin...")
        validation = await plugin_loader.validate_plugin(test_plugin_path)
        logger.info(f"Plugin validation: {validation}")
        
        if validation["valid"]:
            # Load plugin
            logger.info("📥 Loading test plugin...")
            tools = await plugin_loader.load_plugin(test_plugin_path)
            logger.info(f"Loaded {len(tools)} tools from plugin")
            
            # Show plugin info
            plugin_info = plugin_loader.get_plugin_info(test_plugin_path)
            logger.info(f"Plugin info: {plugin_info}")
            
            # Test the loaded tool
            if tools:
                tool = tools[0]
                result = await tool.execute({"name": "Plugin System"})
                logger.info(f"Plugin tool result: {result}")
            
            # Unload plugin
            logger.info("📤 Unloading test plugin...")
            success = await plugin_loader.unload_plugin(test_plugin_path)
            logger.info(f"Plugin unloaded: {success}")
        
    except Exception as e:
        logger.error(f"Plugin demonstration error: {e}")
    
    finally:
        # Clean up test plugin file
        try:
            import os
            if os.path.exists(test_plugin_path):
                os.remove(test_plugin_path)
        except:
            pass
    
    logger.info("✅ Plugin system demonstration completed!")


async def main():
    """Main demonstration function."""
    logger.info("🚀 Starting Coda Tool System Demonstration")
    
    try:
        # Run demonstrations
        await demonstrate_basic_tools()
        await asyncio.sleep(1)
        
        await demonstrate_tool_categories()
        await asyncio.sleep(1)
        
        await demonstrate_error_handling()
        await asyncio.sleep(1)
        
        await demonstrate_performance_monitoring()
        await asyncio.sleep(1)
        
        await demonstrate_websocket_integration()
        await asyncio.sleep(1)
        
        await demonstrate_plugin_system()
        
        logger.info("🎉 All tool system demonstrations completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("🛑 Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
