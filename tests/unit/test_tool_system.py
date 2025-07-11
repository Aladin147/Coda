"""
Unit tests for the tool system.
"""

import pytest
import asyncio
import uuid
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from src.coda.components.tools.models import (
    ToolDefinition,
    ToolParameter,
    ToolCall,
    ToolResult,
    ToolCategory,
    ParameterType,
    ToolConfig,
)
from src.coda.components.tools.base_tool import BaseTool, ToolError, ToolValidationError, ToolExecutionError
from src.coda.components.tools.registry import ToolRegistry
from src.coda.components.tools.executor import ToolExecutor
from src.coda.components.tools.function_calling import FunctionCallingManager
from src.coda.components.tools.manager import ToolManager
from src.coda.components.tools.basic_tools import GetTimeTool, CalculatorTool
from src.coda.components.tools.memory_tools import StoreMemoryTool


class TestToolParameter:
    """Test cases for ToolParameter."""
    
    def test_parameter_creation(self):
        """Test creating a tool parameter."""
        param = ToolParameter(
            name="test_param",
            type=ParameterType.STRING,
            description="A test parameter",
            required=True,
            default="default_value"
        )
        
        assert param.name == "test_param"
        assert param.type == ParameterType.STRING
        assert param.required is True
        assert param.default == "default_value"
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid string parameter
        param = ToolParameter(
            name="text",
            type=ParameterType.STRING,
            description="Text parameter",
            default="test"
        )
        assert param.default == "test"
        
        # Invalid default type should raise error
        with pytest.raises(ValueError):
            ToolParameter(
                name="text",
                type=ParameterType.STRING,
                description="Text parameter",
                default=123  # Invalid type
            )
    
    def test_json_schema_generation(self):
        """Test JSON schema generation."""
        param = ToolParameter(
            name="count",
            type=ParameterType.INTEGER,
            description="Number of items",
            minimum=1,
            maximum=100,
            default=10
        )
        
        schema = param.to_json_schema()
        
        assert schema["type"] == "integer"
        assert schema["description"] == "Number of items"
        assert schema["minimum"] == 1
        assert schema["maximum"] == 100
        assert schema["default"] == 10


class TestToolDefinition:
    """Test cases for ToolDefinition."""
    
    def test_tool_definition_creation(self):
        """Test creating a tool definition."""
        definition = ToolDefinition(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.UTILITY,
            parameters=[
                ToolParameter(
                    name="input",
                    type=ParameterType.STRING,
                    description="Input text",
                    required=True
                )
            ]
        )
        
        assert definition.name == "test_tool"
        assert definition.category == ToolCategory.UTILITY
        assert len(definition.parameters) == 1
    
    def test_function_schema_generation(self):
        """Test OpenAI function schema generation."""
        definition = ToolDefinition(
            name="calculator",
            description="Perform calculations",
            parameters=[
                ToolParameter(
                    name="expression",
                    type=ParameterType.STRING,
                    description="Math expression",
                    required=True
                ),
                ToolParameter(
                    name="precision",
                    type=ParameterType.INTEGER,
                    description="Decimal places",
                    required=False,
                    default=2
                )
            ]
        )
        
        schema = definition.to_function_schema()
        
        assert schema["name"] == "calculator"
        assert schema["description"] == "Perform calculations"
        assert "parameters" in schema
        assert schema["parameters"]["type"] == "object"
        assert "expression" in schema["parameters"]["properties"]
        assert "precision" in schema["parameters"]["properties"]
        assert schema["parameters"]["required"] == ["expression"]
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        definition = ToolDefinition(
            name="test_tool",
            description="Test tool",
            parameters=[
                ToolParameter(
                    name="required_param",
                    type=ParameterType.STRING,
                    description="Required parameter",
                    required=True
                ),
                ToolParameter(
                    name="optional_param",
                    type=ParameterType.INTEGER,
                    description="Optional parameter",
                    required=False,
                    default=42
                )
            ]
        )
        
        # Valid parameters
        validated = definition.validate_parameters({
            "required_param": "test_value",
            "optional_param": 100
        })
        assert validated["required_param"] == "test_value"
        assert validated["optional_param"] == 100
        
        # Missing required parameter
        with pytest.raises(ValueError):
            definition.validate_parameters({"optional_param": 100})
        
        # Invalid parameter type
        with pytest.raises(ValueError):
            definition.validate_parameters({
                "required_param": "test_value",
                "optional_param": "not_an_integer"
            })


class MockTool(BaseTool):
    """Mock tool for testing."""
    
    def _create_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="mock_tool",
            description="A mock tool for testing",
            category=ToolCategory.UTILITY,
            parameters=[
                ToolParameter(
                    name="input",
                    type=ParameterType.STRING,
                    description="Input text",
                    required=True
                )
            ]
        )
    
    async def _execute_impl(self, parameters, context=None):
        return f"Mock result: {parameters['input']}"


class TestBaseTool:
    """Test cases for BaseTool."""
    
    @pytest.fixture
    def mock_tool(self):
        return MockTool()
    
    def test_tool_initialization(self, mock_tool):
        """Test tool initialization."""
        definition = mock_tool.get_definition()
        assert definition.name == "mock_tool"
        assert mock_tool.is_available() is True
        assert mock_tool.get_execution_count() == 0
    
    @pytest.mark.asyncio
    async def test_tool_execution(self, mock_tool):
        """Test tool execution."""
        result = await mock_tool.execute({"input": "test"})
        assert result == "Mock result: test"
        assert mock_tool.get_execution_count() == 1
    
    @pytest.mark.asyncio
    async def test_parameter_validation(self, mock_tool):
        """Test parameter validation."""
        # Valid parameters
        validated = await mock_tool.validate_parameters({"input": "test"})
        assert validated["input"] == "test"
        
        # Missing required parameter
        with pytest.raises(ToolValidationError):
            await mock_tool.validate_parameters({})
    
    def test_availability_control(self, mock_tool):
        """Test tool availability control."""
        assert mock_tool.is_available() is True
        
        mock_tool.set_available(False)
        assert mock_tool.is_available() is False
    
    @pytest.mark.asyncio
    async def test_execution_when_unavailable(self, mock_tool):
        """Test execution when tool is unavailable."""
        mock_tool.set_available(False)
        
        with pytest.raises(ToolError):
            await mock_tool.execute({"input": "test"})


class TestToolRegistry:
    """Test cases for ToolRegistry."""
    
    @pytest.fixture
    def registry(self):
        return ToolRegistry()
    
    @pytest.fixture
    def mock_tool(self):
        return MockTool()
    
    def test_tool_registration(self, registry, mock_tool):
        """Test tool registration."""
        success = registry.register_tool(mock_tool)
        assert success is True
        
        # Check tool is registered
        retrieved_tool = registry.get_tool("mock_tool")
        assert retrieved_tool is mock_tool
        
        # Check definition is stored
        definition = registry.get_tool_definition("mock_tool")
        assert definition.name == "mock_tool"
    
    def test_tool_unregistration(self, registry, mock_tool):
        """Test tool unregistration."""
        registry.register_tool(mock_tool)
        
        success = registry.unregister_tool("mock_tool")
        assert success is True
        
        # Check tool is removed
        retrieved_tool = registry.get_tool("mock_tool")
        assert retrieved_tool is None
    
    def test_tool_listing(self, registry, mock_tool):
        """Test tool listing."""
        registry.register_tool(mock_tool)
        
        # List all tools
        all_tools = registry.list_tools()
        assert len(all_tools) == 1
        assert all_tools[0].name == "mock_tool"
        
        # List by category
        utility_tools = registry.list_tools(ToolCategory.UTILITY)
        assert len(utility_tools) == 1
        assert utility_tools[0].name == "mock_tool"
    
    def test_tool_search(self, registry, mock_tool):
        """Test tool search."""
        registry.register_tool(mock_tool)
        
        # Search by name
        results = registry.search_tools("mock")
        assert len(results) == 1
        assert results[0].name == "mock_tool"
        
        # Search by description
        results = registry.search_tools("testing")
        assert len(results) == 1
    
    def test_function_schema_generation(self, registry, mock_tool):
        """Test function schema generation."""
        registry.register_tool(mock_tool)
        
        schemas = registry.get_function_schemas()
        assert len(schemas) == 1
        
        schema = schemas[0]
        assert schema["name"] == "mock_tool"
        assert "parameters" in schema


@pytest.mark.asyncio
class TestToolExecutor:
    """Test cases for ToolExecutor."""
    
    @pytest.fixture
    def registry(self):
        registry = ToolRegistry()
        registry.register_tool(MockTool())
        return registry
    
    @pytest.fixture
    def executor(self, registry):
        return ToolExecutor(registry)
    
    async def test_tool_execution(self, executor):
        """Test tool execution."""
        tool_call = ToolCall(
            call_id=str(uuid.uuid4()),
            tool_name="mock_tool",
            parameters={"input": "test"}
        )
        
        result = await executor.execute_tool(tool_call)
        
        assert result.success is True
        assert result.result == "Mock result: test"
        assert result.execution_time_ms > 0
    
    async def test_nonexistent_tool_execution(self, executor):
        """Test execution of nonexistent tool."""
        tool_call = ToolCall(
            call_id=str(uuid.uuid4()),
            tool_name="nonexistent_tool",
            parameters={}
        )
        
        result = await executor.execute_tool(tool_call)
        
        assert result.success is False
        assert "not found" in result.error
    
    async def test_multiple_tool_execution(self, executor):
        """Test multiple tool execution."""
        tool_calls = [
            ToolCall(
                call_id=str(uuid.uuid4()),
                tool_name="mock_tool",
                parameters={"input": "test1"}
            ),
            ToolCall(
                call_id=str(uuid.uuid4()),
                tool_name="mock_tool",
                parameters={"input": "test2"}
            )
        ]
        
        results = await executor.execute_tools(tool_calls)
        
        assert len(results) == 2
        assert all(result.success for result in results)
        assert results[0].result == "Mock result: test1"
        assert results[1].result == "Mock result: test2"
    
    def test_execution_stats(self, executor):
        """Test execution statistics."""
        stats = executor.get_execution_stats()
        
        assert stats.total_tools >= 0
        assert stats.total_executions >= 0
        assert stats.successful_executions >= 0
        assert stats.failed_executions >= 0


class TestFunctionCallingManager:
    """Test cases for FunctionCallingManager."""
    
    @pytest.fixture
    def function_calling(self):
        return FunctionCallingManager()
    
    def test_function_call_parsing(self, function_calling):
        """Test function call parsing."""
        function_call_data = {
            "name": "test_tool",
            "arguments": '{"param1": "value1", "param2": 42}'
        }
        
        tool_call = function_calling.parse_function_call(function_call_data)
        
        assert tool_call.tool_name == "test_tool"
        assert tool_call.parameters["param1"] == "value1"
        assert tool_call.parameters["param2"] == 42
    
    def test_tool_result_formatting(self, function_calling):
        """Test tool result formatting."""
        result = ToolResult(
            call_id="test_call",
            tool_name="test_tool",
            success=True,
            result="Test result",
            execution_time_ms=100
        )
        
        formatted = function_calling.format_tool_result(result, for_llm=True)
        
        assert formatted["tool_call_id"] == "test_call"
        assert formatted["role"] == "tool"
        assert formatted["name"] == "test_tool"
        assert formatted["content"] == "Test result"
    
    def test_function_schema_generation(self, function_calling):
        """Test function schema generation."""
        tools = [
            ToolDefinition(
                name="test_tool",
                description="Test tool",
                parameters=[
                    ToolParameter(
                        name="input",
                        type=ParameterType.STRING,
                        description="Input parameter",
                        required=True
                    )
                ]
            )
        ]
        
        schemas = function_calling.generate_function_schemas(tools)
        
        assert len(schemas) == 1
        assert schemas[0]["name"] == "test_tool"


@pytest.mark.asyncio
class TestToolManager:
    """Test cases for ToolManager."""
    
    @pytest.fixture
    def tool_manager(self):
        return ToolManager()
    
    async def test_function_call_processing(self, tool_manager):
        """Test function call processing."""
        function_call = {
            "name": "get_time",
            "arguments": '{"format": "24h"}'
        }
        
        result = await tool_manager.process_function_call(function_call)
        
        assert result.success is True
        assert result.tool_name == "get_time"
    
    def test_available_tools(self, tool_manager):
        """Test getting available tools."""
        tools = tool_manager.get_available_tools()
        
        assert len(tools) > 0
        tool_names = [tool.name for tool in tools]
        assert "get_time" in tool_names
        assert "calculator" in tool_names
    
    def test_function_schemas(self, tool_manager):
        """Test getting function schemas."""
        schemas = tool_manager.get_function_schemas()
        
        assert len(schemas) > 0
        schema_names = [schema["name"] for schema in schemas]
        assert "get_time" in schema_names
    
    def test_tool_analytics(self, tool_manager):
        """Test tool analytics."""
        analytics = tool_manager.get_tool_analytics()
        
        assert "registry" in analytics
        assert "execution" in analytics
        assert "performance" in analytics
        assert "function_calling" in analytics


class TestBuiltinTools:
    """Test cases for built-in tools."""
    
    @pytest.mark.asyncio
    async def test_get_time_tool(self):
        """Test GetTimeTool."""
        tool = GetTimeTool()
        
        # Test default format
        result = await tool.execute({})
        assert isinstance(result, str)
        assert ":" in result  # Should contain time separator
        
        # Test specific format
        result = await tool.execute({"format": "12h"})
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_calculator_tool(self):
        """Test CalculatorTool."""
        tool = CalculatorTool()
        
        # Test simple calculation
        result = await tool.execute({"expression": "2 + 3"})
        assert result == "5"
        
        # Test with precision
        result = await tool.execute({"expression": "10 / 3", "precision": 2})
        assert result == "3.33"
        
        # Test invalid expression
        with pytest.raises(ToolExecutionError):
            await tool.execute({"expression": "invalid"})


if __name__ == "__main__":
    # Run a simple test
    async def simple_test():
        manager = ToolManager()
        
        # Test basic functionality
        tools = manager.get_available_tools()
        print(f"Available tools: {len(tools)}")
        
        # Test function call
        result = await manager.process_function_call({
            "name": "get_time",
            "arguments": "{}"
        })
        print(f"Time result: {result.result}")
        
        # Test calculator
        result = await manager.process_function_call({
            "name": "calculator",
            "arguments": '{"expression": "2 + 3 * 4"}'
        })
        print(f"Calculator result: {result.result}")
        
        # Test analytics
        analytics = manager.get_tool_analytics()
        print(f"Registry stats: {analytics['registry']}")
        
        print("✅ Tool system test passed")
    
    asyncio.run(simple_test())
