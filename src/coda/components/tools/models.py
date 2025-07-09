"""
Data models for the tool system.

This module defines Pydantic models for type-safe tool operations.
"""

import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum
from pydantic import BaseModel, Field, validator


class ToolCategory(str, Enum):
    """Categories of tools."""
    UTILITY = "utility"
    MEMORY = "memory"
    SYSTEM = "system"
    WEB = "web"
    CALCULATION = "calculation"
    COMMUNICATION = "communication"
    FILE = "file"
    DATABASE = "database"
    AI = "ai"
    CUSTOM = "custom"


class ParameterType(str, Enum):
    """Types of tool parameters."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""
    
    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None
    minimum: Optional[Union[int, float]] = None
    maximum: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    examples: List[Any] = Field(default_factory=list)
    
    @validator('default')
    def validate_default_type(cls, v, values):
        """Validate that default value matches parameter type."""
        if v is None:
            return v
        
        param_type = values.get('type')
        if param_type == ParameterType.STRING and not isinstance(v, str):
            raise ValueError(f"Default value must be string for string parameter")
        elif param_type == ParameterType.INTEGER and not isinstance(v, int):
            raise ValueError(f"Default value must be integer for integer parameter")
        elif param_type == ParameterType.FLOAT and not isinstance(v, (int, float)):
            raise ValueError(f"Default value must be number for float parameter")
        elif param_type == ParameterType.BOOLEAN and not isinstance(v, bool):
            raise ValueError(f"Default value must be boolean for boolean parameter")
        elif param_type == ParameterType.ARRAY and not isinstance(v, list):
            raise ValueError(f"Default value must be array for array parameter")
        elif param_type == ParameterType.OBJECT and not isinstance(v, dict):
            raise ValueError(f"Default value must be object for object parameter")
        
        return v
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert parameter to JSON schema format."""
        schema = {
            "type": self.type.value,
            "description": self.description
        }
        
        if self.enum:
            schema["enum"] = self.enum
        
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        
        if self.pattern:
            schema["pattern"] = self.pattern
        
        if self.examples:
            schema["examples"] = self.examples
        
        if self.default is not None:
            schema["default"] = self.default
        
        return schema


class ToolDefinition(BaseModel):
    """Complete definition of a tool."""
    
    name: str = Field(..., pattern=r'^[a-zA-Z][a-zA-Z0-9_]*$')
    description: str
    category: ToolCategory = ToolCategory.CUSTOM
    parameters: List[ToolParameter] = Field(default_factory=list)
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    version: str = Field(default="1.0.0")
    author: Optional[str] = None
    requires_auth: bool = Field(default=False)
    is_dangerous: bool = Field(default=False)
    timeout_seconds: float = Field(default=30.0, gt=0)
    retry_count: int = Field(default=0, ge=0)
    
    def get_required_parameters(self) -> List[ToolParameter]:
        """Get list of required parameters."""
        return [p for p in self.parameters if p.required]
    
    def get_optional_parameters(self) -> List[ToolParameter]:
        """Get list of optional parameters."""
        return [p for p in self.parameters if not p.required]
    
    def to_function_schema(self) -> Dict[str, Any]:
        """Convert tool definition to OpenAI function calling schema."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize parameters."""
        validated = {}
        
        # Check required parameters
        for param in self.get_required_parameters():
            if param.name not in parameters:
                raise ValueError(f"Required parameter '{param.name}' is missing")
        
        # Validate each parameter
        for param in self.parameters:
            value = parameters.get(param.name, param.default)
            
            if value is None and param.required:
                raise ValueError(f"Required parameter '{param.name}' cannot be None")
            
            if value is not None:
                validated[param.name] = self._validate_parameter_value(param, value)
        
        return validated
    
    def _validate_parameter_value(self, param: ToolParameter, value: Any) -> Any:
        """Validate a single parameter value."""
        # Type validation
        if param.type == ParameterType.STRING and not isinstance(value, str):
            raise ValueError(f"Parameter '{param.name}' must be a string")
        elif param.type == ParameterType.INTEGER and not isinstance(value, int):
            raise ValueError(f"Parameter '{param.name}' must be an integer")
        elif param.type == ParameterType.FLOAT and not isinstance(value, (int, float)):
            raise ValueError(f"Parameter '{param.name}' must be a number")
        elif param.type == ParameterType.BOOLEAN and not isinstance(value, bool):
            raise ValueError(f"Parameter '{param.name}' must be a boolean")
        elif param.type == ParameterType.ARRAY and not isinstance(value, list):
            raise ValueError(f"Parameter '{param.name}' must be an array")
        elif param.type == ParameterType.OBJECT and not isinstance(value, dict):
            raise ValueError(f"Parameter '{param.name}' must be an object")
        
        # Enum validation
        if param.enum and value not in param.enum:
            raise ValueError(f"Parameter '{param.name}' must be one of {param.enum}")
        
        # Range validation for numbers
        if param.type in [ParameterType.INTEGER, ParameterType.FLOAT]:
            if param.minimum is not None and value < param.minimum:
                raise ValueError(f"Parameter '{param.name}' must be >= {param.minimum}")
            if param.maximum is not None and value > param.maximum:
                raise ValueError(f"Parameter '{param.name}' must be <= {param.maximum}")
        
        # Pattern validation for strings
        if param.type == ParameterType.STRING and param.pattern:
            import re
            if not re.match(param.pattern, value):
                raise ValueError(f"Parameter '{param.name}' must match pattern {param.pattern}")
        
        return value


class ToolCall(BaseModel):
    """A request to execute a tool."""
    
    call_id: str
    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ToolResult(BaseModel):
    """Result of a tool execution."""
    
    call_id: str
    tool_name: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.success and self.error is None
    
    def get_display_result(self) -> str:
        """Get a human-readable result string."""
        if not self.success:
            return f"Error: {self.error}"
        
        if isinstance(self.result, str):
            return self.result
        elif isinstance(self.result, (dict, list)):
            import json
            return json.dumps(self.result, indent=2)
        else:
            return str(self.result)


class ToolExecution(BaseModel):
    """Context for tool execution."""
    
    call: ToolCall
    definition: ToolDefinition
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "pending"  # pending, running, completed, failed, timeout
    attempts: int = 0
    max_attempts: int = 1
    
    def mark_started(self) -> None:
        """Mark execution as started."""
        self.status = "running"
        self.start_time = time.time()
        self.attempts += 1
    
    def mark_completed(self, result: Any) -> ToolResult:
        """Mark execution as completed."""
        self.status = "completed"
        self.end_time = time.time()
        execution_time = (self.end_time - self.start_time) * 1000
        
        return ToolResult(
            call_id=self.call.call_id,
            tool_name=self.call.tool_name,
            success=True,
            result=result,
            execution_time_ms=execution_time,
            metadata={"attempts": self.attempts}
        )
    
    def mark_failed(self, error: str) -> ToolResult:
        """Mark execution as failed."""
        self.status = "failed"
        self.end_time = time.time()
        execution_time = (self.end_time - self.start_time) * 1000 if self.end_time else 0
        
        return ToolResult(
            call_id=self.call.call_id,
            tool_name=self.call.tool_name,
            success=False,
            error=error,
            execution_time_ms=execution_time,
            metadata={"attempts": self.attempts}
        )
    
    def can_retry(self) -> bool:
        """Check if execution can be retried."""
        return self.attempts < self.max_attempts and self.status == "failed"


class ToolEvent(BaseModel):
    """Event for WebSocket broadcasting."""
    
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    tool_name: Optional[str] = None
    call_id: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ToolStats(BaseModel):
    """Statistics about tool usage."""
    
    total_tools: int = 0
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time_ms: float = 0.0
    most_used_tool: Optional[str] = None
    tool_usage_counts: Dict[str, int] = Field(default_factory=dict)
    category_distribution: Dict[ToolCategory, int] = Field(default_factory=dict)
    error_rate: float = 0.0
    
    def calculate_success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100


# Configuration models

class ToolRegistryConfig(BaseModel):
    """Configuration for tool registry."""
    
    auto_discover_plugins: bool = Field(default=True)
    plugin_directories: List[str] = Field(default_factory=list)
    max_tools: int = Field(default=100, gt=0)
    allow_dangerous_tools: bool = Field(default=False)
    require_auth_for_dangerous: bool = Field(default=True)


class ToolExecutorConfig(BaseModel):
    """Configuration for tool executor."""
    
    default_timeout_seconds: float = Field(default=30.0, gt=0)
    max_concurrent_executions: int = Field(default=10, gt=0)
    enable_retries: bool = Field(default=True)
    max_retry_attempts: int = Field(default=3, ge=0)
    execution_logging: bool = Field(default=True)
    performance_monitoring: bool = Field(default=True)


class ToolConfig(BaseModel):
    """Complete tool system configuration."""
    
    registry: ToolRegistryConfig = Field(default_factory=ToolRegistryConfig)
    executor: ToolExecutorConfig = Field(default_factory=ToolExecutorConfig)
    websocket_events_enabled: bool = Field(default=True)
    analytics_enabled: bool = Field(default=True)
    memory_integration_enabled: bool = Field(default=True)
    personality_integration_enabled: bool = Field(default=True)
