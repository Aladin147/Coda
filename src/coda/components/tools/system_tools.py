"""
System tools for Coda.

This module provides tools for system information, environment variables,
and safe command execution.
"""

import logging
import os
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional

from .base_tool import BaseTool, create_simple_tool_definition
from .models import ToolCategory, ToolDefinition

logger = logging.getLogger("coda.tools.system")


class GetSystemInfoTool(BaseTool):
    """Tool to get system information."""

    def _create_definition(self) -> ToolDefinition:
        """Create tool definition."""
        return create_simple_tool_definition(
            name="get_system_info",
            description="Get information about the system (OS, Python version, etc.)",
            category=ToolCategory.SYSTEM,
            parameters=[
                self.create_string_parameter(
                    "info_type",
                    "Type of system information to retrieve",
                    required=False,
                    default="basic",
                    enum=["basic", "detailed", "python", "platform", "all"],
                    examples=["basic", "detailed", "python"],
                )
            ],
            examples=[{"info_type": "basic"}, {"info_type": "detailed"}, {"info_type": "python"}],
            tags=["system", "info", "platform"],
            timeout_seconds=10.0,
        )

    async def _execute_impl(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the tool."""
        info_type = parameters.get("info_type", "basic")

        result = {}

        if info_type in ["basic", "all"]:
            result["basic"] = {
                "operating_system": platform.system(),
                "os_version": platform.version(),
                "architecture": platform.architecture()[0],
                "machine": platform.machine(),
                "processor": platform.processor(),
                "hostname": platform.node(),
            }

        if info_type in ["python", "all"]:
            result["python"] = {
                "version": sys.version,
                "version_info": {
                    "major": sys.version_info.major,
                    "minor": sys.version_info.minor,
                    "micro": sys.version_info.micro,
                },
                "executable": sys.executable,
                "platform": sys.platform,
                "path": sys.path[:5],  # First 5 paths only
            }

        if info_type in ["platform", "detailed", "all"]:
            result["platform"] = {
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            }

        if info_type in ["detailed", "all"]:
            try:
                import psutil

                result["resources"] = {
                    "cpu_count": psutil.cpu_count(),
                    "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                    "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                    "disk_usage": {
                        "total_gb": round(psutil.disk_usage("/").total / (1024**3), 2),
                        "free_gb": round(psutil.disk_usage("/").free / (1024**3), 2),
                    },
                }
            except ImportError:
                result["resources"] = {"error": "psutil not available for detailed resource info"}

        self.log_info(f"Retrieved {info_type} system information")
        return result


class GetEnvironmentTool(BaseTool):
    """Tool to get environment variables."""

    def _create_definition(self) -> ToolDefinition:
        """Create tool definition."""
        return create_simple_tool_definition(
            name="get_environment",
            description="Get environment variables (filtered for security)",
            category=ToolCategory.SYSTEM,
            parameters=[
                self.create_string_parameter(
                    "variable_name",
                    "Specific environment variable to get (optional)",
                    required=False,
                    examples=["PATH", "HOME", "USER", "PYTHON_PATH"],
                ),
                self.create_boolean_parameter(
                    "include_sensitive",
                    "Include potentially sensitive variables (requires auth)",
                    required=False,
                    default=False,
                ),
                self.create_string_parameter(
                    "filter_pattern",
                    "Pattern to filter variable names (case-insensitive)",
                    required=False,
                    examples=["PYTHON", "PATH", "HOME"],
                ),
            ],
            examples=[
                {"variable_name": "PATH"},
                {"filter_pattern": "PYTHON"},
                {"include_sensitive": False},
            ],
            tags=["system", "environment", "variables"],
            timeout_seconds=5.0,
        )

    async def _execute_impl(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the tool."""
        variable_name = parameters.get("variable_name")
        include_sensitive = parameters.get("include_sensitive", False)
        filter_pattern = parameters.get("filter_pattern")

        # Define sensitive variable patterns
        sensitive_patterns = [
            "password",
            "secret",
            "key",
            "token",
            "auth",
            "credential",
            "api_key",
            "private",
            "cert",
            "ssl",
            "tls",
        ]

        def is_sensitive(var_name: str) -> bool:
            """Check if variable name suggests sensitive content."""
            var_lower = var_name.lower()
            return any(pattern in var_lower for pattern in sensitive_patterns)

        if variable_name:
            # Get specific variable
            value = os.environ.get(variable_name)
            if value is None:
                return {"error": f"Environment variable '{variable_name}' not found"}

            # Check if sensitive
            if is_sensitive(variable_name) and not include_sensitive:
                return {
                    "variable": variable_name,
                    "value": "[SENSITIVE - use include_sensitive=true to view]",
                    "is_sensitive": True,
                }

            return {
                "variable": variable_name,
                "value": value,
                "is_sensitive": is_sensitive(variable_name),
            }

        else:
            # Get all or filtered variables
            env_vars = {}

            for var_name, var_value in os.environ.items():
                # Apply filter pattern if specified
                if filter_pattern and filter_pattern.lower() not in var_name.lower():
                    continue

                # Handle sensitive variables
                if is_sensitive(var_name) and not include_sensitive:
                    env_vars[var_name] = "[SENSITIVE]"
                else:
                    env_vars[var_name] = var_value

            return {
                "environment_variables": env_vars,
                "total_variables": len(env_vars),
                "filter_applied": filter_pattern,
                "sensitive_included": include_sensitive,
            }


class ExecuteCommandTool(BaseTool):
    """Tool to execute safe system commands."""

    def _create_definition(self) -> ToolDefinition:
        """Create tool definition."""
        return create_simple_tool_definition(
            name="execute_command",
            description="Execute safe system commands (restricted for security)",
            category=ToolCategory.SYSTEM,
            parameters=[
                self.create_string_parameter(
                    "command",
                    "Command to execute (must be in allowed list)",
                    required=True,
                    examples=["ls", "pwd", "date", "whoami", "uname -a"],
                ),
                self.create_array_parameter(
                    "args", "Command arguments", required=False, examples=[[], ["-la"], ["-a"]]
                ),
                self.create_integer_parameter(
                    "timeout",
                    "Timeout in seconds",
                    required=False,
                    default=10,
                    minimum=1,
                    maximum=60,
                ),
            ],
            examples=[
                {"command": "ls", "args": ["-la"]},
                {"command": "pwd"},
                {"command": "date", "timeout": 5},
            ],
            tags=["system", "command", "execution"],
            timeout_seconds=60.0,
            is_dangerous=True,  # Mark as dangerous
        )

    async def _execute_impl(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the tool."""
        command = self.validate_string_not_empty(parameters["command"], "command")
        args = parameters.get("args", [])
        timeout = parameters.get("timeout", 10)

        # Define allowed commands for security
        allowed_commands = {
            # File system (read-only)
            "ls",
            "dir",
            "pwd",
            "find",
            "locate",
            "which",
            "whereis",
            # System info
            "date",
            "whoami",
            "id",
            "uname",
            "hostname",
            "uptime",
            "ps",
            "top",
            "df",
            "du",
            "free",
            "lscpu",
            "lsblk",
            # Network (safe)
            "ping",
            "nslookup",
            "dig",
            "curl",
            "wget",
            # Text processing
            "cat",
            "head",
            "tail",
            "grep",
            "wc",
            "sort",
            "uniq",
            # Python
            "python",
            "python3",
            "pip",
            "pip3",
        }

        # Check if command is allowed
        base_command = command.split()[0] if " " in command else command
        if base_command not in allowed_commands:
            raise ValueError(f"Command '{base_command}' is not allowed for security reasons")

        # Additional safety checks
        dangerous_patterns = [
            "rm",
            "del",
            "format",
            "mkfs",
            "dd",
            "sudo",
            "su",
            ">",
            ">>",
            "|",
            "&",
            ";",
        ]
        full_command = f"{command} {' '.join(args)}"

        for pattern in dangerous_patterns:
            if pattern in full_command.lower():
                raise ValueError(f"Command contains dangerous pattern: {pattern}")

        try:
            # Execute the command
            cmd_list = [command] + args

            result = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,  # Don't raise exception on non-zero exit
            )

            return {
                "command": command,
                "args": args,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
                "execution_time": timeout,  # Approximate
            }

        except subprocess.TimeoutExpired:
            raise ValueError(f"Command timed out after {timeout} seconds")
        except FileNotFoundError:
            raise ValueError(f"Command '{command}' not found")
        except PermissionError:
            raise ValueError(f"Permission denied executing command '{command}'")
        except OSError as e:
            raise ValueError(f"OS error executing command '{command}': {e}")
        except Exception as e:
            raise ValueError(f"Command execution failed: {e}")


class GetProcessInfoTool(BaseTool):
    """Tool to get information about running processes."""

    def _create_definition(self) -> ToolDefinition:
        """Create tool definition."""
        return create_simple_tool_definition(
            name="get_process_info",
            description="Get information about running processes",
            category=ToolCategory.SYSTEM,
            parameters=[
                self.create_string_parameter(
                    "filter_name",
                    "Filter processes by name pattern",
                    required=False,
                    examples=["python", "coda", "chrome"],
                ),
                self.create_integer_parameter(
                    "limit",
                    "Maximum number of processes to return",
                    required=False,
                    default=20,
                    minimum=1,
                    maximum=100,
                ),
                self.create_boolean_parameter(
                    "include_details",
                    "Include detailed process information",
                    required=False,
                    default=False,
                ),
            ],
            examples=[
                {"filter_name": "python", "limit": 10},
                {"include_details": True, "limit": 5},
            ],
            tags=["system", "processes", "monitoring"],
            timeout_seconds=15.0,
        )

    async def _execute_impl(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the tool."""
        filter_name = parameters.get("filter_name")
        limit = parameters.get("limit", 20)
        include_details = parameters.get("include_details", False)

        try:
            import psutil
        except ImportError:
            raise ValueError("psutil library not available for process information")

        try:
            processes = []

            for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                try:
                    proc_info = proc.info

                    # Apply name filter if specified
                    if filter_name and filter_name.lower() not in proc_info["name"].lower():
                        continue

                    process_data = {
                        "pid": proc_info["pid"],
                        "name": proc_info["name"],
                        "cpu_percent": proc_info["cpu_percent"],
                        "memory_percent": round(proc_info["memory_percent"], 2),
                    }

                    # Add detailed info if requested
                    if include_details:
                        try:
                            process_data.update(
                                {
                                    "status": proc.status(),
                                    "create_time": proc.create_time(),
                                    "num_threads": proc.num_threads(),
                                    "memory_info": {
                                        "rss": proc.memory_info().rss,
                                        "vms": proc.memory_info().vms,
                                    },
                                }
                            )
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                    processes.append(process_data)

                    # Limit results
                    if len(processes) >= limit:
                        break

                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    # Skip processes we can't access
                    continue

            return {
                "processes": processes,
                "total_found": len(processes),
                "filter_applied": filter_name,
                "include_details": include_details,
            }

        except Exception as e:
            raise ValueError(f"Failed to get process information: {e}")
