"""
Plugin loader for the tool system.

This module provides dynamic loading of tool plugins from Python files.
"""

import os
import sys
import importlib
import importlib.util
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from .interfaces import PluginLoaderInterface, ToolInterface
from .base_tool import BaseTool

logger = logging.getLogger("coda.tools.plugin_loader")


class PluginLoader(PluginLoaderInterface):
    """
    Loads tool plugins from Python files.
    
    Features:
    - Dynamic plugin discovery
    - Safe plugin loading with error handling
    - Plugin reloading support
    - Plugin dependency management
    """
    
    def __init__(self):
        """Initialize the plugin loader."""
        self._loaded_plugins: Dict[str, List[str]] = {}  # plugin_path -> tool_names
        self._plugin_modules: Dict[str, Any] = {}  # plugin_path -> module
        
        logger.info("PluginLoader initialized")
    
    async def load_plugin(self, plugin_path: str) -> List[ToolInterface]:
        """
        Load tools from a plugin file.
        
        Args:
            plugin_path: Path to the plugin file
            
        Returns:
            List of loaded tools
            
        Raises:
            ValueError: If plugin loading fails
        """
        plugin_path = os.path.abspath(plugin_path)
        
        if not os.path.exists(plugin_path):
            raise ValueError(f"Plugin file not found: {plugin_path}")
        
        if not plugin_path.endswith('.py'):
            raise ValueError(f"Plugin file must be a Python file: {plugin_path}")
        
        try:
            # Load the module
            module = await self._load_module(plugin_path)
            
            # Extract tools from the module
            tools = self._extract_tools_from_module(module, plugin_path)
            
            # Store plugin info
            self._loaded_plugins[plugin_path] = [tool.get_definition().name for tool in tools]
            self._plugin_modules[plugin_path] = module
            
            logger.info(f"Loaded plugin: {plugin_path} with {len(tools)} tools")
            return tools
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_path}: {e}")
            raise ValueError(f"Plugin loading failed: {e}")
    
    async def discover_plugins(self, directory: str) -> List[str]:
        """
        Discover plugin files in a directory.
        
        Args:
            directory: Directory to search for plugins
            
        Returns:
            List of plugin file paths
        """
        if not os.path.exists(directory):
            logger.warning(f"Plugin directory not found: {directory}")
            return []
        
        plugin_files = []
        
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        plugin_path = os.path.join(root, file)
                        
                        # Check if it looks like a plugin
                        if await self._is_valid_plugin(plugin_path):
                            plugin_files.append(plugin_path)
            
            logger.info(f"Discovered {len(plugin_files)} plugins in {directory}")
            return plugin_files
            
        except Exception as e:
            logger.error(f"Failed to discover plugins in {directory}: {e}")
            return []
    
    async def reload_plugin(self, plugin_path: str) -> List[ToolInterface]:
        """
        Reload a plugin.
        
        Args:
            plugin_path: Path to the plugin file
            
        Returns:
            List of reloaded tools
        """
        plugin_path = os.path.abspath(plugin_path)
        
        # Unload existing plugin
        if plugin_path in self._plugin_modules:
            await self.unload_plugin(plugin_path)
        
        # Load the plugin again
        return await self.load_plugin(plugin_path)
    
    def get_loaded_plugins(self) -> Dict[str, List[str]]:
        """
        Get list of loaded plugins and their tools.
        
        Returns:
            Dictionary mapping plugin paths to tool names
        """
        return self._loaded_plugins.copy()
    
    async def unload_plugin(self, plugin_path: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            plugin_path: Path to the plugin file
            
        Returns:
            True if unloading successful
        """
        plugin_path = os.path.abspath(plugin_path)
        
        if plugin_path not in self._loaded_plugins:
            logger.warning(f"Plugin not loaded: {plugin_path}")
            return False
        
        try:
            # Remove from loaded plugins
            tool_names = self._loaded_plugins.pop(plugin_path, [])
            
            # Remove module reference
            if plugin_path in self._plugin_modules:
                module = self._plugin_modules.pop(plugin_path)
                
                # Remove from sys.modules if it was added
                module_name = getattr(module, '__name__', None)
                if module_name and module_name in sys.modules:
                    del sys.modules[module_name]
            
            logger.info(f"Unloaded plugin: {plugin_path} ({len(tool_names)} tools)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_path}: {e}")
            return False
    
    async def _load_module(self, plugin_path: str) -> Any:
        """Load a Python module from file path."""
        module_name = f"coda_plugin_{os.path.basename(plugin_path)[:-3]}"
        
        # Load module from file
        spec = importlib.util.spec_from_file_location(module_name, plugin_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load module spec from {plugin_path}")
        
        module = importlib.util.module_from_spec(spec)
        
        # Execute the module
        spec.loader.exec_module(module)
        
        return module
    
    def _extract_tools_from_module(self, module: Any, plugin_path: str) -> List[ToolInterface]:
        """Extract tool instances from a loaded module."""
        tools = []
        
        # Look for tool classes and instances
        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue
            
            attr = getattr(module, attr_name)
            
            # Check if it's a tool class
            if (isinstance(attr, type) and 
                issubclass(attr, BaseTool) and 
                attr != BaseTool):
                try:
                    # Instantiate the tool
                    tool_instance = attr()
                    tools.append(tool_instance)
                    logger.debug(f"Found tool class: {attr_name} in {plugin_path}")
                except Exception as e:
                    logger.warning(f"Failed to instantiate tool {attr_name}: {e}")
            
            # Check if it's already a tool instance
            elif isinstance(attr, ToolInterface):
                tools.append(attr)
                logger.debug(f"Found tool instance: {attr_name} in {plugin_path}")
        
        # Look for a special function that returns tools
        if hasattr(module, 'get_tools'):
            try:
                additional_tools = module.get_tools()
                if isinstance(additional_tools, list):
                    for tool in additional_tools:
                        if isinstance(tool, ToolInterface):
                            tools.append(tool)
                elif isinstance(additional_tools, ToolInterface):
                    tools.append(additional_tools)
            except Exception as e:
                logger.warning(f"Error calling get_tools() in {plugin_path}: {e}")
        
        return tools
    
    async def _is_valid_plugin(self, plugin_path: str) -> bool:
        """Check if a file is a valid plugin."""
        try:
            # Read the file and look for plugin indicators
            with open(plugin_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for common plugin patterns
            plugin_indicators = [
                'BaseTool',
                'ToolInterface',
                'class.*Tool',
                'def get_tools',
                'from.*tools.*import',
                'from coda.components.tools'
            ]
            
            import re
            for indicator in plugin_indicators:
                if re.search(indicator, content, re.IGNORECASE):
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking plugin validity for {plugin_path}: {e}")
            return False
    
    def get_plugin_info(self, plugin_path: str) -> Optional[Dict[str, Any]]:
        """Get information about a loaded plugin."""
        plugin_path = os.path.abspath(plugin_path)
        
        if plugin_path not in self._loaded_plugins:
            return None
        
        module = self._plugin_modules.get(plugin_path)
        
        return {
            "path": plugin_path,
            "tool_names": self._loaded_plugins[plugin_path],
            "tool_count": len(self._loaded_plugins[plugin_path]),
            "module_name": getattr(module, '__name__', 'unknown') if module else 'unknown',
            "module_doc": getattr(module, '__doc__', None) if module else None,
            "loaded": True
        }
    
    def get_all_plugin_info(self) -> List[Dict[str, Any]]:
        """Get information about all loaded plugins."""
        return [
            self.get_plugin_info(plugin_path)
            for plugin_path in self._loaded_plugins.keys()
        ]
    
    async def validate_plugin(self, plugin_path: str) -> Dict[str, Any]:
        """
        Validate a plugin without loading it.
        
        Args:
            plugin_path: Path to the plugin file
            
        Returns:
            Validation results
        """
        validation_result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "tool_count": 0,
            "tool_names": []
        }
        
        try:
            if not os.path.exists(plugin_path):
                validation_result["errors"].append("Plugin file not found")
                return validation_result
            
            if not plugin_path.endswith('.py'):
                validation_result["errors"].append("Plugin must be a Python file")
                return validation_result
            
            # Try to load and validate the module
            module = await self._load_module(plugin_path)
            tools = self._extract_tools_from_module(module, plugin_path)
            
            validation_result["tool_count"] = len(tools)
            validation_result["tool_names"] = [tool.get_definition().name for tool in tools]
            
            if len(tools) == 0:
                validation_result["warnings"].append("No tools found in plugin")
            else:
                validation_result["valid"] = True
            
            # Validate each tool
            for tool in tools:
                try:
                    definition = tool.get_definition()
                    # Basic validation
                    if not definition.name:
                        validation_result["errors"].append(f"Tool has empty name")
                    if not definition.description:
                        validation_result["warnings"].append(f"Tool {definition.name} has no description")
                except Exception as e:
                    validation_result["errors"].append(f"Tool validation error: {e}")
            
            if validation_result["errors"]:
                validation_result["valid"] = False
            
        except Exception as e:
            validation_result["errors"].append(f"Plugin validation failed: {e}")
        
        return validation_result
    
    def clear_all_plugins(self) -> None:
        """Clear all loaded plugins."""
        plugin_paths = list(self._loaded_plugins.keys())
        
        for plugin_path in plugin_paths:
            try:
                self.unload_plugin(plugin_path)
            except Exception as e:
                logger.warning(f"Error unloading plugin {plugin_path}: {e}")
        
        self._loaded_plugins.clear()
        self._plugin_modules.clear()
        
        logger.info("Cleared all loaded plugins")
