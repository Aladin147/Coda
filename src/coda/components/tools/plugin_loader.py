"""
Plugin loader for the tool system.

This module provides dynamic loading of tool plugins from Python files.
"""

import asyncio
import importlib
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base_tool import BaseTool
from .interfaces import PluginLoaderInterface, ToolInterface
from .plugin_metadata import (
    PluginDependencyResolver,
    PluginManifestLoader,
    PluginMetadata,
)

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
        self._plugin_metadata: Dict[str, PluginMetadata] = {}  # plugin_path -> metadata

        # Plugin management components
        self._manifest_loader = PluginManifestLoader()
        self._dependency_resolver = PluginDependencyResolver()

        # Plugin lifecycle hooks
        self._startup_hooks: Dict[str, callable] = {}
        self._shutdown_hooks: Dict[str, callable] = {}

        logger.info("PluginLoader initialized")

    async def load_plugin(
        self, plugin_path: str
    ) -> Tuple[List[ToolInterface], Optional[PluginMetadata]]:
        """
        Load tools from a plugin file or directory.

        Args:
            plugin_path: Path to the plugin file or directory

        Returns:
            Tuple of (loaded tools, plugin metadata)

        Raises:
            ValueError: If plugin loading fails
        """
        plugin_path = os.path.abspath(plugin_path)

        if not os.path.exists(plugin_path):
            raise ValueError(f"Plugin path not found: {plugin_path}")

        try:
            # Handle directory-based plugins with manifests
            if os.path.isdir(plugin_path):
                return await self._load_directory_plugin(plugin_path)

            # Handle single file plugins
            if not plugin_path.endswith(".py"):
                raise ValueError(f"Plugin file must be a Python file: {plugin_path}")

            return await self._load_file_plugin(plugin_path)

        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_path}: {e}")
            raise ValueError(f"Plugin loading failed: {e}")

    async def _load_directory_plugin(
        self, plugin_dir: str
    ) -> Tuple[List[ToolInterface], Optional[PluginMetadata]]:
        """Load plugin from directory with manifest."""
        plugin_path = Path(plugin_dir)

        # Load manifest
        metadata = self._manifest_loader.load_manifest(plugin_path)
        if not metadata:
            # Fallback to scanning for Python files
            logger.warning(f"No manifest found in {plugin_dir}, scanning for Python files")
            return await self._load_directory_fallback(plugin_dir)

        # Check dependencies
        missing_deps = self._dependency_resolver.resolve_dependencies(metadata)
        if missing_deps:
            raise ValueError(f"Missing dependencies: {', '.join(missing_deps)}")

        # Load entry point
        entry_point = plugin_path / metadata.entry_point
        if not entry_point.exists():
            raise ValueError(f"Entry point not found: {entry_point}")

        # Load the module
        module = await self._load_module(str(entry_point))

        # Extract tools
        tools = self._extract_tools_from_module(module, str(entry_point))

        # Execute startup hook if present
        if metadata.startup_hook:
            await self._execute_lifecycle_hook(module, metadata.startup_hook, "startup")

        # Store plugin info
        self._loaded_plugins[plugin_dir] = [tool.get_definition().name for tool in tools]
        self._plugin_modules[plugin_dir] = module
        self._plugin_metadata[plugin_dir] = metadata
        self._dependency_resolver.add_plugin(metadata)

        logger.info(f"Loaded plugin: {metadata.name} v{metadata.version} with {len(tools)} tools")
        return tools, metadata

    async def _load_file_plugin(
        self, plugin_path: str
    ) -> Tuple[List[ToolInterface], Optional[PluginMetadata]]:
        """Load plugin from single Python file."""
        # Load the module
        module = await self._load_module(plugin_path)

        # Extract tools from the module
        tools = self._extract_tools_from_module(module, plugin_path)

        # Store plugin info
        self._loaded_plugins[plugin_path] = [tool.get_definition().name for tool in tools]
        self._plugin_modules[plugin_path] = module

        logger.info(f"Loaded plugin: {plugin_path} with {len(tools)} tools")
        return tools, None

    async def _load_directory_fallback(
        self, plugin_dir: str
    ) -> Tuple[List[ToolInterface], Optional[PluginMetadata]]:
        """Fallback loading for directories without manifests."""
        plugin_path = Path(plugin_dir)
        all_tools = []

        # Look for Python files
        for py_file in plugin_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue

            try:
                module = await self._load_module(str(py_file))
                tools = self._extract_tools_from_module(module, str(py_file))
                all_tools.extend(tools)
            except Exception as e:
                logger.warning(f"Failed to load {py_file}: {e}")

        # Store plugin info
        tool_names = [tool.get_definition().name for tool in all_tools]
        self._loaded_plugins[plugin_dir] = tool_names

        logger.info(f"Loaded directory plugin: {plugin_dir} with {len(all_tools)} tools")
        return all_tools, None

    async def discover_plugins(self, directory: str) -> List[str]:
        """
        Discover plugin files and directories.

        Args:
            directory: Directory to search for plugins

        Returns:
            List of plugin paths (files or directories)
        """
        if not os.path.exists(directory):
            logger.warning(f"Plugin directory not found: {directory}")
            return []

        plugin_paths = []

        try:
            directory_path = Path(directory)

            # Look for directory-based plugins (with manifests)
            for item in directory_path.iterdir():
                if item.is_dir() and not item.name.startswith("."):
                    # Check if directory has a manifest
                    has_manifest = any(
                        (item / filename).exists()
                        for filename in self._manifest_loader.MANIFEST_FILENAMES
                    )
                    if has_manifest:
                        plugin_paths.append(str(item))
                        continue

                    # Check if directory has Python files that look like plugins
                    has_plugin_files = any(
                        py_file.suffix == ".py" and not py_file.name.startswith("__")
                        for py_file in item.glob("*.py")
                    )
                    if has_plugin_files:
                        plugin_paths.append(str(item))

            # Look for single-file plugins
            for py_file in directory_path.glob("*.py"):
                if not py_file.name.startswith("__"):
                    if await self._is_valid_plugin(str(py_file)):
                        plugin_paths.append(str(py_file))

            logger.info(f"Discovered {len(plugin_paths)} plugins in {directory}")
            return plugin_paths

        except Exception as e:
            logger.error(f"Failed to discover plugins in {directory}: {e}")
            return []

    async def reload_plugin(
        self, plugin_path: str
    ) -> Tuple[List[ToolInterface], Optional[PluginMetadata]]:
        """
        Reload a plugin.

        Args:
            plugin_path: Path to the plugin file or directory

        Returns:
            Tuple of (reloaded tools, plugin metadata)
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

    def get_plugin_metadata(self, plugin_path: str) -> Optional[PluginMetadata]:
        """
        Get metadata for a loaded plugin.

        Args:
            plugin_path: Path to the plugin

        Returns:
            Plugin metadata if available
        """
        return self._plugin_metadata.get(os.path.abspath(plugin_path))

    async def load_plugins_with_dependencies(
        self, plugin_paths: List[str]
    ) -> List[Tuple[List[ToolInterface], Optional[PluginMetadata]]]:
        """
        Load multiple plugins in dependency order.

        Args:
            plugin_paths: List of plugin paths

        Returns:
            List of (tools, metadata) tuples in load order
        """
        # First pass: load metadata for all plugins
        plugin_metadata = []
        for path in plugin_paths:
            if os.path.isdir(path):
                metadata = self._manifest_loader.load_manifest(Path(path))
                if metadata:
                    plugin_metadata.append((path, metadata))

        # Calculate load order
        metadata_only = [meta for _, meta in plugin_metadata]
        ordered_metadata = self._dependency_resolver.get_load_order(metadata_only)

        # Create path mapping
        path_map = {meta.name: path for path, meta in plugin_metadata}

        # Load plugins in order
        results = []
        for metadata in ordered_metadata:
            plugin_path = path_map.get(metadata.name)
            if plugin_path:
                try:
                    tools, loaded_metadata = await self.load_plugin(plugin_path)
                    results.append((tools, loaded_metadata))
                except Exception as e:
                    logger.error(f"Failed to load plugin {plugin_path}: {e}")

        # Load remaining plugins without metadata
        for path in plugin_paths:
            if path not in [path_map.get(meta.name) for meta in ordered_metadata]:
                try:
                    tools, metadata = await self.load_plugin(path)
                    results.append((tools, metadata))
                except Exception as e:
                    logger.error(f"Failed to load plugin {path}: {e}")

        return results

    async def _execute_lifecycle_hook(self, module: Any, hook_name: str, hook_type: str) -> None:
        """Execute a plugin lifecycle hook."""
        try:
            if hasattr(module, hook_name):
                hook_func = getattr(module, hook_name)
                if callable(hook_func):
                    if asyncio.iscoroutinefunction(hook_func):
                        await hook_func()
                    else:
                        hook_func()
                    logger.info(f"Executed {hook_type} hook: {hook_name}")
                else:
                    logger.warning(f"{hook_type} hook {hook_name} is not callable")
            else:
                logger.debug(f"No {hook_type} hook {hook_name} found")
        except Exception as e:
            logger.error(f"Failed to execute {hook_type} hook {hook_name}: {e}")

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
            # Execute shutdown hook if present
            metadata = self._plugin_metadata.get(plugin_path)
            if metadata and metadata.shutdown_hook:
                module = self._plugin_modules.get(plugin_path)
                if module:
                    await self._execute_lifecycle_hook(module, metadata.shutdown_hook, "shutdown")

            # Remove from dependency resolver
            if metadata:
                self._dependency_resolver.remove_plugin(metadata.name)

            # Remove from loaded plugins
            tool_names = self._loaded_plugins.pop(plugin_path, [])

            # Remove module reference
            if plugin_path in self._plugin_modules:
                module = self._plugin_modules.pop(plugin_path)

                # Remove from sys.modules if it was added
                module_name = getattr(module, "__name__", None)
                if module_name and module_name in sys.modules:
                    del sys.modules[module_name]

            # Remove metadata
            self._plugin_metadata.pop(plugin_path, None)

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
            if attr_name.startswith("_"):
                continue

            attr = getattr(module, attr_name)

            # Check if it's a tool class
            if isinstance(attr, type) and issubclass(attr, BaseTool) and attr != BaseTool:
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
        if hasattr(module, "get_tools"):
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
            with open(plugin_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Look for common plugin patterns
            plugin_indicators = [
                "BaseTool",
                "ToolInterface",
                "class.*Tool",
                "def get_tools",
                "from.*tools.*import",
                "from coda.components.tools",
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
            "module_name": getattr(module, "__name__", "unknown") if module else "unknown",
            "module_doc": getattr(module, "__doc__", None) if module else None,
            "loaded": True,
        }

    def get_all_plugin_info(self) -> List[Dict[str, Any]]:
        """Get information about all loaded plugins."""
        return [self.get_plugin_info(plugin_path) for plugin_path in self._loaded_plugins.keys()]

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
            "tool_names": [],
        }

        try:
            if not os.path.exists(plugin_path):
                validation_result["errors"].append("Plugin file not found")
                return validation_result

            if not plugin_path.endswith(".py"):
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
                        validation_result["warnings"].append(
                            f"Tool {definition.name} has no description"
                        )
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
