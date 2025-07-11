"""
Plugin metadata system for Coda tools.

This module provides plugin manifest support, dependency management,
and metadata validation for the plugin system.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("coda.tools.plugin_metadata")


class PluginDependency(BaseModel):
    """Plugin dependency specification."""

    name: str = Field(..., description="Name of the dependency")
    version: Optional[str] = Field(None, description="Required version (semver)")
    optional: bool = Field(False, description="Whether dependency is optional")
    source: Optional[str] = Field(None, description="Source URL or package name")


class PluginAuthor(BaseModel):
    """Plugin author information."""

    name: str = Field(..., description="Author name")
    email: Optional[str] = Field(None, description="Author email")
    url: Optional[str] = Field(None, description="Author website")


class PluginMetadata(BaseModel):
    """Plugin metadata from manifest file."""

    # Basic information
    name: str = Field(..., description="Plugin name")
    version: str = Field(..., description="Plugin version")
    description: str = Field(..., description="Plugin description")
    author: PluginAuthor = Field(..., description="Plugin author")

    # Plugin details
    entry_point: str = Field("plugin.py", description="Main plugin file")
    category: str = Field("general", description="Plugin category")
    tags: List[str] = Field(default_factory=list, description="Plugin tags")

    # Requirements
    python_version: str = Field(">=3.8", description="Required Python version")
    coda_version: str = Field(">=0.1.0", description="Required Coda version")
    dependencies: List[PluginDependency] = Field(default_factory=list)

    # Security and permissions
    dangerous: bool = Field(False, description="Whether plugin contains dangerous operations")
    permissions: List[str] = Field(default_factory=list, description="Required permissions")

    # Configuration
    config_schema: Optional[Dict[str, Any]] = Field(None, description="Plugin configuration schema")
    default_config: Optional[Dict[str, Any]] = Field(None, description="Default configuration")

    # Lifecycle hooks
    startup_hook: Optional[str] = Field(None, description="Startup function name")
    shutdown_hook: Optional[str] = Field(None, description="Shutdown function name")

    # Metadata
    created_at: Optional[datetime] = Field(None, description="Plugin creation date")
    updated_at: Optional[datetime] = Field(None, description="Last update date")
    license: Optional[str] = Field(None, description="Plugin license")
    homepage: Optional[str] = Field(None, description="Plugin homepage")
    repository: Optional[str] = Field(None, description="Plugin repository URL")

    @field_validator("version")
    @classmethod
    def validate_version(cls, v):
        """Validate version format."""
        import re

        if not re.match(r"^\d+\.\d+\.\d+", v):
            raise ValueError("Version must follow semantic versioning (x.y.z)")
        return v

    @field_validator("permissions")
    @classmethod
    def validate_permissions(cls, v):
        """Validate permission names."""
        valid_permissions = {
            "file_system",
            "network",
            "system_commands",
            "environment_vars",
            "database",
            "external_apis",
            "user_data",
            "dangerous_operations",
        }

        for permission in v:
            if permission not in valid_permissions:
                raise ValueError(f"Invalid permission: {permission}")

        return v


class PluginManifestLoader:
    """Loads and validates plugin manifests."""

    MANIFEST_FILENAMES = ["plugin.yaml", "plugin.yml", "manifest.yaml", "manifest.yml"]

    def __init__(self):
        """Initialize manifest loader."""
        self.logger = logging.getLogger("coda.tools.plugin_metadata.loader")

    def load_manifest(self, plugin_dir: Path) -> Optional[PluginMetadata]:
        """
        Load plugin manifest from directory.

        Args:
            plugin_dir: Plugin directory path

        Returns:
            Plugin metadata if found and valid, None otherwise
        """
        if not plugin_dir.is_dir():
            return None

        # Find manifest file
        manifest_path = None
        for filename in self.MANIFEST_FILENAMES:
            candidate = plugin_dir / filename
            if candidate.exists():
                manifest_path = candidate
                break

        if not manifest_path:
            self.logger.debug(f"No manifest file found in {plugin_dir}")
            return None

        try:
            # Load YAML content
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = yaml.safe_load(f)

            if not manifest_data:
                self.logger.warning(f"Empty manifest file: {manifest_path}")
                return None

            # Validate and create metadata
            metadata = PluginMetadata(**manifest_data)

            # Resolve entry point path
            entry_point_path = plugin_dir / metadata.entry_point
            if not entry_point_path.exists():
                self.logger.error(f"Entry point not found: {entry_point_path}")
                return None

            self.logger.info(f"Loaded plugin manifest: {metadata.name} v{metadata.version}")
            return metadata

        except yaml.YAMLError as e:
            self.logger.error(f"Invalid YAML in manifest {manifest_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load manifest {manifest_path}: {e}")
            return None

    def create_manifest_template(self, plugin_dir: Path, plugin_name: str) -> Path:
        """
        Create a template manifest file.

        Args:
            plugin_dir: Plugin directory
            plugin_name: Name of the plugin

        Returns:
            Path to created manifest file
        """
        plugin_dir.mkdir(parents=True, exist_ok=True)

        template = {
            "name": plugin_name,
            "version": "1.0.0",
            "description": f"A Coda plugin: {plugin_name}",
            "author": {"name": "Plugin Author", "email": "author@example.com"},
            "entry_point": "plugin.py",
            "category": "general",
            "tags": [],
            "python_version": ">=3.8",
            "coda_version": ">=0.1.0",
            "dependencies": [],
            "dangerous": False,
            "permissions": [],
            "config_schema": {},
            "default_config": {},
        }

        manifest_path = plugin_dir / "plugin.yaml"

        with open(manifest_path, "w", encoding="utf-8") as f:
            yaml.dump(template, f, default_flow_style=False, sort_keys=False)

        self.logger.info(f"Created manifest template: {manifest_path}")
        return manifest_path


class PluginDependencyResolver:
    """Resolves plugin dependencies."""

    def __init__(self):
        """Initialize dependency resolver."""
        self.logger = logging.getLogger("coda.tools.plugin_metadata.resolver")
        self._loaded_plugins: Dict[str, PluginMetadata] = {}

    def add_plugin(self, metadata: PluginMetadata) -> None:
        """Add plugin to dependency tracking."""
        self._loaded_plugins[metadata.name] = metadata

    def remove_plugin(self, plugin_name: str) -> None:
        """Remove plugin from dependency tracking."""
        self._loaded_plugins.pop(plugin_name, None)

    def resolve_dependencies(self, metadata: PluginMetadata) -> List[str]:
        """
        Resolve plugin dependencies.

        Args:
            metadata: Plugin metadata

        Returns:
            List of missing dependencies
        """
        missing_deps = []

        for dep in metadata.dependencies:
            if dep.name not in self._loaded_plugins:
                if not dep.optional:
                    missing_deps.append(dep.name)
                    self.logger.warning(f"Missing required dependency: {dep.name}")
                else:
                    self.logger.info(f"Optional dependency not found: {dep.name}")
            else:
                # Check version compatibility if specified
                if dep.version:
                    loaded_version = self._loaded_plugins[dep.name].version
                    if not self._is_version_compatible(loaded_version, dep.version):
                        missing_deps.append(f"{dep.name} (version {dep.version})")
                        self.logger.error(
                            f"Version mismatch for {dep.name}: need {dep.version}, have {loaded_version}"
                        )

        return missing_deps

    def get_load_order(self, plugins: List[PluginMetadata]) -> List[PluginMetadata]:
        """
        Calculate plugin load order based on dependencies.

        Args:
            plugins: List of plugin metadata

        Returns:
            Plugins sorted by dependency order
        """
        # Simple topological sort
        loaded = set()
        ordered = []
        remaining = plugins.copy()

        while remaining:
            # Find plugins with no unmet dependencies
            ready = []
            for plugin in remaining:
                deps_met = all(dep.name in loaded or dep.optional for dep in plugin.dependencies)
                if deps_met:
                    ready.append(plugin)

            if not ready:
                # Circular dependency or missing dependency
                self.logger.error("Circular dependency detected or missing dependencies")
                break

            # Add ready plugins to load order
            for plugin in ready:
                ordered.append(plugin)
                loaded.add(plugin.name)
                remaining.remove(plugin)

        return ordered

    def _is_version_compatible(self, available: str, required: str) -> bool:
        """Check if available version meets requirement."""
        # Simple version comparison - in production, use proper semver library
        try:
            available_parts = [int(x) for x in available.split(".")]

            if required.startswith(">="):
                required_parts = [int(x) for x in required[2:].split(".")]
                return available_parts >= required_parts
            elif required.startswith(">"):
                required_parts = [int(x) for x in required[1:].split(".")]
                return available_parts > required_parts
            elif required.startswith("<="):
                required_parts = [int(x) for x in required[2:].split(".")]
                return available_parts <= required_parts
            elif required.startswith("<"):
                required_parts = [int(x) for x in required[1:].split(".")]
                return available_parts < required_parts
            else:
                # Exact match
                required_parts = [int(x) for x in required.split(".")]
                return available_parts == required_parts

        except (ValueError, IndexError):
            self.logger.warning(
                f"Invalid version format: available={available}, required={required}"
            )
            return False
