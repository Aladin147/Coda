"""
Tests for the enhanced plugin system.
"""

import pytest
import asyncio
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from src.coda.components.tools.plugin_loader import PluginLoader
from src.coda.components.tools.plugin_metadata import (
    PluginMetadata, PluginManifestLoader, PluginDependencyResolver,
    PluginDependency, PluginAuthor
)
from src.coda.components.tools.manager import ToolManager
from src.coda.components.tools.base_tool import BaseTool
from src.coda.components.tools.models import ToolDefinition, ToolParameter, ToolCategory


class MockPluginTool(BaseTool):
    """Mock tool for plugin testing."""

    def _create_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="test_plugin_tool",
            description="A test tool from plugin",
            category=ToolCategory.UTILITY,
            parameters=[
                ToolParameter(
                    name="message",
                    type="string",
                    description="Test message",
                    required=True
                )
            ]
        )

    async def _execute_impl(self, parameters: dict, context: dict) -> str:
        return f"Plugin tool executed: {parameters.get('message', 'no message')}"


class TestPluginMetadata:
    """Test plugin metadata system."""
    
    def test_plugin_metadata_creation(self):
        """Test creating plugin metadata."""
        author = PluginAuthor(name="Test Author", email="test@example.com")
        
        metadata = PluginMetadata(
            name="test-plugin",
            version="1.0.0",
            description="A test plugin",
            author=author,
            entry_point="plugin.py",
            category="test",
            dangerous=False
        )
        
        assert metadata.name == "test-plugin"
        assert metadata.version == "1.0.0"
        assert metadata.author.name == "Test Author"
        assert not metadata.dangerous
    
    def test_plugin_metadata_validation(self):
        """Test plugin metadata validation."""
        author = PluginAuthor(name="Test Author")
        
        # Test invalid version
        with pytest.raises(ValueError, match="Version must follow semantic versioning"):
            PluginMetadata(
                name="test-plugin",
                version="invalid",
                description="A test plugin",
                author=author
            )
        
        # Test invalid permission
        with pytest.raises(ValueError, match="Invalid permission"):
            PluginMetadata(
                name="test-plugin",
                version="1.0.0",
                description="A test plugin",
                author=author,
                permissions=["invalid_permission"]
            )
    
    def test_plugin_dependency(self):
        """Test plugin dependency specification."""
        dep = PluginDependency(
            name="required-plugin",
            version=">=1.0.0",
            optional=False
        )
        
        assert dep.name == "required-plugin"
        assert dep.version == ">=1.0.0"
        assert not dep.optional


class TestPluginManifestLoader:
    """Test plugin manifest loader."""
    
    def test_load_manifest_success(self):
        """Test successful manifest loading."""
        loader = PluginManifestLoader()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            plugin_dir = Path(temp_dir) / "test_plugin"
            plugin_dir.mkdir()
            
            # Create manifest file
            manifest_data = {
                'name': 'test-plugin',
                'version': '1.0.0',
                'description': 'A test plugin',
                'author': {
                    'name': 'Test Author',
                    'email': 'test@example.com'
                },
                'entry_point': 'plugin.py',
                'category': 'test'
            }
            
            manifest_path = plugin_dir / 'plugin.yaml'
            with open(manifest_path, 'w') as f:
                yaml.dump(manifest_data, f)
            
            # Create entry point file
            (plugin_dir / 'plugin.py').touch()
            
            # Load manifest
            metadata = loader.load_manifest(plugin_dir)
            
            assert metadata is not None
            assert metadata.name == 'test-plugin'
            assert metadata.version == '1.0.0'
            assert metadata.author.name == 'Test Author'
    
    def test_load_manifest_missing_file(self):
        """Test loading manifest when file is missing."""
        loader = PluginManifestLoader()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            plugin_dir = Path(temp_dir) / "test_plugin"
            plugin_dir.mkdir()
            
            metadata = loader.load_manifest(plugin_dir)
            assert metadata is None
    
    def test_create_manifest_template(self):
        """Test creating manifest template."""
        loader = PluginManifestLoader()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            plugin_dir = Path(temp_dir) / "test_plugin"
            
            manifest_path = loader.create_manifest_template(plugin_dir, "test-plugin")
            
            assert manifest_path.exists()
            assert manifest_path.name == 'plugin.yaml'
            
            # Verify template content
            with open(manifest_path) as f:
                template_data = yaml.safe_load(f)
            
            assert template_data['name'] == 'test-plugin'
            assert template_data['version'] == '1.0.0'


class TestPluginDependencyResolver:
    """Test plugin dependency resolver."""
    
    def test_resolve_dependencies_success(self):
        """Test successful dependency resolution."""
        resolver = PluginDependencyResolver()
        
        # Add a dependency
        dep_metadata = PluginMetadata(
            name="dependency-plugin",
            version="1.0.0",
            description="A dependency plugin",
            author=PluginAuthor(name="Author")
        )
        resolver.add_plugin(dep_metadata)
        
        # Create plugin with dependency
        plugin_metadata = PluginMetadata(
            name="main-plugin",
            version="1.0.0",
            description="Main plugin",
            author=PluginAuthor(name="Author"),
            dependencies=[
                PluginDependency(name="dependency-plugin", version=">=1.0.0")
            ]
        )
        
        missing_deps = resolver.resolve_dependencies(plugin_metadata)
        assert len(missing_deps) == 0
    
    def test_resolve_dependencies_missing(self):
        """Test dependency resolution with missing dependencies."""
        resolver = PluginDependencyResolver()
        
        plugin_metadata = PluginMetadata(
            name="main-plugin",
            version="1.0.0",
            description="Main plugin",
            author=PluginAuthor(name="Author"),
            dependencies=[
                PluginDependency(name="missing-plugin", version=">=1.0.0")
            ]
        )
        
        missing_deps = resolver.resolve_dependencies(plugin_metadata)
        assert len(missing_deps) == 1
        assert "missing-plugin" in missing_deps
    
    def test_get_load_order(self):
        """Test plugin load order calculation."""
        resolver = PluginDependencyResolver()
        
        # Create plugins with dependencies
        plugin_a = PluginMetadata(
            name="plugin-a",
            version="1.0.0",
            description="Plugin A",
            author=PluginAuthor(name="Author"),
            dependencies=[]
        )
        
        plugin_b = PluginMetadata(
            name="plugin-b",
            version="1.0.0",
            description="Plugin B",
            author=PluginAuthor(name="Author"),
            dependencies=[
                PluginDependency(name="plugin-a")
            ]
        )
        
        plugins = [plugin_b, plugin_a]  # Intentionally wrong order
        ordered = resolver.get_load_order(plugins)
        
        assert len(ordered) == 2
        assert ordered[0].name == "plugin-a"  # Should be first
        assert ordered[1].name == "plugin-b"  # Should be second


class TestEnhancedPluginLoader:
    """Test enhanced plugin loader."""
    
    @pytest.fixture
    def plugin_loader(self):
        """Create plugin loader for testing."""
        return PluginLoader()
    
    def test_initialization(self, plugin_loader):
        """Test plugin loader initialization."""
        assert plugin_loader._loaded_plugins == {}
        assert plugin_loader._plugin_modules == {}
        assert plugin_loader._plugin_metadata == {}
        assert plugin_loader._manifest_loader is not None
        assert plugin_loader._dependency_resolver is not None
    
    @pytest.mark.asyncio
    async def test_load_file_plugin(self, plugin_loader):
        """Test loading single file plugin."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create plugin file
            plugin_file = Path(temp_dir) / "test_plugin.py"
            plugin_content = '''
from src.coda.components.tools.base_tool import BaseTool
from src.coda.components.tools.models import ToolDefinition, ToolCategory

class TestTool(BaseTool):
    def _create_definition(self):
        return ToolDefinition(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.UTILITY,
            parameters=[]
        )

    async def _execute_impl(self, parameters: dict, context: dict):
        return "test result"
'''
            with open(plugin_file, 'w') as f:
                f.write(plugin_content)
            
            # Load plugin
            tools, metadata = await plugin_loader.load_plugin(str(plugin_file))
            
            assert len(tools) == 1
            assert tools[0].get_definition().name == "test_tool"
            assert metadata is None  # No manifest for file plugins
    
    @pytest.mark.asyncio
    async def test_load_directory_plugin_with_manifest(self, plugin_loader):
        """Test loading directory plugin with manifest."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plugin_dir = Path(temp_dir) / "test_plugin"
            plugin_dir.mkdir()
            
            # Create manifest
            manifest_data = {
                'name': 'test-plugin',
                'version': '1.0.0',
                'description': 'A test plugin',
                'author': {
                    'name': 'Test Author'
                },
                'entry_point': 'plugin.py'
            }
            
            with open(plugin_dir / 'plugin.yaml', 'w') as f:
                yaml.dump(manifest_data, f)
            
            # Create plugin file
            plugin_content = '''
from src.coda.components.tools.base_tool import BaseTool
from src.coda.components.tools.models import ToolDefinition, ToolCategory

class TestTool(BaseTool):
    def _create_definition(self):
        return ToolDefinition(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.UTILITY,
            parameters=[]
        )

    async def _execute_impl(self, parameters: dict, context: dict):
        return "test result"
'''
            with open(plugin_dir / 'plugin.py', 'w') as f:
                f.write(plugin_content)
            
            # Load plugin
            tools, metadata = await plugin_loader.load_plugin(str(plugin_dir))
            
            assert len(tools) == 1
            assert tools[0].get_definition().name == "test_tool"
            assert metadata is not None
            assert metadata.name == 'test-plugin'
    
    @pytest.mark.asyncio
    async def test_discover_plugins(self, plugin_loader):
        """Test plugin discovery."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create directory plugin
            plugin_dir = temp_path / "dir_plugin"
            plugin_dir.mkdir()
            (plugin_dir / 'plugin.yaml').touch()
            (plugin_dir / 'plugin.py').touch()
            
            # Create file plugin
            file_plugin = temp_path / "file_plugin.py"
            plugin_content = '''
from src.coda.components.tools.base_tool import BaseTool

class TestTool(BaseTool):
    pass
'''
            with open(file_plugin, 'w') as f:
                f.write(plugin_content)
            
            # Discover plugins
            discovered = await plugin_loader.discover_plugins(str(temp_path))
            
            assert len(discovered) == 2
            assert str(plugin_dir) in discovered
            assert str(file_plugin) in discovered
    
    @pytest.mark.asyncio
    async def test_plugin_lifecycle_hooks(self, plugin_loader):
        """Test plugin lifecycle hooks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plugin_dir = Path(temp_dir) / "test_plugin"
            plugin_dir.mkdir()
            
            # Create manifest with hooks
            manifest_data = {
                'name': 'test-plugin',
                'version': '1.0.0',
                'description': 'A test plugin',
                'author': {'name': 'Test Author'},
                'entry_point': 'plugin.py',
                'startup_hook': 'startup',
                'shutdown_hook': 'shutdown'
            }
            
            with open(plugin_dir / 'plugin.yaml', 'w') as f:
                yaml.dump(manifest_data, f)
            
            # Create plugin with hooks
            plugin_content = '''
startup_called = False
shutdown_called = False

def startup():
    global startup_called
    startup_called = True

def shutdown():
    global shutdown_called
    shutdown_called = True

from src.coda.components.tools.base_tool import BaseTool
from src.coda.components.tools.models import ToolDefinition, ToolCategory

class TestTool(BaseTool):
    def _create_definition(self):
        return ToolDefinition(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.UTILITY,
            parameters=[]
        )

    async def _execute_impl(self, parameters: dict, context: dict):
        return "test result"
'''
            with open(plugin_dir / 'plugin.py', 'w') as f:
                f.write(plugin_content)
            
            # Load plugin (should call startup hook)
            tools, metadata = await plugin_loader.load_plugin(str(plugin_dir))
            
            # Check startup hook was called
            module = plugin_loader._plugin_modules[str(plugin_dir)]
            assert getattr(module, 'startup_called', False)
            
            # Unload plugin (should call shutdown hook)
            await plugin_loader.unload_plugin(str(plugin_dir))
            
            # Check shutdown hook was called
            assert getattr(module, 'shutdown_called', False)


class TestToolManagerPluginIntegration:
    """Test tool manager plugin integration."""
    
    @pytest.mark.asyncio
    async def test_tool_manager_plugin_loading(self):
        """Test tool manager plugin loading integration."""
        from src.coda.components.tools.models import ToolConfig, ToolRegistryConfig
        
        # Create config with plugin directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ToolConfig(
                registry=ToolRegistryConfig(
                    auto_discover_plugins=True,
                    plugin_directories=[temp_dir]
                )
            )
            
            # Create a test plugin
            plugin_file = Path(temp_dir) / "test_plugin.py"
            plugin_content = '''
from src.coda.components.tools.base_tool import BaseTool
from src.coda.components.tools.models import ToolDefinition, ToolCategory

class MockPluginTool(BaseTool):
    def _create_definition(self):
        return ToolDefinition(
            name="test_plugin_tool",
            description="A test plugin tool",
            category=ToolCategory.UTILITY,
            parameters=[]
        )

    async def _execute_impl(self, parameters: dict, context: dict):
        return "plugin tool result"
'''
            with open(plugin_file, 'w') as f:
                f.write(plugin_content)
            
            # Initialize tool manager
            tool_manager = ToolManager(config)
            await tool_manager.initialize()
            
            # Check that plugin tool was loaded
            tool_names = tool_manager.registry.get_tool_names()
            assert "test_plugin_tool" in tool_names
            
            # Test tool execution
            tool = tool_manager.registry.get_tool("test_plugin_tool")
            assert tool is not None

            result = await tool.execute({})
            assert result == "plugin tool result"
