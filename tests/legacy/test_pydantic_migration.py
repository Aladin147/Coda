#!/usr/bin/env python3
"""
Test Pydantic V2 migration and field validators.
"""

import sys
import os
sys.path.append('./src')

def test_pydantic_v2_migration():
    """Test Pydantic V2 field validators."""
    print("🔍 Pydantic V2 Migration Test")
    print("=" * 50)
    
    try:
        import pydantic
        print(f"Pydantic Version: {pydantic.__version__}")
        
        # Test 1: Memory models
        print("\n📝 Testing memory models...")
        from coda.components.memory.models import MemoryQuery
        from datetime import datetime
        
        # Valid query
        query = MemoryQuery(
            query="test query",
            limit=5,
            min_relevance=0.5,
            time_range=(datetime.now(), datetime.now())
        )
        print("✅ MemoryQuery validation working")
        
        # Test invalid time range
        try:
            invalid_query = MemoryQuery(
                query="test",
                time_range=(datetime.now(), datetime(2020, 1, 1))  # End before start
            )
            print("❌ Time range validation failed - should have raised error")
        except ValueError:
            print("✅ Time range validation working correctly")
        
        # Test 2: Personality models
        print("\n🎭 Testing personality models...")
        from coda.components.personality.models import PersonalityTrait, PersonalityTraitType
        
        trait = PersonalityTrait(
            name=PersonalityTraitType.HUMOR,
            value=0.7,
            default_value=0.5,
            description="Test trait"
        )
        print("✅ PersonalityTrait validation working")
        
        # Test invalid value
        try:
            invalid_trait = PersonalityTrait(
                name=PersonalityTraitType.HUMOR,
                value=1.5,  # Invalid - should be <= 1.0
                default_value=0.5,
                description="Test trait"
            )
            print("❌ Value range validation failed - should have raised error")
        except ValueError:
            print("✅ Value range validation working correctly")
        
        # Test 3: Tool models
        print("\n🔧 Testing tool models...")
        from coda.components.tools.models import ToolParameter, ParameterType
        
        param = ToolParameter(
            name="test_param",
            type=ParameterType.STRING,
            description="Test parameter",
            default="test_value"
        )
        print("✅ ToolParameter validation working")
        
        # Test 4: Plugin metadata
        print("\n🔌 Testing plugin metadata...")
        from coda.components.tools.plugin_metadata import PluginMetadata, PluginAuthor

        author = PluginAuthor(name="Test Author", email="test@example.com")
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            author=author,
            permissions=["file_system"]
        )
        print("✅ PluginMetadata validation working")

        # Test invalid version
        try:
            invalid_metadata = PluginMetadata(
                name="test_plugin",
                version="invalid_version",  # Should follow semantic versioning
                description="Test plugin",
                author=author
            )
            print("❌ Version validation failed - should have raised error")
        except ValueError:
            print("✅ Version validation working correctly")

        # Test invalid permission
        try:
            invalid_metadata = PluginMetadata(
                name="test_plugin",
                version="1.0.0",
                description="Test plugin",
                author=author,
                permissions=["invalid_permission"]  # Should be from valid set
            )
            print("❌ Permission validation failed - should have raised error")
        except ValueError:
            print("✅ Permission validation working correctly")
        
        print("\n🎉 All Pydantic V2 field validators working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_deprecation_warnings():
    """Test for Pydantic deprecation warnings."""
    print("\n⚠️ Checking for Pydantic deprecation warnings...")
    
    import warnings
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Import all models to trigger any deprecation warnings
        try:
            from coda.components.memory.models import MemoryQuery
            from coda.components.personality.models import PersonalityTrait
            from coda.components.tools.models import ToolParameter
            from coda.components.tools.plugin_metadata import PluginMetadata
            
            # Create instances to trigger validators
            MemoryQuery(query="test")
            
            # Check for Pydantic-related warnings
            pydantic_warnings = [warning for warning in w 
                               if 'pydantic' in str(warning.message).lower() 
                               or 'validator' in str(warning.message).lower()]
            
            if pydantic_warnings:
                print(f"⚠️ Found {len(pydantic_warnings)} Pydantic warnings:")
                for warning in pydantic_warnings:
                    print(f"  - {warning.message}")
                return False
            else:
                print("✅ No Pydantic deprecation warnings found")
                return True
                
        except Exception as e:
            print(f"❌ Error checking warnings: {e}")
            return False

if __name__ == "__main__":
    success1 = test_pydantic_v2_migration()
    success2 = test_deprecation_warnings()
    
    if success1 and success2:
        print("\n🎉 Pydantic V2 migration successful!")
        sys.exit(0)
    else:
        print("\n❌ Pydantic migration issues detected")
        sys.exit(1)
