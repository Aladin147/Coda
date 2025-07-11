#!/usr/bin/env python3
"""
System health check for Coda components.
"""

import sys
sys.path.append('src')

def test_system_health():
    """Test core system components."""
    print('🔍 Testing Core System Components...')
    
    results = {}
    
    # Test configuration
    try:
        from coda.core.config import load_config
        config = load_config()
        print('✅ Configuration system: WORKING')
        results['config'] = True
    except Exception as e:
        print(f'❌ Configuration system: FAILED - {e}')
        results['config'] = False
    
    # Test LLM components
    try:
        from coda.components.llm.models import LLMConfig, ProviderConfig, LLMProvider
        llm_config = LLMConfig()
        print('✅ LLM models: WORKING')
        results['llm'] = True
    except Exception as e:
        print(f'❌ LLM models: FAILED - {e}')
        results['llm'] = False
    
    # Test Memory components
    try:
        from coda.components.memory.models import MemoryManagerConfig
        memory_config = MemoryManagerConfig()
        print('✅ Memory models: WORKING')
        results['memory'] = True
    except Exception as e:
        print(f'❌ Memory models: FAILED - {e}')
        results['memory'] = False
    
    # Test Voice components
    try:
        from coda.components.voice.models import VoiceConfig
        voice_config = VoiceConfig()
        print('✅ Voice models: WORKING')
        results['voice'] = True
    except Exception as e:
        print(f'❌ Voice models: FAILED - {e}')
        results['voice'] = False
    
    # Test Tools components
    try:
        from coda.components.tools.models import ToolConfig
        tool_config = ToolConfig()
        print('✅ Tools models: WORKING')
        results['tools'] = True
    except Exception as e:
        print(f'❌ Tools models: FAILED - {e}')
        results['tools'] = False
    
    # Test Personality components
    try:
        from coda.components.personality.models import PersonalityConfig
        personality_config = PersonalityConfig()
        print('✅ Personality models: WORKING')
        results['personality'] = True
    except Exception as e:
        print(f'❌ Personality models: FAILED - {e}')
        results['personality'] = False
    
    print('\n🎉 Core system health check complete!')
    
    # Summary
    working = sum(results.values())
    total = len(results)
    print(f'\n📊 Summary: {working}/{total} components working ({working/total*100:.1f}%)')
    
    if working == total:
        print('🎯 All systems operational!')
        return True
    else:
        print('⚠️  Some components need attention')
        return False

if __name__ == "__main__":
    test_system_health()
