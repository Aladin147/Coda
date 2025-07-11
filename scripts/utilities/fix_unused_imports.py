#!/usr/bin/env python3
"""
Fix unused imports in voice processing system.
"""

import re
from pathlib import Path

# Map of files to unused imports that need to be removed
UNUSED_IMPORTS = {
    "audio_processor.py": [
        "typing.Tuple",
        "datetime.datetime", 
        ".interfaces.VoiceActivityDetectorInterface"
    ],
    "config.py": [
        "typing.Union",
        "pydantic.validator",
        ".models.AudioFormat",
        ".models.VoiceProvider"
    ],
    "context_integration.py": [
        "typing.List",
        "typing.Tuple",
        "..memory.models.Memory",
        "..memory.models.MemoryType", 
        "..memory.models.MemoryMetadata",
        "..personality.models.PersonalityTrait",
        "..personality.models.PersonalityParameters"
    ],
    "conversation_sync.py": [
        "typing.Tuple",
        "..conversation.models.Conversation"
    ],
    "fallback_manager.py": [
        "typing.Union",
        "typing.Type",
        ".models.ConversationState"
    ],
    "hybrid_orchestrator.py": [
        "typing.Tuple",
        "typing.Union", 
        "datetime.datetime",
        "..llm.models.LLMProvider"
    ],
    "inner_monologue.py": [
        ".models.VoiceConfig"
    ],
    "interfaces.py": [
        "datetime.datetime",
        ".models.VoiceEvent"
    ],
    "latency_optimizer.py": [
        "typing.Tuple",
        ".parallel_processor.ParallelProcessingConfig",
        ".mode_manager.ProcessingModeManager",
        ".mode_manager.ProcessingModeType"
    ],
    "llm_integration.py": [
        "typing.Union",
        "datetime.datetime",
        "..llm.models.LLMResponse",
        "..llm.models.LLMStreamChunk",
        "..llm.models.LLMConversation"
    ],
    "llm_manager_integration.py": [
        "typing.Tuple",
        "..llm.models.LLMStreamChunk",
        "..llm.models.MessageRole",
        "..llm.models.LLMProvider",
        "..llm.models.LLMConfig",
        "..llm.models.ProviderConfig",
        "..llm.manager.LLMManager"
    ],
    "manager.py": [
        ".models.VoiceProcessingMode",
        ".audio_processor.AudioProcessor",
        ".pipeline.AudioPipeline"
    ],
    "memory_integration.py": [
        "typing.Tuple",
        "..memory.models.ConversationTurn",
        "..memory.models.ConversationContext",
        "..memory.models.MemoryQuery",
        "..memory.models.MemoryType"
    ],
    "models.py": [
        "typing.Union"
    ],
    "mode_manager.py": [
        "typing.Callable"
    ],
    "moshi_client.py": [
        "typing.Tuple"
    ],
    "moshi_integration.py": [
        "typing.List",
        "typing.Tuple",
        ".interfaces.MoshiInterface",
        ".inner_monologue.ExtractedText"
    ],
    "parallel_processor.py": [
        "typing.List",
        "typing.Tuple",
        "queue.Queue",
        "queue.Empty"
    ],
    "performance_optimizer.py": [
        "typing.List",
        "typing.Callable",
        ".context_integration.ContextConfig",
        ".llm_integration.VoiceLLMConfig"
    ],
    "personality_integration.py": [
        "typing.Tuple",
        "..personality.models.PersonalityAdjustment",
        "..personality.models.BehaviorPattern"
    ],
    "pipeline.py": [
        "typing.Union",
        ".models.VoiceStreamChunk",
        ".utils.StreamingUtils"
    ],
    "processing_modes.py": [
        "typing.Tuple",
        "typing.Union"
    ],
    "tools_integration.py": [
        "typing.Callable"
    ],
    "utils.py": [
        "typing.Union",
        "typing.Tuple",
        ".models.VoiceProcessingMode"
    ]
}

def remove_unused_import(content: str, unused_import: str) -> str:
    """Remove a specific unused import from content."""
    lines = content.split('\n')
    filtered_lines = []
    
    for line in lines:
        # Check if this line imports the unused import
        should_remove = False
        
        # Handle different import patterns
        if 'from' in line and 'import' in line:
            # from ... import ... pattern
            if f' import {unused_import}' in line:
                # Check if this line imports ONLY the unused import
                import_part = line.split(' import ')[1]
                imports = [imp.strip() for imp in import_part.split(',')]
                
                # Remove the unused import from the list
                remaining_imports = [imp for imp in imports if imp != unused_import]
                
                if remaining_imports:
                    # Reconstruct the line with remaining imports
                    prefix = line.split(' import ')[0] + ' import '
                    line = prefix + ', '.join(remaining_imports)
                else:
                    # All imports are unused, remove the line
                    should_remove = True
            elif unused_import in line:
                # Check for partial matches in complex import statements
                pattern = rf'\b{re.escape(unused_import)}\b'
                if re.search(pattern, line):
                    should_remove = True
        elif f'import {unused_import}' in line:
            # Simple import statement
            should_remove = True
        
        if not should_remove:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

def fix_file_imports(file_path: Path) -> None:
    """Fix unused imports in a specific file."""
    filename = file_path.name
    
    if filename not in UNUSED_IMPORTS:
        return
    
    print(f"Fixing unused imports in {filename}...")
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Remove each unused import
        for unused_import in UNUSED_IMPORTS[filename]:
            content = remove_unused_import(content, unused_import)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✓ Fixed imports in {filename}")
        else:
            print(f"  ✓ No changes needed for {filename}")
    
    except Exception as e:
        print(f"  ❌ Error fixing {filename}: {e}")

def main():
    """Main function."""
    print("🔧 Fixing unused imports in voice processing system...")
    print("=" * 50)
    
    voice_dir = Path("src/coda/components/voice")
    
    # Fix each file
    for py_file in voice_dir.glob("*.py"):
        fix_file_imports(py_file)
    
    print()
    print("=" * 50)
    print("🎉 Unused imports cleanup completed!")

if __name__ == "__main__":
    main()
