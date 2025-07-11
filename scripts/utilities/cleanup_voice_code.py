#!/usr/bin/env python3
"""
Voice Code Cleanup Script

This script automatically fixes code quality issues in the voice processing system:
- Removes unused imports
- Fixes trailing whitespace
- Fixes blank lines with whitespace
- Fixes line length issues
- Adds missing newlines at end of files
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Set

def get_voice_python_files() -> List[Path]:
    """Get all Python files in the voice components directory."""
    voice_dir = Path("src/coda/components/voice")
    return list(voice_dir.glob("*.py"))

def remove_trailing_whitespace(content: str) -> str:
    """Remove trailing whitespace from all lines."""
    lines = content.split('\n')
    cleaned_lines = [line.rstrip() for line in lines]
    return '\n'.join(cleaned_lines)

def ensure_newline_at_end(content: str) -> str:
    """Ensure file ends with a newline."""
    if content and not content.endswith('\n'):
        content += '\n'
    return content

def fix_blank_lines_with_whitespace(content: str) -> str:
    """Fix blank lines that contain whitespace."""
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        if line.strip() == '':  # Blank line
            fixed_lines.append('')  # Empty string, no whitespace
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_operator_spacing(content: str) -> str:
    """Fix missing whitespace around arithmetic operators."""
    # Fix arithmetic operators
    content = re.sub(r'(\w)(\+|\-|\*|\/|%)(\w)', r'\1 \2 \3', content)
    content = re.sub(r'(\w)(\+|\-|\*|\/|%)(\d)', r'\1 \2 \3', content)
    content = re.sub(r'(\d)(\+|\-|\*|\/|%)(\w)', r'\1 \2 \3', content)
    
    # Fix comparison operators
    content = re.sub(r'(\w)(==|!=|<=|>=|<|>)(\w)', r'\1 \2 \3', content)
    
    return content

def fix_inline_comments(content: str) -> str:
    """Fix inline comments to have at least two spaces before #."""
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Find inline comments (# not at start of line)
        if '#' in line and not line.strip().startswith('#'):
            # Split on first #
            code_part, comment_part = line.split('#', 1)
            # Ensure at least 2 spaces before #
            if not code_part.endswith('  '):
                code_part = code_part.rstrip() + '  '
            fixed_lines.append(code_part + '#' + comment_part)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_line_length(content: str, max_length: int = 120) -> str:
    """Fix lines that are too long by breaking them appropriately."""
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        if len(line) <= max_length:
            fixed_lines.append(line)
            continue
        
        # Try to break long lines intelligently
        if ',' in line and '(' in line:
            # Function call with multiple parameters
            indent = len(line) - len(line.lstrip())
            base_indent = ' ' * indent
            extra_indent = ' ' * 4
            
            # Find the opening parenthesis
            paren_pos = line.find('(')
            if paren_pos > 0:
                prefix = line[:paren_pos + 1]
                suffix = line[paren_pos + 1:]
                
                if suffix.endswith(')'):
                    params = suffix[:-1]
                    # Split parameters
                    param_parts = [p.strip() for p in params.split(',')]
                    
                    if len(param_parts) > 1:
                        fixed_lines.append(prefix)
                        for i, param in enumerate(param_parts):
                            if i == len(param_parts) - 1:
                                fixed_lines.append(base_indent + extra_indent + param)
                            else:
                                fixed_lines.append(base_indent + extra_indent + param + ',')
                        fixed_lines.append(base_indent + ')')
                        continue
        
        # If we can't break it intelligently, just add it as-is
        # (manual review needed)
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def get_unused_imports(file_path: Path) -> Set[str]:
    """Get unused imports using flake8."""
    try:
        result = subprocess.run(
            ['venv/Scripts/python.exe', '-m', 'flake8', '--select=F401', str(file_path)],
            capture_output=True,
            text=True,
            cwd='.'
        )
        
        unused_imports = set()
        for line in result.stdout.split('\n'):
            if 'F401' in line and 'imported but unused' in line:
                # Extract the import name
                match = re.search(r"'([^']+)' imported but unused", line)
                if match:
                    unused_imports.add(match.group(1))
        
        return unused_imports
    except Exception as e:
        print(f"Error checking unused imports for {file_path}: {e}")
        return set()

def remove_unused_imports(content: str, unused_imports: Set[str]) -> str:
    """Remove unused imports from content."""
    if not unused_imports:
        return content
    
    lines = content.split('\n')
    filtered_lines = []
    
    for line in lines:
        # Check if this line imports any unused imports
        should_remove = False
        
        for unused_import in unused_imports:
            # Handle different import patterns
            patterns = [
                f"from .* import .*{re.escape(unused_import)}",
                f"import {re.escape(unused_import)}",
                f"from .* import {re.escape(unused_import)}",
            ]
            
            for pattern in patterns:
                if re.search(pattern, line):
                    # Check if this line imports ONLY unused imports
                    if 'import' in line:
                        # Extract all imports from this line
                        if ' import ' in line:
                            import_part = line.split(' import ')[1]
                            imports = [imp.strip() for imp in import_part.split(',')]
                            
                            # Remove unused imports from the list
                            remaining_imports = [imp for imp in imports if imp not in unused_imports]
                            
                            if remaining_imports:
                                # Reconstruct the line with remaining imports
                                prefix = line.split(' import ')[0] + ' import '
                                line = prefix + ', '.join(remaining_imports)
                                break
                            else:
                                # All imports are unused, remove the line
                                should_remove = True
                                break
                        else:
                            # Simple import statement
                            should_remove = True
                            break
        
        if not should_remove:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

def fix_multiple_statements_on_line(content: str) -> str:
    """Fix multiple statements on one line (E704)."""
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Look for def statements on one line
        if re.match(r'\s*def\s+\w+\([^)]*\):\s*\w+', line):
            # Split the def and the statement
            match = re.match(r'(\s*def\s+\w+\([^)]*\):\s*)(.*)', line)
            if match:
                def_part = match.group(1)
                statement_part = match.group(2)
                fixed_lines.append(def_part)
                # Add proper indentation for the statement
                indent = len(def_part) - len(def_part.lstrip()) + 4
                fixed_lines.append(' ' * indent + statement_part)
                continue
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_continuation_line_indentation(content: str) -> str:
    """Fix continuation line indentation issues."""
    lines = content.split('\n')
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line has continuation issues
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            
            # If current line ends with comma and next line is indented
            if line.rstrip().endswith(',') and next_line.startswith(' '):
                # Ensure proper indentation for continuation
                base_indent = len(line) - len(line.lstrip())
                if len(next_line) - len(next_line.lstrip()) < base_indent + 4:
                    # Fix indentation
                    next_line = ' ' * (base_indent + 4) + next_line.lstrip()
                    lines[i + 1] = next_line
        
        fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines)

def cleanup_file(file_path: Path) -> None:
    """Clean up a single Python file."""
    print(f"Cleaning up {file_path}...")
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Get unused imports
        unused_imports = get_unused_imports(file_path)
        if unused_imports:
            print(f"  Removing unused imports: {', '.join(unused_imports)}")
            content = remove_unused_imports(content, unused_imports)
        
        # Apply fixes
        content = remove_trailing_whitespace(content)
        content = fix_blank_lines_with_whitespace(content)
        content = fix_operator_spacing(content)
        content = fix_inline_comments(content)
        content = fix_multiple_statements_on_line(content)
        content = fix_continuation_line_indentation(content)
        content = fix_line_length(content)
        content = ensure_newline_at_end(content)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✓ Fixed {file_path}")
        else:
            print(f"  ✓ No changes needed for {file_path}")
    
    except Exception as e:
        print(f"  ❌ Error cleaning {file_path}: {e}")

def main():
    """Main cleanup function."""
    print("🧹 Starting Voice Code Cleanup...")
    print("=" * 50)
    
    # Get all Python files in voice components
    voice_files = get_voice_python_files()
    
    print(f"Found {len(voice_files)} Python files to clean up")
    print()
    
    # Clean up each file
    for file_path in voice_files:
        cleanup_file(file_path)
    
    print()
    print("=" * 50)
    print("🎉 Voice code cleanup completed!")
    print()
    print("Running flake8 to verify fixes...")
    
    # Run flake8 again to check results
    try:
        result = subprocess.run(
            ['venv/Scripts/python.exe', '-m', 'flake8', 'src/coda/components/voice', 
             '--max-line-length=120', '--ignore=E203,W503'],
            capture_output=True,
            text=True,
            cwd='.'
        )
        
        if result.returncode == 0:
            print("✅ All code quality issues fixed!")
        else:
            print("⚠️  Some issues remain:")
            print(result.stdout)
    except Exception as e:
        print(f"❌ Error running flake8: {e}")

if __name__ == "__main__":
    main()
