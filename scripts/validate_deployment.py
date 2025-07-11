#!/usr/bin/env python3
"""
Coda Deployment Validation Script

Comprehensive validation script for Coda deployment environments.
Validates RTX 5090 optimization, dependencies, and system readiness.
"""

import sys
import subprocess
import importlib
import platform
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

def run_command(command: str) -> Tuple[bool, str]:
    """Run a shell command and return success status and output."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)

def check_python_version() -> Tuple[bool, str]:
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.10+)"

def check_gpu_support() -> Tuple[bool, str]:
    """Check RTX 5090 and CUDA support."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        major, minor = torch.cuda.get_device_capability(0)
        compute_capability = f"SM_{major}{minor}"
        
        # Check for RTX 5090 or compatible
        is_rtx_5090 = "RTX 5090" in gpu_name or major >= 12
        
        info = f"{gpu_name} ({gpu_memory:.1f}GB, {compute_capability})"
        
        if is_rtx_5090:
            return True, f"RTX 5090 Compatible: {info}"
        else:
            return False, f"Not RTX 5090 compatible: {info}"
            
    except ImportError:
        return False, "PyTorch not installed"
    except Exception as e:
        return False, f"GPU check failed: {e}"

def check_dependencies() -> Tuple[bool, List[str]]:
    """Check critical dependencies."""
    critical_deps = [
        'torch', 'transformers', 'chromadb', 'ollama', 'moshi',
        'aiohttp', 'websockets', 'pydantic', 'numpy', 'sounddevice'
    ]
    
    results = []
    all_good = True
    
    for dep in critical_deps:
        try:
            module = importlib.import_module(dep)
            version = getattr(module, '__version__', 'unknown')
            results.append(f"✅ {dep}: {version}")
        except ImportError:
            results.append(f"❌ {dep}: Not installed")
            all_good = False
        except Exception as e:
            results.append(f"⚠️  {dep}: {e}")
            all_good = False
    
    return all_good, results

def check_coda_components() -> Tuple[bool, List[str]]:
    """Check Coda component availability."""
    sys.path.append('src')
    
    components = [
        ('LLM Manager', 'coda.components.llm.manager', 'LLMManager'),
        ('Memory Manager', 'coda.components.memory.manager', 'MemoryManager'),
        ('Personality Manager', 'coda.components.personality.manager', 'PersonalityManager'),
        ('Tool Manager', 'coda.components.tools.manager', 'ToolManager'),
        ('Voice Manager', 'coda.components.voice.manager', 'VoiceManager'),
        ('Core Assistant', 'coda.core.assistant', 'CodaAssistant'),
        ('WebSocket Server', 'coda.interfaces.websocket.server', 'CodaWebSocketServer'),
        ('Dashboard Server', 'coda.interfaces.dashboard.server', 'CodaDashboardServer')
    ]
    
    results = []
    all_good = True
    
    for name, module_path, class_name in components:
        try:
            module = importlib.import_module(module_path)
            component_class = getattr(module, class_name)
            if callable(component_class):
                results.append(f"✅ {name}: Available")
            else:
                results.append(f"❌ {name}: Not callable")
                all_good = False
        except Exception as e:
            results.append(f"❌ {name}: {e}")
            all_good = False
    
    return all_good, results

def check_configuration() -> Tuple[bool, str]:
    """Check configuration files."""
    config_files = [
        'configs/default.yaml',
        'configs/production.yaml',
        '.env.example'
    ]
    
    missing_files = []
    for config_file in config_files:
        if not Path(config_file).exists():
            missing_files.append(config_file)
    
    if missing_files:
        return False, f"Missing config files: {', '.join(missing_files)}"
    
    return True, "All configuration files present"

def check_system_resources() -> Tuple[bool, str]:
    """Check system resources."""
    try:
        import psutil
        
        # Check RAM
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        # Check disk space
        disk = psutil.disk_usage('.')
        disk_free_gb = disk.free / (1024**3)
        
        # Check CPU
        cpu_count = psutil.cpu_count()
        
        info = f"RAM: {memory_gb:.1f}GB, Disk: {disk_free_gb:.1f}GB free, CPU: {cpu_count} cores"
        
        if memory_gb >= 16 and disk_free_gb >= 10:
            return True, info
        else:
            return False, f"Insufficient resources: {info}"
            
    except ImportError:
        return False, "psutil not available for resource check"
    except Exception as e:
        return False, f"Resource check failed: {e}"

def main():
    """Run comprehensive deployment validation."""
    print("🔍 CODA DEPLOYMENT VALIDATION")
    print("=" * 50)
    print()
    
    validation_results = {}
    overall_success = True
    
    # Check Python version
    print("1. Python Version Check...")
    success, info = check_python_version()
    validation_results['python'] = {'success': success, 'info': info}
    print(f"   {'✅' if success else '❌'} {info}")
    if not success:
        overall_success = False
    print()
    
    # Check GPU support
    print("2. RTX 5090 GPU Support Check...")
    success, info = check_gpu_support()
    validation_results['gpu'] = {'success': success, 'info': info}
    print(f"   {'✅' if success else '❌'} {info}")
    if not success:
        overall_success = False
    print()
    
    # Check dependencies
    print("3. Dependencies Check...")
    success, deps = check_dependencies()
    validation_results['dependencies'] = {'success': success, 'details': deps}
    for dep in deps:
        print(f"   {dep}")
    if not success:
        overall_success = False
    print()
    
    # Check Coda components
    print("4. Coda Components Check...")
    success, components = check_coda_components()
    validation_results['components'] = {'success': success, 'details': components}
    for component in components:
        print(f"   {component}")
    if not success:
        overall_success = False
    print()
    
    # Check configuration
    print("5. Configuration Check...")
    success, info = check_configuration()
    validation_results['configuration'] = {'success': success, 'info': info}
    print(f"   {'✅' if success else '❌'} {info}")
    if not success:
        overall_success = False
    print()
    
    # Check system resources
    print("6. System Resources Check...")
    success, info = check_system_resources()
    validation_results['resources'] = {'success': success, 'info': info}
    print(f"   {'✅' if success else '❌'} {info}")
    if not success:
        overall_success = False
    print()
    
    # Overall result
    print("=" * 50)
    if overall_success:
        print("🎉 DEPLOYMENT VALIDATION: PASSED")
        print("   System is ready for Coda deployment!")
    else:
        print("❌ DEPLOYMENT VALIDATION: FAILED")
        print("   Please address the issues above before deployment.")
    
    print()
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    
    # Save validation report
    report_path = Path("deployment_validation_report.json")
    with open(report_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    print(f"Validation report saved to: {report_path}")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main())
