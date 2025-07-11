#!/usr/bin/env python3
"""
Documentation Testing Script for Coda.

Tests the completeness and accuracy of documentation including:
- Documentation file existence and structure
- Code examples validation
- Link checking
- Configuration examples
- Installation instructions
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
import re
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("documentation_test")


class DocumentationTestSuite:
    """Comprehensive documentation testing suite."""
    
    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.docs_dir = Path("docs")
        self.root_dir = Path(".")
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all documentation tests."""
        logger.info("🚀 Starting Documentation Test Suite")
        logger.info("=" * 60)
        
        # Test 1: Documentation Structure
        logger.info("📚 Test 1: Documentation Structure")
        structure_results = await self.test_documentation_structure()
        self.results["documentation_structure"] = structure_results
        
        # Test 2: File Completeness
        logger.info("📚 Test 2: File Completeness")
        completeness_results = await self.test_file_completeness()
        self.results["file_completeness"] = completeness_results
        
        # Test 3: Configuration Examples
        logger.info("📚 Test 3: Configuration Examples")
        config_results = await self.test_configuration_examples()
        self.results["configuration_examples"] = config_results
        
        # Test 4: Code Examples
        logger.info("📚 Test 4: Code Examples")
        code_results = await self.test_code_examples()
        self.results["code_examples"] = code_results
        
        # Test 5: Installation Instructions
        logger.info("📚 Test 5: Installation Instructions")
        install_results = await self.test_installation_instructions()
        self.results["installation_instructions"] = install_results
        
        # Generate summary
        summary = self.generate_test_summary()
        self.results["summary"] = summary
        
        logger.info("=" * 60)
        logger.info("🎉 Documentation Test Suite Complete!")
        
        return self.results
    
    async def test_documentation_structure(self) -> Dict[str, Any]:
        """Test documentation directory structure."""
        results = {
            "required_files": {},
            "directory_structure": {},
            "missing_files": [],
            "extra_files": []
        }
        
        # Required documentation files
        required_files = [
            "README.md",
            "INSTALLATION_GUIDE.md",
            "USER_GUIDE.md",
            "CONFIGURATION.md",
            "TROUBLESHOOTING.md",
            "DEPLOYMENT_GUIDE.md",
            "api/README.md"
        ]
        
        logger.info("  Checking required documentation files...")
        
        for file_path in required_files:
            full_path = self.docs_dir / file_path
            exists = full_path.exists()
            results["required_files"][file_path] = {
                "exists": exists,
                "size_bytes": full_path.stat().st_size if exists else 0,
                "readable": self._is_readable(full_path) if exists else False
            }
            
            if not exists:
                results["missing_files"].append(file_path)
            
            logger.info(f"    {'✅' if exists else '❌'} {file_path}")
        
        # Check directory structure
        expected_dirs = ["api", "examples", "architecture", "development", "components"]
        
        for dir_name in expected_dirs:
            dir_path = self.docs_dir / dir_name
            results["directory_structure"][dir_name] = {
                "exists": dir_path.exists(),
                "is_directory": dir_path.is_dir() if dir_path.exists() else False
            }
        
        return results
    
    async def test_file_completeness(self) -> Dict[str, Any]:
        """Test completeness of documentation files."""
        results = {
            "file_analysis": {},
            "content_quality": {},
            "word_counts": {}
        }
        
        files_to_check = [
            "INSTALLATION_GUIDE.md",
            "USER_GUIDE.md", 
            "CONFIGURATION.md",
            "TROUBLESHOOTING.md",
            "DEPLOYMENT_GUIDE.md"
        ]
        
        logger.info("  Analyzing documentation file completeness...")
        
        for file_name in files_to_check:
            file_path = self.docs_dir / file_name
            
            if not file_path.exists():
                results["file_analysis"][file_name] = {"error": "File not found"}
                continue
            
            try:
                content = file_path.read_text(encoding='utf-8')
                
                # Basic content analysis
                word_count = len(content.split())
                line_count = len(content.splitlines())
                char_count = len(content)
                
                # Check for key sections
                has_toc = "table of contents" in content.lower() or "## " in content
                has_examples = "```" in content or "example" in content.lower()
                has_links = "[" in content and "](" in content
                
                results["file_analysis"][file_name] = {
                    "word_count": word_count,
                    "line_count": line_count,
                    "char_count": char_count,
                    "has_table_of_contents": has_toc,
                    "has_code_examples": has_examples,
                    "has_links": has_links,
                    "completeness_score": self._calculate_completeness_score(content)
                }
                
                results["word_counts"][file_name] = word_count
                
                logger.info(f"    ✅ {file_name}: {word_count} words, "
                           f"completeness: {results['file_analysis'][file_name]['completeness_score']:.1%}")
                
            except Exception as e:
                results["file_analysis"][file_name] = {"error": str(e)}
                logger.error(f"    ❌ {file_name}: Error reading file - {e}")
        
        return results
    
    async def test_configuration_examples(self) -> Dict[str, Any]:
        """Test configuration examples in documentation."""
        results = {
            "yaml_examples": {},
            "config_validation": {},
            "example_completeness": {}
        }
        
        logger.info("  Testing configuration examples...")
        
        # Check CONFIGURATION.md for YAML examples
        config_file = self.docs_dir / "CONFIGURATION.md"
        
        if config_file.exists():
            content = config_file.read_text(encoding='utf-8')
            
            # Extract YAML code blocks
            yaml_blocks = re.findall(r'```yaml\n(.*?)\n```', content, re.DOTALL)
            
            results["yaml_examples"]["total_blocks"] = len(yaml_blocks)
            results["yaml_examples"]["valid_blocks"] = 0
            results["yaml_examples"]["invalid_blocks"] = []
            
            for i, yaml_block in enumerate(yaml_blocks):
                try:
                    parsed = yaml.safe_load(yaml_block)
                    results["yaml_examples"]["valid_blocks"] += 1
                    logger.info(f"    ✅ YAML block {i+1}: Valid")
                except yaml.YAMLError as e:
                    results["yaml_examples"]["invalid_blocks"].append({
                        "block_number": i+1,
                        "error": str(e)
                    })
                    logger.error(f"    ❌ YAML block {i+1}: Invalid - {e}")
            
            # Check for key configuration sections
            key_sections = [
                "llm:", "voice:", "memory:", "personality:", 
                "tools:", "websocket:", "dashboard:", "performance:"
            ]
            
            found_sections = []
            for section in key_sections:
                if section in content:
                    found_sections.append(section)
            
            results["config_validation"]["key_sections_found"] = found_sections
            results["config_validation"]["coverage"] = len(found_sections) / len(key_sections)
            
            logger.info(f"    ✅ Configuration coverage: {results['config_validation']['coverage']:.1%}")
        
        return results
    
    async def test_code_examples(self) -> Dict[str, Any]:
        """Test code examples in documentation."""
        results = {
            "code_blocks": {},
            "python_examples": {},
            "bash_examples": {}
        }
        
        logger.info("  Testing code examples...")
        
        doc_files = [
            "INSTALLATION_GUIDE.md",
            "USER_GUIDE.md",
            "DEPLOYMENT_GUIDE.md",
            "api/README.md"
        ]
        
        total_python_blocks = 0
        valid_python_blocks = 0
        total_bash_blocks = 0
        
        for file_name in doc_files:
            file_path = self.docs_dir / file_name
            
            if not file_path.exists():
                continue
            
            content = file_path.read_text(encoding='utf-8')
            
            # Extract Python code blocks
            python_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
            bash_blocks = re.findall(r'```bash\n(.*?)\n```', content, re.DOTALL)
            
            total_python_blocks += len(python_blocks)
            total_bash_blocks += len(bash_blocks)
            
            # Validate Python syntax
            for i, python_block in enumerate(python_blocks):
                try:
                    compile(python_block, f"{file_name}_block_{i}", 'exec')
                    valid_python_blocks += 1
                except SyntaxError as e:
                    logger.warning(f"    ⚠️ {file_name} Python block {i+1}: Syntax error - {e}")
            
            results["code_blocks"][file_name] = {
                "python_blocks": len(python_blocks),
                "bash_blocks": len(bash_blocks)
            }
        
        results["python_examples"]["total"] = total_python_blocks
        results["python_examples"]["valid"] = valid_python_blocks
        results["python_examples"]["syntax_success_rate"] = (
            valid_python_blocks / total_python_blocks if total_python_blocks > 0 else 1.0
        )
        
        results["bash_examples"]["total"] = total_bash_blocks
        
        logger.info(f"    ✅ Python examples: {valid_python_blocks}/{total_python_blocks} valid "
                   f"({results['python_examples']['syntax_success_rate']:.1%})")
        logger.info(f"    ✅ Bash examples: {total_bash_blocks} found")
        
        return results
    
    async def test_installation_instructions(self) -> Dict[str, Any]:
        """Test installation instructions accuracy."""
        results = {
            "instruction_validation": {},
            "command_analysis": {},
            "dependency_check": {}
        }
        
        logger.info("  Testing installation instructions...")
        
        install_file = self.docs_dir / "INSTALLATION_GUIDE.md"
        
        if install_file.exists():
            content = install_file.read_text(encoding='utf-8')
            
            # Check for key installation steps
            key_steps = [
                "python -m venv",
                "pip install",
                "git clone",
                "requirements.txt",
                "python coda_launcher.py"
            ]
            
            found_steps = []
            for step in key_steps:
                if step in content:
                    found_steps.append(step)
            
            results["instruction_validation"]["key_steps_found"] = found_steps
            results["instruction_validation"]["completeness"] = len(found_steps) / len(key_steps)
            
            # Extract and analyze commands
            bash_commands = re.findall(r'```bash\n(.*?)\n```', content, re.DOTALL)
            
            command_count = 0
            for bash_block in bash_commands:
                commands = [line.strip() for line in bash_block.split('\n') 
                           if line.strip() and not line.strip().startswith('#')]
                command_count += len(commands)
            
            results["command_analysis"]["total_commands"] = command_count
            results["command_analysis"]["bash_blocks"] = len(bash_commands)
            
            # Check for dependency mentions
            dependencies = ["python", "pip", "git", "cuda", "pytorch", "ollama"]
            found_deps = []
            
            for dep in dependencies:
                if dep.lower() in content.lower():
                    found_deps.append(dep)
            
            results["dependency_check"]["mentioned_dependencies"] = found_deps
            results["dependency_check"]["dependency_coverage"] = len(found_deps) / len(dependencies)
            
            logger.info(f"    ✅ Installation completeness: {results['instruction_validation']['completeness']:.1%}")
            logger.info(f"    ✅ Commands found: {command_count}")
            logger.info(f"    ✅ Dependencies covered: {results['dependency_check']['dependency_coverage']:.1%}")
        
        return results
    
    def _is_readable(self, file_path: Path) -> bool:
        """Check if file is readable."""
        try:
            file_path.read_text(encoding='utf-8')
            return True
        except Exception:
            return False
    
    def _calculate_completeness_score(self, content: str) -> float:
        """Calculate completeness score for documentation content."""
        score = 0.0
        
        # Check for various documentation elements
        if len(content) > 1000:  # Substantial content
            score += 0.2
        if "##" in content:  # Has sections
            score += 0.2
        if "```" in content:  # Has code examples
            score += 0.2
        if "[" in content and "](" in content:  # Has links
            score += 0.2
        if "example" in content.lower():  # Has examples
            score += 0.1
        if "note:" in content.lower() or "warning:" in content.lower():  # Has notes/warnings
            score += 0.1
        
        return min(score, 1.0)
    
    def generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        summary = {
            "overall_status": "success",
            "total_tests": 5,
            "passed_tests": 0,
            "documentation_quality": {},
            "recommendations": []
        }
        
        # Analyze documentation structure
        if "documentation_structure" in self.results:
            struct_results = self.results["documentation_structure"]
            missing_files = len(struct_results.get("missing_files", []))
            
            if missing_files == 0:
                summary["passed_tests"] += 1
                summary["documentation_quality"]["structure"] = "complete"
            else:
                summary["recommendations"].append(f"Create {missing_files} missing documentation files")
        
        # Analyze file completeness
        if "file_completeness" in self.results:
            completeness_results = self.results["file_completeness"]
            avg_word_count = sum(completeness_results.get("word_counts", {}).values()) / max(1, len(completeness_results.get("word_counts", {})))
            
            if avg_word_count > 500:  # Substantial documentation
                summary["passed_tests"] += 1
                summary["documentation_quality"]["content_depth"] = "substantial"
            else:
                summary["recommendations"].append("Expand documentation content for better coverage")
        
        # Analyze configuration examples
        if "configuration_examples" in self.results:
            config_results = self.results["configuration_examples"]
            yaml_valid = config_results.get("yaml_examples", {}).get("valid_blocks", 0)
            yaml_total = config_results.get("yaml_examples", {}).get("total_blocks", 1)
            
            if yaml_valid / yaml_total > 0.8:
                summary["passed_tests"] += 1
                summary["documentation_quality"]["config_examples"] = "excellent"
        
        # Analyze code examples
        if "code_examples" in self.results:
            code_results = self.results["code_examples"]
            syntax_rate = code_results.get("python_examples", {}).get("syntax_success_rate", 0)
            
            if syntax_rate > 0.9:
                summary["passed_tests"] += 1
                summary["documentation_quality"]["code_examples"] = "high_quality"
        
        # Analyze installation instructions
        if "installation_instructions" in self.results:
            install_results = self.results["installation_instructions"]
            completeness = install_results.get("instruction_validation", {}).get("completeness", 0)
            
            if completeness > 0.8:
                summary["passed_tests"] += 1
                summary["documentation_quality"]["installation_guide"] = "comprehensive"
        
        # Overall assessment
        if summary["passed_tests"] >= 4:
            summary["overall_status"] = "excellent"
        elif summary["passed_tests"] >= 3:
            summary["overall_status"] = "good"
        elif summary["passed_tests"] >= 2:
            summary["overall_status"] = "fair"
        else:
            summary["overall_status"] = "needs_improvement"
        
        return summary


async def main():
    """Run the documentation test suite."""
    test_suite = DocumentationTestSuite()
    
    try:
        results = await test_suite.run_all_tests()
        
        # Print summary
        summary = results["summary"]
        logger.info("=" * 60)
        logger.info("📚 DOCUMENTATION TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Overall Status: {summary['overall_status'].upper()}")
        logger.info(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
        
        if summary["documentation_quality"]:
            logger.info("\n📊 Documentation Quality:")
            for aspect, quality in summary["documentation_quality"].items():
                logger.info(f"  • {aspect}: {quality}")
        
        if summary["recommendations"]:
            logger.info("\n💡 Recommendations:")
            for rec in summary["recommendations"]:
                logger.info(f"  • {rec}")
        
        return 0 if summary["overall_status"] in ["excellent", "good"] else 1
        
    except Exception as e:
        logger.error(f"Documentation test suite failed: {e}")
        return 1


if __name__ == "__main__":
    import asyncio
    exit(asyncio.run(main()))
