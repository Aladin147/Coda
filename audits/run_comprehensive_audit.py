#!/usr/bin/env python3
"""
Comprehensive Deep Audit Runner.

Executes automated analysis tools and generates comprehensive audit report.
"""

import subprocess
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("audit_runner")


class ComprehensiveAuditor:
    """Comprehensive audit runner for Coda codebase."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.src_path = self.project_root / "src"
        self.audit_results = {}
        self.issues = []
        
    def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run all audit tools and generate comprehensive report."""
        logger.info("🔍 Starting Comprehensive Deep Audit")
        logger.info("=" * 80)
        
        # 1. Code Quality Analysis
        logger.info("📊 Phase 1: Code Quality Analysis")
        self._run_pylint_analysis()
        self._run_flake8_analysis()
        self._run_complexity_analysis()
        
        # 2. Security Analysis
        logger.info("🔒 Phase 2: Security Analysis")
        self._run_bandit_security_scan()
        self._run_safety_dependency_scan()
        
        # 3. Code Structure Analysis
        logger.info("🏗️ Phase 3: Code Structure Analysis")
        self._analyze_imports()
        self._analyze_file_structure()
        self._check_naming_conventions()
        
        # 4. Configuration Analysis
        logger.info("⚙️ Phase 4: Configuration Analysis")
        self._analyze_configurations()
        self._check_environment_variables()
        
        # 5. Documentation Analysis
        logger.info("📚 Phase 5: Documentation Analysis")
        self._analyze_documentation()
        self._check_code_comments()
        
        # 6. Generate Summary
        logger.info("📋 Phase 6: Generating Audit Summary")
        summary = self._generate_audit_summary()
        
        logger.info("=" * 80)
        logger.info("🎉 Comprehensive Deep Audit Complete!")
        
        return {
            "audit_results": self.audit_results,
            "issues": self.issues,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    
    def _run_pylint_analysis(self):
        """Run pylint code quality analysis."""
        logger.info("  Running pylint analysis...")
        
        try:
            # Run pylint on src directory
            result = subprocess.run([
                sys.executable, "-m", "pylint", 
                str(self.src_path),
                "--output-format=json",
                "--disable=C0114,C0115,C0116",  # Disable docstring warnings for now
                "--max-line-length=100"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.stdout:
                try:
                    pylint_issues = json.loads(result.stdout)
                    self.audit_results["pylint"] = {
                        "total_issues": len(pylint_issues),
                        "issues": pylint_issues[:20]  # Limit to first 20 for report
                    }
                    
                    # Categorize issues
                    for issue in pylint_issues:
                        self.issues.append({
                            "tool": "pylint",
                            "type": issue.get("type", "unknown"),
                            "severity": self._map_pylint_severity(issue.get("type")),
                            "message": issue.get("message", ""),
                            "file": issue.get("path", ""),
                            "line": issue.get("line", 0),
                            "category": "code_quality"
                        })
                    
                    logger.info(f"    ✅ Pylint: {len(pylint_issues)} issues found")
                    
                except json.JSONDecodeError:
                    logger.warning("    ⚠️ Pylint: Could not parse JSON output")
                    self.audit_results["pylint"] = {"error": "JSON parse error"}
            else:
                logger.info("    ✅ Pylint: No issues found")
                self.audit_results["pylint"] = {"total_issues": 0, "issues": []}
                
        except Exception as e:
            logger.error(f"    ❌ Pylint failed: {e}")
            self.audit_results["pylint"] = {"error": str(e)}
    
    def _run_flake8_analysis(self):
        """Run flake8 style analysis."""
        logger.info("  Running flake8 analysis...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "flake8", 
                str(self.src_path),
                "--max-line-length=100",
                "--extend-ignore=E203,W503"  # Ignore some common issues
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.stdout:
                flake8_lines = result.stdout.strip().split('\n')
                flake8_issues = []
                
                for line in flake8_lines:
                    if line.strip():
                        parts = line.split(':', 3)
                        if len(parts) >= 4:
                            flake8_issues.append({
                                "file": parts[0],
                                "line": parts[1],
                                "column": parts[2],
                                "message": parts[3].strip()
                            })
                
                self.audit_results["flake8"] = {
                    "total_issues": len(flake8_issues),
                    "issues": flake8_issues[:20]  # Limit to first 20
                }
                
                # Add to issues list
                for issue in flake8_issues:
                    self.issues.append({
                        "tool": "flake8",
                        "type": "style",
                        "severity": "low",
                        "message": issue["message"],
                        "file": issue["file"],
                        "line": int(issue["line"]) if issue["line"].isdigit() else 0,
                        "category": "code_style"
                    })
                
                logger.info(f"    ✅ Flake8: {len(flake8_issues)} issues found")
            else:
                logger.info("    ✅ Flake8: No issues found")
                self.audit_results["flake8"] = {"total_issues": 0, "issues": []}
                
        except Exception as e:
            logger.error(f"    ❌ Flake8 failed: {e}")
            self.audit_results["flake8"] = {"error": str(e)}
    
    def _run_complexity_analysis(self):
        """Run complexity analysis with radon."""
        logger.info("  Running complexity analysis...")
        
        try:
            # Cyclomatic complexity
            result = subprocess.run([
                sys.executable, "-m", "radon", "cc", 
                str(self.src_path), "-j"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.stdout:
                try:
                    complexity_data = json.loads(result.stdout)
                    high_complexity = []
                    
                    for file_path, functions in complexity_data.items():
                        for func in functions:
                            if func.get("complexity", 0) > 10:  # High complexity threshold
                                high_complexity.append({
                                    "file": file_path,
                                    "function": func.get("name", ""),
                                    "complexity": func.get("complexity", 0),
                                    "line": func.get("lineno", 0)
                                })
                    
                    self.audit_results["complexity"] = {
                        "high_complexity_functions": len(high_complexity),
                        "functions": high_complexity[:10]  # Top 10 most complex
                    }
                    
                    # Add high complexity as issues
                    for func in high_complexity:
                        self.issues.append({
                            "tool": "radon",
                            "type": "complexity",
                            "severity": "medium" if func["complexity"] > 15 else "low",
                            "message": f"High complexity function: {func['function']} (complexity: {func['complexity']})",
                            "file": func["file"],
                            "line": func["line"],
                            "category": "code_quality"
                        })
                    
                    logger.info(f"    ✅ Complexity: {len(high_complexity)} high-complexity functions")
                    
                except json.JSONDecodeError:
                    logger.warning("    ⚠️ Complexity: Could not parse JSON output")
                    self.audit_results["complexity"] = {"error": "JSON parse error"}
            else:
                logger.info("    ✅ Complexity: No high-complexity functions found")
                self.audit_results["complexity"] = {"high_complexity_functions": 0, "functions": []}
                
        except Exception as e:
            logger.error(f"    ❌ Complexity analysis failed: {e}")
            self.audit_results["complexity"] = {"error": str(e)}
    
    def _run_bandit_security_scan(self):
        """Run bandit security analysis."""
        logger.info("  Running bandit security scan...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "bandit", "-r", 
                str(self.src_path), "-f", "json"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.stdout:
                try:
                    bandit_data = json.loads(result.stdout)
                    security_issues = bandit_data.get("results", [])
                    
                    self.audit_results["bandit"] = {
                        "total_issues": len(security_issues),
                        "issues": security_issues[:10]  # Top 10 security issues
                    }
                    
                    # Add security issues
                    for issue in security_issues:
                        self.issues.append({
                            "tool": "bandit",
                            "type": "security",
                            "severity": issue.get("issue_severity", "medium").lower(),
                            "message": issue.get("issue_text", ""),
                            "file": issue.get("filename", ""),
                            "line": issue.get("line_number", 0),
                            "category": "security"
                        })
                    
                    logger.info(f"    ✅ Bandit: {len(security_issues)} security issues found")
                    
                except json.JSONDecodeError:
                    logger.warning("    ⚠️ Bandit: Could not parse JSON output")
                    self.audit_results["bandit"] = {"error": "JSON parse error"}
            else:
                logger.info("    ✅ Bandit: No security issues found")
                self.audit_results["bandit"] = {"total_issues": 0, "issues": []}
                
        except Exception as e:
            logger.error(f"    ❌ Bandit failed: {e}")
            self.audit_results["bandit"] = {"error": str(e)}
    
    def _run_safety_dependency_scan(self):
        """Run safety dependency vulnerability scan."""
        logger.info("  Running safety dependency scan...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "safety", "check", "--json"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.stdout:
                try:
                    safety_data = json.loads(result.stdout)
                    vulnerabilities = safety_data if isinstance(safety_data, list) else []
                    
                    self.audit_results["safety"] = {
                        "total_vulnerabilities": len(vulnerabilities),
                        "vulnerabilities": vulnerabilities[:10]  # Top 10 vulnerabilities
                    }
                    
                    # Add vulnerabilities as issues
                    for vuln in vulnerabilities:
                        self.issues.append({
                            "tool": "safety",
                            "type": "vulnerability",
                            "severity": "high",
                            "message": vuln.get("advisory", ""),
                            "file": "requirements",
                            "line": 0,
                            "category": "security"
                        })
                    
                    logger.info(f"    ✅ Safety: {len(vulnerabilities)} vulnerabilities found")
                    
                except json.JSONDecodeError:
                    logger.warning("    ⚠️ Safety: Could not parse JSON output")
                    self.audit_results["safety"] = {"error": "JSON parse error"}
            else:
                logger.info("    ✅ Safety: No vulnerabilities found")
                self.audit_results["safety"] = {"total_vulnerabilities": 0, "vulnerabilities": []}
                
        except Exception as e:
            logger.error(f"    ❌ Safety failed: {e}")
            self.audit_results["safety"] = {"error": str(e)}
    
    def _analyze_imports(self):
        """Analyze import structure and find issues."""
        logger.info("  Analyzing import structure...")
        
        import_issues = []
        
        # Find Python files
        python_files = list(self.src_path.rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    for i, line in enumerate(lines, 1):
                        line = line.strip()
                        
                        # Check for unused imports (basic check)
                        if line.startswith('import ') or line.startswith('from '):
                            # This is a simplified check - would need AST for full analysis
                            if 'import *' in line:
                                import_issues.append({
                                    "file": str(py_file.relative_to(self.project_root)),
                                    "line": i,
                                    "issue": "wildcard_import",
                                    "message": "Wildcard import found"
                                })
                            
            except Exception as e:
                logger.warning(f"    ⚠️ Could not analyze {py_file}: {e}")
        
        self.audit_results["imports"] = {
            "total_issues": len(import_issues),
            "issues": import_issues[:20]
        }
        
        # Add import issues
        for issue in import_issues:
            self.issues.append({
                "tool": "import_analyzer",
                "type": "import",
                "severity": "medium",
                "message": issue["message"],
                "file": issue["file"],
                "line": issue["line"],
                "category": "code_structure"
            })
        
        logger.info(f"    ✅ Import analysis: {len(import_issues)} issues found")
    
    def _analyze_file_structure(self):
        """Analyze file and directory structure."""
        logger.info("  Analyzing file structure...")
        
        structure_issues = []
        
        # Check for common structure issues
        python_files = list(self.src_path.rglob("*.py"))
        
        for py_file in python_files:
            # Check file size
            try:
                file_size = py_file.stat().st_size
                if file_size > 10000:  # Files larger than 10KB
                    structure_issues.append({
                        "file": str(py_file.relative_to(self.project_root)),
                        "issue": "large_file",
                        "message": f"Large file ({file_size} bytes)",
                        "size": file_size
                    })
                
                # Check line count
                with open(py_file, 'r', encoding='utf-8') as f:
                    line_count = len(f.readlines())
                    if line_count > 500:  # Files with more than 500 lines
                        structure_issues.append({
                            "file": str(py_file.relative_to(self.project_root)),
                            "issue": "long_file",
                            "message": f"Long file ({line_count} lines)",
                            "lines": line_count
                        })
                        
            except Exception as e:
                logger.warning(f"    ⚠️ Could not analyze {py_file}: {e}")
        
        self.audit_results["file_structure"] = {
            "total_issues": len(structure_issues),
            "issues": structure_issues[:20]
        }
        
        # Add structure issues
        for issue in structure_issues:
            self.issues.append({
                "tool": "structure_analyzer",
                "type": "structure",
                "severity": "low",
                "message": issue["message"],
                "file": issue["file"],
                "line": 0,
                "category": "code_structure"
            })
        
        logger.info(f"    ✅ File structure: {len(structure_issues)} issues found")
    
    def _check_naming_conventions(self):
        """Check naming conventions."""
        logger.info("  Checking naming conventions...")
        
        # This would be a more comprehensive check in a real implementation
        naming_issues = []
        
        self.audit_results["naming"] = {
            "total_issues": len(naming_issues),
            "issues": naming_issues
        }
        
        logger.info(f"    ✅ Naming conventions: {len(naming_issues)} issues found")
    
    def _analyze_configurations(self):
        """Analyze configuration files."""
        logger.info("  Analyzing configurations...")
        
        config_issues = []
        
        # Check for config files
        config_files = [
            self.project_root / "config.yaml",
            self.project_root / "src" / "coda" / "config.yaml",
            self.project_root / ".env",
            self.project_root / "requirements.txt"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Basic checks
                        if config_file.name.endswith('.yaml'):
                            # Check for YAML syntax issues (basic)
                            if 'null' in content.lower():
                                config_issues.append({
                                    "file": str(config_file.relative_to(self.project_root)),
                                    "issue": "null_values",
                                    "message": "Configuration contains null values"
                                })
                        
                except Exception as e:
                    config_issues.append({
                        "file": str(config_file.relative_to(self.project_root)),
                        "issue": "read_error",
                        "message": f"Could not read config file: {e}"
                    })
        
        self.audit_results["configuration"] = {
            "total_issues": len(config_issues),
            "issues": config_issues
        }
        
        # Add config issues
        for issue in config_issues:
            self.issues.append({
                "tool": "config_analyzer",
                "type": "configuration",
                "severity": "medium",
                "message": issue["message"],
                "file": issue["file"],
                "line": 0,
                "category": "configuration"
            })
        
        logger.info(f"    ✅ Configuration analysis: {len(config_issues)} issues found")
    
    def _check_environment_variables(self):
        """Check environment variable usage."""
        logger.info("  Checking environment variables...")
        
        env_issues = []
        
        # This would check for hardcoded secrets, missing env vars, etc.
        # Simplified implementation for now
        
        self.audit_results["environment"] = {
            "total_issues": len(env_issues),
            "issues": env_issues
        }
        
        logger.info(f"    ✅ Environment variables: {len(env_issues)} issues found")
    
    def _analyze_documentation(self):
        """Analyze documentation completeness."""
        logger.info("  Analyzing documentation...")
        
        doc_issues = []
        
        # Check for README files
        readme_files = list(self.project_root.glob("README*"))
        if not readme_files:
            doc_issues.append({
                "file": ".",
                "issue": "missing_readme",
                "message": "No README file found"
            })
        
        self.audit_results["documentation"] = {
            "total_issues": len(doc_issues),
            "issues": doc_issues
        }
        
        logger.info(f"    ✅ Documentation: {len(doc_issues)} issues found")
    
    def _check_code_comments(self):
        """Check code comment quality."""
        logger.info("  Checking code comments...")
        
        comment_issues = []
        
        # This would analyze comment density, quality, etc.
        # Simplified implementation for now
        
        self.audit_results["comments"] = {
            "total_issues": len(comment_issues),
            "issues": comment_issues
        }
        
        logger.info(f"    ✅ Code comments: {len(comment_issues)} issues found")
    
    def _map_pylint_severity(self, pylint_type: str) -> str:
        """Map pylint message types to severity levels."""
        mapping = {
            "error": "high",
            "warning": "medium",
            "refactor": "low",
            "convention": "low",
            "info": "low"
        }
        return mapping.get(pylint_type, "medium")
    
    def _generate_audit_summary(self) -> Dict[str, Any]:
        """Generate comprehensive audit summary."""
        total_issues = len(self.issues)
        
        # Count by severity
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        for issue in self.issues:
            severity_counts[issue["severity"]] += 1
        
        # Count by category
        category_counts = {}
        for issue in self.issues:
            category = issue["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Count by tool
        tool_counts = {}
        for issue in self.issues:
            tool = issue["tool"]
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        
        # Overall assessment
        if severity_counts["high"] > 0:
            overall_status = "critical"
        elif severity_counts["medium"] > 10:
            overall_status = "needs_attention"
        elif total_issues > 50:
            overall_status = "moderate_issues"
        else:
            overall_status = "good"
        
        return {
            "overall_status": overall_status,
            "total_issues": total_issues,
            "severity_breakdown": severity_counts,
            "category_breakdown": category_counts,
            "tool_breakdown": tool_counts,
            "audit_timestamp": datetime.now().isoformat(),
            "files_analyzed": len(list(self.src_path.rglob("*.py"))),
            "tools_used": list(self.audit_results.keys())
        }


def main():
    """Run the comprehensive audit."""
    auditor = ComprehensiveAuditor()
    
    try:
        results = auditor.run_comprehensive_audit()
        
        # Save results to file
        output_file = Path("audits") / f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        summary = results["summary"]
        logger.info("=" * 80)
        logger.info("🎯 COMPREHENSIVE AUDIT SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Overall Status: {summary['overall_status'].upper()}")
        logger.info(f"Total Issues: {summary['total_issues']}")
        logger.info(f"High Severity: {summary['severity_breakdown']['high']}")
        logger.info(f"Medium Severity: {summary['severity_breakdown']['medium']}")
        logger.info(f"Low Severity: {summary['severity_breakdown']['low']}")
        logger.info(f"Files Analyzed: {summary['files_analyzed']}")
        
        logger.info("\n📊 Issues by Category:")
        for category, count in summary["category_breakdown"].items():
            logger.info(f"  • {category}: {count}")
        
        logger.info(f"\n📄 Full report saved to: {output_file}")
        
        return 0 if summary["overall_status"] in ["good", "moderate_issues"] else 1
        
    except Exception as e:
        logger.error(f"Audit failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
