#!/usr/bin/env python3
"""
Comprehensive test runner for the voice processing system.

This script runs all test suites including unit tests, integration tests,
performance benchmarks, load tests, and generates detailed reports.
"""

import sys
import os
import time
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestRunner:
    """Comprehensive test runner with reporting."""
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_test_suite(self, suite_name: str, test_path: str, markers: str = None, 
                      timeout: int = 300) -> Dict[str, Any]:
        """Run a specific test suite and collect results."""
        print(f"\n🧪 Running {suite_name}...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            test_path,
            "-v",
            "--tb=short",
            "--json-report",
            f"--json-report-file={self.output_dir}/{suite_name}_report.json"
        ]
        
        if markers:
            cmd.extend(["-m", markers])
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path(__file__).parent.parent
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Load JSON report if available
            report_file = self.output_dir / f"{suite_name}_report.json"
            json_report = {}
            if report_file.exists():
                try:
                    with open(report_file) as f:
                        json_report = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load JSON report: {e}")
            
            suite_result = {
                "suite_name": suite_name,
                "duration": duration,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "json_report": json_report,
                "success": result.returncode == 0
            }
            
            if suite_result["success"]:
                print(f"✅ {suite_name} completed successfully in {duration:.1f}s")
            else:
                print(f"❌ {suite_name} failed in {duration:.1f}s")
                if result.stderr:
                    print(f"Error output: {result.stderr[:500]}...")
            
            return suite_result
            
        except subprocess.TimeoutExpired:
            print(f"⏰ {suite_name} timed out after {timeout}s")
            return {
                "suite_name": suite_name,
                "duration": timeout,
                "return_code": -1,
                "stdout": "",
                "stderr": f"Test suite timed out after {timeout}s",
                "json_report": {},
                "success": False,
                "timeout": True
            }
        
        except Exception as e:
            print(f"💥 {suite_name} crashed: {e}")
            return {
                "suite_name": suite_name,
                "duration": 0,
                "return_code": -2,
                "stdout": "",
                "stderr": str(e),
                "json_report": {},
                "success": False,
                "error": str(e)
            }
    
    def run_all_tests(self, test_types: List[str] = None) -> Dict[str, Any]:
        """Run all test suites."""
        self.start_time = time.time()
        
        # Define test suites
        test_suites = {
            "unit_voice_manager": {
                "path": "tests/voice/test_comprehensive_voice_manager.py",
                "timeout": 300
            },
            "unit_websocket": {
                "path": "tests/voice/test_comprehensive_websocket.py",
                "timeout": 300
            },
            "integration_websocket": {
                "path": "tests/voice/test_websocket_integration.py",
                "timeout": 600
            },
            "performance_benchmarks": {
                "path": "tests/voice/test_performance_benchmarks.py",
                "markers": "performance",
                "timeout": 900
            },
            "load_tests": {
                "path": "tests/voice/test_load_stress.py",
                "markers": "load_test",
                "timeout": 1200
            },
            "stress_tests": {
                "path": "tests/voice/test_load_stress.py",
                "markers": "stress_test",
                "timeout": 1200
            },
            "recovery_tests": {
                "path": "tests/voice/test_load_stress.py",
                "markers": "recovery_test",
                "timeout": 600
            }
        }
        
        # Filter test types if specified
        if test_types:
            test_suites = {k: v for k, v in test_suites.items() if k in test_types}
        
        print(f"🚀 Starting comprehensive test run with {len(test_suites)} test suites...")
        
        # Run each test suite
        for suite_name, suite_config in test_suites.items():
            result = self.run_test_suite(
                suite_name,
                suite_config["path"],
                suite_config.get("markers"),
                suite_config.get("timeout", 300)
            )
            self.results[suite_name] = result
        
        self.end_time = time.time()
        
        # Generate summary report
        return self.generate_summary_report()
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        total_duration = self.end_time - self.start_time if self.end_time else 0
        
        summary = {
            "test_run_info": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "total_duration": total_duration,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "overall_results": {
                "total_suites": len(self.results),
                "passed_suites": sum(1 for r in self.results.values() if r["success"]),
                "failed_suites": sum(1 for r in self.results.values() if not r["success"]),
                "success_rate": 0.0
            },
            "suite_results": self.results,
            "performance_metrics": self._extract_performance_metrics(),
            "recommendations": self._generate_recommendations()
        }
        
        # Calculate success rate
        if summary["overall_results"]["total_suites"] > 0:
            summary["overall_results"]["success_rate"] = (
                summary["overall_results"]["passed_suites"] / 
                summary["overall_results"]["total_suites"]
            )
        
        # Save detailed report
        report_file = self.output_dir / "comprehensive_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Generate human-readable report
        self._generate_human_readable_report(summary)
        
        return summary
    
    def _extract_performance_metrics(self) -> Dict[str, Any]:
        """Extract performance metrics from test results."""
        metrics = {
            "latency_metrics": {},
            "throughput_metrics": {},
            "resource_usage": {},
            "error_rates": {}
        }
        
        for suite_name, result in self.results.items():
            if "performance" in suite_name or "load" in suite_name:
                # Extract metrics from stdout/stderr
                output = result.get("stdout", "")
                
                # Look for performance indicators in output
                if "Performance:" in output:
                    # Extract performance data (simplified)
                    lines = output.split('\n')
                    for line in lines:
                        if "ms" in line and ":" in line:
                            try:
                                metric_name = line.split(':')[0].strip()
                                metric_value = float(line.split(':')[1].replace('ms', '').strip())
                                metrics["latency_metrics"][f"{suite_name}_{metric_name}"] = metric_value
                            except (ValueError, IndexError):
                                continue
        
        return metrics
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        failed_suites = [name for name, result in self.results.items() if not result["success"]]
        
        if failed_suites:
            recommendations.append(f"❌ {len(failed_suites)} test suite(s) failed: {', '.join(failed_suites)}")
            recommendations.append("🔧 Review failed test logs and fix underlying issues")
        
        # Check for performance issues
        slow_suites = [
            name for name, result in self.results.items() 
            if result["duration"] > 300 and result["success"]
        ]
        
        if slow_suites:
            recommendations.append(f"⏰ Slow test suites detected: {', '.join(slow_suites)}")
            recommendations.append("🚀 Consider optimizing test execution or system performance")
        
        # Check for timeouts
        timeout_suites = [
            name for name, result in self.results.items() 
            if result.get("timeout", False)
        ]
        
        if timeout_suites:
            recommendations.append(f"⏰ Test timeouts detected: {', '.join(timeout_suites)}")
            recommendations.append("🔧 Investigate performance bottlenecks or increase timeout limits")
        
        if not failed_suites and not slow_suites and not timeout_suites:
            recommendations.append("🎉 All tests passed successfully!")
            recommendations.append("✅ System is ready for production deployment")
        
        return recommendations
    
    def _generate_human_readable_report(self, summary: Dict[str, Any]) -> None:
        """Generate human-readable test report."""
        report_file = self.output_dir / "test_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Comprehensive Test Report\n\n")
            
            # Overview
            f.write("## Test Run Overview\n\n")
            f.write(f"- **Start Time:** {summary['test_run_info']['timestamp']}\n")
            f.write(f"- **Total Duration:** {summary['test_run_info']['total_duration']:.1f} seconds\n")
            f.write(f"- **Total Test Suites:** {summary['overall_results']['total_suites']}\n")
            f.write(f"- **Passed:** {summary['overall_results']['passed_suites']}\n")
            f.write(f"- **Failed:** {summary['overall_results']['failed_suites']}\n")
            f.write(f"- **Success Rate:** {summary['overall_results']['success_rate']:.1%}\n\n")
            
            # Suite Results
            f.write("## Test Suite Results\n\n")
            for suite_name, result in summary['suite_results'].items():
                status = "✅ PASSED" if result["success"] else "❌ FAILED"
                f.write(f"### {suite_name} - {status}\n")
                f.write(f"- Duration: {result['duration']:.1f}s\n")
                f.write(f"- Return Code: {result['return_code']}\n")
                
                if not result["success"] and result.get("stderr"):
                    f.write(f"- Error: {result['stderr'][:200]}...\n")
                
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            for rec in summary['recommendations']:
                f.write(f"- {rec}\n")
            
            f.write("\n")
        
        print(f"📊 Detailed reports saved to {self.output_dir}/")


def main():
    """Main function to run comprehensive tests."""
    parser = argparse.ArgumentParser(description="Run comprehensive voice system tests")
    parser.add_argument(
        "--test-types",
        nargs="+",
        choices=[
            "unit_voice_manager", "unit_websocket", "integration_websocket",
            "performance_benchmarks", "load_tests", "stress_tests", "recovery_tests"
        ],
        help="Specific test types to run (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        default="test_results",
        help="Output directory for test results"
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner(args.output_dir)
    
    # Run tests
    summary = runner.run_all_tests(args.test_types)
    
    # Print summary
    print(f"\n📊 Test Run Summary:")
    print(f"Total Suites: {summary['overall_results']['total_suites']}")
    print(f"Passed: {summary['overall_results']['passed_suites']}")
    print(f"Failed: {summary['overall_results']['failed_suites']}")
    print(f"Success Rate: {summary['overall_results']['success_rate']:.1%}")
    print(f"Total Duration: {summary['test_run_info']['total_duration']:.1f}s")
    
    # Print recommendations
    print(f"\n💡 Recommendations:")
    for rec in summary['recommendations']:
        print(f"  {rec}")
    
    # Exit with appropriate code
    if summary['overall_results']['success_rate'] == 1.0:
        print(f"\n🎉 All tests passed! System is ready for deployment.")
        sys.exit(0)
    else:
        print(f"\n⚠️  Some tests failed. Please review the results.")
        sys.exit(1)


if __name__ == "__main__":
    main()
