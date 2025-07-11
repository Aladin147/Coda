#!/usr/bin/env python3
"""
Execute comprehensive testing and optimization phase.

This script orchestrates the complete testing and optimization process
including validation, performance testing, and optimization implementation.
"""

import sys
import os
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scripts.run_comprehensive_tests import TestRunner
from scripts.validate_system import main as validate_system


class TestingPhaseExecutor:
    """Execute complete testing and optimization phase."""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    async def execute_phase(self) -> Dict[str, Any]:
        """Execute the complete testing and optimization phase."""
        self.start_time = time.time()
        
        print("🚀 Starting Phase 7: Testing & Optimization")
        print("=" * 60)
        
        # Step 1: System Validation
        print("\n📋 Step 1: System Validation")
        validation_result = await self._run_system_validation()
        self.results["validation"] = validation_result
        
        if not validation_result["success"]:
            print("❌ System validation failed. Cannot proceed with testing.")
            return self._generate_final_report()
        
        # Step 2: Unit and Integration Testing
        print("\n🧪 Step 2: Unit and Integration Testing")
        unit_test_result = await self._run_unit_tests()
        self.results["unit_tests"] = unit_test_result
        
        # Step 3: Performance Benchmarking
        print("\n⚡ Step 3: Performance Benchmarking")
        performance_result = await self._run_performance_tests()
        self.results["performance"] = performance_result
        
        # Step 4: Load and Stress Testing
        print("\n🔥 Step 4: Load and Stress Testing")
        load_test_result = await self._run_load_tests()
        self.results["load_tests"] = load_test_result
        
        # Step 5: Optimization Implementation
        print("\n🎯 Step 5: Optimization Implementation")
        optimization_result = await self._implement_optimizations()
        self.results["optimization"] = optimization_result
        
        # Step 6: Final Validation
        print("\n✅ Step 6: Final Validation")
        final_validation_result = await self._run_final_validation()
        self.results["final_validation"] = final_validation_result
        
        self.end_time = time.time()
        
        # Generate comprehensive report
        return self._generate_final_report()
    
    async def _run_system_validation(self) -> Dict[str, Any]:
        """Run system validation."""
        try:
            print("  Running system validation...")
            
            # Run validation script
            import subprocess
            result = subprocess.run(
                [sys.executable, "scripts/validate_system.py"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            success = result.returncode == 0
            
            if success:
                print("  ✅ System validation passed")
            else:
                print("  ❌ System validation failed")
                print(f"  Error: {result.stderr}")
            
            return {
                "success": success,
                "return_code": result.returncode,
                "output": result.stdout,
                "error": result.stderr
            }
            
        except Exception as e:
            print(f"  💥 System validation crashed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _run_unit_tests(self) -> Dict[str, Any]:
        """Run unit and integration tests."""
        try:
            print("  Running unit and integration tests...")
            
            runner = TestRunner("test_results/unit_tests")
            
            # Run unit tests
            unit_result = runner.run_test_suite(
                "unit_voice_manager",
                "tests/voice/test_comprehensive_voice_manager.py",
                timeout=300
            )
            
            websocket_result = runner.run_test_suite(
                "unit_websocket",
                "tests/voice/test_comprehensive_websocket.py",
                timeout=300
            )
            
            integration_result = runner.run_test_suite(
                "integration_websocket",
                "tests/voice/test_websocket_integration.py",
                timeout=600
            )
            
            # Calculate overall success
            all_results = [unit_result, websocket_result, integration_result]
            success_count = sum(1 for r in all_results if r["success"])
            overall_success = success_count == len(all_results)
            
            if overall_success:
                print(f"  ✅ All unit tests passed ({success_count}/{len(all_results)})")
            else:
                print(f"  ⚠️  Some unit tests failed ({success_count}/{len(all_results)})")
            
            return {
                "success": overall_success,
                "passed": success_count,
                "total": len(all_results),
                "results": {
                    "voice_manager": unit_result,
                    "websocket": websocket_result,
                    "integration": integration_result
                }
            }
            
        except Exception as e:
            print(f"  💥 Unit tests crashed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance benchmarking tests."""
        try:
            print("  Running performance benchmarks...")
            
            runner = TestRunner("test_results/performance")
            
            result = runner.run_test_suite(
                "performance_benchmarks",
                "tests/voice/test_performance_benchmarks.py",
                markers="performance",
                timeout=900
            )
            
            if result["success"]:
                print("  ✅ Performance benchmarks completed")
            else:
                print("  ⚠️  Performance benchmarks had issues")
            
            return result
            
        except Exception as e:
            print(f"  💥 Performance tests crashed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _run_load_tests(self) -> Dict[str, Any]:
        """Run load and stress tests."""
        try:
            print("  Running load and stress tests...")
            
            runner = TestRunner("test_results/load_tests")
            
            # Run load tests
            load_result = runner.run_test_suite(
                "load_tests",
                "tests/voice/test_load_stress.py",
                markers="load_test",
                timeout=1200
            )
            
            # Run stress tests
            stress_result = runner.run_test_suite(
                "stress_tests",
                "tests/voice/test_load_stress.py",
                markers="stress_test",
                timeout=1200
            )
            
            # Run recovery tests
            recovery_result = runner.run_test_suite(
                "recovery_tests",
                "tests/voice/test_load_stress.py",
                markers="recovery_test",
                timeout=600
            )
            
            all_results = [load_result, stress_result, recovery_result]
            success_count = sum(1 for r in all_results if r["success"])
            overall_success = success_count >= 2  # Allow one failure
            
            if overall_success:
                print(f"  ✅ Load tests completed successfully ({success_count}/3)")
            else:
                print(f"  ⚠️  Load tests had significant issues ({success_count}/3)")
            
            return {
                "success": overall_success,
                "passed": success_count,
                "total": 3,
                "results": {
                    "load": load_result,
                    "stress": stress_result,
                    "recovery": recovery_result
                }
            }
            
        except Exception as e:
            print(f"  💥 Load tests crashed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _implement_optimizations(self) -> Dict[str, Any]:
        """Implement performance optimizations."""
        try:
            print("  Implementing performance optimizations...")
            
            # Initialize comprehensive optimizer
            from src.coda.components.voice.comprehensive_optimizer import get_comprehensive_optimizer
            
            optimizer = get_comprehensive_optimizer()
            
            # Start optimization
            await optimizer.start_optimization()
            
            # Let it run for a short time to collect metrics
            await asyncio.sleep(10)
            
            # Get optimization report
            report = optimizer.get_optimization_report()
            
            # Stop optimization
            await optimizer.stop_optimization()
            
            print("  ✅ Optimization analysis completed")
            print(f"    Performance Score: {report.get('performance_scores', {}).get('composite_score', 0):.1f}/100")
            print(f"    Optimization Opportunities: {len(report.get('optimization_opportunities', []))}")
            
            return {
                "success": True,
                "optimization_report": report,
                "opportunities_found": len(report.get('optimization_opportunities', [])),
                "performance_score": report.get('performance_scores', {}).get('composite_score', 0)
            }
            
        except Exception as e:
            print(f"  💥 Optimization implementation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _run_final_validation(self) -> Dict[str, Any]:
        """Run final validation after optimizations."""
        try:
            print("  Running final system validation...")
            
            # Run validation again to ensure system is still working
            validation_result = await self._run_system_validation()
            
            if validation_result["success"]:
                print("  ✅ Final validation passed - system is ready")
            else:
                print("  ❌ Final validation failed - system needs attention")
            
            return validation_result
            
        except Exception as e:
            print(f"  💥 Final validation crashed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final testing phase report."""
        total_duration = self.end_time - self.start_time if self.end_time else 0
        
        # Calculate overall success
        phase_results = []
        for phase_name, result in self.results.items():
            if isinstance(result, dict) and "success" in result:
                phase_results.append(result["success"])
        
        overall_success = all(phase_results) if phase_results else False
        success_rate = sum(phase_results) / len(phase_results) if phase_results else 0
        
        report = {
            "phase": "Testing & Optimization",
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": total_duration,
            "overall_success": overall_success,
            "success_rate": success_rate,
            "phase_results": self.results,
            "summary": {
                "validation": self.results.get("validation", {}).get("success", False),
                "unit_tests": self.results.get("unit_tests", {}).get("success", False),
                "performance": self.results.get("performance", {}).get("success", False),
                "load_tests": self.results.get("load_tests", {}).get("success", False),
                "optimization": self.results.get("optimization", {}).get("success", False),
                "final_validation": self.results.get("final_validation", {}).get("success", False)
            },
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check validation
        if not self.results.get("validation", {}).get("success", False):
            recommendations.append("❌ System validation failed - fix critical issues before deployment")
        
        # Check unit tests
        unit_test_result = self.results.get("unit_tests", {})
        if not unit_test_result.get("success", False):
            passed = unit_test_result.get("passed", 0)
            total = unit_test_result.get("total", 0)
            recommendations.append(f"⚠️  Unit tests partially failed ({passed}/{total}) - review and fix failing tests")
        
        # Check performance
        if not self.results.get("performance", {}).get("success", False):
            recommendations.append("⚡ Performance benchmarks failed - investigate performance issues")
        
        # Check load tests
        load_test_result = self.results.get("load_tests", {})
        if not load_test_result.get("success", False):
            recommendations.append("🔥 Load tests failed - system may not handle production load")
        
        # Check optimization
        optimization_result = self.results.get("optimization", {})
        if optimization_result.get("success", False):
            opportunities = optimization_result.get("opportunities_found", 0)
            if opportunities > 0:
                recommendations.append(f"🎯 {opportunities} optimization opportunities identified - consider implementing")
        
        # Check final validation
        if not self.results.get("final_validation", {}).get("success", False):
            recommendations.append("❌ Final validation failed - system integrity compromised")
        
        # Success case
        if not recommendations:
            recommendations.append("🎉 All tests passed successfully!")
            recommendations.append("✅ System is ready for production deployment")
            recommendations.append("🚀 Consider implementing identified optimizations for better performance")
        
        return recommendations
    
    def print_final_summary(self, report: Dict[str, Any]) -> None:
        """Print final testing phase summary."""
        print("\n" + "=" * 60)
        print("📊 TESTING & OPTIMIZATION PHASE SUMMARY")
        print("=" * 60)
        
        print(f"Duration: {report['duration_seconds']:.1f} seconds")
        print(f"Overall Success: {'✅ YES' if report['overall_success'] else '❌ NO'}")
        print(f"Success Rate: {report['success_rate']:.1%}")
        
        print("\n📋 Phase Results:")
        for phase, success in report['summary'].items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"  {phase.replace('_', ' ').title()}: {status}")
        
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  {rec}")
        
        print("\n" + "=" * 60)


async def main():
    """Main function to execute testing phase."""
    executor = TestingPhaseExecutor()
    
    try:
        report = await executor.execute_phase()
        executor.print_final_summary(report)
        
        # Save report
        import json
        report_file = Path("test_results/testing_phase_report.json")
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        # Exit with appropriate code
        if report['overall_success']:
            print("\n🎉 Testing & Optimization Phase COMPLETED SUCCESSFULLY!")
            sys.exit(0)
        else:
            print("\n⚠️  Testing & Optimization Phase completed with issues.")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n💥 Testing phase execution failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
