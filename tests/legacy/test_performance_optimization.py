#!/usr/bin/env python3
"""
Performance Optimization Testing Script for Coda.

Tests the comprehensive performance optimization system including:
- Connection pooling efficiency
- Cache performance and hit rates
- Memory optimization
- Response time improvements
- Resource management
"""

import asyncio
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Any
import statistics

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from coda.core.performance_optimizer import PerformanceOptimizer, OptimizationLevel, PerformanceTarget, ConnectionPoolConfig
from coda.core.connection_pool import ConnectionPool
from coda.core.optimized_cache import OptimizedCache, CachePolicy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("performance_test")


class PerformanceTestSuite:
    """Comprehensive performance testing suite."""
    
    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.optimizer: Optional[PerformanceOptimizer] = None
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all performance tests."""
        logger.info("🚀 Starting Performance Optimization Test Suite")
        logger.info("=" * 60)
        
        # Test 1: Cache Performance
        logger.info("📊 Test 1: Cache Performance")
        cache_results = await self.test_cache_performance()
        self.results["cache_performance"] = cache_results
        
        # Test 2: Connection Pool Performance
        logger.info("📊 Test 2: Connection Pool Performance")
        pool_results = await self.test_connection_pool_performance()
        self.results["connection_pool_performance"] = pool_results
        
        # Test 3: Performance Optimizer Integration
        logger.info("📊 Test 3: Performance Optimizer Integration")
        optimizer_results = await self.test_performance_optimizer()
        self.results["optimizer_performance"] = optimizer_results
        
        # Test 4: Memory Optimization
        logger.info("📊 Test 4: Memory Optimization")
        memory_results = await self.test_memory_optimization()
        self.results["memory_optimization"] = memory_results
        
        # Test 5: Response Time Optimization
        logger.info("📊 Test 5: Response Time Optimization")
        response_time_results = await self.test_response_time_optimization()
        self.results["response_time_optimization"] = response_time_results
        
        # Generate summary
        summary = self.generate_test_summary()
        self.results["summary"] = summary
        
        logger.info("=" * 60)
        logger.info("🎉 Performance Test Suite Complete!")
        
        return self.results
    
    async def test_cache_performance(self) -> Dict[str, Any]:
        """Test cache performance with different policies."""
        results = {}
        
        # Test different cache policies
        policies = [CachePolicy.LRU, CachePolicy.LFU, CachePolicy.HYBRID]
        
        for policy in policies:
            logger.info(f"  Testing {policy.value} cache policy...")
            
            cache = OptimizedCache(
                max_memory_mb=10.0,
                max_entries=1000,
                policy=policy,
                name=f"test_{policy.value}"
            )
            
            await cache.start()
            
            # Performance test
            start_time = time.time()
            
            # Write test
            for i in range(500):
                await cache.set(f"key_{i}", f"value_{i}_{'x' * 100}")
            
            # Read test (mix of hits and misses)
            hits = 0
            for i in range(1000):
                key = f"key_{i % 600}"  # Some keys won't exist
                result = await cache.get(key)
                if result is not None:
                    hits += 1
            
            end_time = time.time()
            
            stats = cache.get_stats()
            
            results[policy.value] = {
                "total_time_ms": (end_time - start_time) * 1000,
                "hit_rate": stats["hit_rate"],
                "memory_usage_mb": stats["memory_usage_mb"],
                "operations_per_second": 1500 / (end_time - start_time),
                "stats": stats
            }
            
            await cache.stop()
            
            logger.info(f"    ✅ {policy.value}: {stats['hit_rate']:.2%} hit rate, "
                       f"{stats['memory_usage_mb']:.1f}MB used")
        
        return results
    
    async def test_connection_pool_performance(self) -> Dict[str, Any]:
        """Test connection pool performance and efficiency."""
        logger.info("  Testing connection pool efficiency...")
        
        # Create a simple config object for testing
        class SimpleConfig:
            def __init__(self):
                self.max_connections = 20
                self.min_connections = 5
                self.connection_timeout = 10.0
                self.idle_timeout = 300.0
                self.health_check_interval = 30.0

        config = SimpleConfig()
        
        pool = ConnectionPool("test_service", config)
        await pool.initialize()
        
        # Simulate concurrent connection usage
        start_time = time.time()
        
        async def simulate_request():
            try:
                connection = await pool.get_connection()
                # Simulate work
                await asyncio.sleep(0.01)
                await pool.return_connection(connection)
                return True
            except Exception as e:
                logger.error(f"Connection error: {e}")
                return False
        
        # Run concurrent requests
        tasks = [simulate_request() for _ in range(100)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        successful_requests = sum(1 for r in results if r is True)
        stats = pool.get_stats()
        
        await pool.close()
        
        pool_results = {
            "total_time_ms": (end_time - start_time) * 1000,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / len(tasks),
            "requests_per_second": len(tasks) / (end_time - start_time),
            "pool_stats": stats
        }
        
        logger.info(f"    ✅ Pool: {successful_requests}/{len(tasks)} successful, "
                   f"{pool_results['requests_per_second']:.1f} req/s")
        
        return pool_results
    
    async def test_performance_optimizer(self) -> Dict[str, Any]:
        """Test the integrated performance optimizer."""
        logger.info("  Testing performance optimizer integration...")
        
        # Create optimizer with aggressive settings
        targets = PerformanceTarget(
            max_response_time_ms=100.0,
            max_memory_usage_percent=70.0,
            max_cpu_usage_percent=60.0,
            min_cache_hit_rate=0.8
        )
        
        self.optimizer = PerformanceOptimizer(
            optimization_level=OptimizationLevel.AGGRESSIVE,
            targets=targets
        )
        
        await self.optimizer.initialize()
        
        # Register test components
        self.optimizer.register_component("test_llm", "llm")
        self.optimizer.register_component("test_cache", "memory")
        
        # Simulate some load and optimization
        start_time = time.time()
        
        # Cache some responses
        for i in range(100):
            await self.optimizer.cache_response(
                "response", f"test_key_{i}", f"test_response_{i}"
            )
        
        # Get cached responses (should have high hit rate)
        cache_hits = 0
        for i in range(150):
            key = f"test_key_{i % 120}"  # Some hits, some misses
            result = await self.optimizer.get_cached_response("response", key)
            if result is not None:
                cache_hits += 1
        
        # Get performance report
        report = self.optimizer.get_performance_report()
        
        end_time = time.time()
        
        optimizer_results = {
            "total_time_ms": (end_time - start_time) * 1000,
            "cache_hit_rate": cache_hits / 150,
            "performance_report": report,
            "optimization_status": report.get("status", "unknown")
        }
        
        logger.info(f"    ✅ Optimizer: {optimizer_results['cache_hit_rate']:.2%} hit rate, "
                   f"status: {report.get('status', 'unknown')}")
        
        return optimizer_results
    
    async def test_memory_optimization(self) -> Dict[str, Any]:
        """Test memory optimization features."""
        logger.info("  Testing memory optimization...")
        
        if not self.optimizer:
            return {"error": "Optimizer not initialized"}
        
        # Create memory pressure
        large_cache = OptimizedCache(
            max_memory_mb=50.0,
            max_entries=5000,
            name="memory_test"
        )
        
        await large_cache.start()
        
        # Fill cache with large objects
        start_memory = large_cache.get_stats()["memory_usage_mb"]
        
        for i in range(1000):
            large_data = "x" * 1000  # 1KB per entry
            await large_cache.set(f"large_key_{i}", large_data)
        
        peak_memory = large_cache.get_stats()["memory_usage_mb"]
        
        # Trigger optimization
        await large_cache.optimize()
        
        optimized_memory = large_cache.get_stats()["memory_usage_mb"]
        
        await large_cache.stop()
        
        memory_results = {
            "start_memory_mb": start_memory,
            "peak_memory_mb": peak_memory,
            "optimized_memory_mb": optimized_memory,
            "memory_saved_mb": peak_memory - optimized_memory,
            "optimization_efficiency": (peak_memory - optimized_memory) / peak_memory if peak_memory > 0 else 0
        }
        
        logger.info(f"    ✅ Memory: {memory_results['memory_saved_mb']:.1f}MB saved "
                   f"({memory_results['optimization_efficiency']:.1%} efficiency)")
        
        return memory_results
    
    async def test_response_time_optimization(self) -> Dict[str, Any]:
        """Test response time optimization."""
        logger.info("  Testing response time optimization...")
        
        # Test with and without caching
        response_times_no_cache = []
        response_times_with_cache = []
        
        # Simulate expensive operation
        async def expensive_operation(key: str) -> str:
            await asyncio.sleep(0.01)  # Simulate 10ms operation
            return f"result_for_{key}"
        
        # Test without caching
        for i in range(50):
            start = time.time()
            result = await expensive_operation(f"key_{i % 10}")  # Repeat some keys
            end = time.time()
            response_times_no_cache.append((end - start) * 1000)
        
        # Test with caching
        cache = OptimizedCache(max_memory_mb=5.0, name="response_time_test")
        await cache.start()
        
        for i in range(50):
            key = f"key_{i % 10}"
            start = time.time()
            
            # Check cache first
            result = await cache.get(key)
            if result is None:
                result = await expensive_operation(key)
                await cache.set(key, result)
            
            end = time.time()
            response_times_with_cache.append((end - start) * 1000)
        
        await cache.stop()
        
        avg_no_cache = statistics.mean(response_times_no_cache)
        avg_with_cache = statistics.mean(response_times_with_cache)
        improvement = (avg_no_cache - avg_with_cache) / avg_no_cache
        
        response_time_results = {
            "avg_response_time_no_cache_ms": avg_no_cache,
            "avg_response_time_with_cache_ms": avg_with_cache,
            "improvement_percent": improvement,
            "cache_stats": cache.get_stats()
        }
        
        logger.info(f"    ✅ Response Time: {improvement:.1%} improvement with caching "
                   f"({avg_no_cache:.1f}ms → {avg_with_cache:.1f}ms)")
        
        return response_time_results
    
    def generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        summary = {
            "overall_status": "success",
            "total_tests": 5,
            "passed_tests": 0,
            "performance_improvements": {},
            "recommendations": []
        }
        
        # Analyze cache performance
        if "cache_performance" in self.results:
            cache_results = self.results["cache_performance"]
            best_policy = max(cache_results.keys(), 
                            key=lambda k: cache_results[k]["hit_rate"])
            summary["performance_improvements"]["best_cache_policy"] = best_policy
            summary["passed_tests"] += 1
        
        # Analyze connection pool
        if "connection_pool_performance" in self.results:
            pool_results = self.results["connection_pool_performance"]
            if pool_results["success_rate"] > 0.95:
                summary["passed_tests"] += 1
                summary["performance_improvements"]["connection_pooling"] = "excellent"
            else:
                summary["recommendations"].append("Optimize connection pool configuration")
        
        # Analyze optimizer
        if "optimizer_performance" in self.results:
            opt_results = self.results["optimizer_performance"]
            if opt_results["cache_hit_rate"] > 0.7:
                summary["passed_tests"] += 1
                summary["performance_improvements"]["optimizer_integration"] = "effective"
        
        # Analyze memory optimization
        if "memory_optimization" in self.results:
            mem_results = self.results["memory_optimization"]
            if mem_results["optimization_efficiency"] > 0.1:
                summary["passed_tests"] += 1
                summary["performance_improvements"]["memory_optimization"] = f"{mem_results['optimization_efficiency']:.1%}"
        
        # Analyze response time
        if "response_time_optimization" in self.results:
            rt_results = self.results["response_time_optimization"]
            if rt_results["improvement_percent"] > 0.3:
                summary["passed_tests"] += 1
                summary["performance_improvements"]["response_time"] = f"{rt_results['improvement_percent']:.1%}"
        
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
    """Run the performance test suite."""
    test_suite = PerformanceTestSuite()
    
    try:
        results = await test_suite.run_all_tests()
        
        # Print summary
        summary = results["summary"]
        logger.info("=" * 60)
        logger.info("📊 PERFORMANCE TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Overall Status: {summary['overall_status'].upper()}")
        logger.info(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
        
        if summary["performance_improvements"]:
            logger.info("\n🚀 Performance Improvements:")
            for improvement, value in summary["performance_improvements"].items():
                logger.info(f"  • {improvement}: {value}")
        
        if summary["recommendations"]:
            logger.info("\n💡 Recommendations:")
            for rec in summary["recommendations"]:
                logger.info(f"  • {rec}")
        
        # Cleanup
        if test_suite.optimizer:
            await test_suite.optimizer.shutdown()
        
        return 0 if summary["overall_status"] in ["excellent", "good"] else 1
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
