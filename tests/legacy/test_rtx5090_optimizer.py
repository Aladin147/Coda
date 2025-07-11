#!/usr/bin/env python3
"""Test RTX 5090 optimizer functionality."""

import sys
sys.path.append('src')

from coda.core.rtx5090_optimizer import get_rtx5090_optimizer, apply_rtx5090_optimizations

def test_rtx5090_optimizer():
    """Test RTX 5090 optimizer functionality."""
    print('RTX 5090 Optimizer Test')
    print('=' * 50)
    print()

    # Get optimizer instance
    optimizer = get_rtx5090_optimizer()
    print(f'RTX 5090 Detected: {optimizer.is_rtx5090}')
    print(f'Total Memory: {optimizer.total_memory_gb:.1f} GB')
    print()

    # Apply optimizations
    print('Applying RTX 5090 optimizations...')
    success = apply_rtx5090_optimizations()
    print(f'Optimizations Applied: {success}')
    print()

    # Get performance stats
    stats = optimizer.get_performance_stats()
    if stats:
        print('Current Performance Stats:')
        print(f'  Memory Allocated: {stats.memory_allocated_gb:.2f} GB')
        print(f'  Memory Reserved: {stats.memory_reserved_gb:.2f} GB')
        print(f'  Memory Free: {stats.memory_free_gb:.2f} GB')
        print(f'  TF32 Enabled: {stats.tf32_enabled}')
        print(f'  cuDNN Benchmark: {stats.cudnn_benchmark}')
        print()

    # Run benchmark
    print('Running Performance Benchmark...')
    results = optimizer.benchmark_performance(matrix_size=4096, iterations=50)
    if 'error' not in results:
        print('Benchmark Results:')
        print(f'  Average Time: {results["avg_time_ms"]:.2f} ms')
        print(f'  Performance: {results["gflops"]:.1f} GFLOPS')
        print(f'  Operations/sec: {results["operations_per_sec"]:.1f}')
        print()
        
        # Test workload optimization
        print('Testing workload optimizations...')
        optimizer.optimize_for_workload('llm_inference')
        optimizer.optimize_for_workload('voice_processing')
        print()
        
        print('RTX 5090 optimization test complete!')
    else:
        print(f'Benchmark Error: {results["error"]}')

if __name__ == "__main__":
    test_rtx5090_optimizer()
