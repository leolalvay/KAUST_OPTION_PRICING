"""
Cost Metrics Utility Module

Provides hardware-independent cost metrics for comparing computational methods.
Designed to complement wall-clock timing with reproducible metrics.

Location: Comparison_Between_Methods/methods/cost_metrics.py

Key Metrics
-----------
1. total_function_calls : int
   Total Python function calls (hardware-independent, like counting steps)
   
2. primitive_calls : int
   Non-recursive function calls (avoids double-counting recursion)
   
3. cpu_time : float
   CPU time excluding sleep/IO (better than wall-clock)
   
4. peak_memory : int
   Peak memory allocation in bytes

Usage
-----
Simple (context manager):
    from methods.cost_metrics import CostProfiler
    
    with CostProfiler() as profiler:
        result = your_method(args)
    print(profiler.summary())

Simple (function wrapper):
    from methods.cost_metrics import profile_function
    
    result, costs = profile_function(your_method, args)
    
For experiments:
    from methods.cost_metrics import CostMetrics
    
    costs = CostMetrics()  # dataclass to store results
    
Threading control:
    from methods.cost_metrics import single_threaded
    
    with single_threaded():
        # All BLAS operations use 1 thread here
        result = your_method(args)

Author: Wadoud (KAUST Internship)
"""

import os
import time
import cProfile
import pstats
import io
import tracemalloc
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Dict, Tuple
from contextlib import contextmanager


# =============================================================================
# Data Classes for Storing Results
# =============================================================================

@dataclass
class CostMetrics:
    """
    Container for all cost metrics from a single run.
    
    Attributes
    ----------
    wall_time : float
        Wall-clock time in seconds (hardware-dependent, for reference)
    cpu_time : float
        CPU time in seconds (excludes sleep/IO, more consistent)
    total_calls : int
        Total Python function calls (hardware-independent!)
    primitive_calls : int
        Non-recursive function calls
    peak_memory_bytes : int
        Peak memory usage in bytes
    method_name : str
        Name of the method being profiled
    """
    wall_time: float = 0.0
    cpu_time: float = 0.0
    total_calls: int = 0
    primitive_calls: int = 0
    peak_memory_bytes: int = 0
    method_name: str = ""
    
    @property
    def peak_memory_mb(self) -> float:
        """Peak memory in megabytes."""
        return self.peak_memory_bytes / (1024 * 1024)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialisation."""
        return {
            'method': self.method_name,
            'wall_time_s': self.wall_time,
            'cpu_time_s': self.cpu_time,
            'total_function_calls': self.total_calls,
            'primitive_calls': self.primitive_calls,
            'peak_memory_mb': self.peak_memory_mb,
        }
    
    def __str__(self) -> str:
        return (
            f"{self.method_name}:\n"
            f"  Wall time:      {self.wall_time:.3f}s\n"
            f"  CPU time:       {self.cpu_time:.3f}s\n"
            f"  Function calls: {self.total_calls:,}\n"
            f"  Primitive calls: {self.primitive_calls:,}\n"
            f"  Peak memory:    {self.peak_memory_mb:.2f} MB"
        )


# =============================================================================
# Main Profiler Class
# =============================================================================

class CostProfiler:
    """
    Context manager for profiling computational cost.
    
    Captures multiple cost metrics simultaneously:
    - Wall-clock time
    - CPU time  
    - Total function calls (hardware-independent!)
    - Peak memory usage
    
    Example
    -------
    >>> from methods.cost_metrics import CostProfiler
    >>> with CostProfiler("MLMC") as p:
    ...     result = estimate_volatility_mlmc(params, t_grid, s_grid)
    >>> print(p.metrics)
    >>> print(f"Function calls: {p.metrics.total_calls:,}")
    """
    
    def __init__(self, method_name: str = ""):
        self.method_name = method_name
        self._profiler = cProfile.Profile()
        self._metrics = CostMetrics(method_name=method_name)
        
        # Timing
        self._wall_start = 0.0
        self._cpu_start = 0.0
        
    def __enter__(self) -> 'CostProfiler':
        # Start memory tracking
        tracemalloc.start()
        
        # Start timing
        self._wall_start = time.perf_counter()
        self._cpu_start = time.process_time()
        
        # Start profiling
        self._profiler.enable()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop profiling
        self._profiler.disable()
        
        # Stop timing
        self._metrics.wall_time = time.perf_counter() - self._wall_start
        self._metrics.cpu_time = time.process_time() - self._cpu_start
        
        # Get memory stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._metrics.peak_memory_bytes = peak
        
        # Extract call counts from profiler
        stats = pstats.Stats(self._profiler)
        
        # Sum up all function calls
        total_calls = 0
        primitive_calls = 0
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            # cc = call count, nc = primitive (non-recursive) count
            total_calls += cc
            primitive_calls += nc
        
        self._metrics.total_calls = total_calls
        self._metrics.primitive_calls = primitive_calls
        
        return False  # Don't suppress exceptions
    
    @property
    def metrics(self) -> CostMetrics:
        """Return the collected metrics."""
        return self._metrics
    
    def summary(self) -> str:
        """Return a formatted summary string."""
        return str(self._metrics)
    
    def get_top_functions(self, n: int = 10, sort_by: str = 'calls') -> str:
        """
        Get the top N functions by the specified metric.
        
        Parameters
        ----------
        n : int
            Number of functions to show
        sort_by : str
            Sort key: 'calls', 'time', 'cumulative'
        
        Returns
        -------
        str
            Formatted table of top functions
        """
        sort_key = {
            'calls': 'calls',
            'time': 'tottime',
            'cumulative': 'cumtime',
        }.get(sort_by, 'calls')
        
        stream = io.StringIO()
        stats = pstats.Stats(self._profiler, stream=stream)
        stats.sort_stats(sort_key)
        stats.print_stats(n)
        return stream.getvalue()


# =============================================================================
# Simple Function Wrapper
# =============================================================================

def profile_function(
    func: Callable,
    *args,
    method_name: str = "",
    **kwargs
) -> Tuple[Any, CostMetrics]:
    """
    Profile a single function call and return result with metrics.
    
    Parameters
    ----------
    func : callable
        Function to profile
    *args : tuple
        Positional arguments for the function
    method_name : str
        Name for the metrics (defaults to function name)
    **kwargs : dict
        Keyword arguments for the function
    
    Returns
    -------
    result : Any
        Return value of the function
    metrics : CostMetrics
        Collected cost metrics
    
    Example
    -------
    >>> from methods.cost_metrics import profile_function
    >>> result, costs = profile_function(
    ...     estimate_volatility_mlmc, 
    ...     params, t_grid, s_grid,
    ...     method_name="MLMC"
    ... )
    >>> print(f"Total calls: {costs.total_calls:,}")
    """
    name = method_name or func.__name__
    
    with CostProfiler(name) as profiler:
        result = func(*args, **kwargs)
    
    return result, profiler.metrics


# =============================================================================
# Quick Cost Counter (Simplest Usage)
# =============================================================================

def count_function_calls(func: Callable, *args, **kwargs) -> Tuple[Any, int]:
    """
    Ultra-simple function call counter.
    
    Like time.process_time() but for counting operations.
    
    Parameters
    ----------
    func : callable
        Function to profile
    *args, **kwargs
        Arguments for the function
    
    Returns
    -------
    result : Any
        Return value of the function
    call_count : int
        Total number of Python function calls
    
    Example
    -------
    >>> from methods.cost_metrics import count_function_calls
    >>> result, n_calls = count_function_calls(my_method, arg1, arg2)
    >>> print(f"Operations: {n_calls:,}")
    """
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()
    
    # Count total calls
    stats = pstats.Stats(profiler)
    total_calls = sum(cc for (cc, nc, tt, ct, callers) in stats.stats.values())
    
    return result, total_calls


# =============================================================================
# Threading Control
# =============================================================================

@contextmanager
def single_threaded():
    """
    Context manager to disable BLAS/OpenMP parallelisation.
    
    Useful for fair comparison between methods when your Mac's Accelerate
    framework automatically parallelises operations.
    
    Example
    -------
    >>> from methods.cost_metrics import single_threaded, profile_function
    >>> with single_threaded():
    ...     result_mlmc, costs_mlmc = profile_function(mlmc_method, ...)
    ...     result_laplace, costs_laplace = profile_function(laplace_method, ...)
    
    Note
    ----
    Requires `threadpoolctl` package: pip install threadpoolctl
    If not installed, this context manager does nothing but prints a warning.
    """
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1, user_api='blas'):
            yield
    except ImportError:
        print("Warning: threadpoolctl not installed. "
              "Parallelisation not disabled. "
              "Install with: pip install threadpoolctl")
        yield


@contextmanager  
def set_threading(n_threads: int = 1):
    """
    Context manager to set specific number of threads for BLAS operations.
    
    Parameters
    ----------
    n_threads : int
        Number of threads to use. Default 1 (single-threaded).
    
    Note
    ----
    Requires `threadpoolctl` package: pip install threadpoolctl
    """
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=n_threads, user_api='blas'):
            yield
    except ImportError:
        print(f"Warning: threadpoolctl not installed. Cannot set {n_threads} threads.")
        yield


# =============================================================================
# Comparison Helper
# =============================================================================

def compare_methods(
    methods: Dict[str, Callable],
    *args,
    single_thread: bool = False,
    **kwargs
) -> Dict[str, CostMetrics]:
    """
    Compare multiple methods on the same inputs.
    
    Parameters
    ----------
    methods : dict
        Dictionary mapping method names to callables
    *args : tuple
        Arguments to pass to each method
    single_thread : bool
        Whether to disable parallelisation for fair comparison
    **kwargs : dict
        Keyword arguments to pass to each method
    
    Returns
    -------
    dict
        Dictionary mapping method names to CostMetrics
    
    Example
    -------
    >>> from methods.cost_metrics import compare_methods
    >>> methods = {
    ...     'MLMC': lambda: estimate_volatility_mlmc(params, t_grid, s_grid),
    ...     'Laplace': lambda: estimate_volatility_laplace(params, t_grid, s_grid),
    ... }
    >>> results = compare_methods(methods, single_thread=True)
    >>> for name, metrics in results.items():
    ...     print(f"{name}: {metrics.total_calls:,} calls in {metrics.cpu_time:.2f}s")
    """
    all_metrics = {}
    
    if single_thread:
        ctx = single_threaded()
    else:
        ctx = contextmanager(lambda: (yield))()
    
    with ctx:
        for name, method in methods.items():
            with CostProfiler(name) as profiler:
                if callable(method):
                    method(*args, **kwargs) if args or kwargs else method()
            all_metrics[name] = profiler.metrics
    
    return all_metrics


# =============================================================================
# Pretty Printing
# =============================================================================

def print_comparison_table(metrics_dict: Dict[str, CostMetrics]) -> None:
    """
    Print a formatted comparison table.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary mapping method names to CostMetrics
    """
    print()
    print("=" * 80)
    print("COST METRICS COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Method':<15} {'Wall (s)':<12} {'CPU (s)':<12} "
          f"{'Calls':<15} {'Memory (MB)':<12}")
    print("-" * 80)
    
    for name, m in metrics_dict.items():
        print(f"{name:<15} {m.wall_time:<12.3f} {m.cpu_time:<12.3f} "
              f"{m.total_calls:<15,} {m.peak_memory_mb:<12.2f}")
    
    print("-" * 80)
    
    # Compute ratios if exactly 2 methods
    if len(metrics_dict) == 2:
        names = list(metrics_dict.keys())
        m1, m2 = metrics_dict[names[0]], metrics_dict[names[1]]
        
        print()
        print(f"Ratios ({names[1]} / {names[0]}):")
        if m1.wall_time > 0:
            print(f"  Wall time:      {m2.wall_time / m1.wall_time:.2f}x")
        if m1.cpu_time > 0:
            print(f"  CPU time:       {m2.cpu_time / m1.cpu_time:.2f}x")
        if m1.total_calls > 0:
            print(f"  Function calls: {m2.total_calls / m1.total_calls:.2f}x")
    
    print("=" * 80)


# =============================================================================
# Integration with Existing Code
# =============================================================================

def add_cost_metrics_to_result(result_obj: Any, metrics: CostMetrics) -> Any:
    """
    Add cost metrics to an existing result object.
    
    If the result object has a 'computation_time' attribute, this adds
    additional cost metrics alongside it.
    
    Parameters
    ----------
    result_obj : Any
        Result object (e.g., VolatilitySurfaceResult)
    metrics : CostMetrics
        Collected cost metrics
    
    Returns
    -------
    result_obj : Any
        Modified result object with added metrics
    """
    # Add metrics as attributes
    result_obj.cost_metrics = metrics
    result_obj.total_function_calls = metrics.total_calls
    result_obj.cpu_time = metrics.cpu_time
    result_obj.peak_memory_mb = metrics.peak_memory_mb
    
    return result_obj


# =============================================================================
# Example Usage / Self-Test
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    
    print("=" * 60)
    print("COST METRICS UTILITY - DEMO")
    print("=" * 60)
    
    # Define two example methods
    def method_a(n):
        """Simulate MLMC-like computation."""
        result = 0
        for level in range(5):
            samples = n // (2 ** level)
            arr = np.random.randn(samples, 10)
            result += np.mean(arr @ arr.T)
        return result
    
    def method_b(n):
        """Simulate Laplace-like computation."""
        arr = np.random.randn(n, 10)
        # More matrix operations
        cov = arr.T @ arr
        eigvals = np.linalg.eigvalsh(cov)
        return np.sum(eigvals)
    
    n = 10000
    
    # Method 1: Simple call counting
    print("\n1. Simple call counting:")
    print("-" * 40)
    result_a, calls_a = count_function_calls(method_a, n)
    result_b, calls_b = count_function_calls(method_b, n)
    print(f"Method A: {calls_a:,} function calls")
    print(f"Method B: {calls_b:,} function calls")
    print(f"Ratio B/A: {calls_b / calls_a:.2f}x")
    
    # Method 2: Full profiling
    print("\n2. Full profiling with CostProfiler:")
    print("-" * 40)
    with CostProfiler("Method A") as p_a:
        method_a(n)
    with CostProfiler("Method B") as p_b:
        method_b(n)
    
    print(p_a.summary())
    print()
    print(p_b.summary())
    
    # Method 3: Comparison table
    print("\n3. Comparison table:")
    print_comparison_table({
        "Method A": p_a.metrics,
        "Method B": p_b.metrics,
    })
    
    # Method 4: With threading control
    print("\n4. With threading control:")
    print("-" * 40)
    try:
        with single_threaded():
            _, costs_single = profile_function(method_a, n, method_name="A (1 thread)")
        print(f"Single-threaded: {costs_single.total_calls:,} calls, "
              f"{costs_single.cpu_time:.3f}s CPU")
    except Exception as e:
        print(f"Threading control skipped: {e}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
