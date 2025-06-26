"""
Performance Testing Module for Agentic Edu-RAG System

This module provides comprehensive performance testing capabilities including
response time measurement, scalability testing, API usage analysis, and
resource monitoring for the multi-agent RAG system.

Author: Agentic Edu-RAG System
"""

import asyncio
import time
import json
import logging
import statistics
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openai import AsyncOpenAI, OpenAI
import aiohttp
import memory_profiler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""
    response_time: float
    api_calls_count: int
    tokens_used: Dict[str, int]
    memory_usage: float
    cpu_usage: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ConcurrencyTestResult:
    """Results from concurrency testing."""
    concurrent_users: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput_rps: float
    error_rate: float
    resource_usage: Dict[str, float]

@dataclass
class ScalabilityTestResult:
    """Results from scalability testing."""
    query_loads: List[int]
    response_times: List[float]
    throughput_metrics: List[float]
    resource_usage_progression: List[Dict[str, float]]
    failure_points: List[Dict[str, Any]]
    scaling_efficiency: float

@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    min_users: int = 1
    max_users: int = 50
    step_size: int = 5
    duration_per_step: int = 30  # seconds
    ramp_up_time: int = 10  # seconds
    test_queries: List[str] = field(default_factory=list)

class SystemResourceMonitor:
    """Monitor system resources during testing."""
    
    def __init__(self, monitoring_interval: float = 0.5):
        self.monitoring_interval = monitoring_interval
        self.monitoring = False
        self.resource_data = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        self.monitoring = True
        self.resource_data = []
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> List[Dict[str, float]]:
        """Stop monitoring and return collected data."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return self.resource_data
    
    def _monitor_resources(self):
        """Monitor system resources continuously."""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                net_io = psutil.net_io_counters()
                
                resource_snapshot = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_mb': memory.used / (1024 * 1024),
                    'memory_available_mb': memory.available / (1024 * 1024),
                    'disk_read_mb': disk_io.read_bytes / (1024 * 1024) if disk_io else 0,
                    'disk_write_mb': disk_io.write_bytes / (1024 * 1024) if disk_io else 0,
                    'network_sent_mb': net_io.bytes_sent / (1024 * 1024) if net_io else 0,
                    'network_recv_mb': net_io.bytes_recv / (1024 * 1024) if net_io else 0
                }
                
                self.resource_data.append(resource_snapshot)
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                time.sleep(self.monitoring_interval)

class APIUsageTracker:
    """Track OpenAI API usage and costs."""
    
    def __init__(self):
        self.reset_tracking()
    
    def reset_tracking(self):
        """Reset tracking counters."""
        self.api_calls = 0
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.estimated_cost = 0.0
        self.call_timestamps = []
        
    def track_api_call(self, response: Any, model: str = "gpt-4o"):
        """Track an OpenAI API response."""
        self.api_calls += 1
        self.call_timestamps.append(datetime.now())
        
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            self.total_tokens += usage.total_tokens
            
            # Estimate cost (rates as of 2024)
            cost_per_input_token = self._get_input_cost_per_token(model)
            cost_per_output_token = self._get_output_cost_per_token(model)
            
            call_cost = (usage.prompt_tokens * cost_per_input_token + 
                        usage.completion_tokens * cost_per_output_token)
            self.estimated_cost += call_cost
    
    def _get_input_cost_per_token(self, model: str) -> float:
        """Get input token cost for model."""
        costs = {
            "gpt-4o": 0.000005,  # $5 per 1M tokens
            "gpt-4o-mini": 0.00000015,  # $0.15 per 1M tokens
            "gpt-3.5-turbo": 0.0000005,  # $0.5 per 1M tokens
        }
        return costs.get(model, 0.000005)
    
    def _get_output_cost_per_token(self, model: str) -> float:
        """Get output token cost for model."""
        costs = {
            "gpt-4o": 0.000015,  # $15 per 1M tokens
            "gpt-4o-mini": 0.0000006,  # $0.6 per 1M tokens
            "gpt-3.5-turbo": 0.0000015,  # $1.5 per 1M tokens
        }
        return costs.get(model, 0.000015)
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary."""
        return {
            'api_calls': self.api_calls,
            'total_tokens': self.total_tokens,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'estimated_cost_usd': self.estimated_cost,
            'calls_per_minute': len(self.call_timestamps) / 
                              ((datetime.now() - self.call_timestamps[0]).total_seconds() / 60) 
                              if self.call_timestamps else 0
        }

class PerformanceTester:
    """
    Comprehensive performance testing for the Agentic Edu-RAG system.
    
    Tests response times, scalability, concurrency, and resource usage
    across different load patterns and query types.
    """
    
    def __init__(self, 
                 system_endpoint: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 output_dir: Path = Path("performance_results")):
        """
        Initialize performance tester.
        
        Args:
            system_endpoint: URL endpoint for the RAG system
            openai_api_key: OpenAI API key for direct testing
            output_dir: Directory for test results
        """
        self.system_endpoint = system_endpoint
        self.openai_client = AsyncOpenAI(api_key=openai_api_key) if openai_api_key else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.resource_monitor = SystemResourceMonitor()
        self.api_tracker = APIUsageTracker()
        
        # Test configurations
        self.test_queries = self._load_default_test_queries()
        self.test_results = {}
        
    def _load_default_test_queries(self) -> List[Dict[str, Any]]:
        """Load default test queries for different SRL phases."""
        return [
            # Implementation queries (Forethought phase)
            {
                "query": "How do I implement a binary search algorithm in Python?",
                "type": "implementation",
                "complexity": "medium"
            },
            {
                "query": "What's the best approach to sort a list of dictionaries by multiple keys?",
                "type": "implementation", 
                "complexity": "easy"
            },
            {
                "query": "How can I design a class hierarchy for a simple game?",
                "type": "implementation",
                "complexity": "hard"
            },
            
            # Debugging queries (Performance phase)
            {
                "query": "My code gives 'IndexError: list index out of range'. Here's the code: for i in range(len(lst)+1): print(lst[i])",
                "type": "debugging",
                "complexity": "easy",
                "code": "for i in range(len(lst)+1): print(lst[i])",
                "error": "IndexError: list index out of range"
            },
            {
                "query": "Why does my recursive function cause a stack overflow?",
                "type": "debugging",
                "complexity": "medium",
                "code": "def factorial(n): return n * factorial(n-1)"
            },
            {
                "query": "My sorting algorithm is very slow on large datasets. How can I optimize it?",
                "type": "debugging",
                "complexity": "hard"
            }
        ]

    async def run_response_time_tests(self, 
                                    num_iterations: int = 10,
                                    warm_up_iterations: int = 3) -> Dict[str, PerformanceMetrics]:
        """
        Test response times for single queries across different types.
        
        Args:
            num_iterations: Number of test iterations per query type
            warm_up_iterations: Number of warm-up calls before measurement
            
        Returns:
            Dictionary of performance metrics by query type
        """
        logger.info(f"Starting response time tests with {num_iterations} iterations")
        
        results = {}
        
        for query_data in self.test_queries:
            query_type = f"{query_data['type']}_{query_data['complexity']}"
            logger.info(f"Testing {query_type}")
            
            # Warm-up iterations
            for _ in range(warm_up_iterations):
                try:
                    await self._make_test_request(query_data)
                except Exception as e:
                    logger.warning(f"Warm-up iteration failed: {e}")
            
            # Actual test iterations
            metrics_list = []
            for iteration in range(num_iterations):
                try:
                    self.resource_monitor.start_monitoring()
                    start_time = time.perf_counter()
                    
                    response = await self._make_test_request(query_data)
                    
                    end_time = time.perf_counter()
                    resource_data = self.resource_monitor.stop_monitoring()
                    
                    # Calculate metrics
                    response_time = end_time - start_time
                    memory_usage = statistics.mean([r['memory_used_mb'] for r in resource_data]) if resource_data else 0
                    cpu_usage = statistics.mean([r['cpu_percent'] for r in resource_data]) if resource_data else 0
                    
                    metrics = PerformanceMetrics(
                        response_time=response_time,
                        api_calls_count=1,
                        tokens_used=self._extract_token_usage(response),
                        memory_usage=memory_usage,
                        cpu_usage=cpu_usage,
                        success=True
                    )
                    
                    metrics_list.append(metrics)
                    logger.debug(f"Iteration {iteration + 1}: {response_time:.3f}s")
                    
                except Exception as e:
                    logger.error(f"Test iteration {iteration + 1} failed: {e}")
                    metrics = PerformanceMetrics(
                        response_time=0.0,
                        api_calls_count=0,
                        tokens_used={},
                        memory_usage=0.0,
                        cpu_usage=0.0,
                        success=False,
                        error_message=str(e)
                    )
                    metrics_list.append(metrics)
            
            # Aggregate results
            successful_metrics = [m for m in metrics_list if m.success]
            if successful_metrics:
                avg_metrics = self._aggregate_metrics(successful_metrics)
                results[query_type] = avg_metrics
                
                logger.info(f"{query_type} average response time: {avg_metrics.response_time:.3f}s")
        
        self.test_results['response_time'] = results
        return results

    async def run_concurrency_tests(self, 
                                   load_config: LoadTestConfig) -> Dict[int, ConcurrencyTestResult]:
        """
        Test system performance under concurrent load.
        
        Args:
            load_config: Configuration for load testing
            
        Returns:
            Dictionary of results by concurrent user count
        """
        logger.info("Starting concurrency tests")
        
        results = {}
        
        for concurrent_users in range(load_config.min_users, 
                                    load_config.max_users + 1, 
                                    load_config.step_size):
            
            logger.info(f"Testing with {concurrent_users} concurrent users")
            
            # Start resource monitoring
            self.resource_monitor.start_monitoring()
            
            # Create concurrent tasks
            tasks = []
            start_time = time.perf_counter()
            
            for user_id in range(concurrent_users):
                query_data = self.test_queries[user_id % len(self.test_queries)]
                task = asyncio.create_task(
                    self._timed_request(query_data, user_id)
                )
                tasks.append(task)
            
            # Execute all tasks concurrently
            try:
                results_list = await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Concurrency test failed: {e}")
                continue
            
            end_time = time.perf_counter()
            resource_data = self.resource_monitor.stop_monitoring()
            
            # Analyze results
            successful_results = [r for r in results_list if isinstance(r, tuple) and r[1] is None]
            failed_results = [r for r in results_list if not isinstance(r, tuple) or r[1] is not None]
            
            if successful_results:
                response_times = [r[0] for r in successful_results]
                
                # Calculate statistics
                avg_response_time = statistics.mean(response_times)
                median_response_time = statistics.median(response_times)
                p95_response_time = np.percentile(response_times, 95)
                p99_response_time = np.percentile(response_times, 99)
                
                total_time = end_time - start_time
                throughput_rps = len(successful_results) / total_time
                error_rate = len(failed_results) / len(results_list)
                
                # Average resource usage
                avg_cpu = statistics.mean([r['cpu_percent'] for r in resource_data]) if resource_data else 0
                avg_memory = statistics.mean([r['memory_used_mb'] for r in resource_data]) if resource_data else 0
                
                concurrency_result = ConcurrencyTestResult(
                    concurrent_users=concurrent_users,
                    total_requests=len(results_list),
                    successful_requests=len(successful_results),
                    failed_requests=len(failed_results),
                    average_response_time=avg_response_time,
                    median_response_time=median_response_time,
                    p95_response_time=p95_response_time,
                    p99_response_time=p99_response_time,
                    throughput_rps=throughput_rps,
                    error_rate=error_rate,
                    resource_usage={
                        'avg_cpu_percent': avg_cpu,
                        'avg_memory_mb': avg_memory
                    }
                )
                
                results[concurrent_users] = concurrency_result
                
                logger.info(f"Concurrent users: {concurrent_users}, "
                          f"Success rate: {(1-error_rate)*100:.1f}%, "
                          f"Avg response time: {avg_response_time:.3f}s, "
                          f"Throughput: {throughput_rps:.2f} RPS")
            
            # Brief pause between tests
            await asyncio.sleep(2)
        
        self.test_results['concurrency'] = results
        return results

    async def run_scalability_tests(self, 
                                  query_loads: List[int],
                                  queries_per_load: int = 10) -> ScalabilityTestResult:
        """
        Test system scalability across different query loads.
        
        Args:
            query_loads: List of batch sizes to test
            queries_per_load: Number of queries per batch size
            
        Returns:
            Scalability test results
        """
        logger.info("Starting scalability tests")
        
        response_times = []
        throughput_metrics = []
        resource_usage_progression = []
        failure_points = []
        
        for load in query_loads:
            logger.info(f"Testing load: {load} queries")
            
            try:
                # Prepare batch of queries
                query_batch = []
                for i in range(load):
                    query_data = self.test_queries[i % len(self.test_queries)]
                    query_batch.append(query_data)
                
                # Start monitoring
                self.resource_monitor.start_monitoring()
                start_time = time.perf_counter()
                
                # Execute batch
                tasks = [self._make_test_request(query) for query in query_batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                end_time = time.perf_counter()
                resource_data = self.resource_monitor.stop_monitoring()
                
                # Analyze results
                total_time = end_time - start_time
                successful_results = [r for r in results if not isinstance(r, Exception)]
                failed_results = [r for r in results if isinstance(r, Exception)]
                
                # Calculate metrics
                avg_response_time = total_time / load
                throughput = len(successful_results) / total_time
                
                response_times.append(avg_response_time)
                throughput_metrics.append(throughput)
                
                # Resource usage
                if resource_data:
                    avg_resources = {
                        'cpu_percent': statistics.mean([r['cpu_percent'] for r in resource_data]),
                        'memory_mb': statistics.mean([r['memory_used_mb'] for r in resource_data]),
                        'network_mb': statistics.mean([r['network_sent_mb'] + r['network_recv_mb'] for r in resource_data])
                    }
                    resource_usage_progression.append(avg_resources)
                else:
                    resource_usage_progression.append({'cpu_percent': 0, 'memory_mb': 0, 'network_mb': 0})
                
                # Check for failure points
                error_rate = len(failed_results) / load
                if error_rate > 0.1:  # More than 10% errors
                    failure_points.append({
                        'load': load,
                        'error_rate': error_rate,
                        'errors': [str(e) for e in failed_results[:5]]  # First 5 errors
                    })
                
                logger.info(f"Load {load}: {avg_response_time:.3f}s avg response, "
                          f"{throughput:.2f} RPS, {error_rate*100:.1f}% errors")
                
            except Exception as e:
                logger.error(f"Scalability test failed at load {load}: {e}")
                failure_points.append({
                    'load': load,
                    'error': str(e),
                    'critical_failure': True
                })
                # Fill with zeros for failed test
                response_times.append(0.0)
                throughput_metrics.append(0.0)
                resource_usage_progression.append({'cpu_percent': 0, 'memory_mb': 0, 'network_mb': 0})
            
            # Brief pause between tests
            await asyncio.sleep(3)
        
        # Calculate scaling efficiency
        scaling_efficiency = self._calculate_scaling_efficiency(query_loads, throughput_metrics)
        
        scalability_result = ScalabilityTestResult(
            query_loads=query_loads,
            response_times=response_times,
            throughput_metrics=throughput_metrics,
            resource_usage_progression=resource_usage_progression,
            failure_points=failure_points,
            scaling_efficiency=scaling_efficiency
        )
        
        self.test_results['scalability'] = scalability_result
        return scalability_result

    async def run_api_usage_analysis(self, 
                                   analysis_duration: int = 300) -> Dict[str, Any]:
        """
        Analyze API usage patterns and costs over time.
        
        Args:
            analysis_duration: Duration in seconds for the analysis
            
        Returns:
            API usage analysis results
        """
        logger.info(f"Starting {analysis_duration}s API usage analysis")
        
        self.api_tracker.reset_tracking()
        start_time = time.time()
        end_time = start_time + analysis_duration
        
        # Run continuous queries for the specified duration
        query_count = 0
        while time.time() < end_time:
            try:
                query_data = self.test_queries[query_count % len(self.test_queries)]
                response = await self._make_test_request(query_data)
                self.api_tracker.track_api_call(response)
                query_count += 1
                
                # Brief pause to simulate realistic usage
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.warning(f"API usage test query failed: {e}")
        
        # Generate usage analysis
        usage_summary = self.api_tracker.get_usage_summary()
        
        analysis_result = {
            'test_duration_seconds': analysis_duration,
            'total_queries_sent': query_count,
            'usage_summary': usage_summary,
            'cost_projections': {
                'cost_per_hour': usage_summary['estimated_cost_usd'] * (3600 / analysis_duration),
                'cost_per_day': usage_summary['estimated_cost_usd'] * (86400 / analysis_duration),
                'cost_per_1000_queries': usage_summary['estimated_cost_usd'] * (1000 / query_count) if query_count > 0 else 0
            },
            'rate_limiting_info': {
                'avg_calls_per_minute': usage_summary['calls_per_minute'],
                'tokens_per_minute': usage_summary['total_tokens'] / (analysis_duration / 60),
                'recommended_batch_size': min(20, max(1, int(60 / usage_summary['calls_per_minute']))) if usage_summary['calls_per_minute'] > 0 else 1
            }
        }
        
        self.test_results['api_usage'] = analysis_result
        logger.info(f"API analysis complete. Estimated cost: ${usage_summary['estimated_cost_usd']:.4f}")
        
        return analysis_result

    async def _make_test_request(self, query_data: Dict[str, Any]) -> Any:
        """Make a test request to the system or OpenAI API."""
        if self.system_endpoint:
            # Make request to RAG system endpoint
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.system_endpoint,
                    json={'query': query_data['query']},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise Exception(f"HTTP {response.status}: {await response.text()}")
        
        elif self.openai_client:
            # Direct OpenAI API call for testing
            messages = [
                {"role": "system", "content": "You are a helpful programming tutor."},
                {"role": "user", "content": query_data['query']}
            ]
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=300,
                temperature=0.1
            )
            
            return response
        
        else:
            raise ValueError("No system endpoint or OpenAI client configured")

    async def _timed_request(self, query_data: Dict[str, Any], user_id: int) -> Tuple[float, Optional[str]]:
        """Make a timed request and return (response_time, error_message)."""
        start_time = time.perf_counter()
        try:
            await self._make_test_request(query_data)
            end_time = time.perf_counter()
            return (end_time - start_time, None)
        except Exception as e:
            end_time = time.perf_counter()
            return (end_time - start_time, str(e))

    def _extract_token_usage(self, response: Any) -> Dict[str, int]:
        """Extract token usage from API response."""
        if hasattr(response, 'usage') and response.usage:
            return {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        return {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

    def _aggregate_metrics(self, metrics_list: List[PerformanceMetrics]) -> PerformanceMetrics:
        """Aggregate multiple performance metrics into averages."""
        return PerformanceMetrics(
            response_time=statistics.mean([m.response_time for m in metrics_list]),
            api_calls_count=sum([m.api_calls_count for m in metrics_list]),
            tokens_used={
                'prompt_tokens': sum([m.tokens_used.get('prompt_tokens', 0) for m in metrics_list]),
                'completion_tokens': sum([m.tokens_used.get('completion_tokens', 0) for m in metrics_list]),
                'total_tokens': sum([m.tokens_used.get('total_tokens', 0) for m in metrics_list])
            },
            memory_usage=statistics.mean([m.memory_usage for m in metrics_list]),
            cpu_usage=statistics.mean([m.cpu_usage for m in metrics_list]),
            success=all([m.success for m in metrics_list])
        )

    def _calculate_scaling_efficiency(self, loads: List[int], throughputs: List[float]) -> float:
        """Calculate scaling efficiency metric."""
        if len(loads) < 2 or len(throughputs) < 2:
            return 0.0
        
        # Linear scaling would maintain constant throughput per unit load
        # Efficiency = actual throughput increase / ideal throughput increase
        ideal_scaling = throughputs[0] * (loads[-1] / loads[0])
        actual_throughput = throughputs[-1]
        
        return actual_throughput / ideal_scaling if ideal_scaling > 0 else 0.0

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        report = {
            'test_summary': {
                'timestamp': datetime.now().isoformat(),
                'tests_conducted': list(self.test_results.keys()),
                'total_test_duration': 'N/A'  # Could track this if needed
            },
            'response_time_analysis': {},
            'concurrency_analysis': {},
            'scalability_analysis': {},
            'api_usage_analysis': {},
            'recommendations': []
        }
        
        # Response time analysis
        if 'response_time' in self.test_results:
            rt_data = self.test_results['response_time']
            report['response_time_analysis'] = {
                'average_response_times': {k: v.response_time for k, v in rt_data.items()},
                'fastest_query_type': min(rt_data.keys(), key=lambda k: rt_data[k].response_time),
                'slowest_query_type': max(rt_data.keys(), key=lambda k: rt_data[k].response_time),
                'meets_target_3s': all(m.response_time < 3.0 for m in rt_data.values())
            }
        
        # Concurrency analysis
        if 'concurrency' in self.test_results:
            conc_data = self.test_results['concurrency']
            report['concurrency_analysis'] = {
                'max_concurrent_users_tested': max(conc_data.keys()) if conc_data else 0,
                'optimal_concurrency_level': self._find_optimal_concurrency(conc_data),
                'error_rate_progression': {k: v.error_rate for k, v in conc_data.items()},
                'throughput_peak': max(v.throughput_rps for v in conc_data.values()) if conc_data else 0
            }
        
        # Generate recommendations
        recommendations = []
        
        if 'response_time' in self.test_results:
            avg_response_time = statistics.mean([m.response_time for m in self.test_results['response_time'].values()])
            if avg_response_time > 3.0:
                recommendations.append("Response times exceed 3s target. Consider optimizing RAG retrieval or using faster models.")
        
        if 'concurrency' in self.test_results:
            max_error_rate = max(v.error_rate for v in self.test_results['concurrency'].values()) if self.test_results['concurrency'] else 0
            if max_error_rate > 0.1:
                recommendations.append("High error rates under load. Implement better error handling and rate limiting.")
        
        if 'api_usage' in self.test_results:
            cost_per_1000 = self.test_results['api_usage']['cost_projections']['cost_per_1000_queries']
            if cost_per_1000 > 1.0:
                recommendations.append("API costs are high. Consider using more efficient models or optimizing prompts.")
        
        report['recommendations'] = recommendations
        
        return report

    def _find_optimal_concurrency(self, concurrency_data: Dict[int, ConcurrencyTestResult]) -> int:
        """Find optimal concurrency level based on throughput and error rate."""
        best_level = 1
        best_score = 0
        
        for level, result in concurrency_data.items():
            # Score based on throughput with penalty for errors
            score = result.throughput_rps * (1 - result.error_rate)
            if score > best_score:
                best_score = score
                best_level = level
        
        return best_level

    def save_results(self, filename_prefix: str = "performance_test"):
        """Save all test results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        results_file = self.output_dir / f"{filename_prefix}_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        # Generate and save report
        report = self.generate_performance_report()
        report_file = self.output_dir / f"{filename_prefix}_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        logger.info(f"Report saved to {report_file}")
        
        return results_file, report_file

    def create_performance_visualizations(self):
        """Create visualizations of performance test results."""
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Agentic Edu-RAG Performance Analysis', fontsize=16)
        
        # Response time comparison
        if 'response_time' in self.test_results:
            rt_data = self.test_results['response_time']
            types = list(rt_data.keys())
            times = [rt_data[t].response_time for t in types]
            
            axes[0, 0].bar(types, times)
            axes[0, 0].set_title('Response Times by Query Type')
            axes[0, 0].set_ylabel('Response Time (seconds)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Concurrency performance
        if 'concurrency' in self.test_results:
            conc_data = self.test_results['concurrency']
            users = list(conc_data.keys())
            throughput = [conc_data[u].throughput_rps for u in users]
            error_rates = [conc_data[u].error_rate * 100 for u in users]
            
            ax1 = axes[0, 1]
            ax2 = ax1.twinx()
            
            ax1.plot(users, throughput, 'b-', label='Throughput (RPS)')
            ax2.plot(users, error_rates, 'r--', label='Error Rate (%)')
            
            ax1.set_xlabel('Concurrent Users')
            ax1.set_ylabel('Throughput (RPS)', color='b')
            ax2.set_ylabel('Error Rate (%)', color='r')
            ax1.set_title('Concurrency Performance')
        
        # Scalability analysis
        if 'scalability' in self.test_results:
            scal_data = self.test_results['scalability']
            loads = scal_data.query_loads
            times = scal_data.response_times
            
            axes[1, 0].plot(loads, times, 'g-o')
            axes[1, 0].set_title('Scalability: Response Time vs Load')
            axes[1, 0].set_xlabel('Query Load')
            axes[1, 0].set_ylabel('Response Time (seconds)')
        
        # Resource usage
        if 'concurrency' in self.test_results:
            conc_data = self.test_results['concurrency']
            users = list(conc_data.keys())
            cpu_usage = [conc_data[u].resource_usage['avg_cpu_percent'] for u in users]
            memory_usage = [conc_data[u].resource_usage['avg_memory_mb'] for u in users]
            
            ax1 = axes[1, 1]
            ax2 = ax1.twinx()
            
            ax1.plot(users, cpu_usage, 'orange', label='CPU %')
            ax2.plot(users, memory_usage, 'purple', label='Memory (MB)')
            
            ax1.set_xlabel('Concurrent Users')
            ax1.set_ylabel('CPU Usage (%)', color='orange')
            ax2.set_ylabel('Memory Usage (MB)', color='purple')
            ax1.set_title('Resource Usage vs Load')
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_file = self.output_dir / f"performance_analysis_{timestamp}.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance visualization saved to {viz_file}")
        return viz_file

# Example usage and testing
async def main():
    """Example usage of PerformanceTester."""
    
    # Initialize tester
    tester = PerformanceTester(
        openai_api_key="your-api-key-here",  # Replace with actual API key
        output_dir=Path("performance_results")
    )
    
    try:
        # Run response time tests
        logger.info("Running response time tests...")
        await tester.run_response_time_tests(num_iterations=5)
        
        # Run concurrency tests
        logger.info("Running concurrency tests...")
        load_config = LoadTestConfig(min_users=1, max_users=10, step_size=2)
        await tester.run_concurrency_tests(load_config)
        
        # Run scalability tests
        logger.info("Running scalability tests...")
        await tester.run_scalability_tests(query_loads=[1, 5, 10, 20])
        
        # Run API usage analysis (shorter duration for example)
        logger.info("Running API usage analysis...")
        await tester.run_api_usage_analysis(analysis_duration=60)
        
        # Generate report and save results
        report = tester.generate_performance_report()
        results_file, report_file = tester.save_results()
        
        # Create visualizations
        viz_file = tester.create_performance_visualizations()
        
        print("Performance testing completed successfully!")
        print(f"Results saved to: {results_file}")
        print(f"Report saved to: {report_file}")
        print(f"Visualizations saved to: {viz_file}")
        
        # Print key metrics
        if 'response_time' in tester.test_results:
            avg_time = statistics.mean([m.response_time for m in tester.test_results['response_time'].values()])
            print(f"Average response time: {avg_time:.3f}s")
        
        if 'api_usage' in tester.test_results:
            cost_per_1000 = tester.test_results['api_usage']['cost_projections']['cost_per_1000_queries']
            print(f"Estimated cost per 1000 queries: ${cost_per_1000:.4f}")
        
    except Exception as e:
        logger.error(f"Performance testing failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
