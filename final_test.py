"""
Final System Test - Comprehensive testing of optimized GridAttention System
Tests performance, memory usage, stability, and all optimizations
"""

import asyncio
import psutil
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any
from main import GridTradingSystem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemTester:
    """Comprehensive system testing"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        
    async def run_comprehensive_test(self):
        """Run all system tests"""
        print("🧪 GridAttention Final System Test")
        print("=" * 50)
        
        # Test configurations
        configs = [
            ('config_production.yaml', 'Production Config'),
            ('config_minimal.yaml', 'Minimal Config')
        ]
        
        for config_file, config_name in configs:
            print(f"\n🔬 Testing {config_name}")
            print("-" * 30)
            
            try:
                results = await self._test_configuration(config_file, config_name)
                self.test_results[config_name] = results
                self._print_test_results(config_name, results)
                
            except Exception as e:
                print(f"❌ {config_name} test failed: {e}")
                self.test_results[config_name] = {'error': str(e)}
        
        # Generate final report
        self._generate_final_report()
    
    async def _test_configuration(self, config_file: str, config_name: str) -> Dict[str, Any]:
        """Test specific configuration"""
        results = {
            'config_file': config_file,
            'config_name': config_name,
            'start_time': datetime.now().isoformat()
        }
        
        # Initialize metrics tracking
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        initial_cpu = process.cpu_percent()
        
        try:
            # Create and initialize system
            print(f"   Initializing system...")
            system = GridTradingSystem(config_file)
            
            initialization_start = time.time()
            await system.initialize()
            initialization_time = time.time() - initialization_start
            
            print(f"   ✓ Initialized in {initialization_time:.2f}s")
            
            # Test memory management
            memory_stats = await self._test_memory_management(system)
            results['memory_management'] = memory_stats
            
            # Test unified monitoring
            monitoring_stats = await self._test_unified_monitoring(system)
            results['unified_monitoring'] = monitoring_stats
            
            # Test dashboard optimization
            dashboard_stats = await self._test_dashboard_optimization(system)
            results['dashboard_optimization'] = dashboard_stats
            
            # Performance test - run for 2 minutes
            print(f"   Running performance test (2 minutes)...")
            perf_stats = await self._test_performance(system, duration=120)
            results['performance'] = perf_stats
            
            # Final metrics
            final_memory = process.memory_info().rss / 1024 / 1024
            avg_cpu = process.cpu_percent(interval=1)
            
            results.update({
                'initialization_time': initialization_time,
                'memory_increase': final_memory - initial_memory,
                'avg_cpu_usage': avg_cpu,
                'total_test_time': time.time() - initialization_start,
                'success': True
            })
            
            # Cleanup
            if hasattr(system, 'unified_monitor'):
                await system.unified_monitor.stop()
            
            await self._cleanup_system(system)
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
            raise
        
        return results
    
    async def _test_memory_management(self, system) -> Dict[str, Any]:
        """Test memory management features"""
        print("     Testing memory management...")
        
        # Check if memory manager is available
        if not hasattr(system, 'memory_manager'):
            return {'enabled': False, 'error': 'Memory manager not found'}
        
        memory_manager = system.memory_manager
        
        # Get initial stats
        initial_usage = memory_manager.get_memory_usage()
        buffer_stats = memory_manager.get_buffer_stats()
        
        # Test cleanup
        cleanup_result = memory_manager.cleanup_memory(force=True)
        
        # Final stats
        final_usage = memory_manager.get_memory_usage()
        
        return {
            'enabled': True,
            'initial_memory_mb': initial_usage['rss_mb'],
            'final_memory_mb': final_usage['rss_mb'],
            'memory_freed_mb': cleanup_result.get('freed_mb', 0),
            'total_buffers': buffer_stats['summary']['total_buffers'],
            'buffer_items': buffer_stats['summary']['total_items']
        }
    
    async def _test_unified_monitoring(self, system) -> Dict[str, Any]:
        """Test unified monitoring system"""
        print("     Testing unified monitoring...")
        
        if not hasattr(system, 'unified_monitor'):
            return {'enabled': False, 'note': 'Using traditional monitoring'}
        
        monitor = system.unified_monitor
        
        # Start monitoring briefly
        monitor_task = asyncio.create_task(monitor.start())
        await asyncio.sleep(10)  # Run for 10 seconds
        
        # Stop monitoring
        await monitor.stop()
        monitor_task.cancel()
        
        # Get statistics
        status = monitor.get_status()
        
        return {
            'enabled': True,
            'active_tasks': len(status['enabled_tasks']),
            'error_counts': status['error_counts'],
            'last_metrics': bool(status['last_metrics'])
        }
    
    async def _test_dashboard_optimization(self, system) -> Dict[str, Any]:
        """Test dashboard optimization"""
        print("     Testing dashboard optimization...")
        
        if not hasattr(system, 'optimized_dashboard_collector'):
            return {'enabled': False, 'note': 'Dashboard optimization not found'}
        
        collector = system.optimized_dashboard_collector
        
        # Test data collection performance
        start_time = time.time()
        data1 = await collector.collect_all_data()  # First call - cache miss
        first_call_time = time.time() - start_time
        
        start_time = time.time()
        data2 = await collector.collect_all_data()  # Second call - cache hit
        second_call_time = time.time() - start_time
        
        # Get cache statistics
        cache_stats = collector.get_cache_stats()
        
        return {
            'enabled': True,
            'first_call_time': first_call_time,
            'second_call_time': second_call_time,
            'speedup_ratio': first_call_time / max(second_call_time, 0.001),
            'cache_hit_rate': cache_stats['hit_rate'],
            'cache_items': cache_stats['cached_items']
        }
    
    async def _test_performance(self, system, duration: int = 120) -> Dict[str, Any]:
        """Test system performance over time"""
        process = psutil.Process()
        
        # Performance tracking
        memory_samples = []
        cpu_samples = []
        error_count = 0
        
        start_time = time.time()
        end_time = start_time + duration
        
        # Sample every 10 seconds
        sample_interval = 10
        next_sample = start_time + sample_interval
        
        while time.time() < end_time:
            current_time = time.time()
            
            # Take memory and CPU samples
            if current_time >= next_sample:
                try:
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    cpu_percent = process.cpu_percent()
                    
                    memory_samples.append(memory_mb)
                    cpu_samples.append(cpu_percent)
                    
                    next_sample += sample_interval
                    
                except Exception:
                    error_count += 1
            
            await asyncio.sleep(1)
        
        # Calculate statistics
        if memory_samples and cpu_samples:
            avg_memory = sum(memory_samples) / len(memory_samples)
            max_memory = max(memory_samples)
            min_memory = min(memory_samples)
            
            avg_cpu = sum(cpu_samples) / len(cpu_samples)
            max_cpu = max(cpu_samples)
            
            # Memory trend (increasing/stable/decreasing)
            if len(memory_samples) > 2:
                memory_trend = memory_samples[-1] - memory_samples[0]
            else:
                memory_trend = 0
                
        else:
            avg_memory = max_memory = min_memory = 0
            avg_cpu = max_cpu = 0
            memory_trend = 0
        
        return {
            'duration_seconds': duration,
            'samples_taken': len(memory_samples),
            'avg_memory_mb': avg_memory,
            'max_memory_mb': max_memory,
            'min_memory_mb': min_memory,
            'memory_trend_mb': memory_trend,
            'avg_cpu_percent': avg_cpu,
            'max_cpu_percent': max_cpu,
            'errors': error_count
        }
    
    async def _cleanup_system(self, system):
        """Clean up system resources"""
        try:
            if hasattr(system, 'stop'):
                await system.stop()
            elif hasattr(system, 'shutdown'):
                await system.shutdown()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
    
    def _print_test_results(self, config_name: str, results: Dict[str, Any]):
        """Print test results in readable format"""
        if not results.get('success', False):
            print(f"   ❌ {config_name} failed: {results.get('error', 'Unknown error')}")
            return
        
        print(f"   ✅ {config_name} Results:")
        print(f"      Initialization: {results.get('initialization_time', 0):.2f}s")
        print(f"      Memory increase: {results.get('memory_increase', 0):.1f}MB")
        print(f"      CPU usage: {results.get('avg_cpu_usage', 0):.1f}%")
        
        # Memory management
        memory = results.get('memory_management', {})
        if memory.get('enabled'):
            print(f"      Memory freed: {memory.get('memory_freed_mb', 0):.1f}MB")
            print(f"      Active buffers: {memory.get('total_buffers', 0)}")
        
        # Unified monitoring
        monitoring = results.get('unified_monitoring', {})
        if monitoring.get('enabled'):
            print(f"      Monitor tasks: {monitoring.get('active_tasks', 0)}")
        
        # Dashboard optimization
        dashboard = results.get('dashboard_optimization', {})
        if dashboard.get('enabled'):
            speedup = dashboard.get('speedup_ratio', 1)
            print(f"      Dashboard speedup: {speedup:.1f}x")
            print(f"      Cache hit rate: {dashboard.get('cache_hit_rate', 0):.1%}")
        
        # Performance
        perf = results.get('performance', {})
        print(f"      Avg memory: {perf.get('avg_memory_mb', 0):.1f}MB")
        print(f"      Memory trend: {perf.get('memory_trend_mb', 0):+.1f}MB")
        print(f"      Avg CPU: {perf.get('avg_cpu_percent', 0):.1f}%")
    
    def _generate_final_report(self):
        """Generate comprehensive final report"""
        print(f"\n📊 FINAL TEST REPORT")
        print("=" * 50)
        
        successful_tests = [name for name, results in self.test_results.items() 
                          if results.get('success', False)]
        failed_tests = [name for name, results in self.test_results.items() 
                       if not results.get('success', False)]
        
        print(f"Total tests: {len(self.test_results)}")
        print(f"Successful: {len(successful_tests)}")
        print(f"Failed: {len(failed_tests)}")
        
        if failed_tests:
            print(f"\n❌ Failed tests: {', '.join(failed_tests)}")
        
        if successful_tests:
            print(f"\n✅ Successful tests: {', '.join(successful_tests)}")
            
            # Performance summary
            print(f"\n📈 Performance Summary:")
            for test_name in successful_tests:
                results = self.test_results[test_name]
                perf = results.get('performance', {})
                memory = results.get('memory_management', {})
                
                print(f"\n{test_name}:")
                print(f"  Memory usage: {perf.get('avg_memory_mb', 0):.1f}MB avg, {perf.get('memory_trend_mb', 0):+.1f}MB trend")
                print(f"  CPU usage: {perf.get('avg_cpu_percent', 0):.1f}% avg")
                if memory.get('enabled'):
                    print(f"  Memory management: {memory.get('total_buffers', 0)} buffers, {memory.get('memory_freed_mb', 0):.1f}MB freed")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if len(successful_tests) == len(self.test_results):
            print("✅ ALL TESTS PASSED - System ready for production!")
        elif len(successful_tests) >= len(self.test_results) * 0.8:
            print("⚠️ MOST TESTS PASSED - System mostly ready, minor issues remain")
        else:
            print("❌ MULTIPLE FAILURES - System needs significant work")
        
        # Save detailed report
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        print(f"Total test time: {time.time() - self.start_time:.1f} seconds")


async def main():
    """Main test execution"""
    tester = SystemTester()
    await tester.run_comprehensive_test()


if __name__ == "__main__":
    asyncio.run(main())