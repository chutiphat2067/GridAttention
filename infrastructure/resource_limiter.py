"""
Resource Limiter - Set system resource limits for performance
"""
import resource
import psutil
import os
import gc
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def set_resource_limits(config: Dict[str, Any] = None):
    """Set system resource limits"""
    
    if config is None:
        config = {
            'max_memory_mb': 2048,
            'max_cpu_cores': 2,
            'process_priority': 10,
            'max_open_files': 1024
        }
    
    try:
        # 1. Memory limit
        max_memory = config.get('max_memory_mb', 2048) * 1024 * 1024  # Convert to bytes
        resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
        logger.info(f"Set memory limit: {config.get('max_memory_mb', 2048)}MB")
        
        # 2. Process priority (lower = higher priority)
        process = psutil.Process()
        priority = config.get('process_priority', 10)
        process.nice(priority)
        logger.info(f"Set process priority: {priority}")
        
        # 3. CPU affinity (limit cores)
        max_cores = config.get('max_cpu_cores', 2)
        cpu_count = psutil.cpu_count()
        if max_cores < cpu_count:
            cores = list(range(min(max_cores, cpu_count)))
            process.cpu_affinity(cores)
            logger.info(f"Limited to {len(cores)} CPU cores: {cores}")
        
        # 4. File descriptor limit
        max_files = config.get('max_open_files', 1024)
        resource.setrlimit(resource.RLIMIT_NOFILE, (max_files, max_files))
        logger.info(f"Set max open files: {max_files}")
        
        logger.info("✓ Resource limits applied successfully")
        
    except Exception as e:
        logger.warning(f"Failed to set some resource limits: {e}")

def monitor_resource_usage() -> Dict[str, Any]:
    """Monitor current resource usage"""
    process = psutil.Process()
    
    return {
        'memory_mb': process.memory_info().rss / 1024 / 1024,
        'memory_percent': process.memory_percent(),
        'cpu_percent': process.cpu_percent(),
        'num_threads': process.num_threads(),
        'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0,
        'cpu_affinity': process.cpu_affinity() if hasattr(process, 'cpu_affinity') else [],
        'nice': process.nice()
    }

def optimize_memory():
    """Force memory optimization"""
    stats_before = monitor_resource_usage()
    
    # Force garbage collection
    collected = gc.collect()
    
    # Get new stats
    stats_after = monitor_resource_usage()
    
    memory_freed = stats_before['memory_mb'] - stats_after['memory_mb']
    
    logger.info(f"Memory optimization: freed {memory_freed:.1f}MB, collected {collected} objects")
    
    return {
        'memory_freed_mb': memory_freed,
        'objects_collected': collected,
        'memory_before': stats_before['memory_mb'],
        'memory_after': stats_after['memory_mb']
    }

def check_resource_limits() -> Dict[str, Any]:
    """Check if resource usage is within limits"""
    usage = monitor_resource_usage()
    
    warnings = []
    
    # Check memory (warn at 80% of limit)
    if usage['memory_percent'] > 80:
        warnings.append(f"High memory usage: {usage['memory_percent']:.1f}%")
    
    # Check if too many threads
    if usage['num_threads'] > 50:
        warnings.append(f"High thread count: {usage['num_threads']}")
    
    # Check file descriptors (warn at 80% of limit)
    max_fds = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    if usage['num_fds'] > max_fds * 0.8:
        warnings.append(f"High file descriptor usage: {usage['num_fds']}/{max_fds}")
    
    return {
        'usage': usage,
        'warnings': warnings,
        'healthy': len(warnings) == 0
    }

class ResourceMonitor:
    """Continuous resource monitoring"""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.running = False
        self.alerts = []
        
    async def start_monitoring(self):
        """Start resource monitoring loop"""
        import asyncio
        
        self.running = True
        logger.info("Started resource monitoring")
        
        while self.running:
            try:
                status = check_resource_limits()
                
                if not status['healthy']:
                    for warning in status['warnings']:
                        logger.warning(f"Resource warning: {warning}")
                        self.alerts.append({
                            'timestamp': time.time(),
                            'warning': warning
                        })
                
                # Auto-optimize if memory usage too high
                if status['usage']['memory_percent'] > 90:
                    logger.warning("High memory usage detected, optimizing...")
                    optimize_memory()
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(self.check_interval)
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.running = False
        logger.info("Stopped resource monitoring")
    
    def get_alerts(self) -> list:
        """Get recent alerts"""
        return self.alerts[-10:]  # Return last 10 alerts

# Global resource monitor
resource_monitor = ResourceMonitor()

if __name__ == "__main__":
    print("Testing resource limiter...")
    
    print("Before limits:")
    print(monitor_resource_usage())
    
    set_resource_limits()
    
    print("\nAfter limits:")
    print(monitor_resource_usage())
    
    print("\nResource check:")
    print(check_resource_limits())