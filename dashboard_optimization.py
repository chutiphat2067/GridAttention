"""
Dashboard Optimization - Async batch queries and caching
Improves dashboard responsiveness with parallel data collection
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dashboard_integration import DashboardDataCollector
from performance_cache import metrics_cache

logger = logging.getLogger(__name__)


class OptimizedDashboardCollector(DashboardDataCollector):
    """Enhanced dashboard collector with async batching and caching"""
    
    def __init__(self, grid_system, cache_ttl: int = 15):
        super().__init__(grid_system)
        self.cache_ttl = cache_ttl
        self.cache = {}
        self.cache_timestamps = {}
        self.collection_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_collection_time': 0,
            'last_collection_time': 0
        }
        
    async def collect_all_data(self) -> Dict[str, Any]:
        """Collect data with async batching and intelligent caching"""
        start_time = time.time()
        self.collection_stats['total_requests'] += 1
        
        try:
            # Check cache first
            cache_key = 'dashboard_data'
            if self._is_cache_valid(cache_key):
                self.collection_stats['cache_hits'] += 1
                logger.debug("Dashboard data served from cache")
                return self.cache[cache_key]
            
            self.collection_stats['cache_misses'] += 1
            
            # Run critical queries in parallel (high priority)
            critical_tasks = [
                self._get_system_status_async(),
                self._get_critical_metrics_async(),
                self._get_system_health_async()
            ]
            
            # Run secondary queries in parallel (lower priority)
            secondary_tasks = [
                self._get_learning_status_async(),
                self._get_trading_activity_async(),
                self._get_market_analysis_async()
            ]
            
            # Execute critical tasks first
            critical_results = await asyncio.gather(*critical_tasks, return_exceptions=True)
            
            # Execute secondary tasks with timeout
            try:
                secondary_results = await asyncio.wait_for(
                    asyncio.gather(*secondary_tasks, return_exceptions=True),
                    timeout=5.0  # 5 second timeout for secondary data
                )
            except asyncio.TimeoutError:
                logger.warning("Secondary dashboard queries timed out")
                secondary_results = [{}, {}, {}]  # Empty results
            
            # Construct response with error handling
            data = {
                'timestamp': datetime.now().isoformat(),
                'system_status': self._safe_result(critical_results[0], 'UNKNOWN'),
                'critical_metrics': self._safe_result(critical_results[1], {}),
                'system_health': self._safe_result(critical_results[2], {}),
                'learning_status': self._safe_result(secondary_results[0], {}),
                'trading_activity': self._safe_result(secondary_results[1], {}),
                'market_analysis': self._safe_result(secondary_results[2], {}),
                'logs': await self._get_recent_logs_async(limit=50),  # Reduced limit
                'collection_stats': self.collection_stats.copy()
            }
            
            # Cache the result
            self.cache[cache_key] = data
            self.cache_timestamps[cache_key] = time.time()
            
            # Calculate performance stats
            collection_time = time.time() - start_time
            self.collection_stats['last_collection_time'] = collection_time
            
            # Update average (simple moving average)
            if self.collection_stats['avg_collection_time'] == 0:
                self.collection_stats['avg_collection_time'] = collection_time
            else:
                self.collection_stats['avg_collection_time'] = (
                    self.collection_stats['avg_collection_time'] * 0.9 + 
                    collection_time * 0.1
                )
            
            logger.debug(f"Dashboard data collected in {collection_time:.3f}s")
            return data
            
        except Exception as e:
            logger.error(f"Dashboard collection error: {e}")
            return self._get_error_response(str(e))
    
    def _safe_result(self, result: Any, default: Any) -> Any:
        """Safely extract result or return default"""
        if isinstance(result, Exception):
            logger.warning(f"Dashboard query failed: {result}")
            return default
        return result if result is not None else default
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        cache_age = time.time() - self.cache_timestamps.get(cache_key, 0)
        return cache_age < self.cache_ttl
    
    async def _get_system_status_async(self) -> str:
        """Async version of system status check"""
        try:
            return await super()._get_system_status()
        except Exception as e:
            logger.error(f"System status check failed: {e}")
            return 'ERROR'
    
    async def _get_critical_metrics_async(self) -> Dict[str, Any]:
        """Async version with timeout and error handling"""
        try:
            # Use cached metrics if available
            cache_key = 'critical_metrics'
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]
            
            metrics = await asyncio.wait_for(
                super()._get_critical_metrics(),
                timeout=3.0  # 3 second timeout
            )
            
            # Cache the result
            self.cache[cache_key] = metrics
            self.cache_timestamps[cache_key] = time.time()
            
            return metrics
            
        except asyncio.TimeoutError:
            logger.warning("Critical metrics query timed out")
            return self._get_default_metrics()
        except Exception as e:
            logger.error(f"Critical metrics error: {e}")
            return self._get_default_metrics()
    
    async def _get_system_health_async(self) -> Dict[str, Any]:
        """Async system health with reduced queries"""
        try:
            # Simplified health check for speed
            import psutil
            
            # Only essential metrics
            cpu_percent = psutil.cpu_percent(interval=0)  # Non-blocking
            memory = psutil.virtual_memory()
            
            # Quick network check (if available)
            net_latency = 0
            market_data = self.system.components.get('market_data')
            if market_data and hasattr(market_data, 'last_latency'):
                net_latency = getattr(market_data, 'last_latency', 0)
            
            return {
                'cpuUsage': cpu_percent,
                'memUsage': memory.percent,
                'netLatency': net_latency,
                'apiStatus': 'OK' if market_data else 'ERROR',
                'uptime': int(time.time() - self.start_time),
                'errorCount': getattr(self.system, 'error_count', 0)
            }
            
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    async def _get_learning_status_async(self) -> Dict[str, Any]:
        """Async learning status with caching"""
        try:
            cache_key = 'learning_status'
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]
            
            status = await super()._get_learning_status()
            
            # Cache with longer TTL for learning status (changes slowly)
            self.cache[cache_key] = status
            self.cache_timestamps[cache_key] = time.time()
            
            return status
            
        except Exception as e:
            logger.error(f"Learning status error: {e}")
            return {'phase': 'error', 'observations': 0}
    
    async def _get_trading_activity_async(self) -> Dict[str, Any]:
        """Async trading activity with pagination"""
        try:
            perf_monitor = self.system.components.get('performance_monitor')
            
            # Quick metrics only
            activity = {
                'tradesToday': 0,
                'volumeToday': 0,
                'openPositions': []
            }
            
            if perf_monitor and hasattr(perf_monitor, 'get_quick_metrics'):
                # Use quick metrics if available
                quick_metrics = await perf_monitor.get_quick_metrics()
                activity.update(quick_metrics)
            elif perf_monitor:
                # Fallback to regular metrics with timeout
                try:
                    metrics = await asyncio.wait_for(
                        perf_monitor.get_daily_metrics(),
                        timeout=2.0
                    )
                    activity.update(metrics)
                except asyncio.TimeoutError:
                    logger.warning("Trading activity query timed out")
            
            return activity
            
        except Exception as e:
            logger.error(f"Trading activity error: {e}")
            return {'tradesToday': 0, 'volumeToday': 0}
    
    async def _get_market_analysis_async(self) -> Dict[str, Any]:
        """Async market analysis with caching"""
        try:
            cache_key = 'market_analysis'
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]
            
            analysis = await super()._get_market_analysis()
            
            # Cache market analysis (changes moderately)
            self.cache[cache_key] = analysis
            self.cache_timestamps[cache_key] = time.time()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            return {'regime': 'unknown', 'confidence': 0}
    
    async def _get_recent_logs_async(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent logs with limit for performance"""
        try:
            # Simplified log retrieval
            if hasattr(self.system, 'log_buffer'):
                # Get from bounded log buffer if available
                return self.system.log_buffer.get_recent(limit)
            else:
                # Return empty list for performance
                return []
                
        except Exception as e:
            logger.error(f"Log retrieval error: {e}")
            return []
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """Default metrics when queries fail"""
        return {
            'pnl': 0,
            'winRate': 0,
            'drawdown': 0,
            'overfittingScore': 0,
            'openPositions': 0,
            'latency': 0
        }
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info("Dashboard cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        hit_rate = 0
        if self.collection_stats['total_requests'] > 0:
            hit_rate = self.collection_stats['cache_hits'] / self.collection_stats['total_requests']
        
        return {
            'hit_rate': hit_rate,
            'total_requests': self.collection_stats['total_requests'],
            'cache_hits': self.collection_stats['cache_hits'],
            'cache_misses': self.collection_stats['cache_misses'],
            'avg_collection_time': self.collection_stats['avg_collection_time'],
            'cached_items': len(self.cache),
            'cache_ttl': self.cache_ttl
        }


class PaginatedDataCollector:
    """Collector for paginated data requests"""
    
    def __init__(self, system):
        self.system = system
    
    async def get_performance_history(self, page: int = 1, page_size: int = 100) -> Dict[str, Any]:
        """Get paginated performance history"""
        try:
            perf_monitor = self.system.components.get('performance_monitor')
            if not perf_monitor:
                return {'data': [], 'total': 0, 'page': page}
            
            # Get total count
            if hasattr(perf_monitor, 'performance_history'):
                history = perf_monitor.performance_history
                total = len(history)
                
                # Calculate pagination
                start_idx = (page - 1) * page_size
                end_idx = start_idx + page_size
                
                # Get page data
                if hasattr(history, 'get_recent'):
                    # BoundedBuffer
                    all_data = history.get_all()
                    page_data = all_data[start_idx:end_idx]
                else:
                    # Regular list
                    page_data = history[start_idx:end_idx]
                
                return {
                    'data': page_data,
                    'total': total,
                    'page': page,
                    'page_size': page_size,
                    'total_pages': (total + page_size - 1) // page_size
                }
            else:
                return {'data': [], 'total': 0, 'page': page}
                
        except Exception as e:
            logger.error(f"Performance history pagination error: {e}")
            return {'data': [], 'total': 0, 'page': page, 'error': str(e)}
    
    async def get_trade_history(self, page: int = 1, page_size: int = 50) -> Dict[str, Any]:
        """Get paginated trade history"""
        try:
            execution = self.system.components.get('execution')
            if not execution or not hasattr(execution, 'trade_history'):
                return {'data': [], 'total': 0, 'page': page}
            
            history = execution.trade_history
            total = len(history)
            
            # Calculate pagination
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            
            # Get page data
            if hasattr(history, 'get_recent'):
                all_data = history.get_all()
                page_data = all_data[start_idx:end_idx]
            else:
                page_data = history[start_idx:end_idx]
            
            return {
                'data': page_data,
                'total': total,
                'page': page,
                'page_size': page_size,
                'total_pages': (total + page_size - 1) // page_size
            }
            
        except Exception as e:
            logger.error(f"Trade history pagination error: {e}")
            return {'data': [], 'total': 0, 'page': page, 'error': str(e)}


# Integration function
def optimize_dashboard_performance(system, cache_ttl: int = 15):
    """Replace dashboard collector with optimized version"""
    
    # Create optimized collector
    optimized_collector = OptimizedDashboardCollector(system, cache_ttl)
    
    # Create paginated collector
    paginated_collector = PaginatedDataCollector(system)
    
    # Store in system
    system.optimized_dashboard_collector = optimized_collector
    system.paginated_data_collector = paginated_collector
    
    logger.info(f"✓ Dashboard optimization enabled (cache TTL: {cache_ttl}s)")
    
    return optimized_collector, paginated_collector