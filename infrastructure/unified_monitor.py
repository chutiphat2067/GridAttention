"""
Unified Monitor - Single monitoring loop for all system checks
Consolidates multiple monitoring loops to reduce CPU overhead
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class UnifiedMonitor:
    """Single monitoring loop for all system checks"""
    
    def __init__(self, system):
        self.system = system
        self.components = system.components
        self.running = False
        
        # Monitoring schedules (intervals in seconds)
        self.schedules = {
            'health_check': {'interval': 30, 'last_run': 0, 'enabled': True},
            'performance': {'interval': 60, 'last_run': 0, 'enabled': True},
            'checkpoint': {'interval': 300, 'last_run': 0, 'enabled': True},
            'overfitting': {'interval': 60, 'last_run': 0, 'enabled': True},
            'augmentation': {'interval': 30, 'last_run': 0, 'enabled': True},
            'resource_check': {'interval': 120, 'last_run': 0, 'enabled': True}
        }
        
        # Performance tracking
        self.last_metrics = {}
        self.error_counts = {task: 0 for task in self.schedules.keys()}
        
    async def start(self):
        """Start unified monitoring loop"""
        self.running = True
        logger.info("🔍 Starting unified monitoring system...")
        
        # Log schedule info
        for task, schedule in self.schedules.items():
            if schedule['enabled']:
                logger.info(f"   {task}: every {schedule['interval']}s")
        
        await self.run()
        
    async def run(self):
        """Single loop for all monitoring tasks"""
        loop_count = 0
        
        while self.running:
            try:
                now = time.time()
                loop_count += 1
                
                # Run scheduled tasks
                for task, schedule in self.schedules.items():
                    if not schedule['enabled']:
                        continue
                        
                    if now - schedule['last_run'] >= schedule['interval']:
                        try:
                            await self._run_task(task)
                            schedule['last_run'] = now
                            self.error_counts[task] = 0  # Reset error count on success
                        except Exception as e:
                            self.error_counts[task] += 1
                            logger.error(f"Monitor task {task} failed (#{self.error_counts[task]}): {e}")
                            
                            # Disable task if too many errors
                            if self.error_counts[task] >= 5:
                                schedule['enabled'] = False
                                logger.warning(f"Disabled {task} monitoring due to repeated failures")
                
                # Log status every 10 minutes
                if loop_count % 600 == 0:  # Every 10 minutes (600 seconds)
                    await self._log_monitoring_status()
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Unified monitor error: {e}")
                await asyncio.sleep(5)  # Longer sleep on error
                
    async def stop(self):
        """Stop monitoring"""
        logger.info("Stopping unified monitoring...")
        self.running = False
        
    async def _run_task(self, task: str):
        """Run specific monitoring task"""
        start_time = time.time()
        
        try:
            if task == 'health_check':
                await self._check_health()
            elif task == 'performance':
                await self._update_performance()
            elif task == 'checkpoint':
                await self._save_checkpoint()
            elif task == 'overfitting':
                await self._check_overfitting()
            elif task == 'augmentation':
                await self._monitor_augmentation()
            elif task == 'resource_check':
                await self._check_resources()
            else:
                logger.warning(f"Unknown monitoring task: {task}")
                
        except Exception as e:
            raise  # Re-raise for error counting
        finally:
            # Track task execution time
            execution_time = time.time() - start_time
            if execution_time > 5:  # Warn if task takes too long
                logger.warning(f"Monitoring task {task} took {execution_time:.2f}s")
    
    async def _check_health(self):
        """Check component health"""
        unhealthy_components = []
        
        for name, component in self.components.items():
            try:
                if hasattr(component, 'health_check'):
                    health = await component.health_check()
                    if not health.get('healthy', True):
                        unhealthy_components.append(name)
                elif hasattr(component, 'is_healthy'):
                    if not await component.is_healthy():
                        unhealthy_components.append(name)
            except Exception as e:
                logger.warning(f"Health check failed for {name}: {e}")
                unhealthy_components.append(name)
        
        # Take action if components unhealthy
        if unhealthy_components:
            logger.warning(f"Unhealthy components: {unhealthy_components}")
            
            # Try to recover unhealthy components
            for component_name in unhealthy_components[:3]:  # Limit recovery attempts
                await self._recover_component(component_name)
    
    async def _update_performance(self):
        """Update performance metrics"""
        try:
            perf_monitor = self.components.get('performance_monitor')
            if perf_monitor and hasattr(perf_monitor, 'get_current_metrics'):
                current_metrics = await perf_monitor.get_current_metrics()
                
                # Check for performance alerts
                if current_metrics:
                    win_rate = current_metrics.get('win_rate', 0)
                    drawdown = current_metrics.get('current_drawdown', 0)
                    
                    if win_rate < 0.4:  # Below 40% win rate
                        logger.warning(f"Low win rate detected: {win_rate:.2%}")
                    
                    if drawdown > 0.1:  # Above 10% drawdown
                        logger.warning(f"High drawdown detected: {drawdown:.2%}")
                        
                self.last_metrics = current_metrics or {}
                
        except Exception as e:
            logger.error(f"Performance update failed: {e}")
    
    async def _save_checkpoint(self):
        """Save system checkpoint"""
        try:
            checkpoint_manager = self.components.get('checkpoint_manager')
            if checkpoint_manager:
                # Save attention layer checkpoint
                if 'attention' in self.components:
                    checkpoint_id = await checkpoint_manager.save_checkpoint(
                        model_name='attention_layer',
                        component=self.components['attention'],
                        performance_metrics=self.last_metrics
                    )
                    logger.info(f"Saved checkpoint: {checkpoint_id}")
                    
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")
    
    async def _check_overfitting(self):
        """Check for overfitting"""
        try:
            overfitting_detector = self.components.get('overfitting_detector')
            if overfitting_detector:
                detection = await overfitting_detector.detect_overfitting()
                
                if detection.get('is_overfitting', False):
                    severity = detection.get('severity', 'MEDIUM')
                    logger.warning(f"Overfitting detected - Severity: {severity}")
                    
                    # Update risk manager
                    risk_manager = self.components.get('risk_manager')
                    if risk_manager:
                        risk_manager.overfitting_detected = True
                        risk_manager.overfitting_severity = severity
                        
        except Exception as e:
            logger.error(f"Overfitting check failed: {e}")
    
    async def _monitor_augmentation(self):
        """Monitor data augmentation"""
        try:
            if hasattr(self.system, 'augmentation_manager') and self.system.augmentation_manager:
                stats = self.system.augmentation_manager.get_stats()
                
                # Log augmentation statistics periodically
                if stats.get('total_augmented', 0) > 0:
                    logger.debug(f"Augmentation stats: {stats.get('total_augmented', 0)} samples")
                    
                # Check for alerts
                dashboard_data = self.system.augmentation_manager.get_monitoring_dashboard()
                alerts = dashboard_data.get('recent_alerts', [])
                
                for alert in alerts[-5:]:  # Check last 5 alerts
                    if alert.get('severity') == 'ERROR':
                        logger.error(f"Augmentation alert: {alert.get('message')}")
                        
        except Exception as e:
            logger.error(f"Augmentation monitoring failed: {e}")
    
    async def _check_resources(self):
        """Check system resources"""
        try:
            import psutil
            
            # Memory check
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                logger.warning(f"High memory usage: {memory.percent:.1f}%")
                
                # Force garbage collection
                if hasattr(self.system, 'optimize_memory'):
                    await self.system.optimize_memory()
            
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
            
            # Disk check
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                logger.error(f"Low disk space: {disk.percent:.1f}% used")
                
        except Exception as e:
            logger.error(f"Resource check failed: {e}")
    
    async def _recover_component(self, component_name: str):
        """Try to recover a failed component"""
        try:
            component = self.components.get(component_name)
            
            if component and hasattr(component, 'recover'):
                await component.recover()
                logger.info(f"Component {component_name} recovered")
            else:
                logger.warning(f"Component {component_name} has no recovery method")
                
        except Exception as e:
            logger.error(f"Failed to recover {component_name}: {e}")
    
    async def _log_monitoring_status(self):
        """Log monitoring system status"""
        enabled_tasks = [task for task, schedule in self.schedules.items() if schedule['enabled']]
        disabled_tasks = [task for task, schedule in self.schedules.items() if not schedule['enabled']]
        
        logger.info(f"📊 Monitoring Status:")
        logger.info(f"   Active tasks: {len(enabled_tasks)} - {', '.join(enabled_tasks)}")
        
        if disabled_tasks:
            logger.warning(f"   Disabled tasks: {len(disabled_tasks)} - {', '.join(disabled_tasks)}")
        
        # Component health summary
        healthy_count = 0
        total_count = len(self.components)
        
        for name, component in self.components.items():
            try:
                if hasattr(component, 'health_check'):
                    health = await component.health_check()
                    if health.get('healthy', True):
                        healthy_count += 1
                else:
                    healthy_count += 1  # Assume healthy if no health check
            except:
                pass  # Skip failed health checks
        
        logger.info(f"   Component health: {healthy_count}/{total_count} healthy")
        
        # Performance summary
        if self.last_metrics:
            pnl = self.last_metrics.get('total_pnl', 0)
            win_rate = self.last_metrics.get('win_rate', 0)
            logger.info(f"   Performance: PnL=${pnl:.2f}, Win Rate={win_rate:.1%}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring system status"""
        enabled_tasks = [task for task, schedule in self.schedules.items() if schedule['enabled']]
        error_summary = {task: count for task, count in self.error_counts.items() if count > 0}
        
        return {
            'running': self.running,
            'enabled_tasks': enabled_tasks,
            'error_counts': error_summary,
            'last_metrics': self.last_metrics
        }


# Integration helper
def replace_monitoring_loops(system):
    """Replace multiple monitoring loops with unified monitor"""
    # Create unified monitor
    unified_monitor = UnifiedMonitor(system)
    
    # Store reference in system
    system.unified_monitor = unified_monitor
    
    return unified_monitor