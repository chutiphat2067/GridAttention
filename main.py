# main.py
"""
Main orchestration for grid trading system with overfitting protection
Integrates all components including overfitting detection and checkpointing

Author: Grid Trading System
Date: 2024
"""

import asyncio
import logging
import signal
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Core components
from data.market_data_input import MarketDataInput
from data.feature_engineering_pipeline import FeatureEngineeringPipeline
from core.attention_learning_layer import AttentionLearningLayer
from core.market_regime_detector import MarketRegimeDetector
from core.grid_strategy_selector import GridStrategySelector
from core.risk_management_system import RiskManagementSystem
from core.execution_engine import ExecutionEngine
from core.performance_monitor import PerformanceMonitor
from core.feedback_loop import FeedbackLoop
from monitoring.scaling_monitor import create_scaling_monitor, ScalingMonitor

# Overfitting protection components
from core.overfitting_detector import OverfittingDetector, OverfittingMonitor, OverfittingRecovery
from utils.checkpoint_manager import CheckpointManager
from data.data_augmentation import MarketDataAugmenter, FeatureAugmenter, create_augmentation_pipeline
from utils.adaptive_learning_scheduler import AdaptiveLearningScheduler, LearningRateMonitor
from infrastructure.essential_fixes import apply_essential_fixes, KillSwitch
from infrastructure.system_coordinator import SystemCoordinator
from infrastructure.resource_limiter import set_resource_limits, resource_monitor
from monitoring.performance_cache import optimize_memory

# Phase-aware augmentation components
from data.phase_aware_data_augmenter import (
    PhaseAwareDataAugmenter, 
    AugmentationManager,
    create_phase_aware_augmentation_config
)
from monitoring.augmentation_monitor import AugmentationDashboard

# Dashboard integration
from monitoring.dashboard_integration import integrate_dashboard_optimized
from monitoring.dashboard_optimization import optimize_dashboard_performance, OptimizedDashboardCollector
from infrastructure.unified_monitor import replace_monitoring_loops
from infrastructure.memory_manager import patch_system_buffers, memory_manager, BoundedBuffer

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/grid_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GridTradingSystem:
    """Main grid trading system with comprehensive overfitting protection"""
    
    def __init__(self, config_path: str, overfitting_config_path: str = None):
        # Load configurations
        self.config = self._load_config(config_path)
        self.overfitting_config = self._load_config(
            overfitting_config_path or 'config/overfitting_config.yaml'
        )
        
        # System state
        self.components = {}
        self.coordinator = None
        self._running = False
        self._shutdown_requested = False
        self._tasks = []
        self._in_safe_mode = False
        
        # Performance monitoring
        self.start_time = datetime.now()
        self.error_count = 0
        self.last_heartbeat = datetime.now()
        self.scaling_monitor = None
        self.tick_count = 0
        
        # Overfitting protection
        self.overfitting_detector = None
        self.overfitting_monitor = None
        self.checkpoint_manager = None
        self.recovery_manager = None
        self.components = apply_essential_fixes(self.components)
        
        # Phase-aware augmentation
        self.augmentation_manager = None
        self.augmentation_dashboard = None
        self.training_mode = self.config.get('augmentation', {}).get('training_mode_default', True)
        
        # Dashboard integration
        self.dashboard_enabled = True
        self.dashboard_server = None
        
        # Performance optimization
        self.performance_mode = 'normal'  # 'normal', 'minimal', 'high_performance'
        self.resource_limits_applied = False
        
    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = Path(path)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {path}")
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    async def initialize(self):
        """Initialize with SystemCoordinator"""
        try:
            logger.info("Initializing Grid Trading System with SystemCoordinator...")
            
            # Create coordinator with merged configuration
            coordinator_config = {
                'attention_learning_layer': {
                    **self.config.get('attention', {}),
                    **self.overfitting_config.get('regularization', {})
                },
                'market_regime_detector': {
                    **self.config.get('regime_detector', {}),
                    'ensemble': self.overfitting_config.get('ensemble', {})
                },
                'grid_strategy_selector': self.config.get('strategy_selector', {}),
                'risk_management_system': {
                    **self.config.get('risk_management', {}),
                    'overfitting_aware': True
                },
                'execution_engine': self.config.get('execution', {}),
                'performance_monitor': {
                    **self.config.get('performance', {}),
                    'overfitting_config': self.overfitting_config
                },
                'overfitting_detector': self.overfitting_config.get('detector', {}),
                'feedback_loop': self.config.get('feedback', {})
            }
            
            # Initialize coordinator
            self.coordinator = SystemCoordinator(coordinator_config)
            await self.coordinator.initialize_components()
            self.components = self.coordinator.components
            
            logger.info("✓ System coordinator initialized")
            
            # Add legacy components for compatibility
            await self._initialize_legacy_components()
            
            # Start coordinator
            await self.coordinator.start()
            logger.info("✓ System coordinator started")
            
            # === Phase-Aware Augmentation Setup ===
            
            # Get augmentation configuration from main config
            aug_config = self.config.get('augmentation', {})
            
            # Use default config if not provided
            if not aug_config:
                aug_config = create_phase_aware_augmentation_config()['augmentation']
                logger.warning("No augmentation config found, using defaults")
            
            # Merge with overfitting config if provided
            if 'augmentation' in self.overfitting_config:
                aug_config.update(self.overfitting_config['augmentation'])
                
            # Initialize augmentation manager
            self.augmentation_manager = AugmentationManager(aug_config)
            await self.augmentation_manager.initialize(self.components['attention'])
            
            # Initialize augmentation dashboard if enabled
            if aug_config.get('monitoring', {}).get('dashboard_enabled', True):
                self.augmentation_dashboard = AugmentationDashboard(
                    self.augmentation_manager.monitor
                )
                
            logger.info("✓ Phase-Aware Data Augmentation with Monitoring initialized")
            
            # === Memory Management ===
            
            # Patch all buffers with size limits
            patch_system_buffers(self)
            logger.info("✓ Memory management patches applied")
            
            # === Dashboard Integration ===
            
            if self.dashboard_enabled:
                # Use optimized dashboard with reduced update frequency
                update_interval = self.config.get('dashboard', {}).get('update_interval', 10)
                
                # Initialize optimized dashboard collector
                self.optimized_dashboard_collector = OptimizedDashboardCollector(self)
                
                self.dashboard_server = integrate_dashboard_optimized(self, update_interval)
                logger.info(f"✓ Optimized dashboard server initialized (update interval: {update_interval}s)")
                logger.info("✓ Optimized dashboard collector integrated")
            
            # === System Integration ===
            
            # Initialize Performance Monitor with all components
            await self.components['performance_monitor'].initialize(self.components)
            
            # Link components
            self.components['feedback_loop'].set_components(self.components)
            
            # Initialize Scaling Monitor if configured
            if self.config.get('scaling_monitor', {}).get('enabled', False):
                self.scaling_monitor = await create_scaling_monitor(
                    self.config['scaling_monitor']
                )
                logger.info("✓ Scaling Monitor initialized")
                
            # Load checkpoint if exists
            await self._load_latest_checkpoint()
            
            logger.info("✅ All components initialized successfully!")
            
            # === System Optimizations ===
            
            # Enable memory management
            patch_system_buffers(self)
            logger.info("✓ Memory management enabled")
            
            # Optimize dashboard performance
            if self.dashboard_enabled:
                optimize_dashboard_performance(self, cache_ttl=self.config.get('dashboard', {}).get('cache_ttl', 15))
                logger.info("✓ Dashboard optimization enabled")
            
            logger.info("✅ All components initialized with integration!")
            
        except Exception as e:
            logger.critical(f"Failed to initialize system: {e}")
            raise
    
    async def _initialize_legacy_components(self):
        """Initialize legacy components for backward compatibility"""
        try:
            # Map coordinator components to legacy names
            if 'attention_learning_layer' in self.components:
                self.components['attention'] = self.components['attention_learning_layer']
            if 'market_regime_detector' in self.components:
                self.components['regime_detector'] = self.components['market_regime_detector']
            if 'grid_strategy_selector' in self.components:
                self.components['strategy_selector'] = self.components['grid_strategy_selector']
            if 'risk_management_system' in self.components:
                self.components['risk_manager'] = self.components['risk_management_system']
            if 'execution_engine' in self.components:
                self.components['execution'] = self.components['execution_engine']
                
            # Initialize additional legacy components
            # Overfitting Monitor
            self.overfitting_detector = self.components.get('overfitting_detector')
            if self.overfitting_detector:
                self.overfitting_monitor = OverfittingMonitor(self.overfitting_detector)
                
                # Register alert handler
                async def overfitting_alert_handler(alert):
                    await self._handle_overfitting_alert(alert)
                    
                self.overfitting_monitor.register_alert_handler(overfitting_alert_handler)
                logger.info("✓ Overfitting Monitor initialized")
            
            # Checkpoint Manager
            checkpoint_config = self.overfitting_config.get('checkpointing', {})
            self.checkpoint_manager = CheckpointManager(
                checkpoint_config.get('checkpoint_dir', './checkpoints')
            )
            self.components['checkpoint_manager'] = self.checkpoint_manager
            logger.info("✓ Checkpoint Manager initialized")
            
            # Recovery Manager
            self.recovery_manager = OverfittingRecovery(self.components)
            logger.info("✓ Recovery Manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize legacy components: {e}")
    
    def apply_performance_optimizations(self):
        """Apply performance optimizations based on config"""
        try:
            # Check if minimal config is being used
            config_file = getattr(self, '_config_file', '')
            if 'minimal' in config_file:
                self.performance_mode = 'minimal'
                
            # Apply resource limits
            if not self.resource_limits_applied:
                perf_config = self.config.get('performance_optimization', {})
                
                resource_config = {
                    'max_memory_mb': self.config.get('memory', {}).get('max_memory_mb', 2048),
                    'max_cpu_cores': perf_config.get('max_workers', 2),
                    'process_priority': 10
                }
                
                set_resource_limits(resource_config)
                self.resource_limits_applied = True
                logger.info(f"✓ Performance optimizations applied (mode: {self.performance_mode})")
                
            # Start resource monitoring if enabled
            monitoring_config = self.config.get('monitoring', {})
            if monitoring_config.get('resource_monitoring', True):
                import asyncio
                asyncio.create_task(resource_monitor.start_monitoring())
                
        except Exception as e:
            logger.warning(f"Failed to apply performance optimizations: {e}")
    
    async def optimize_memory(self):
        """Perform memory optimization"""
        try:
            result = optimize_memory()
            logger.info(f"Memory optimized: freed {result['memory_freed_mb']:.1f}MB")
            return result
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return None
            
    async def start(self):
        """Start the trading system"""
        try:
            # Apply performance optimizations first
            self.apply_performance_optimizations()
            
            # Initialize components
            await self.initialize()
            
            # Setup signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            logger.info("Starting Grid Trading System...")
            self._running = True
            
            # Optimize memory after initialization
            await self.optimize_memory()
            
            # Check if using unified monitoring
            use_unified = self.config.get('monitoring', {}).get('unified_monitor', {}).get('enabled', True)
            
            if use_unified:
                # Use unified monitoring system
                self.unified_monitor = replace_monitoring_loops(self)
                self._tasks = [
                    asyncio.create_task(self._main_trading_loop()),
                    asyncio.create_task(self.unified_monitor.start())
                ]
                logger.info("✓ Using unified monitoring system")
            else:
                # Use traditional multiple monitoring loops
                self._tasks = [
                    asyncio.create_task(self._main_trading_loop()),
                    asyncio.create_task(self._monitoring_loop()),
                    asyncio.create_task(self._checkpoint_loop()),
                    asyncio.create_task(self._health_check_loop()),
                    asyncio.create_task(self._overfitting_monitoring_loop()),
                    asyncio.create_task(self._augmentation_monitoring_loop())
                ]
                logger.info("✓ Using traditional monitoring loops")
            
            # Start component-specific tasks
            for name, component in self.components.items():
                if hasattr(component, 'start'):
                    self._tasks.append(asyncio.create_task(component.start()))
                    
            # Start overfitting monitor
            await self.overfitting_monitor.start_monitoring()
            
            # Start dashboard server if enabled
            if self.dashboard_enabled and self.dashboard_server:
                await self.dashboard_server.start()
            
            logger.info("🚀 System started successfully!")
            
            # Run until shutdown
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
        except Exception as e:
            logger.critical(f"System error: {e}")
            await self.shutdown()
            raise
            
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        self._shutdown_requested = True
        asyncio.create_task(self.shutdown())
        
    async def _main_trading_loop(self):
        """Main trading logic loop"""
        logger.info("Starting main trading loop...")
        
        while self._running:
            try:
                # Check if in safe mode
                if self._in_safe_mode:
                    await asyncio.sleep(5)
                    continue
                    
                # Get market data
                market_data = await self.components['market_data'].get_latest_data()
                
                if market_data:
                    # Process through pipeline
                    await self._process_market_tick(market_data)
                    
                # Reduced frequency to prevent CPU overload
                await asyncio.sleep(1.0)  # Changed from 0.1 to 1.0 seconds
                
            except Exception as e:
                logger.error(f"Error in main trading loop: {e}")
                self.error_count += 1
                
                # Enter safe mode if too many errors
                if self.error_count > 10:
                    await self._enter_safe_mode("Too many errors")
                    
                await asyncio.sleep(1)
                
    async def _process_market_tick(self, tick):
        """Process single market tick through entire pipeline"""
        try:
            # Increment tick counter
            self.tick_count += 1
            
            # 1. Store original tick
            await self.components['market_data'].process_tick(tick)
            
            # 2. Get recent data for feature extraction
            recent_data = await self.components['market_data'].get_recent_data(tick.symbol, 100)
            
            # 3. Extract features
            features = await self.components['features'].extract_features(recent_data)
            
            if not features:
                return
                
            # 4. Detect regime
            regime, confidence = await self.components['regime_detector'].detect_regime(
                features.features
            )
            
            # 5. Prepare context
            context = {
                'timestamp': tick.timestamp,
                'regime': regime,
                'regime_confidence': confidence,
                'performance': await self._get_recent_performance(),
                'trade_id': f"{tick.symbol}_{tick.timestamp}"
            }
            
            # 6. Apply phase-aware augmentation and process
            if self.training_mode and self.augmentation_manager:
                # Process with augmentation based on current phase
                aug_result = await self.augmentation_manager.process_tick(
                    tick,
                    features.features,
                    regime,
                    context
                )
                
                # Use the processed results
                enhanced_features = aug_result['results'][0] if aug_result['results'] else features.features
                
                # Log augmentation stats periodically
                if self.tick_count % 1000 == 0:
                    aug_stats = self.augmentation_manager.get_stats()
                    logger.info(f"Augmentation stats: {aug_stats}")
            else:
                # Production mode - process without augmentation
                enhanced_features = await self.components['attention'].process_market_data(
                    features.features,
                    recent_data
                )
                
            # 7. Select strategy
            strategy = await self.components['strategy_selector'].select_strategy(
                regime, 
                enhanced_features
            )
            
            # 5. Generate grid levels
            grid_levels = strategy.calculate_grid_levels(tick.price)
            
            # 6. Risk check and execute
            for level in grid_levels:
                order_params = {
                    'symbol': tick.symbol,
                    'side': level.side,
                    'price': level.price,
                    'quantity': level.quantity
                }
                
                # Risk check
                risk_check = await self.components['risk_manager'].check_order_risk(order_params)
                
                if risk_check['approved']:
                    # Execute order
                    await self.components['execution'].execute_order(order_params)
                    
            # 7. Update performance metrics
            await self.components['performance_monitor'].update_tick_metrics(tick)
            
        except Exception as e:
            logger.error(f"Error processing market tick: {e}")
            
    async def _get_recent_performance(self) -> Dict[str, float]:
        """Get recent performance metrics for augmentation decisions"""
        if not self.components.get('performance_monitor'):
            return {'win_rate': 0.5, 'sharpe_ratio': 0.0}
            
        try:
            metrics = await self.components['performance_monitor'].get_current_metrics()
            
            return {
                'win_rate': metrics.get('win_rate', 0.5),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                'total_pnl': metrics.get('total_pnl', 0.0),
                'drawdown': metrics.get('current_drawdown', 0.0)
            }
        except Exception as e:
            logger.warning(f"Could not get performance metrics: {e}")
            return {'win_rate': 0.5, 'sharpe_ratio': 0.0, 'total_pnl': 0.0, 'drawdown': 0.0}
            
    async def _monitoring_loop(self):
        """Performance monitoring loop"""
        logger.info("Starting monitoring loop...")
        
        while self._running:
            try:
                # Generate performance report
                report = await self.components['performance_monitor'].generate_performance_report()
                
                # Check for issues
                if report.get('alerts'):
                    for alert in report['alerts']:
                        logger.warning(f"Performance alert: {alert}")
                        
                # Log summary
                if report.get('summary'):
                    summary = report['summary']
                    logger.info(
                        f"Performance - Trades: {summary.get('total_trades', 0)}, "
                        f"Win Rate: {summary.get('win_rate', 0):.2%}, "
                        f"PnL: ${summary.get('total_pnl', 0):.2f}"
                    )
                    
                await asyncio.sleep(60)  # Every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
                
    async def _checkpoint_loop(self):
        """Checkpoint saving loop"""
        logger.info("Starting checkpoint loop...")
        
        checkpoint_interval = self.overfitting_config.get('checkpointing', {}).get(
            'checkpoint_interval', 3600
        )
        
        while self._running:
            try:
                await asyncio.sleep(checkpoint_interval)
                
                # Save checkpoints for key components
                await self._save_system_checkpoint()
                
            except Exception as e:
                logger.error(f"Error in checkpoint loop: {e}")
                
    async def _health_check_loop(self):
        """System health check loop"""
        logger.info("Starting health check loop...")
        
        while self._running:
            try:
                # Update heartbeat
                self.last_heartbeat = datetime.now()
                
                # Check component health
                unhealthy_components = []
                
                for name, component in self.components.items():
                    if hasattr(component, 'health_check'):
                        health = await component.health_check()
                        if not health.get('healthy', True):
                            unhealthy_components.append(name)
                            
                # Take action if components unhealthy
                if unhealthy_components:
                    logger.warning(f"Unhealthy components: {unhealthy_components}")
                    
                    # Try to recover
                    for component_name in unhealthy_components:
                        await self._recover_component(component_name)
                        
                await asyncio.sleep(30)  # Every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(30)
                
    async def _overfitting_monitoring_loop(self):
        """Dedicated overfitting monitoring loop"""
        logger.info("Starting overfitting monitoring loop...")
        
        while self._running:
            try:
                # Get current overfitting metrics
                detection = await self.overfitting_detector.detect_overfitting()
                
                # Update components with overfitting status
                if detection['is_overfitting']:
                    # Update risk manager
                    self.components['risk_manager'].overfitting_detected = True
                    self.components['risk_manager'].overfitting_severity = detection['severity']
                    
                    # Update feedback loop
                    self.components['feedback_loop'].overfitting_factor = detection['metrics'].get(
                        'performance_gap', 0
                    )
                    
                # Log status
                if detection['is_overfitting']:
                    logger.warning(
                        f"Overfitting detected - Severity: {detection['severity']}, "
                        f"Types: {detection['overfitting_types']}"
                    )
                    
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in overfitting monitoring: {e}")
                await asyncio.sleep(300)
                
    async def _augmentation_monitoring_loop(self):
        """Enhanced monitoring task with alerts"""
        while self._running:
            try:
                if self.augmentation_manager:
                    # Get dashboard data
                    dashboard_data = self.augmentation_manager.get_monitoring_dashboard()
                    
                    # Check for alerts
                    alerts = dashboard_data.get('recent_alerts', [])
                    for alert in alerts:
                        if alert['severity'] == 'ERROR':
                            logger.error(f"Augmentation Alert: {alert['message']}")
                            
                            # Take action based on alert type
                            if alert['type'] == 'EXCESSIVE_AUGMENTATION':
                                # Reduce augmentation factor
                                await self._handle_excessive_augmentation()
                            elif alert['type'] == 'ACTIVE_PHASE_AUGMENTATION':
                                # Log for investigation
                                await self._investigate_active_augmentation(alert)
                                
                    # Update dashboard if enabled
                    if self.augmentation_dashboard:
                        # Could serve this via HTTP endpoint
                        pass
                        
                    # Regular statistics logging
                    stats = self.augmentation_manager.get_stats()
                    phase = self.components['attention'].phase
                    try:
                        progress = self.components['attention'].get_learning_progress()
                    except:
                        progress = 0.0
                    
                    logger.info(f"""
                    Augmentation Statistics:
                    - Phase: {phase} (Progress: {progress:.1%})
                    - Total Augmented: {stats.get('total_augmented', 0):,}
                    - Learning Phase: {stats.get('augmentation_by_phase', {}).get('learning', 0):,}
                    - Shadow Phase: {stats.get('augmentation_by_phase', {}).get('shadow', 0):,}
                    - Active Phase: {stats.get('augmentation_by_phase', {}).get('active', 0):,}
                    """)
                        
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in augmentation monitoring: {e}")
                await asyncio.sleep(60)
                
    async def _handle_excessive_augmentation(self):
        """Handle excessive augmentation alert"""
        logger.warning("Handling excessive augmentation - reducing factors")
        # Could implement automatic adjustment here
        
    async def _investigate_active_augmentation(self, alert: Dict[str, Any]):
        """Investigate why augmentation happened in active phase"""
        performance = alert.get('performance', {})
        logger.info(f"Active phase augmentation investigation:")
        logger.info(f"  Win rate: {performance.get('win_rate', 'N/A')}")
        logger.info(f"  Sharpe ratio: {performance.get('sharpe_ratio', 'N/A')}")
        
        # Could trigger deeper analysis or notifications
                
    async def _handle_overfitting_alert(self, alert: Dict[str, Any]):
        """Handle overfitting alerts"""
        logger.warning(f"Overfitting alert received: {alert['message']}")
        
        severity = alert.get('severity', 'MEDIUM')
        
        # Take action based on severity
        if severity in ['CRITICAL', 'HIGH']:
            # Save checkpoint before recovery
            await self._save_system_checkpoint(reason="Pre-recovery checkpoint")
            
            # Execute recovery
            detection = alert.get('details', {})
            recovery_result = await self.recovery_manager.recover_from_overfitting(
                detection,
                severity
            )
            
            if recovery_result['success']:
                logger.info(f"Recovery successful: {recovery_result['actions_taken']}")
            else:
                logger.error(f"Recovery failed: {recovery_result.get('error')}")
                await self._enter_safe_mode("Recovery failed")
                
    async def _save_system_checkpoint(self, reason: str = "Scheduled"):
        """Save checkpoint for all key components"""
        try:
            logger.info(f"Saving system checkpoint: {reason}")
            
            # Get current performance metrics
            performance_metrics = await self.components['performance_monitor'].get_current_metrics()
            
            # Save attention layer checkpoint
            if 'attention' in self.components:
                checkpoint_id = await self.checkpoint_manager.save_checkpoint(
                    model_name='attention_layer',
                    component=self.components['attention'],
                    performance_metrics=performance_metrics
                )
                logger.info(f"Saved attention layer checkpoint: {checkpoint_id}")
                
            # Save regime detector checkpoint
            if 'regime_detector' in self.components:
                checkpoint_id = await self.checkpoint_manager.save_checkpoint(
                    model_name='regime_detector',
                    component=self.components['regime_detector'],
                    performance_metrics=performance_metrics
                )
                logger.info(f"Saved regime detector checkpoint: {checkpoint_id}")
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            
    async def _load_latest_checkpoint(self):
        """Load latest checkpoint if available"""
        try:
            # Load attention layer
            if 'attention' in self.components:
                success = await self.checkpoint_manager.load_checkpoint(
                    model_name='attention_layer',
                    component=self.components['attention']
                )
                if success:
                    logger.info("Loaded attention layer from checkpoint")
                    
            # Load regime detector
            if 'regime_detector' in self.components:
                success = await self.checkpoint_manager.load_checkpoint(
                    model_name='regime_detector',
                    component=self.components['regime_detector']
                )
                if success:
                    logger.info("Loaded regime detector from checkpoint")
                    
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
            
    async def _enter_safe_mode(self, reason: str):
        """Enter safe mode to protect capital"""
        logger.warning(f"Entering SAFE MODE: {reason}")
        self._in_safe_mode = True
        
        # Reduce all risks
        if 'risk_manager' in self.components:
            self.components['risk_manager'].risk_reduction_mode = True
            
        # Close all positions gradually
        if 'execution' in self.components:
            await self.components['execution'].close_all_positions(gradual=True)
            
        # Notify
        logger.critical(f"SAFE MODE ACTIVATED: {reason}")
        
    async def _recover_component(self, component_name: str):
        """Try to recover a failed component"""
        logger.info(f"Attempting to recover component: {component_name}")
        
        try:
            component = self.components.get(component_name)
            
            if component and hasattr(component, 'recover'):
                await component.recover()
                logger.info(f"Component {component_name} recovered")
            else:
                # Try to reinitialize
                logger.warning(f"Reinitializing component: {component_name}")
                # Component-specific reinitialization logic here
                
        except Exception as e:
            logger.error(f"Failed to recover {component_name}: {e}")
            
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Initiating graceful shutdown...")
        self._running = False
        
        # Save final checkpoint
        await self._save_system_checkpoint(reason="Shutdown checkpoint")
        
        # Stop overfitting monitor
        if self.overfitting_monitor:
            await self.overfitting_monitor.stop_monitoring()
            
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            
        # Stop all components
        for name, component in self.components.items():
            try:
                if hasattr(component, 'stop'):
                    await component.stop()
                    logger.info(f"Stopped {name}")
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
                
        # Stop scaling monitor
        if self.scaling_monitor:
            await self.scaling_monitor.stop()
            
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Generate final report
        try:
            final_report = await self.components['performance_monitor'].generate_performance_report()
            
            # Save to file
            report_path = f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                import json
                json.dump(final_report, f, indent=2, default=str)
                
            logger.info(f"Final report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate final report: {e}")
            
        logger.info("✅ Shutdown complete")
    
    async def stop(self):
        """Alias for shutdown method for compatibility"""
        await self.shutdown()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Grid Trading System')
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to main configuration file'
    )
    parser.add_argument(
        '--overfitting-config',
        default='config/overfitting_config.yaml',
        help='Path to overfitting configuration file'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--training-mode', 
        action='store_true',
        help='Enable training mode with augmentation'
    )
    parser.add_argument(
        '--production',
        action='store_true',
        help='Production mode - no augmentation'
    )
    parser.add_argument(
        '--no-dashboard',
        action='store_true',
        help='Disable dashboard server'
    )
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        # Create system instance
        system = GridTradingSystem(
            config_path=args.config,
            overfitting_config_path=args.overfitting_config
        )
        
        # Set mode
        system.training_mode = not args.production
        if args.training_mode:
            system.training_mode = True
            
        # Set dashboard mode
        if args.no_dashboard:
            system.dashboard_enabled = False
            
        logger.info(f"Starting in {'TRAINING' if system.training_mode else 'PRODUCTION'} mode")
        logger.info(f"Dashboard: {'ENABLED' if system.dashboard_enabled else 'DISABLED'}")
        
        # Run system
        asyncio.run(system.start())
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.critical(f"System failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()