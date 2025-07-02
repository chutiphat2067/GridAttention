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
from market_data_input import MarketDataInput
from feature_engineering_pipeline import FeatureEngineeringPipeline
from attention_learning_layer import AttentionLearningLayer
from market_regime_detector import MarketRegimeDetector
from grid_strategy_selector import GridStrategySelector
from risk_management_system import RiskManagementSystem
from execution_engine import ExecutionEngine
from performance_monitor import PerformanceMonitor
from feedback_loop import FeedbackLoop
from scaling_monitor import create_scaling_monitor, ScalingMonitor

# Overfitting protection components
from overfitting_detector import OverfittingDetector, OverfittingMonitor, OverfittingRecovery
from checkpoint_manager import CheckpointManager
from data_augmentation import MarketDataAugmenter, FeatureAugmenter, create_augmentation_pipeline
from adaptive_learning_scheduler import AdaptiveLearningScheduler, LearningRateMonitor
from essential_fixes import apply_essential_fixes, KillSwitch

# Phase-aware augmentation components
from phase_aware_data_augmenter import (
    PhaseAwareDataAugmenter, 
    AugmentationManager,
    create_phase_aware_augmentation_config
)
from augmentation_monitor import AugmentationDashboard

# Dashboard integration
from dashboard_integration import integrate_dashboard

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('grid_trading.log'),
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
            overfitting_config_path or 'overfitting_config.yaml'
        )
        
        # System state
        self.components = {}
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
        
    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = Path(path)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {path}")
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    async def initialize(self):
        """Initialize all components with proper error handling"""
        try:
            logger.info("Initializing Grid Trading System with Overfitting Protection...")
            
            # === Core Trading Components ===
            
            # 1. Market Data Input
            self.components['market_data'] = MarketDataInput(
                self.config.get('market_data', {})
            )
            logger.info("✓ Market Data Input initialized")
            
            # 2. Feature Engineering with Augmentation
            self.components['features'] = FeatureEngineeringPipeline(
                self.config.get('features', {})
            )
            
            # Initialize data augmentation
            market_augmenter, feature_augmenter = create_augmentation_pipeline(
                self.overfitting_config.get('data_augmentation', {})
            )
            self.components['market_augmenter'] = market_augmenter
            self.components['feature_augmenter'] = feature_augmenter
            logger.info("✓ Feature Engineering with Augmentation initialized")
            
            # 3. Attention Layer with Regularization
            attention_config = self.config.get('attention', {})
            attention_config.update(self.overfitting_config.get('regularization', {}))
            self.components['attention'] = AttentionLearningLayer(attention_config)
            logger.info("✓ Attention Layer with Enhanced Regularization initialized")
            
            # 4. Market Regime Detector with Ensemble
            regime_config = self.config.get('regime_detector', {})
            regime_config['ensemble'] = self.overfitting_config.get('ensemble', {})
            self.components['regime_detector'] = MarketRegimeDetector(regime_config)
            logger.info("✓ Ensemble Market Regime Detector initialized")
            
            # 5. Grid Strategy Selector
            self.components['strategy_selector'] = GridStrategySelector(
                self.config.get('strategy_selector', {})
            )
            logger.info("✓ Grid Strategy Selector initialized")
            
            # 6. Risk Management System
            risk_config = self.config.get('risk_management', {})
            risk_config['overfitting_aware'] = True
            self.components['risk_manager'] = RiskManagementSystem(risk_config)
            logger.info("✓ Risk Management System initialized")
            
            # 7. Execution Engine
            self.components['execution'] = ExecutionEngine(
                self.config.get('execution', {})
            )
            logger.info("✓ Execution Engine initialized")
            
            # 8. Performance Monitor
            self.components['performance_monitor'] = PerformanceMonitor(
                self.config.get('performance_monitor', {})
            )
            logger.info("✓ Performance Monitor initialized")
            
            # 9. Feedback Loop
            self.components['feedback_loop'] = FeedbackLoop(
                self.config.get('feedback_loop', {})
            )
            logger.info("✓ Feedback Loop initialized")
            
            # === Overfitting Protection Components ===
            
            # 10. Overfitting Detector
            overfitting_detector_config = self.overfitting_config.get('overfitting_detection', {})
            self.overfitting_detector = OverfittingDetector(overfitting_detector_config)
            self.components['overfitting_detector'] = self.overfitting_detector
            logger.info("✓ Overfitting Detector initialized")
            
            # 11. Overfitting Monitor
            self.overfitting_monitor = OverfittingMonitor(self.overfitting_detector)
            
            # Register alert handler
            async def overfitting_alert_handler(alert):
                await self._handle_overfitting_alert(alert)
                
            self.overfitting_monitor.register_alert_handler(overfitting_alert_handler)
            logger.info("✓ Overfitting Monitor initialized")
            
            # 12. Checkpoint Manager
            checkpoint_config = self.overfitting_config.get('checkpointing', {})
            self.checkpoint_manager = CheckpointManager(
                checkpoint_config.get('checkpoint_dir', './checkpoints')
            )
            self.components['checkpoint_manager'] = self.checkpoint_manager
            logger.info("✓ Checkpoint Manager initialized")
            
            # 13. Recovery Manager
            self.recovery_manager = OverfittingRecovery(self.components)
            logger.info("✓ Recovery Manager initialized")
            
            # 14. Adaptive Learning Scheduler
            if hasattr(self.components['attention'], 'optimizer'):
                scheduler_config = self.overfitting_config.get('adaptive_learning', {})
                self.components['learning_scheduler'] = AdaptiveLearningScheduler(
                    self.components['attention'].optimizer,
                    scheduler_config
                )
                logger.info("✓ Adaptive Learning Scheduler initialized")
            
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
            
            # === Dashboard Integration ===
            
            if self.dashboard_enabled:
                self.dashboard_server = integrate_dashboard(self)
                logger.info("✓ Dashboard server initialized")
            
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
            
            # Apply essential fixes after initialization
            logger.info("Applying essential fixes...")
            self.components = apply_essential_fixes(self.components)
            logger.info("✓ Essential fixes applied")
            
            # Add kill switch to risk manager if not present
            if 'risk_manager' in self.components and not hasattr(self.components['risk_manager'], 'kill_switch'):
                self.components['risk_manager'].kill_switch = KillSwitch()
                logger.info("✓ Kill switch added to risk manager")
            
        except Exception as e:
            logger.critical(f"Failed to initialize system: {e}")
            raise
            
    async def start(self):
        """Start the trading system"""
        try:
            # Initialize components
            await self.initialize()
            
            # Setup signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            logger.info("Starting Grid Trading System...")
            self._running = True
            
            # Start core tasks
            self._tasks = [
                asyncio.create_task(self._main_trading_loop()),
                asyncio.create_task(self._monitoring_loop()),
                asyncio.create_task(self._checkpoint_loop()),
                asyncio.create_task(self._health_check_loop()),
                asyncio.create_task(self._overfitting_monitoring_loop()),
                asyncio.create_task(self._augmentation_monitoring_loop())
            ]
            
            # Start component-specific tasks
            for name, component in self.components.items():
                if hasattr(component, 'start'):
                    self._tasks.append(asyncio.create_task(component.start()))
                    
            # Start overfitting monitor
            await self.overfitting_monitor.start_monitoring()
            
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
                    
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
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


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Grid Trading System')
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to main configuration file'
    )
    parser.add_argument(
        '--overfitting-config',
        default='overfitting_config.yaml',
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