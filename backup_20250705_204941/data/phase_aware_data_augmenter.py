"""
Phase-Based Data Augmentation Implementation for GridAttention
Integrates data augmentation with attention learning phases
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from core.attention_learning_layer import AttentionPhase, AttentionLearningLayer
from data.data_augmentation import MarketDataAugmenter, FeatureAugmenter, AugmentationConfig
from data.market_data_input import MarketTick
from monitoring.augmentation_monitor import AugmentationMonitor

logger = logging.getLogger(__name__)


@dataclass
class PhaseAugmentationConfig:
    """Configuration for phase-specific augmentation"""
    learning_config: AugmentationConfig
    shadow_config: AugmentationConfig
    active_config: AugmentationConfig
    

class PhaseAwareDataAugmenter:
    """Data augmenter that adapts based on attention phase"""
    
    def __init__(self, phase_config: Optional[PhaseAugmentationConfig] = None):
        # Default configurations for each phase
        self.phase_config = phase_config or self._create_default_configs()
        
        # Create augmenters for each phase
        self.augmenters = {
            AttentionPhase.LEARNING: MarketDataAugmenter(self.phase_config.learning_config),
            AttentionPhase.SHADOW: MarketDataAugmenter(self.phase_config.shadow_config),
            AttentionPhase.ACTIVE: MarketDataAugmenter(self.phase_config.active_config)
        }
        
    def _create_default_configs(self) -> PhaseAugmentationConfig:
        """Create default configurations for each phase"""
        
        # Learning phase: Maximum augmentation
        learning_config = AugmentationConfig(
            noise_level='moderate',
            time_warp_factor=0.1,
            magnitude_warp_factor=0.1,
            bootstrap_ratio=0.5,
            synthetic_ratio=0.3,
            feature_dropout_rate=0.1,
            preserve_correlations=True,
            max_augmentation_factor=3  # 3x data
        )
        
        # Shadow phase: Moderate augmentation
        shadow_config = AugmentationConfig(
            noise_level='conservative',
            time_warp_factor=0.05,
            magnitude_warp_factor=0.05,
            bootstrap_ratio=0.2,
            synthetic_ratio=0.1,
            feature_dropout_rate=0.05,
            preserve_correlations=True,
            max_augmentation_factor=1.5  # 1.5x data
        )
        
        # Active phase: Minimal or no augmentation
        active_config = AugmentationConfig(
            noise_level='conservative',
            time_warp_factor=0.0,
            magnitude_warp_factor=0.0,
            bootstrap_ratio=0.0,
            synthetic_ratio=0.0,
            feature_dropout_rate=0.0,
            preserve_correlations=True,
            max_augmentation_factor=1.0  # No augmentation
        )
        
        return PhaseAugmentationConfig(
            learning_config=learning_config,
            shadow_config=shadow_config,
            active_config=active_config
        )
        
    async def augment_based_on_phase(
        self,
        data: Union[List[MarketTick], Any],
        current_phase: AttentionPhase,
        force_augment: bool = False
    ) -> Tuple[Any, Dict[str, Any]]:
        """Augment data based on current attention phase"""
        
        # Active phase - typically no augmentation unless forced
        if current_phase == AttentionPhase.ACTIVE and not force_augment:
            logger.debug("Active phase - returning original data")
            return data, {"augmented": False, "phase": current_phase.value}
            
        # Get appropriate augmenter
        augmenter = self.augmenters[current_phase]
        
        # Select methods based on phase
        methods = self._get_methods_for_phase(current_phase)
        
        # Perform augmentation
        augmented_data, metadata = augmenter.augment_market_data(data, methods)
        
        # Add phase info to metadata
        augmentation_info = {
            "augmented": True,
            "phase": current_phase.value,
            "methods": methods,
            "augmentation_factor": metadata.augmented_size / metadata.original_size,
            "quality_score": metadata.quality_score
        }
        
        logger.info(f"Phase {current_phase.value}: Augmented {metadata.original_size} → "
                   f"{metadata.augmented_size} samples (factor: {augmentation_info['augmentation_factor']:.2f})")
        
        return augmented_data, augmentation_info
        
    def _get_methods_for_phase(self, phase: AttentionPhase) -> List[str]:
        """Get augmentation methods appropriate for each phase"""
        
        if phase == AttentionPhase.LEARNING:
            # Use all methods for maximum diversity
            return [
                'noise_injection',
                'time_warping',
                'magnitude_warping',
                'bootstrap_sampling',
                'synthetic_patterns',
                'feature_dropout'
            ]
            
        elif phase == AttentionPhase.SHADOW:
            # Use only conservative methods
            return [
                'noise_injection',
                'bootstrap_sampling'
            ]
            
        else:  # ACTIVE
            # Typically no methods, but can be overridden
            return []


class AugmentationScheduler:
    """Schedule augmentation based on learning progress and performance"""
    
    def __init__(self):
        self.schedule_history = []
        self.performance_metrics = {}
        
    def should_augment(
        self,
        phase: AttentionPhase,
        learning_progress: float,
        recent_performance: Dict[str, float]
    ) -> Tuple[bool, float]:
        """Determine if augmentation should be applied and at what level"""
        
        # Always augment in learning phase
        if phase == AttentionPhase.LEARNING:
            return True, self._calculate_learning_factor(learning_progress)
            
        # Conditional augmentation in shadow phase
        elif phase == AttentionPhase.SHADOW:
            # Reduce augmentation as we approach active phase
            if learning_progress > 0.8:
                return True, 0.5  # Light augmentation
            elif learning_progress > 0.6:
                return True, 1.0  # Moderate augmentation
            else:
                return True, 1.5  # Higher augmentation
                
        # Active phase - augment only if performance drops
        else:  # ACTIVE
            if self._detect_performance_degradation(recent_performance):
                logger.warning("Performance degradation detected - enabling augmentation")
                return True, 0.3  # Very light augmentation
            return False, 0.0
            
    def _calculate_learning_factor(self, progress: float) -> float:
        """Calculate augmentation factor based on learning progress"""
        # Start high, reduce as learning progresses
        return max(1.0, 3.0 - (progress * 2.0))
        
    def _detect_performance_degradation(self, metrics: Dict[str, float]) -> bool:
        """Detect if model performance is degrading"""
        # Check win rate drop
        if metrics.get('win_rate', 0.5) < 0.45:
            return True
            
        # Check Sharpe ratio drop
        if metrics.get('sharpe_ratio', 0) < 0.5:
            return True
            
        return False


class IntegratedAugmentationPipeline:
    """Complete pipeline integrating augmentation with attention phases"""
    
    def __init__(
        self,
        attention_layer: AttentionLearningLayer,
        augmenter: Optional[PhaseAwareDataAugmenter] = None,
        scheduler: Optional[AugmentationScheduler] = None
    ):
        self.attention = attention_layer
        self.augmenter = augmenter or PhaseAwareDataAugmenter()
        self.scheduler = scheduler or AugmentationScheduler()
        
    async def process_data_with_augmentation(
        self,
        market_data: List[MarketTick],
        features: Dict[str, float],
        regime: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process data with phase-appropriate augmentation"""
        
        # Get current phase and progress
        current_phase = self.attention.phase
        learning_progress = self.attention.get_learning_progress()
        
        # Get recent performance metrics
        recent_performance = context.get('performance', {})
        
        # Determine if augmentation needed
        should_augment, aug_factor = self.scheduler.should_augment(
            current_phase,
            learning_progress,
            recent_performance
        )
        
        # Apply augmentation if needed
        if should_augment:
            # Augment market data
            augmented_data, aug_info = await self.augmenter.augment_based_on_phase(
                market_data,
                current_phase,
                force_augment=True
            )
            
            # Log augmentation
            logger.info(f"Applied {aug_info['methods']} augmentation "
                       f"(factor: {aug_factor:.2f}) in {current_phase.value} phase")
                       
            # Process augmented data through attention
            results = []
            for tick in augmented_data:
                # Extract features from augmented tick
                aug_features = self._extract_features_from_tick(tick, features)
                
                # Process through attention
                result = await self.attention.process(
                    aug_features,
                    regime,
                    {**context, 'augmented': True}
                )
                results.append(result)
                
            return {
                'processed_count': len(results),
                'augmentation_applied': True,
                'augmentation_info': aug_info,
                'results': results
            }
            
        else:
            # Process without augmentation
            result = await self.attention.process(features, regime, context)
            
            return {
                'processed_count': 1,
                'augmentation_applied': False,
                'results': [result]
            }
            
    def _extract_features_from_tick(
        self,
        tick: MarketTick,
        base_features: Dict[str, float]
    ) -> Dict[str, float]:
        """Extract features from augmented tick"""
        # Simple feature extraction - in practice would use FeatureEngineeringPipeline
        return {
            **base_features,
            'price': tick.price,
            'volume': tick.volume,
            'spread': tick.ask - tick.bid,
            'augmented': True
        }


# Example configuration for main.py integration
def create_phase_aware_augmentation_config() -> Dict[str, Any]:
    """Create configuration for phase-aware augmentation"""
    return {
        'augmentation': {
            'enabled': True,
            'phase_configs': {
                'learning': {
                    'noise_level': 'moderate',
                    'augmentation_factor': 3.0,
                    'methods': ['all']
                },
                'shadow': {
                    'noise_level': 'conservative',
                    'augmentation_factor': 1.5,
                    'methods': ['noise_injection', 'bootstrap']
                },
                'active': {
                    'noise_level': 'none',
                    'augmentation_factor': 1.0,
                    'methods': []
                }
            },
            'scheduler': {
                'performance_threshold': 0.45,
                'sharpe_threshold': 0.5,
                'enable_recovery_augmentation': True
            }
        }
    }


# Integration with main.py
class AugmentationManager:
    """Manager to handle augmentation in main trading loop with monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline = None
        self.stats = {
            'total_augmented': 0,
            'augmentation_by_phase': {
                'learning': 0,
                'shadow': 0,
                'active': 0
            }
        }
        
        # Initialize monitor
        self.monitor = AugmentationMonitor(
            window_size=config.get('monitoring', {}).get('window_size', 1000)
        )
        
        # Start monitoring task if enabled
        if config.get('monitoring', {}).get('enabled', True):
            self.monitoring_task = None
        
    async def initialize(self, attention_layer: AttentionLearningLayer):
        """Initialize augmentation pipeline with monitoring"""
        augmenter = PhaseAwareDataAugmenter()
        scheduler = AugmentationScheduler()
        
        self.pipeline = IntegratedAugmentationPipeline(
            attention_layer,
            augmenter,
            scheduler
        )
        
        # Start monitoring loop
        if self.config.get('monitoring', {}).get('enabled', True):
            interval = self.config.get('monitoring', {}).get('log_interval', 300)
            self.monitoring_task = asyncio.create_task(
                self.monitor.start_monitoring_loop(interval)
            )
        
        logger.info("Augmentation manager initialized with monitoring")
        
    async def process_tick(
        self,
        tick: MarketTick,
        features: Dict[str, float],
        regime: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process single tick with appropriate augmentation and monitoring"""
        
        if not self.pipeline:
            raise RuntimeError("Augmentation pipeline not initialized")
            
        # Get performance metrics from context
        performance_metrics = context.get('performance', {
            'win_rate': 0.5,
            'sharpe_ratio': 0.0
        })
        
        # Process with augmentation
        result = await self.pipeline.process_data_with_augmentation(
            [tick],  # Single tick as list
            features,
            regime,
            context
        )
        
        # Update monitor
        current_phase = self.pipeline.attention.phase.value
        await self.monitor.update(
            result,
            performance_metrics,
            current_phase
        )
        
        # Update statistics
        if result['augmentation_applied']:
            phase = self.pipeline.attention.phase.value
            self.stats['total_augmented'] += result['processed_count']
            self.stats['augmentation_by_phase'][phase] += result['processed_count']
            
        return result
        
    def get_stats(self) -> Dict[str, Any]:
        """Get augmentation statistics"""
        return {
            **self.stats,
            'current_phase': self.pipeline.attention.phase.value if self.pipeline else None,
            'learning_progress': self.pipeline.attention.get_learning_progress() if self.pipeline else 0
        }
        
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get monitoring dashboard data"""
        return self.monitor.get_dashboard_data()
        
    async def stop(self):
        """Stop monitoring tasks"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            await asyncio.gather(self.monitoring_task, return_exceptions=True)


# Example usage
async def example_usage():
    """Example of phase-aware augmentation"""
    
    # Initialize components
    attention = AttentionLearningLayer({
        'min_trades_learning': 1000,
        'min_trades_shadow': 500,
        'min_trades_active': 200
    })
    
    # Create augmentation manager
    aug_manager = AugmentationManager(create_phase_aware_augmentation_config())
    await aug_manager.initialize(attention)
    
    # Simulate trading loop
    for i in range(2000):
        # Create tick
        tick = MarketTick(
            symbol='BTC/USDT',
            price=50000 + np.random.randn() * 100,
            volume=100 + np.random.exponential(50),
            timestamp=time.time() + i,
            bid=49995,
            ask=50005,
            exchange='binance'
        )
        
        # Extract features (simplified)
        features = {
            'volatility': 0.001 + np.random.rand() * 0.001,
            'trend': np.random.randn() * 0.1,
            'volume_ratio': 0.8 + np.random.rand() * 0.4
        }
        
        # Process with augmentation
        result = await aug_manager.process_tick(
            tick,
            features,
            'ranging',
            {'performance': {'win_rate': 0.52, 'sharpe_ratio': 1.2}}
        )
        
        # Log progress periodically
        if i % 100 == 0:
            stats = aug_manager.get_stats()
            print(f"Tick {i}: Phase={stats['current_phase']}, "
                  f"Progress={stats['learning_progress']:.1%}, "
                  f"Augmented={stats['total_augmented']}")


if __name__ == "__main__":
    import time
    asyncio.run(example_usage())