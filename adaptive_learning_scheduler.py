# adaptive_learning_scheduler.py
"""
Adaptive learning rate scheduling system for overfitting prevention
Dynamically adjusts learning rates based on validation performance

Author: Grid Trading System
Date: 2024
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    ExponentialLR, 
    ReduceLROnPlateau,
    CyclicLR,
    OneCycleLR,
    CosineAnnealingWarmRestarts
)

logger = logging.getLogger(__name__)

# Constants
MIN_LEARNING_RATE = 1e-6
MAX_LEARNING_RATE = 1e-2
DEFAULT_PATIENCE = 20
DEFAULT_REDUCTION_FACTOR = 0.5


@dataclass
class LearningRateMetrics:
    """Metrics for learning rate adjustment"""
    current_lr: float
    best_performance: float
    patience_counter: int
    total_adjustments: int
    improvement_rate: float
    timestamp: float = field(default_factory=time.time)


@dataclass 
class SchedulerConfig:
    """Configuration for learning rate scheduler"""
    scheduler_type: str = 'adaptive_plateau'  # adaptive_plateau, cosine, exponential, cyclic, one_cycle
    initial_lr: float = 0.001
    min_lr: float = MIN_LEARNING_RATE
    max_lr: float = MAX_LEARNING_RATE
    patience: int = DEFAULT_PATIENCE
    reduction_factor: float = DEFAULT_REDUCTION_FACTOR
    warmup_epochs: int = 10
    cycle_length: int = 50
    adaptive_threshold: float = 0.01  # Min improvement to reset patience


class AdaptiveLearningScheduler:
    """Advanced learning rate scheduling with multiple strategies"""
    
    def __init__(self, optimizer: optim.Optimizer, config: Optional[SchedulerConfig] = None):
        self.optimizer = optimizer
        self.config = config or SchedulerConfig()
        
        # Metrics tracking
        self.metrics = LearningRateMetrics(
            current_lr=self.config.initial_lr,
            best_performance=-float('inf'),
            patience_counter=0,
            total_adjustments=0,
            improvement_rate=0.0
        )
        
        # Performance history
        self.performance_history = deque(maxlen=100)
        self.lr_history = deque(maxlen=1000)
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        self.warmup_steps = 0
        self.is_warmup_complete = False
        
        # Callbacks
        self.callbacks = []
        
    def _create_scheduler(self):
        """Create the appropriate scheduler based on config"""
        scheduler_type = self.config.scheduler_type.lower()
        
        if scheduler_type == 'adaptive_plateau':
            return AdaptivePlateauScheduler(
                self.optimizer,
                patience=self.config.patience,
                factor=self.config.reduction_factor,
                min_lr=self.config.min_lr,
                threshold=self.config.adaptive_threshold
            )
            
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.cycle_length,
                eta_min=self.config.min_lr
            )
            
        elif scheduler_type == 'cosine_warm_restarts':
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.cycle_length,
                T_mult=2,
                eta_min=self.config.min_lr
            )
            
        elif scheduler_type == 'exponential':
            gamma = np.exp(np.log(self.config.min_lr / self.config.initial_lr) / 1000)
            return ExponentialLR(self.optimizer, gamma=gamma)
            
        elif scheduler_type == 'cyclic':
            return CyclicLR(
                self.optimizer,
                base_lr=self.config.min_lr,
                max_lr=self.config.max_lr,
                step_size_up=self.config.cycle_length // 2,
                mode='triangular2'
            )
            
        elif scheduler_type == 'one_cycle':
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.max_lr,
                total_steps=1000,  # Will be adjusted
                pct_start=0.3,
                anneal_strategy='cos',
                final_div_factor=1000
            )
            
        else:
            # Default to ReduceLROnPlateau
            return ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=self.config.patience,
                factor=self.config.reduction_factor,
                min_lr=self.config.min_lr
            )
            
    async def step(self, current_performance: float, epoch: Optional[int] = None) -> float:
        """Step the scheduler based on performance"""
        
        # Track performance
        self.performance_history.append(current_performance)
        
        # Handle warmup
        if not self.is_warmup_complete:
            new_lr = await self._warmup_step()
            if self.warmup_steps >= self.config.warmup_epochs:
                self.is_warmup_complete = True
                logger.info(f"Warmup complete after {self.warmup_steps} steps")
        else:
            # Regular scheduling
            new_lr = await self._regular_step(current_performance, epoch)
            
        # Update metrics
        self._update_metrics(current_performance, new_lr)
        
        # Store history
        self.lr_history.append({
            'lr': new_lr,
            'performance': current_performance,
            'timestamp': time.time()
        })
        
        # Execute callbacks
        await self._execute_callbacks(new_lr, current_performance)
        
        return new_lr
        
    async def _warmup_step(self) -> float:
        """Handle warmup phase"""
        self.warmup_steps += 1
        
        # Linear warmup
        warmup_factor = min(1.0, self.warmup_steps / self.config.warmup_epochs)
        new_lr = self.config.initial_lr * warmup_factor
        
        # Set learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
            
        return new_lr
        
    async def _regular_step(self, current_performance: float, epoch: Optional[int] = None) -> float:
        """Regular scheduling step"""
        
        if isinstance(self.scheduler, AdaptivePlateauScheduler):
            self.scheduler.step(current_performance)
        elif isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(current_performance)
        elif epoch is not None:
            self.scheduler.step(epoch)
        else:
            self.scheduler.step()
            
        # Get current learning rate
        new_lr = self.optimizer.param_groups[0]['lr']
        
        # Additional adaptive adjustments
        if self.config.scheduler_type == 'adaptive_plateau':
            new_lr = await self._adaptive_adjustments(new_lr, current_performance)
            
        return new_lr
        
    async def _adaptive_adjustments(self, base_lr: float, current_performance: float) -> float:
        """Additional adaptive adjustments based on performance trends"""
        
        if len(self.performance_history) < 10:
            return base_lr
            
        # Analyze recent performance trend
        recent_performance = list(self.performance_history)[-10:]
        performance_trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
        
        # Adjust based on trend
        if performance_trend < -0.01:  # Declining performance
            # More aggressive reduction
            adjusted_lr = base_lr * 0.8
            logger.info(f"Declining performance detected, extra LR reduction: {base_lr:.6f} -> {adjusted_lr:.6f}")
            
        elif performance_trend > 0.05:  # Rapid improvement
            # Slight increase to exploit momentum
            adjusted_lr = min(base_lr * 1.1, self.config.max_lr)
            logger.info(f"Rapid improvement detected, slight LR increase: {base_lr:.6f} -> {adjusted_lr:.6f}")
            
        else:
            adjusted_lr = base_lr
            
        # Set the adjusted learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = adjusted_lr
            
        return adjusted_lr
        
    def _update_metrics(self, current_performance: float, new_lr: float):
        """Update internal metrics"""
        
        # Update best performance
        if current_performance > self.metrics.best_performance:
            improvement = current_performance - self.metrics.best_performance
            self.metrics.best_performance = current_performance
            self.metrics.patience_counter = 0
            
            # Calculate improvement rate
            if len(self.performance_history) > 1:
                self.metrics.improvement_rate = improvement / abs(self.metrics.best_performance)
        else:
            self.metrics.patience_counter += 1
            
        # Update current LR
        if new_lr != self.metrics.current_lr:
            self.metrics.total_adjustments += 1
            self.metrics.current_lr = new_lr
            
        self.metrics.timestamp = time.time()
        
    async def _execute_callbacks(self, new_lr: float, performance: float):
        """Execute registered callbacks"""
        for callback in self.callbacks:
            try:
                await callback(new_lr, performance, self.metrics)
            except Exception as e:
                logger.error(f"Error in LR scheduler callback: {e}")
                
    def register_callback(self, callback: Callable):
        """Register a callback for LR changes"""
        self.callbacks.append(callback)
        
    def get_metrics(self) -> LearningRateMetrics:
        """Get current metrics"""
        return self.metrics
        
    def get_history(self) -> Dict[str, Any]:
        """Get learning rate history"""
        return {
            'lr_history': list(self.lr_history),
            'performance_history': list(self.performance_history),
            'current_metrics': {
                'lr': self.metrics.current_lr,
                'best_performance': self.metrics.best_performance,
                'patience_counter': self.metrics.patience_counter,
                'total_adjustments': self.metrics.total_adjustments
            }
        }
        
    def reset(self):
        """Reset scheduler state"""
        self.metrics = LearningRateMetrics(
            current_lr=self.config.initial_lr,
            best_performance=-float('inf'),
            patience_counter=0,
            total_adjustments=0,
            improvement_rate=0.0
        )
        
        self.performance_history.clear()
        self.lr_history.clear()
        self.warmup_steps = 0
        self.is_warmup_complete = False
        
        # Reset optimizer LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.initial_lr
            
        # Recreate scheduler
        self.scheduler = self._create_scheduler()
        
    def suggest_scheduler_type(self, training_profile: Dict[str, Any]) -> str:
        """Suggest best scheduler type based on training profile"""
        
        data_size = training_profile.get('data_size', 1000)
        model_complexity = training_profile.get('model_complexity', 'medium')
        training_stability = training_profile.get('stability', 'stable')
        
        if data_size < 1000:
            # Small dataset - use adaptive plateau
            return 'adaptive_plateau'
            
        elif training_stability == 'unstable':
            # Unstable training - use cosine with warm restarts
            return 'cosine_warm_restarts'
            
        elif model_complexity == 'high':
            # Complex model - use one cycle
            return 'one_cycle'
            
        else:
            # Default to cyclic
            return 'cyclic'


class AdaptivePlateauScheduler:
    """Custom adaptive plateau scheduler with enhanced features"""
    
    def __init__(self, optimizer, patience=10, factor=0.5, min_lr=1e-6, threshold=0.01):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.threshold = threshold
        
        self.best = -float('inf')
        self.num_bad_epochs = 0
        self.last_epoch = -1
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        
        # Additional adaptive features
        self.performance_window = deque(maxlen=patience * 2)
        self.lr_adjustments = deque(maxlen=50)
        
    def step(self, metrics):
        """Step based on metrics"""
        current = float(metrics)
        self.performance_window.append(current)
        
        # Check if we should reduce
        if self.is_better(current):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
            
    def is_better(self, current):
        """Check if current is better than best"""
        return current > self.best * (1 + self.threshold)
        
    def _reduce_lr(self):
        """Reduce learning rate"""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            
            self.lr_adjustments.append({
                'timestamp': time.time(),
                'old_lr': old_lr,
                'new_lr': new_lr,
                'reason': 'plateau'
            })
            
            logger.info(f'Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}')


class LearningRateMonitor:
    """Monitor and analyze learning rate behavior"""
    
    def __init__(self):
        self.schedulers = {}
        self.analysis_results = deque(maxlen=100)
        
    def register_scheduler(self, name: str, scheduler: AdaptiveLearningScheduler):
        """Register a scheduler for monitoring"""
        self.schedulers[name] = scheduler
        
    async def analyze_all(self) -> Dict[str, Any]:
        """Analyze all registered schedulers"""
        results = {}
        
        for name, scheduler in self.schedulers.items():
            results[name] = await self._analyze_scheduler(scheduler)
            
        self.analysis_results.append({
            'timestamp': time.time(),
            'results': results
        })
        
        return results
        
    async def _analyze_scheduler(self, scheduler: AdaptiveLearningScheduler) -> Dict[str, Any]:
        """Analyze a single scheduler"""
        history = scheduler.get_history()
        metrics = scheduler.get_metrics()
        
        analysis = {
            'current_lr': metrics.current_lr,
            'total_adjustments': metrics.total_adjustments,
            'patience_usage': metrics.patience_counter / scheduler.config.patience,
            'improvement_rate': metrics.improvement_rate
        }
        
        # Analyze LR history
        if len(history['lr_history']) > 10:
            lr_values = [h['lr'] for h in history['lr_history']]
            
            analysis['lr_trend'] = self._calculate_trend(lr_values)
            analysis['lr_volatility'] = np.std(lr_values) / np.mean(lr_values)
            analysis['convergence_indicator'] = self._calculate_convergence(lr_values)
            
        # Analyze performance correlation
        if len(history['performance_history']) > 10:
            analysis['performance_lr_correlation'] = self._calculate_correlation(
                history['lr_history'],
                history['performance_history']
            )
            
        return analysis
        
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'unknown'
            
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope < -1e-6:
            return 'decreasing'
        elif slope > 1e-6:
            return 'increasing'
        else:
            return 'stable'
            
    def _calculate_convergence(self, lr_values: List[float]) -> float:
        """Calculate convergence indicator (0-1, higher = more converged)"""
        if len(lr_values) < 10:
            return 0.0
            
        recent = lr_values[-10:]
        early = lr_values[:10]
        
        recent_var = np.var(recent)
        early_var = np.var(early)
        
        if early_var == 0:
            return 1.0
            
        return 1.0 - min(recent_var / early_var, 1.0)
        
    def _calculate_correlation(self, lr_history: List[Dict], performance_history: List[float]) -> float:
        """Calculate correlation between LR and performance"""
        if len(lr_history) != len(performance_history):
            return 0.0
            
        lr_values = [h['lr'] for h in lr_history]
        
        if len(lr_values) < 10:
            return 0.0
            
        return np.corrcoef(lr_values, performance_history)[0, 1]
        
    def get_recommendations(self) -> List[str]:
        """Get recommendations based on analysis"""
        if not self.analysis_results:
            return []
            
        recommendations = []
        latest = self.analysis_results[-1]['results']
        
        for name, analysis in latest.items():
            # Check for issues
            if analysis.get('lr_volatility', 0) > 0.5:
                recommendations.append(f"{name}: High LR volatility - consider more stable scheduler")
                
            if analysis.get('convergence_indicator', 0) < 0.3:
                recommendations.append(f"{name}: Poor convergence - increase patience or reduce factor")
                
            if abs(analysis.get('performance_lr_correlation', 0)) < 0.1:
                recommendations.append(f"{name}: LR changes not correlated with performance - review scheduler type")
                
        return recommendations


# Example usage
async def example_usage():
    """Example of using AdaptiveLearningScheduler"""
    
    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create scheduler config
    config = SchedulerConfig(
        scheduler_type='adaptive_plateau',
        initial_lr=0.001,
        min_lr=1e-6,
        patience=20,
        reduction_factor=0.5,
        warmup_epochs=5
    )
    
    # Initialize scheduler
    scheduler = AdaptiveLearningScheduler(optimizer, config)
    
    # Register callback
    async def lr_callback(new_lr, performance, metrics):
        print(f"LR changed to {new_lr:.6f}, performance: {performance:.4f}")
        
    scheduler.register_callback(lr_callback)
    
    # Simulate training
    for epoch in range(100):
        # Simulate performance (improving then plateauing)
        if epoch < 30:
            performance = 0.5 + epoch * 0.01 + np.random.normal(0, 0.02)
        else:
            performance = 0.8 + np.random.normal(0, 0.02)
            
        # Step scheduler
        new_lr = await scheduler.step(performance, epoch)
        
        if epoch % 10 == 0:
            metrics = scheduler.get_metrics()
            print(f"Epoch {epoch}: LR={new_lr:.6f}, Best={metrics.best_performance:.4f}, Patience={metrics.patience_counter}")
            
    # Get final history
    history = scheduler.get_history()
    print(f"\nTotal LR adjustments: {len(history['lr_history'])}")
    
    # Create monitor and analyze
    monitor = LearningRateMonitor()
    monitor.register_scheduler('main', scheduler)
    
    analysis = await monitor.analyze_all()
    print(f"\nAnalysis: {analysis}")
    
    recommendations = monitor.get_recommendations()
    print(f"\nRecommendations: {recommendations}")


if __name__ == "__main__":
    asyncio.run(example_usage())