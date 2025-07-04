"""
grid_strategy_selector.py
Select and configure grid strategy based on regime for grid trading system

Author: Grid Trading System
Date: 2024
"""

# Standard library imports
import asyncio
import time
import logging
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

# Local imports
from core.market_regime_detector import MarketRegime
from core.attention_learning_layer import AttentionLearningLayer, AttentionPhase

# Setup logger
logger = logging.getLogger(__name__)


# Constants
DEFAULT_GRID_SPACING = 0.001  # 0.1%
DEFAULT_GRID_LEVELS = 5
DEFAULT_POSITION_SIZE = 0.01  # 1% of capital per level
MIN_GRID_SPACING = 0.0001  # 0.01%
MAX_GRID_SPACING = 0.01  # 1%
MIN_GRID_LEVELS = 2
MAX_GRID_LEVELS = 20
STRATEGY_CACHE_TTL = 60  # seconds


# Enums
class GridType(Enum):
    """Types of grid strategies"""
    SYMMETRIC = "symmetric"      # Equal spacing above and below
    ASYMMETRIC = "asymmetric"    # Different spacing for buy/sell
    GEOMETRIC = "geometric"      # Geometric progression spacing
    FIBONACCI = "fibonacci"      # Fibonacci-based spacing
    DYNAMIC = "dynamic"          # Dynamically adjusted spacing


class OrderDistribution(Enum):
    """How to distribute orders in the grid"""
    UNIFORM = "uniform"          # Equal size at each level
    PYRAMID = "pyramid"          # Larger positions at better prices
    INVERSE_PYRAMID = "inverse"  # Smaller positions at better prices
    WEIGHTED = "weighted"        # Custom weight distribution


@dataclass
class GridLevel:
    """Single grid level configuration"""
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    level_index: int
    distance_from_mid: float
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GridStrategyConfig:
    """Complete grid strategy configuration"""
    regime: MarketRegime
    grid_type: GridType
    spacing: float  # Base spacing between levels
    levels: int  # Number of levels each side
    position_size: float  # Total position size
    order_distribution: OrderDistribution
    risk_limits: Dict[str, float]
    execution_rules: Dict[str, Any]
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'regime': self.regime.value,
            'grid_type': self.grid_type.value,
            'spacing': self.spacing,
            'levels': self.levels,
            'position_size': self.position_size,
            'order_distribution': self.order_distribution.value,
            'risk_limits': self.risk_limits.copy(),
            'execution_rules': self.execution_rules.copy(),
            'enabled': self.enabled,
            'metadata': self.metadata.copy()
        }


class StrategyCache:
    """LRU cache for strategy configurations"""
    
    def __init__(self, max_size: int = 100, ttl: int = STRATEGY_CACHE_TTL):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
        self.hit_count = 0
        self.miss_count = 0
        self._lock = asyncio.Lock()
        
    async def get(self, key: str) -> Optional[GridStrategyConfig]:
        """Get cached strategy"""
        async with self._lock:
            if key in self.cache:
                # Check if expired
                if time.time() - self.timestamps[key] > self.ttl:
                    del self.cache[key]
                    del self.timestamps[key]
                    self.miss_count += 1
                    return None
                    
                self.hit_count += 1
                return self.cache[key]
                
            self.miss_count += 1
            return None
            
    async def set(self, key: str, config: GridStrategyConfig) -> None:
        """Cache strategy configuration"""
        async with self._lock:
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.timestamps, key=self.timestamps.get)
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
                
            self.cache[key] = config
            self.timestamps[key] = time.time()
            
    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


class BaseGridStrategy(ABC):
    """Base class for grid strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.execution_count = 0
        self.success_count = 0
        self.total_profit = 0.0
        self.parameter_history = deque(maxlen=100)
        
    @abstractmethod
    def get_base_parameters(self) -> Dict[str, Any]:
        """Get base strategy parameters"""
        pass
        
    @abstractmethod
    def calculate_grid_levels(
        self, 
        current_price: float, 
        params: Dict[str, Any]
    ) -> List[GridLevel]:
        """Calculate grid levels"""
        pass
        
    @abstractmethod
    def get_risk_profile(self) -> Dict[str, float]:
        """Get risk profile for this strategy"""
        pass
        
    def update_performance(self, profit: float, success: bool) -> None:
        """Update strategy performance metrics"""
        self.execution_count += 1
        if success:
            self.success_count += 1
        self.total_profit += profit
        
    def get_success_rate(self) -> float:
        """Get strategy success rate"""
        if self.execution_count == 0:
            return 0.5
        return self.success_count / self.execution_count
        
    def get_average_profit(self) -> float:
        """Get average profit per execution"""
        if self.execution_count == 0:
            return 0.0
        return self.total_profit / self.execution_count


class RangingGridStrategy(BaseGridStrategy):
    """Grid strategy for ranging/sideways markets"""
    
    def __init__(self):
        super().__init__('ranging_grid')
        
    def get_base_parameters(self) -> Dict[str, Any]:
        """Get base parameters for ranging market - CONSERVATIVE"""
        return {
            'grid_type': GridType.SYMMETRIC,
            'spacing': 0.002,  # 0.2% - increased from 0.15%
            'levels': 5,  # Reduced from 8 for safety
            'order_distribution': OrderDistribution.UNIFORM,
            'position_multiplier': 0.8,  # Reduced from 1.2
            'take_profit': 0.0015,  # 0.15% - increased from 0.1%
            'stop_loss': 0.003,  # 0.3% - tighter from 0.5%
            'rebalance_threshold': 0.005,  # Rebalance at 0.5% drift
            'enabled': True
        }
        
    def calculate_grid_levels(
        self, 
        current_price: float, 
        params: Dict[str, Any]
    ) -> List[GridLevel]:
        """Calculate symmetric grid levels for ranging market"""
        levels = []
        spacing = params.get('spacing', 0.0015)
        num_levels = params.get('levels', 8)
        distribution = params.get('order_distribution', OrderDistribution.UNIFORM)
        
        # Calculate buy levels (below current price)
        for i in range(1, num_levels + 1):
            price = current_price * (1 - spacing * i)
            weight = self._calculate_weight(i, num_levels, distribution, 'buy')
            
            level = GridLevel(
                price=price,
                size=weight,  # Will be multiplied by position size
                side='buy',
                level_index=i,
                distance_from_mid=spacing * i,
                weight=weight,
                metadata={'strategy': 'ranging', 'level_type': 'support'}
            )
            levels.append(level)
            
        # Calculate sell levels (above current price)
        for i in range(1, num_levels + 1):
            price = current_price * (1 + spacing * i)
            weight = self._calculate_weight(i, num_levels, distribution, 'sell')
            
            level = GridLevel(
                price=price,
                size=weight,
                side='sell',
                level_index=i,
                distance_from_mid=spacing * i,
                weight=weight,
                metadata={'strategy': 'ranging', 'level_type': 'resistance'}
            )
            levels.append(level)
            
        return levels
        
    def _calculate_weight(
        self, 
        level_index: int, 
        total_levels: int, 
        distribution: OrderDistribution,
        side: str
    ) -> float:
        """Calculate weight for order distribution"""
        if distribution == OrderDistribution.UNIFORM:
            return 1.0 / total_levels
            
        elif distribution == OrderDistribution.PYRAMID:
            # More weight on levels closer to current price
            return (total_levels - level_index + 1) / sum(range(1, total_levels + 1))
            
        elif distribution == OrderDistribution.INVERSE_PYRAMID:
            # More weight on levels further from current price
            return level_index / sum(range(1, total_levels + 1))
            
        return 1.0 / total_levels
        
    def get_risk_profile(self) -> Dict[str, float]:
        """Get risk profile for ranging strategy"""
        return {
            'max_position_size': 0.15,  # 15% of capital
            'max_drawdown': 0.03,  # 3%
            'position_timeout': 3600,  # 1 hour
            'correlation_limit': 0.7,
            'leverage_limit': 3.0
        }


class TrendingGridStrategy(BaseGridStrategy):
    """Grid strategy for trending markets"""
    
    def __init__(self):
        super().__init__('trending_grid')
        
    def get_base_parameters(self) -> Dict[str, Any]:
        """Get base parameters for trending market - CONSERVATIVE"""
        return {
            'grid_type': GridType.ASYMMETRIC,
            'spacing_with_trend': 0.0025,  # 0.25% - increased from 0.2%
            'spacing_against_trend': 0.005,  # 0.5% - increased from 0.4%
            'levels_with_trend': 4,  # Reduced from 6
            'levels_against_trend': 2,  # Reduced from 3
            'order_distribution': OrderDistribution.PYRAMID,
            'position_multiplier': 0.6,  # Reduced from 0.8
            'take_profit': 0.004,  # 0.4% - increased from 0.3%
            'stop_loss': 0.0015,  # 0.15% - tighter from 0.2%
            'trail_stop': True,  # Use trailing stops
            'enabled': True
        }
        
    def calculate_grid_levels(
        self, 
        current_price: float, 
        params: Dict[str, Any]
    ) -> List[GridLevel]:
        """Calculate asymmetric grid levels for trending market"""
        levels = []
        
        # Determine trend direction (would come from features)
        trend_direction = params.get('trend_direction', 'up')
        
        if trend_direction == 'up':
            # More buy levels, fewer sell levels
            buy_spacing = params.get('spacing_with_trend', 0.002)
            sell_spacing = params.get('spacing_against_trend', 0.004)
            buy_levels = params.get('levels_with_trend', 6)
            sell_levels = params.get('levels_against_trend', 3)
        else:
            # More sell levels, fewer buy levels
            buy_spacing = params.get('spacing_against_trend', 0.004)
            sell_spacing = params.get('spacing_with_trend', 0.002)
            buy_levels = params.get('levels_against_trend', 3)
            sell_levels = params.get('levels_with_trend', 6)
            
        distribution = params.get('order_distribution', OrderDistribution.PYRAMID)
        
        # Calculate buy levels
        for i in range(1, buy_levels + 1):
            price = current_price * (1 - buy_spacing * i)
            weight = self._calculate_trend_weight(i, buy_levels, distribution, 'buy', trend_direction)
            
            level = GridLevel(
                price=price,
                size=weight,
                side='buy',
                level_index=i,
                distance_from_mid=buy_spacing * i,
                weight=weight,
                metadata={
                    'strategy': 'trending',
                    'trend_direction': trend_direction,
                    'with_trend': trend_direction == 'up'
                }
            )
            levels.append(level)
            
        # Calculate sell levels
        for i in range(1, sell_levels + 1):
            price = current_price * (1 + sell_spacing * i)
            weight = self._calculate_trend_weight(i, sell_levels, distribution, 'sell', trend_direction)
            
            level = GridLevel(
                price=price,
                size=weight,
                side='sell',
                level_index=i,
                distance_from_mid=sell_spacing * i,
                weight=weight,
                metadata={
                    'strategy': 'trending',
                    'trend_direction': trend_direction,
                    'with_trend': trend_direction == 'down'
                }
            )
            levels.append(level)
            
        return levels
        
    def _calculate_trend_weight(
        self, 
        level_index: int, 
        total_levels: int, 
        distribution: OrderDistribution,
        side: str,
        trend_direction: str
    ) -> float:
        """Calculate weight considering trend direction"""
        base_weight = 1.0 / (total_levels * 2)  # Divided by 2 for both sides
        
        # Adjust weight based on trend alignment
        if (side == 'buy' and trend_direction == 'up') or (side == 'sell' and trend_direction == 'down'):
            # With trend - higher weight
            trend_multiplier = 1.5
        else:
            # Against trend - lower weight
            trend_multiplier = 0.5
            
        if distribution == OrderDistribution.PYRAMID:
            position_weight = (total_levels - level_index + 1) / sum(range(1, total_levels + 1))
        else:
            position_weight = 1.0 / total_levels
            
        return base_weight * position_weight * trend_multiplier
        
    def get_risk_profile(self) -> Dict[str, float]:
        """Get risk profile for trending strategy"""
        return {
            'max_position_size': 0.1,  # 10% of capital - lower for trending
            'max_drawdown': 0.02,  # 2% - tighter
            'position_timeout': 7200,  # 2 hours - hold longer
            'correlation_limit': 0.8,
            'leverage_limit': 2.0  # Less leverage in trends
        }


class VolatileGridStrategy(BaseGridStrategy):
    """Grid strategy for volatile markets"""
    
    def __init__(self):
        super().__init__('volatile_grid')
        
    def get_base_parameters(self) -> Dict[str, Any]:
        """Get base parameters for volatile market - VERY CONSERVATIVE"""
        return {
            'grid_type': GridType.GEOMETRIC,
            'base_spacing': 0.005,  # 0.5% - increased from 0.3%
            'spacing_multiplier': 1.2,  # Geometric progression
            'levels': 3,  # Reduced from 5
            'order_distribution': OrderDistribution.INVERSE_PYRAMID,
            'position_multiplier': 0.3,  # Reduced from 0.5
            'take_profit': 0.007,  # 0.7% - increased from 0.5%
            'stop_loss': 0.002,  # 0.2% - tighter from 0.3%
            'max_exposure': 0.03,  # 3% max exposure - reduced from 5%
            'enabled': True
        }
        
    def calculate_grid_levels(
        self, 
        current_price: float, 
        params: Dict[str, Any]
    ) -> List[GridLevel]:
        """Calculate geometric grid levels for volatile market"""
        levels = []
        base_spacing = params.get('base_spacing', 0.003)
        multiplier = params.get('spacing_multiplier', 1.2)
        num_levels = params.get('levels', 5)
        distribution = params.get('order_distribution', OrderDistribution.INVERSE_PYRAMID)
        
        # Calculate buy levels with geometric progression
        current_spacing = base_spacing
        for i in range(1, num_levels + 1):
            price = current_price * (1 - current_spacing)
            weight = self._calculate_volatile_weight(i, num_levels, distribution)
            
            level = GridLevel(
                price=price,
                size=weight,
                side='buy',
                level_index=i,
                distance_from_mid=current_spacing,
                weight=weight,
                metadata={
                    'strategy': 'volatile',
                    'spacing_type': 'geometric',
                    'risk_adjusted': True
                }
            )
            levels.append(level)
            
            # Increase spacing for next level
            current_spacing *= multiplier
            
        # Calculate sell levels with geometric progression
        current_spacing = base_spacing
        for i in range(1, num_levels + 1):
            price = current_price * (1 + current_spacing)
            weight = self._calculate_volatile_weight(i, num_levels, distribution)
            
            level = GridLevel(
                price=price,
                size=weight,
                side='sell',
                level_index=i,
                distance_from_mid=current_spacing,
                weight=weight,
                metadata={
                    'strategy': 'volatile',
                    'spacing_type': 'geometric',
                    'risk_adjusted': True
                }
            )
            levels.append(level)
            
            # Increase spacing for next level
            current_spacing *= multiplier
            
        return levels
        
    def _calculate_volatile_weight(
        self, 
        level_index: int, 
        total_levels: int, 
        distribution: OrderDistribution
    ) -> float:
        """Calculate weight for volatile market"""
        # In volatile markets, prefer smaller positions at extremes
        if distribution == OrderDistribution.INVERSE_PYRAMID:
            # More weight on outer levels (better prices)
            weight = level_index / sum(range(1, total_levels + 1))
        else:
            weight = 1.0 / total_levels
            
        # Apply volatility dampening
        volatility_dampener = 0.5  # Reduce all weights
        return weight * volatility_dampener
        
    def get_risk_profile(self) -> Dict[str, float]:
        """Get risk profile for volatile strategy"""
        return {
            'max_position_size': 0.05,  # 5% of capital - very conservative
            'max_drawdown': 0.015,  # 1.5% - very tight
            'position_timeout': 1800,  # 30 minutes - quick exits
            'correlation_limit': 0.5,  # Lower correlation tolerance
            'leverage_limit': 1.0  # No leverage in volatile markets
        }


class DormantStrategy(BaseGridStrategy):
    """Strategy for dormant/inactive markets"""
    
    def __init__(self):
        super().__init__('dormant_strategy')
        
    def get_base_parameters(self) -> Dict[str, Any]:
        """Get base parameters for dormant market"""
        return {
            'grid_type': GridType.SYMMETRIC,
            'spacing': 0.002,  # 0.2% - moderate spacing
            'levels': 3,  # Very few levels
            'order_distribution': OrderDistribution.UNIFORM,
            'position_multiplier': 0.3,  # Very small positions
            'take_profit': 0.002,  # 0.2%
            'stop_loss': 0.004,  # 0.4%
            'enabled': False  # Often disabled in dormant markets
        }
        
    def calculate_grid_levels(
        self, 
        current_price: float, 
        params: Dict[str, Any]
    ) -> List[GridLevel]:
        """Calculate minimal grid for dormant market"""
        if not params.get('enabled', False):
            return []  # No levels if disabled
            
        levels = []
        spacing = params.get('spacing', 0.002)
        num_levels = params.get('levels', 3)
        
        # Minimal grid setup
        for i in range(1, num_levels + 1):
            # Buy level
            buy_price = current_price * (1 - spacing * i)
            levels.append(GridLevel(
                price=buy_price,
                size=1.0 / (num_levels * 2),
                side='buy',
                level_index=i,
                distance_from_mid=spacing * i,
                weight=1.0 / num_levels,
                metadata={'strategy': 'dormant', 'minimal': True}
            ))
            
            # Sell level
            sell_price = current_price * (1 + spacing * i)
            levels.append(GridLevel(
                price=sell_price,
                size=1.0 / (num_levels * 2),
                side='sell',
                level_index=i,
                distance_from_mid=spacing * i,
                weight=1.0 / num_levels,
                metadata={'strategy': 'dormant', 'minimal': True}
            ))
            
        return levels
        
    def get_risk_profile(self) -> Dict[str, float]:
        """Get risk profile for dormant strategy"""
        return {
            'max_position_size': 0.02,  # 2% of capital - minimal
            'max_drawdown': 0.01,  # 1%
            'position_timeout': 14400,  # 4 hours - very patient
            'correlation_limit': 0.9,  # High correlation OK
            'leverage_limit': 1.0  # No leverage
        }


class GridStrategySelector:
    """
    Select and configure grid strategy based on regime
    """
    
    def __init__(self, attention_layer: Optional[AttentionLearningLayer] = None):
        self.attention = attention_layer
        self.strategies = self._init_strategies()
        self.strategy_cache = StrategyCache()
        self.performance_history = defaultdict(list)
        self.current_strategy = None
        self.strategy_adjustments = defaultdict(dict)
        self._lock = asyncio.Lock()
        
        logger.info("Initialized Grid Strategy Selector")
        
    def _init_strategies(self) -> Dict[MarketRegime, BaseGridStrategy]:
        """Initialize regime-specific strategies"""
        return {
            MarketRegime.RANGING: RangingGridStrategy(),
            MarketRegime.TRENDING: TrendingGridStrategy(),
            MarketRegime.VOLATILE: VolatileGridStrategy(),
            MarketRegime.DORMANT: DormantStrategy()
        }
        
    async def select_strategy(
        self, 
        regime: MarketRegime, 
        features: Dict[str, float], 
        context: Dict[str, Any]
    ) -> GridStrategyConfig:
        """Select appropriate grid strategy"""
        async with self._lock:
            # Generate cache key
            cache_key = self._generate_cache_key(regime, features)
            
            # Check cache
            cached_config = await self.strategy_cache.get(cache_key)
            if cached_config:
                return cached_config
                
            # Get base strategy
            base_strategy = self.strategies.get(regime)
            if not base_strategy:
                logger.error(f"No strategy found for regime {regime}")
                base_strategy = self.strategies[MarketRegime.RANGING]  # Default
                
            # Get base parameters
            params = base_strategy.get_base_parameters()
            
            # Apply attention adjustments if active
            if self.attention and self.attention.phase == AttentionPhase.ACTIVE:
                params = await self._apply_attention_adjustments(params, regime, context)
                
            # Apply dynamic adjustments based on current conditions
            params = self._apply_dynamic_adjustments(params, features, context)
            
            # Apply learned adjustments
            if regime in self.strategy_adjustments:
                params = self._apply_learned_adjustments(params, self.strategy_adjustments[regime])
                
            # Calculate risk limits
            risk_limits = self._calculate_risk_limits(params, context, base_strategy)
            
            # Get execution rules
            execution_rules = self._get_execution_rules(regime, features)
            
            # Create final strategy configuration
            strategy_config = GridStrategyConfig(
                regime=regime,
                grid_type=params.get('grid_type', GridType.SYMMETRIC),
                spacing=params.get('spacing', DEFAULT_GRID_SPACING),
                levels=params.get('levels', DEFAULT_GRID_LEVELS),
                position_size=params.get('position_size', DEFAULT_POSITION_SIZE),
                order_distribution=params.get('order_distribution', OrderDistribution.UNIFORM),
                risk_limits=risk_limits,
                execution_rules=execution_rules,
                enabled=params.get('enabled', True),
                metadata={
                    'base_strategy': base_strategy.name,
                    'adjustments_applied': True,
                    'cache_key': cache_key,
                    'timestamp': time.time()
                }
            )
            
            # Cache configuration
            await self.strategy_cache.set(cache_key, strategy_config)
            
            # Update current strategy
            self.current_strategy = strategy_config
            
            return strategy_config
            
    def _generate_cache_key(self, regime: MarketRegime, features: Dict[str, float]) -> str:
        """Generate cache key for strategy configuration"""
        # Use regime and key feature values
        key_features = ['volatility_5m', 'trend_strength', 'volume_ratio', 'spread_bps']
        
        key_data = {
            'regime': regime.value,
            'features': {k: round(features.get(k, 0), 4) for k in key_features}
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
        
    async def _apply_attention_adjustments(
        self, 
        params: Dict[str, Any], 
        regime: MarketRegime, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply attention-based adjustments to parameters"""
        if not self.attention:
            return params
            
        # Get regime-specific adjustments from attention
        attention_context = {
            'strategy_params': params,
            'regime': regime.value
        }
        
        adjusted_params = await self.attention.regime_attention.apply_adjustments(
            regime.value, params
        )
        
        return adjusted_params
        
    def _apply_dynamic_adjustments(
        self, 
        params: Dict[str, Any], 
        features: Dict[str, float], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply real-time adjustments to parameters"""
        adjusted_params = params.copy()
        
        # Volatility adjustment
        current_vol = features.get('volatility_5m', 0.001)
        baseline_vol = 0.001
        vol_ratio = current_vol / baseline_vol
        
        # Adjust spacing based on volatility
        if vol_ratio > 1.5:  # High volatility
            adjusted_params['spacing'] *= (1 + (vol_ratio - 1) * 0.5)  # Increase spacing
            adjusted_params['levels'] = max(MIN_GRID_LEVELS, params['levels'] - 2)  # Fewer levels
            
        elif vol_ratio < 0.5:  # Low volatility
            adjusted_params['spacing'] *= (1 - (1 - vol_ratio) * 0.3)  # Decrease spacing
            adjusted_params['levels'] = min(MAX_GRID_LEVELS, params['levels'] + 2)  # More levels
            
        # Spread adjustment
        spread_bps = features.get('spread_bps', 1)
        if spread_bps > 5:  # Wide spread
            adjusted_params['spacing'] = max(adjusted_params['spacing'], spread_bps / 10000 * 2)
            adjusted_params['position_multiplier'] = params.get('position_multiplier', 1) * 0.7
            
        # Volume adjustment
        volume_ratio = features.get('volume_ratio', 1.0)
        if volume_ratio < 0.3:  # Very low volume
            adjusted_params['enabled'] = False  # Disable in low volume
            logger.warning(f"Strategy disabled due to low volume: {volume_ratio}")
            
        elif volume_ratio < 0.5:  # Low volume
            adjusted_params['position_multiplier'] = params.get('position_multiplier', 1) * 0.5
            
        # Trend strength adjustment for trending regime
        if params.get('grid_type') == GridType.ASYMMETRIC:
            trend_strength = features.get('trend_strength', 0)
            adjusted_params['trend_direction'] = 'up' if trend_strength > 0 else 'down'
            
            # Stronger trends = more asymmetry
            trend_magnitude = abs(trend_strength)
            if trend_magnitude > 0.7:
                ratio = 2.0  # 2:1 with:against trend
            elif trend_magnitude > 0.4:
                ratio = 1.5
            else:
                ratio = 1.2
                
            if trend_strength > 0:  # Uptrend
                adjusted_params['levels_with_trend'] = int(params.get('levels', 6) * ratio / (ratio + 1))
                adjusted_params['levels_against_trend'] = params.get('levels', 6) - adjusted_params['levels_with_trend']
            else:  # Downtrend
                adjusted_params['levels_against_trend'] = int(params.get('levels', 6) * ratio / (ratio + 1))
                adjusted_params['levels_with_trend'] = params.get('levels', 6) - adjusted_params['levels_against_trend']
                
        # Ensure parameters are within bounds
        adjusted_params['spacing'] = np.clip(adjusted_params['spacing'], MIN_GRID_SPACING, MAX_GRID_SPACING)
        adjusted_params['levels'] = np.clip(adjusted_params['levels'], MIN_GRID_LEVELS, MAX_GRID_LEVELS)
        
        return adjusted_params
        
    def _apply_learned_adjustments(
        self, 
        params: Dict[str, Any], 
        learned_adjustments: Dict[str, float]
    ) -> Dict[str, Any]:
        """Apply adjustments learned from historical performance"""
        adjusted_params = params.copy()
        
        for param, adjustment in learned_adjustments.items():
            if param in adjusted_params and isinstance(adjusted_params[param], (int, float)):
                # Apply multiplicative adjustment
                adjusted_params[param] *= adjustment
                
        return adjusted_params
        
    def _calculate_risk_limits(
        self, 
        params: Dict[str, Any], 
        context: Dict[str, Any],
        strategy: BaseGridStrategy
    ) -> Dict[str, float]:
        """Calculate risk limits for strategy"""
        # Get base risk profile from strategy
        risk_profile = strategy.get_risk_profile()
        
        # Adjust based on account status
        account_balance = context.get('account_balance', 10000)
        daily_loss = context.get('daily_loss', 0)
        
        # Tighten limits if already have losses
        if daily_loss < -0.01 * account_balance:  # Lost 1% today
            risk_profile['max_position_size'] *= 0.5
            risk_profile['max_drawdown'] *= 0.5
            
        # Consider correlation with existing positions
        existing_positions = context.get('existing_positions', [])
        if len(existing_positions) > 0:
            risk_profile['max_position_size'] *= (1 - 0.1 * len(existing_positions))
            
        # Apply parameter-based adjustments
        position_multiplier = params.get('position_multiplier', 1.0)
        risk_profile['max_position_size'] *= position_multiplier
        
        return risk_profile
        
    def _get_execution_rules(self, regime: MarketRegime, features: Dict[str, float]) -> Dict[str, Any]:
        """Get execution rules for regime"""
        rules = {
            'order_type': 'limit',
            'time_in_force': 'GTC',
            'post_only': True,  # Maker only
            'reduce_only': False,
            'iceberg': False,
            'execution_delay': 0  # milliseconds
        }
        
        # Regime-specific rules
        if regime == MarketRegime.VOLATILE:
            rules['post_only'] = False  # Allow taker in volatile markets
            rules['time_in_force'] = 'IOC'  # Immediate or cancel
            rules['execution_delay'] = 100  # Add delay to avoid spikes
            
        elif regime == MarketRegime.TRENDING:
            rules['iceberg'] = True  # Hide large orders in trends
            rules['iceberg_visible_size'] = 0.2  # Show only 20%
            
        elif regime == MarketRegime.DORMANT:
            rules['execution_delay'] = 500  # Slower execution in dormant
            
        # Spread-based adjustments
        spread_bps = features.get('spread_bps', 1)
        if spread_bps > 10:
            rules['post_only'] = True  # Force maker in wide spreads
            rules['price_improvement'] = 0.0001  # Improve price by 1 bps
            
        return rules
        
    async def calculate_grid_levels(
        self, 
        strategy_config: GridStrategyConfig, 
        current_price: float
    ) -> List[GridLevel]:
        """Calculate actual grid levels from configuration"""
        if not strategy_config.enabled:
            return []
            
        # Get the appropriate strategy
        strategy = self.strategies.get(strategy_config.regime)
        if not strategy:
            logger.error(f"No strategy found for regime {strategy_config.regime}")
            return []
            
        # Prepare parameters
        params = {
            'spacing': strategy_config.spacing,
            'levels': strategy_config.levels,
            'order_distribution': strategy_config.order_distribution,
            'grid_type': strategy_config.grid_type
        }
        
        # Add any metadata parameters
        params.update(strategy_config.metadata)
        
        # Calculate levels
        levels = strategy.calculate_grid_levels(current_price, params)
        
        # Apply position sizing
        total_size = strategy_config.position_size
        for level in levels:
            level.size *= total_size
            
        return levels
        
    async def update_performance(
        self, 
        regime: MarketRegime, 
        profit: float, 
        success: bool,
        execution_metadata: Dict[str, Any]
    ) -> None:
        """Update strategy performance metrics"""
        async with self._lock:
            # Update strategy performance
            if regime in self.strategies:
                self.strategies[regime].update_performance(profit, success)
                
            # Track performance history
            performance_entry = {
                'timestamp': time.time(),
                'regime': regime.value,
                'profit': profit,
                'success': success,
                'metadata': execution_metadata
            }
            self.performance_history[regime].append(performance_entry)
            
            # Learn from performance
            await self._learn_from_performance(regime)
            
    async def _learn_from_performance(self, regime: MarketRegime) -> None:
        """Learn parameter adjustments from performance"""
        history = self.performance_history[regime]
        
        if len(history) < 20:  # Need sufficient history
            return
            
        # Analyze recent performance
        recent_history = history[-50:]
        
        # Calculate performance metrics
        total_profit = sum(h['profit'] for h in recent_history)
        success_rate = sum(1 for h in recent_history if h['success']) / len(recent_history)
        
        # Simple learning: adjust parameters based on performance
        adjustments = {}
        
        if success_rate < 0.4:  # Poor performance
            adjustments['spacing'] = 1.1  # Increase spacing
            adjustments['levels'] = 0.9  # Reduce levels
            adjustments['position_multiplier'] = 0.8  # Reduce size
            
        elif success_rate > 0.6:  # Good performance
            adjustments['spacing'] = 0.95  # Decrease spacing slightly
            adjustments['levels'] = 1.05  # Increase levels slightly
            adjustments['position_multiplier'] = 1.1  # Increase size
            
        # Store learned adjustments
        self.strategy_adjustments[regime] = adjustments
        
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive strategy statistics"""
        stats = {
            'cache_hit_rate': self.strategy_cache.get_hit_rate(),
            'current_strategy': self.current_strategy.to_dict() if self.current_strategy else None,
            'strategy_performance': {},
            'regime_performance': {},
            'learned_adjustments': dict(self.strategy_adjustments)
        }
        
        # Get performance for each strategy
        for regime, strategy in self.strategies.items():
            stats['strategy_performance'][regime.value] = {
                'execution_count': strategy.execution_count,
                'success_rate': strategy.get_success_rate(),
                'average_profit': strategy.get_average_profit(),
                'total_profit': strategy.total_profit
            }
            
        # Get performance by regime
        for regime, history in self.performance_history.items():
            if history:
                recent = history[-100:]  # Last 100 trades
                stats['regime_performance'][regime.value] = {
                    'trade_count': len(recent),
                    'total_profit': sum(h['profit'] for h in recent),
                    'success_rate': sum(1 for h in history if h['success']) / len(recent)
                }
                
        return stats
        
    async def save_state(self, filepath: str) -> None:
        """Save selector state to file"""
        state = {
            'performance_history': {
                k.value: v for k, v in self.performance_history.items()
            },
            'strategy_adjustments': {
                k.value: v for k, v in self.strategy_adjustments.items()
            },
            'statistics': await self.get_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Saved strategy selector state to {filepath}")


# Example usage
async def main():
    """Example usage of GridStrategySelector"""
    
    # Initialize selector
    selector = GridStrategySelector()
    
    # Simulate different market conditions
    scenarios = [
        {
            'regime': MarketRegime.RANGING,
            'features': {
                'trend_strength': 0.1,
                'volatility_5m': 0.0008,
                'volume_ratio': 1.2,
                'spread_bps': 2,
                'rsi_14': 0.5
            }
        },
        {
            'regime': MarketRegime.TRENDING,
            'features': {
                'trend_strength': 0.7,
                'volatility_5m': 0.0015,
                'volume_ratio': 1.8,
                'spread_bps': 3,
                'rsi_14': 0.7
            }
        },
        {
            'regime': MarketRegime.VOLATILE,
            'features': {
                'trend_strength': -0.2,
                'volatility_5m': 0.003,
                'volume_ratio': 2.5,
                'spread_bps': 8,
                'rsi_14': 0.3
            }
        }
    ]
    
    for scenario in scenarios:
        # Select strategy
        context = {
            'account_balance': 10000,
            'daily_loss': -50,
            'existing_positions': []
        }
        
        config = await selector.select_strategy(
            scenario['regime'],
            scenario['features'],
            context
        )
        
        print(f"\nRegime: {scenario['regime'].value}")
        print(f"Strategy Config:")
        print(f"  Grid Type: {config.grid_type.value}")
        print(f"  Spacing: {config.spacing:.4f}")
        print(f"  Levels: {config.levels}")
        print(f"  Enabled: {config.enabled}")
        
        # Calculate grid levels
        current_price = 50000
        levels = await selector.calculate_grid_levels(config, current_price)
        
        print(f"\nGrid Levels ({len(levels)} total):")
        for i, level in enumerate(levels[:3]):  # Show first 3
            print(f"  Level {i+1}: {level.side} @ ${level.price:.2f} (size: {level.size:.4f})")
            
        # Simulate performance update
        await selector.update_performance(
            scenario['regime'],
            profit=np.random.randn() * 10,
            success=np.random.rand() > 0.4,
            execution_metadata={'execution_time': 0.5}
        )
        
    # Get statistics
    stats = await selector.get_statistics()
    print(f"\nStrategy Statistics:")
    print(f"  Cache Hit Rate: {stats['cache_hit_rate']:.2%}")
    print(f"  Performance by Regime:")
    for regime, perf in stats['strategy_performance'].items():
        print(f"    {regime}: {perf['success_rate']:.2%} success rate")


if __name__ == "__main__":
    asyncio.run(main())
