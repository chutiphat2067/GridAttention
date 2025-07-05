"""
Dynamic Grid Parameter Optimization
Adjusts grid parameters based on market conditions and Kelly criterion
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass
import logging
from scipy import optimize
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class OptimizedGridParams:
    """Optimized grid parameters container"""
    spacing: float
    levels: int
    position_size: float
    spacing_multiplier: float
    risk_adjustment: float
    fill_probability: float
    expected_profit: float
    max_drawdown_risk: float
    optimal_entry_zones: List[float]
    stop_loss_level: Optional[float] = None
    take_profit_level: Optional[float] = None

@dataclass
class GridPerformanceMetrics:
    """Grid performance analysis metrics"""
    hit_rate: float
    average_profit_per_fill: float
    grid_efficiency: float
    risk_reward_ratio: float
    drawdown_frequency: float
    optimal_fill_distribution: Dict[int, float]

@dataclass
class MarketConditions:
    """Market conditions for grid optimization"""
    regime: str
    sub_regime: str
    volatility_percentile: float
    volume_profile: str
    session: str
    trend_strength: float
    support_resistance_levels: List[float]
    correlation_factor: float
    news_impact_expected: bool

class DynamicGridOptimizer:
    """Advanced grid parameter optimizer with multiple optimization methods"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Base parameters by regime
        self.base_params = {
            'TRENDING': {
                'atr_multiplier': 1.5,
                'base_levels': 10,
                'position_multiplier': 1.0,
                'risk_factor': 0.8,
                'spacing_bias': 1.2  # Wider spacing in trends
            },
            'RANGING': {
                'atr_multiplier': 1.0,
                'base_levels': 15,
                'position_multiplier': 1.2,
                'risk_factor': 1.0,
                'spacing_bias': 0.9  # Tighter spacing in ranges
            },
            'VOLATILE': {
                'atr_multiplier': 2.0,
                'base_levels': 8,
                'position_multiplier': 0.7,
                'risk_factor': 0.6,
                'spacing_bias': 1.5  # Much wider in volatility
            },
            'UNCERTAIN': {
                'atr_multiplier': 1.2,
                'base_levels': 12,
                'position_multiplier': 0.8,
                'risk_factor': 0.7,
                'spacing_bias': 1.0
            }
        }
        
        # Volatility regime adjustments
        self.volatility_adjustments = {
            'very_high': {'spacing_mult': 1.8, 'level_mult': 0.6, 'size_mult': 0.5},
            'high': {'spacing_mult': 1.4, 'level_mult': 0.8, 'size_mult': 0.7},
            'normal': {'spacing_mult': 1.0, 'level_mult': 1.0, 'size_mult': 1.0},
            'low': {'spacing_mult': 0.7, 'level_mult': 1.3, 'size_mult': 1.2},
            'very_low': {'spacing_mult': 0.5, 'level_mult': 1.5, 'size_mult': 1.4}
        }
        
        # Session-based multipliers
        self.session_multipliers = {
            'ASIAN': {'spacing': 0.8, 'levels': 1.2, 'size': 0.9},
            'LONDON': {'spacing': 1.0, 'levels': 1.0, 'size': 1.0},
            'US': {'spacing': 1.1, 'levels': 0.9, 'size': 1.1},
            'US_OPEN': {'spacing': 1.3, 'levels': 0.7, 'size': 0.8},
            'OVERLAP': {'spacing': 1.1, 'levels': 1.0, 'size': 1.0},
            'OFF_HOURS': {'spacing': 0.6, 'levels': 1.4, 'size': 0.7}
        }
        
        # Performance history for adaptive learning
        self.performance_history = deque(maxlen=config.get('history_size', 500))
        self.parameter_effectiveness = {}
        
        # Kelly criterion parameters
        self.kelly_config = {
            'lookback_periods': [50, 100, 200],
            'confidence_threshold': 0.6,
            'max_kelly_fraction': 0.25,
            'conservative_factor': 0.5
        }
        
        # Risk management constraints
        self.risk_constraints = {
            'max_position_size': 0.05,  # 5% of account
            'max_total_exposure': 0.15,  # 15% total grid exposure
            'max_correlation_exposure': 0.10,  # 10% correlated exposure
            'min_liquidity_ratio': 0.1,  # Minimum liquidity requirement
            'max_drawdown_single_grid': 0.03  # 3% max drawdown per grid
        }
        
    def optimize_grid_parameters(
        self,
        regime: str,
        features: Dict[str, float],
        account_info: Dict[str, float],
        market_context: Dict[str, Any],
        historical_data: Optional[pd.DataFrame] = None
    ) -> OptimizedGridParams:
        """Calculate optimal grid parameters using multiple optimization methods"""
        
        try:
            # Extract market conditions
            conditions = self._extract_market_conditions(regime, features, market_context)
            
            # Get base parameters
            base_params = self.base_params.get(regime, self.base_params['UNCERTAIN'])
            
            # Optimize spacing
            optimal_spacing = self._optimize_spacing(base_params, features, conditions, historical_data)
            
            # Optimize levels
            optimal_levels = self._optimize_levels(
                base_params, account_info, features, optimal_spacing, conditions
            )
            
            # Calculate position size using Kelly criterion
            position_size = self._calculate_kelly_position_size(
                account_info, features, conditions, optimal_levels
            )
            
            # Calculate additional metrics
            fill_probability = self._estimate_fill_probability(
                optimal_spacing, features, conditions, historical_data
            )
            
            expected_profit = self._estimate_expected_profit(
                optimal_spacing, optimal_levels, position_size, features, conditions
            )
            
            drawdown_risk = self._estimate_drawdown_risk(
                optimal_spacing, optimal_levels, position_size, features
            )
            
            # Find optimal entry zones
            entry_zones = self._calculate_optimal_entry_zones(
                optimal_spacing, optimal_levels, features, conditions
            )
            
            # Calculate stop loss and take profit levels
            stop_loss, take_profit = self._calculate_exit_levels(
                optimal_spacing, optimal_levels, features, conditions
            )
            
            # Apply risk constraints
            spacing_multiplier = self._calculate_spacing_multiplier(features, conditions)
            risk_adjustment = self._calculate_risk_adjustment(features, conditions)
            
            # Create optimized parameters
            optimized_params = OptimizedGridParams(
                spacing=optimal_spacing,
                levels=optimal_levels,
                position_size=position_size,
                spacing_multiplier=spacing_multiplier,
                risk_adjustment=risk_adjustment,
                fill_probability=fill_probability,
                expected_profit=expected_profit,
                max_drawdown_risk=drawdown_risk,
                optimal_entry_zones=entry_zones,
                stop_loss_level=stop_loss,
                take_profit_level=take_profit
            )
            
            # Validate parameters
            validated_params = self._validate_and_adjust_parameters(
                optimized_params, account_info, conditions
            )
            
            # Store for learning
            self._store_optimization_result(validated_params, features, conditions)
            
            return validated_params
            
        except Exception as e:
            logger.error(f"Grid optimization failed: {e}")
            # Return safe fallback parameters
            return self._get_fallback_parameters(regime, features, account_info)
    
    def _extract_market_conditions(
        self, 
        regime: str, 
        features: Dict[str, float], 
        market_context: Dict[str, Any]
    ) -> MarketConditions:
        """Extract and structure market conditions"""
        
        # Determine volatility level
        vol_percentile = features.get('volatility_percentile', 50)
        if vol_percentile > 85:
            vol_level = 'very_high'
        elif vol_percentile > 70:
            vol_level = 'high'
        elif vol_percentile < 15:
            vol_level = 'very_low'
        elif vol_percentile < 30:
            vol_level = 'low'
        else:
            vol_level = 'normal'
        
        # Extract support/resistance levels
        sr_levels = market_context.get('support_resistance_levels', [])
        
        return MarketConditions(
            regime=regime,
            sub_regime=market_context.get('sub_regime', 'unknown'),
            volatility_percentile=vol_percentile,
            volume_profile=market_context.get('volume_profile', 'normal'),
            session=market_context.get('session', 'UNKNOWN'),
            trend_strength=features.get('trend_strength', 0),
            support_resistance_levels=sr_levels,
            correlation_factor=features.get('correlation_factor', 0.5),
            news_impact_expected=market_context.get('news_in_next_hour', False)
        )
    
    def _optimize_spacing(
        self,
        base_params: Dict[str, float],
        features: Dict[str, float],
        conditions: MarketConditions,
        historical_data: Optional[pd.DataFrame] = None
    ) -> float:
        """Optimize grid spacing using multiple methods"""
        
        # Method 1: ATR-based spacing
        atr = features.get('atr_14', 0.001)
        atr_spacing = atr * base_params['atr_multiplier']
        
        # Method 2: Volatility-adjusted spacing
        current_vol = features.get('volatility_5m', 0.001)
        avg_vol = features.get('volatility_avg', 0.001)
        vol_ratio = current_vol / (avg_vol + 1e-8)
        vol_spacing = atr * vol_ratio * base_params['spacing_bias']
        
        # Method 3: Historical optimization
        hist_spacing = atr_spacing
        if historical_data is not None and len(historical_data) > 100:
            hist_spacing = self._optimize_spacing_historically(historical_data, atr)
        
        # Method 4: Support/Resistance aware spacing
        sr_spacing = self._calculate_sr_aware_spacing(
            atr_spacing, conditions.support_resistance_levels, features
        )
        
        # Combine methods with weights
        spacing_weights = {
            'atr': 0.3,
            'volatility': 0.3,
            'historical': 0.2,
            'sr_aware': 0.2
        }
        
        optimal_spacing = (
            spacing_weights['atr'] * atr_spacing +
            spacing_weights['volatility'] * vol_spacing +
            spacing_weights['historical'] * hist_spacing +
            spacing_weights['sr_aware'] * sr_spacing
        )
        
        # Apply session and volatility adjustments
        session_adj = self.session_multipliers.get(conditions.session, {}).get('spacing', 1.0)
        vol_level = self._get_volatility_level(conditions.volatility_percentile)
        vol_adj = self.volatility_adjustments[vol_level]['spacing_mult']
        
        optimal_spacing *= session_adj * vol_adj
        
        # Apply bounds
        min_spacing = atr * 0.3
        max_spacing = atr * 4.0
        optimal_spacing = np.clip(optimal_spacing, min_spacing, max_spacing)
        
        return optimal_spacing
    
    def _optimize_levels(
        self,
        base_params: Dict[str, float],
        account_info: Dict[str, float],
        features: Dict[str, float],
        spacing: float,
        conditions: MarketConditions
    ) -> int:
        """Optimize number of grid levels"""
        
        # Risk-based calculation
        account_size = account_info.get('balance', 10000)
        risk_tolerance = account_info.get('risk_tolerance', 0.02)
        max_exposure = account_size * risk_tolerance
        
        # Estimate position size per level
        estimated_position_size = self._estimate_position_size_per_level(account_info, features)
        
        # Calculate maximum levels based on risk
        max_risk_levels = int(max_exposure / (estimated_position_size * spacing + 1e-8))
        
        # Base levels from regime
        base_levels = base_params['base_levels']
        
        # Session adjustment
        session_adj = self.session_multipliers.get(conditions.session, {}).get('levels', 1.0)
        
        # Volatility adjustment
        vol_level = self._get_volatility_level(conditions.volatility_percentile)
        vol_adj = self.volatility_adjustments[vol_level]['level_mult']
        
        # Trend strength adjustment
        trend_adj = 1.0
        if abs(conditions.trend_strength) > 0.5:
            trend_adj = 0.8  # Fewer levels in strong trends
        elif abs(conditions.trend_strength) < 0.2:
            trend_adj = 1.2  # More levels in weak trends
        
        # Calculate optimal levels
        optimal_levels = int(base_levels * session_adj * vol_adj * trend_adj)
        
        # Apply constraints
        optimal_levels = min(optimal_levels, max_risk_levels, 50)  # Max 50 levels
        optimal_levels = max(optimal_levels, 3)  # Min 3 levels
        
        return optimal_levels
    
    def _calculate_kelly_position_size(
        self,
        account_info: Dict[str, float],
        features: Dict[str, float],
        conditions: MarketConditions,
        num_levels: int
    ) -> float:
        """Calculate position size using Kelly criterion"""
        
        # Get historical performance metrics
        win_rate = account_info.get('win_rate', 0.52)
        avg_win = account_info.get('avg_win', 1.0)
        avg_loss = abs(account_info.get('avg_loss', 0.8))
        
        # Adjust metrics based on current conditions
        adjusted_metrics = self._adjust_kelly_metrics(
            win_rate, avg_win, avg_loss, features, conditions
        )
        
        win_rate_adj, avg_win_adj, avg_loss_adj = adjusted_metrics
        
        # Kelly formula with safety adjustments
        if avg_win_adj > 0:
            kelly_fraction = (
                (win_rate_adj * avg_win_adj - (1 - win_rate_adj) * avg_loss_adj) / avg_win_adj
            )
        else:
            kelly_fraction = 0.01
        
        # Apply conservative factor
        conservative_kelly = kelly_fraction * self.kelly_config['conservative_factor']
        
        # Apply confidence adjustment
        confidence = min(account_info.get('confidence', 0.7), 1.0)
        confidence_adjusted_kelly = conservative_kelly * confidence
        
        # Apply bounds
        max_kelly = self.kelly_config['max_kelly_fraction']
        bounded_kelly = np.clip(confidence_adjusted_kelly, 0.001, max_kelly)
        
        # Calculate position size
        account_size = account_info.get('balance', 10000)
        base_position = account_size * bounded_kelly
        
        # Adjust for number of levels (smaller position per level)
        level_adjustment = np.sqrt(num_levels) / num_levels  # Diversification benefit
        position_per_level = base_position * level_adjustment
        
        # Apply regime-specific adjustments
        regime_adj = self.base_params.get(conditions.regime, {}).get('position_multiplier', 1.0)
        session_adj = self.session_multipliers.get(conditions.session, {}).get('size', 1.0)
        vol_level = self._get_volatility_level(conditions.volatility_percentile)
        vol_adj = self.volatility_adjustments[vol_level]['size_mult']
        
        final_position_size = position_per_level * regime_adj * session_adj * vol_adj
        
        # Apply absolute constraints
        max_position = account_size * self.risk_constraints['max_position_size']
        min_position = account_size * 0.0001  # 0.01% minimum
        
        return np.clip(final_position_size, min_position, max_position)
    
    def _estimate_fill_probability(
        self,
        spacing: float,
        features: Dict[str, float],
        conditions: MarketConditions,
        historical_data: Optional[pd.DataFrame] = None
    ) -> float:
        """Estimate probability of grid fills"""
        
        # Base probability from volatility and range
        atr = features.get('atr_14', 0.001)
        fill_ratio = atr / spacing
        
        # Base probability curve
        if fill_ratio > 3.0:
            base_prob = 0.95
        elif fill_ratio > 2.0:
            base_prob = 0.80
        elif fill_ratio > 1.5:
            base_prob = 0.65
        elif fill_ratio > 1.0:
            base_prob = 0.50
        elif fill_ratio > 0.5:
            base_prob = 0.35
        else:
            base_prob = 0.20
        
        # Volume adjustment
        volume_ratio = features.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            volume_adj = 1.2
        elif volume_ratio > 1.2:
            volume_adj = 1.1
        elif volume_ratio < 0.5:
            volume_adj = 0.7
        elif volume_ratio < 0.8:
            volume_adj = 0.9
        else:
            volume_adj = 1.0
        
        # Session adjustment
        session_factor = {
            'ASIAN': 0.8,
            'LONDON': 1.0,
            'US': 1.1,
            'US_OPEN': 1.3,
            'OVERLAP': 1.2,
            'OFF_HOURS': 0.6
        }.get(conditions.session, 1.0)
        
        # Historical validation if available
        hist_factor = 1.0
        if historical_data is not None and len(historical_data) > 50:
            hist_factor = self._calculate_historical_fill_rate(historical_data, spacing)
        
        # Combine factors
        final_probability = base_prob * volume_adj * session_factor * hist_factor
        
        return np.clip(final_probability, 0.05, 0.98)
    
    def _estimate_expected_profit(
        self,
        spacing: float,
        levels: int,
        position_size: float,
        features: Dict[str, float],
        conditions: MarketConditions
    ) -> float:
        """Estimate expected profit from grid setup"""
        
        # Average profit per fill
        avg_profit_per_fill = spacing * position_size * 0.5  # Conservative estimate
        
        # Fill probability
        fill_probability = self._estimate_fill_probability(spacing, features, conditions)
        
        # Expected fills per period
        volume_factor = features.get('volume_ratio', 1.0)
        session_factor = {
            'ASIAN': 0.6,
            'LONDON': 1.0,
            'US': 1.2,
            'US_OPEN': 1.5,
            'OVERLAP': 1.3,
            'OFF_HOURS': 0.3
        }.get(conditions.session, 1.0)
        
        expected_fills_per_hour = levels * fill_probability * volume_factor * session_factor * 0.1
        
        # Time horizon (assume 24 hours)
        time_horizon = 24
        
        # Expected profit
        expected_profit = avg_profit_per_fill * expected_fills_per_hour * time_horizon
        
        # Risk adjustment
        volatility_risk = features.get('volatility_5m', 0.001) / 0.002  # Normalize to 0.2% base
        risk_factor = max(0.3, 1.0 - volatility_risk * 0.3)
        
        return expected_profit * risk_factor
    
    def _estimate_drawdown_risk(
        self,
        spacing: float,
        levels: int,
        position_size: float,
        features: Dict[str, float]
    ) -> float:
        """Estimate maximum drawdown risk"""
        
        # Worst case: all levels filled on one side and market moves against
        max_adverse_move = features.get('atr_14', 0.001) * 3  # 3x ATR adverse move
        
        # Total exposure if all levels filled
        total_exposure = levels * position_size
        
        # Drawdown calculation
        adverse_pnl = total_exposure * max_adverse_move
        
        # Grid recovery potential
        recovery_potential = (levels * spacing * position_size) * 0.5
        
        # Net drawdown
        net_drawdown = adverse_pnl - recovery_potential
        
        # As percentage of total grid investment
        total_investment = levels * position_size * spacing
        drawdown_percentage = net_drawdown / (total_investment + 1e-8)
        
        return max(0, drawdown_percentage)
    
    def _calculate_optimal_entry_zones(
        self,
        spacing: float,
        levels: int,
        features: Dict[str, float],
        conditions: MarketConditions
    ) -> List[float]:
        """Calculate optimal price zones for grid entries"""
        
        current_price = features.get('current_price', 100.0)
        
        # Center the grid around current price with bias
        trend_bias = 0
        if abs(conditions.trend_strength) > 0.3:
            # Bias grid placement based on trend
            trend_bias = spacing * conditions.trend_strength * 2
        
        # Support/resistance aware placement
        sr_adjustment = 0
        if conditions.support_resistance_levels:
            sr_levels = np.array(conditions.support_resistance_levels)
            # Find nearest SR level
            nearest_sr = sr_levels[np.argmin(np.abs(sr_levels - current_price))]
            sr_adjustment = (nearest_sr - current_price) * 0.3
        
        # Calculate grid center
        grid_center = current_price + trend_bias + sr_adjustment
        
        # Generate entry zones
        entry_zones = []
        half_levels = levels // 2
        
        for i in range(-half_levels, levels - half_levels):
            zone_price = grid_center + (i * spacing)
            entry_zones.append(zone_price)
        
        return sorted(entry_zones)
    
    def _calculate_exit_levels(
        self,
        spacing: float,
        levels: int,
        features: Dict[str, float],
        conditions: MarketConditions
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels"""
        
        current_price = features.get('current_price', 100.0)
        atr = features.get('atr_14', 0.001)
        
        # Stop loss calculation
        stop_multiplier = {
            'TRENDING': 2.5,
            'RANGING': 1.5,
            'VOLATILE': 3.0,
            'UNCERTAIN': 2.0
        }.get(conditions.regime, 2.0)
        
        stop_distance = atr * stop_multiplier
        
        # Take profit calculation
        tp_multiplier = {
            'TRENDING': 4.0,
            'RANGING': 2.0,
            'VOLATILE': 3.5,
            'UNCERTAIN': 3.0
        }.get(conditions.regime, 3.0)
        
        tp_distance = atr * tp_multiplier
        
        # Direction bias
        if conditions.trend_strength > 0.3:
            # Uptrend bias
            stop_loss = current_price - stop_distance
            take_profit = current_price + tp_distance
        elif conditions.trend_strength < -0.3:
            # Downtrend bias
            stop_loss = current_price + stop_distance
            take_profit = current_price - tp_distance
        else:
            # No clear direction - use grid boundaries
            stop_loss = current_price - (levels * spacing * 0.6)
            take_profit = current_price + (levels * spacing * 0.6)
        
        return stop_loss, take_profit
    
    def _optimize_spacing_historically(
        self, 
        historical_data: pd.DataFrame, 
        base_atr: float
    ) -> float:
        """Optimize spacing based on historical performance"""
        
        if len(historical_data) < 100:
            return base_atr
        
        # Test different spacing multiples
        test_multiples = np.arange(0.5, 3.0, 0.1)
        best_multiple = 1.0
        best_score = 0
        
        returns = historical_data['close'].pct_change().dropna()
        
        for multiple in test_multiples:
            test_spacing = base_atr * multiple
            score = self._score_spacing_performance(returns, test_spacing)
            
            if score > best_score:
                best_score = score
                best_multiple = multiple
        
        return base_atr * best_multiple
    
    def _score_spacing_performance(self, returns: pd.Series, spacing: float) -> float:
        """Score spacing performance based on historical data"""
        
        # Simulate grid fills
        price_moves = returns.cumsum()
        fills = 0
        profit = 0
        
        current_level = 0
        for move in price_moves:
            level = int(move / spacing)
            if level != current_level:
                fills += abs(level - current_level)
                profit += abs(level - current_level) * spacing * 0.5
                current_level = level
        
        # Score based on fills and profit
        if fills == 0:
            return 0
        
        avg_profit_per_fill = profit / fills
        fill_frequency = fills / len(returns)
        
        return avg_profit_per_fill * fill_frequency
    
    def _calculate_sr_aware_spacing(
        self, 
        base_spacing: float, 
        sr_levels: List[float], 
        features: Dict[str, float]
    ) -> float:
        """Adjust spacing based on support/resistance levels"""
        
        if not sr_levels or len(sr_levels) < 2:
            return base_spacing
        
        current_price = features.get('current_price', 100.0)
        
        # Find relevant SR levels (within reasonable range)
        relevant_levels = [
            level for level in sr_levels 
            if abs(level - current_price) / current_price < 0.05  # Within 5%
        ]
        
        if len(relevant_levels) < 2:
            return base_spacing
        
        # Calculate average distance between levels
        level_distances = []
        relevant_levels.sort()
        
        for i in range(len(relevant_levels) - 1):
            distance = relevant_levels[i + 1] - relevant_levels[i]
            level_distances.append(distance)
        
        if level_distances:
            avg_sr_distance = np.mean(level_distances)
            # Use 60% of SR distance as spacing
            sr_spacing = avg_sr_distance * 0.6
            
            # Blend with base spacing
            return (base_spacing + sr_spacing) / 2
        
        return base_spacing
    
    def _calculate_historical_fill_rate(
        self, 
        historical_data: pd.DataFrame, 
        spacing: float
    ) -> float:
        """Calculate historical fill rate for given spacing"""
        
        if len(historical_data) < 50:
            return 1.0
        
        # Simulate fills
        price_moves = historical_data['close'].pct_change().cumsum()
        fills = 0
        total_opportunities = 0
        
        for i in range(1, len(price_moves)):
            move_size = abs(price_moves.iloc[i] - price_moves.iloc[i-1])
            opportunities = int(move_size / spacing)
            fills += opportunities
            total_opportunities += max(opportunities, 1)
        
        if total_opportunities > 0:
            return min(fills / total_opportunities, 1.0)
        
        return 0.5
    
    def _adjust_kelly_metrics(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        features: Dict[str, float],
        conditions: MarketConditions
    ) -> Tuple[float, float, float]:
        """Adjust Kelly metrics based on current market conditions"""
        
        # Volatility adjustment
        vol_ratio = features.get('volatility_5m', 0.001) / features.get('volatility_avg', 0.001)
        if vol_ratio > 1.5:
            # Higher volatility = lower win rate, higher avg win/loss
            win_rate_adj = win_rate * 0.9
            avg_win_adj = avg_win * 1.2
            avg_loss_adj = avg_loss * 1.2
        elif vol_ratio < 0.7:
            # Lower volatility = higher win rate, lower avg win/loss
            win_rate_adj = min(win_rate * 1.1, 0.95)
            avg_win_adj = avg_win * 0.8
            avg_loss_adj = avg_loss * 0.8
        else:
            win_rate_adj = win_rate
            avg_win_adj = avg_win
            avg_loss_adj = avg_loss
        
        # Regime adjustment
        if conditions.regime == 'TRENDING':
            # Trending markets may have lower win rate but higher rewards
            win_rate_adj *= 0.95
            avg_win_adj *= 1.3
        elif conditions.regime == 'RANGING':
            # Ranging markets may have higher win rate but lower rewards
            win_rate_adj *= 1.05
            avg_win_adj *= 0.9
        elif conditions.regime == 'VOLATILE':
            # Volatile markets are unpredictable
            win_rate_adj *= 0.9
            avg_loss_adj *= 1.2
        
        # Bounds
        win_rate_adj = np.clip(win_rate_adj, 0.2, 0.95)
        avg_win_adj = max(avg_win_adj, 0.1)
        avg_loss_adj = max(avg_loss_adj, 0.1)
        
        return win_rate_adj, avg_win_adj, avg_loss_adj
    
    def _get_volatility_level(self, percentile: float) -> str:
        """Convert volatility percentile to level"""
        if percentile > 85:
            return 'very_high'
        elif percentile > 70:
            return 'high'
        elif percentile < 15:
            return 'very_low'
        elif percentile < 30:
            return 'low'
        else:
            return 'normal'
    
    def _estimate_position_size_per_level(
        self, 
        account_info: Dict[str, float], 
        features: Dict[str, float]
    ) -> float:
        """Estimate position size per grid level"""
        account_size = account_info.get('balance', 10000)
        return account_size * 0.001  # 0.1% per level as base estimate
    
    def _calculate_spacing_multiplier(
        self, 
        features: Dict[str, float], 
        conditions: MarketConditions
    ) -> float:
        """Calculate spacing multiplier for dynamic adjustment"""
        
        # Base multiplier
        multiplier = 1.0
        
        # Volatility adjustment
        vol_level = self._get_volatility_level(conditions.volatility_percentile)
        multiplier *= self.volatility_adjustments[vol_level]['spacing_mult']
        
        # Session adjustment
        session_mult = self.session_multipliers.get(conditions.session, {}).get('spacing', 1.0)
        multiplier *= session_mult
        
        # News adjustment
        if conditions.news_impact_expected:
            multiplier *= 1.3  # Wider spacing before news
        
        return multiplier
    
    def _calculate_risk_adjustment(
        self, 
        features: Dict[str, float], 
        conditions: MarketConditions
    ) -> float:
        """Calculate risk adjustment factor"""
        
        risk_factors = []
        
        # Volatility risk
        vol_risk = min(features.get('volatility_5m', 0.001) / 0.003, 1.0)
        risk_factors.append(vol_risk)
        
        # Correlation risk
        corr_risk = abs(conditions.correlation_factor - 0.5) * 2  # Higher when correlation extreme
        risk_factors.append(corr_risk)
        
        # Trend exhaustion risk
        if abs(conditions.trend_strength) > 0.8:
            risk_factors.append(0.8)  # High risk when trend very strong
        
        # Average risk
        avg_risk = np.mean(risk_factors) if risk_factors else 0.5
        
        # Convert to adjustment factor (lower risk = higher adjustment)
        return 1.0 - avg_risk * 0.5
    
    def _validate_and_adjust_parameters(
        self,
        params: OptimizedGridParams,
        account_info: Dict[str, float],
        conditions: MarketConditions
    ) -> OptimizedGridParams:
        """Validate and adjust parameters against constraints"""
        
        account_size = account_info.get('balance', 10000)
        
        # Check position size constraint
        max_position = account_size * self.risk_constraints['max_position_size']
        if params.position_size > max_position:
            params.position_size = max_position
        
        # Check total exposure constraint
        total_exposure = params.position_size * params.levels
        max_exposure = account_size * self.risk_constraints['max_total_exposure']
        if total_exposure > max_exposure:
            # Reduce levels to fit exposure
            params.levels = int(max_exposure / params.position_size)
            params.levels = max(params.levels, 3)  # Minimum levels
        
        # Check drawdown constraint
        if params.max_drawdown_risk > self.risk_constraints['max_drawdown_single_grid']:
            # Reduce position size to limit drawdown
            adjustment_factor = self.risk_constraints['max_drawdown_single_grid'] / params.max_drawdown_risk
            params.position_size *= adjustment_factor
            params.max_drawdown_risk *= adjustment_factor
        
        # Recalculate dependent values
        params.expected_profit *= (params.position_size / max(params.position_size * 2, 0.001))
        
        return params
    
    def _store_optimization_result(
        self,
        params: OptimizedGridParams,
        features: Dict[str, float],
        conditions: MarketConditions
    ) -> None:
        """Store optimization result for learning"""
        
        result = {
            'timestamp': pd.Timestamp.now(),
            'regime': conditions.regime,
            'spacing': params.spacing,
            'levels': params.levels,
            'position_size': params.position_size,
            'expected_profit': params.expected_profit,
            'fill_probability': params.fill_probability,
            'volatility_percentile': conditions.volatility_percentile,
            'trend_strength': conditions.trend_strength
        }
        
        self.performance_history.append(result)
    
    def _get_fallback_parameters(
        self,
        regime: str,
        features: Dict[str, float],
        account_info: Dict[str, float]
    ) -> OptimizedGridParams:
        """Get safe fallback parameters when optimization fails"""
        
        atr = features.get('atr_14', 0.001)
        account_size = account_info.get('balance', 10000)
        
        return OptimizedGridParams(
            spacing=atr * 1.5,
            levels=10,
            position_size=account_size * 0.001,
            spacing_multiplier=1.0,
            risk_adjustment=0.8,
            fill_probability=0.5,
            expected_profit=account_size * 0.01,
            max_drawdown_risk=0.02,
            optimal_entry_zones=[],
            stop_loss_level=None,
            take_profit_level=None
        )
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get statistics about optimization performance"""
        
        if not self.performance_history:
            return {}
        
        df = pd.DataFrame(self.performance_history)
        
        return {
            'total_optimizations': len(df),
            'avg_expected_profit': df['expected_profit'].mean(),
            'avg_fill_probability': df['fill_probability'].mean(),
            'regime_distribution': df['regime'].value_counts(normalize=True).to_dict(),
            'parameter_ranges': {
                'spacing': {'min': df['spacing'].min(), 'max': df['spacing'].max()},
                'levels': {'min': df['levels'].min(), 'max': df['levels'].max()},
                'position_size': {'min': df['position_size'].min(), 'max': df['position_size'].max()}
            }
        }