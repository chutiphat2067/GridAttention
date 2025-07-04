"""
assertion_helpers.py
Comprehensive assertion utilities for GridAttention trading system tests
Provides specialized assertions for trading system components

Author: GridAttention Test Suite
Date: 2024
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import pytest
from unittest.mock import Mock
import warnings


# ============= Type Definitions =============
class MarketRegime(Enum):
    """Market regime types for testing"""
    RANGING = "ranging"
    TRENDING = "trending"
    VOLATILE = "volatile"
    DORMANT = "dormant"


class AttentionPhase(Enum):
    """Attention system phases"""
    LEARNING = "learning"
    SHADOW = "shadow"
    ACTIVE = "active"


class OrderStatus(Enum):
    """Order status types"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class PriceRange:
    """Price range for assertions"""
    min_price: float
    max_price: float
    
    def contains(self, price: float) -> bool:
        return self.min_price <= price <= self.max_price


# ============= Base Assertion Helpers =============
class AssertionHelpers:
    """Base class for common assertion utilities"""
    
    @staticmethod
    def assert_within_tolerance(
        actual: float, 
        expected: float, 
        tolerance: float = 0.01,
        message: str = ""
    ) -> None:
        """Assert value is within tolerance"""
        diff = abs(actual - expected)
        assert diff <= tolerance, (
            f"Value {actual} not within {tolerance} of {expected}. "
            f"Difference: {diff}. {message}"
        )
    
    @staticmethod
    def assert_percentage_within(
        actual: float,
        expected: float,
        percentage: float = 1.0,
        message: str = ""
    ) -> None:
        """Assert value is within percentage of expected"""
        if expected == 0:
            assert actual == 0, f"Expected 0 but got {actual}. {message}"
            return
            
        diff_percentage = abs((actual - expected) / expected) * 100
        assert diff_percentage <= percentage, (
            f"Value {actual} not within {percentage}% of {expected}. "
            f"Difference: {diff_percentage:.2f}%. {message}"
        )
    
    @staticmethod
    def assert_monotonic_increasing(
        values: List[float],
        strict: bool = False,
        message: str = ""
    ) -> None:
        """Assert values are monotonically increasing"""
        for i in range(1, len(values)):
            if strict:
                assert values[i] > values[i-1], (
                    f"Values not strictly increasing at index {i}: "
                    f"{values[i-1]} >= {values[i]}. {message}"
                )
            else:
                assert values[i] >= values[i-1], (
                    f"Values not monotonically increasing at index {i}: "
                    f"{values[i-1]} > {values[i]}. {message}"
                )
    
    @staticmethod
    def assert_bounded(
        value: float,
        min_val: float,
        max_val: float,
        inclusive: bool = True,
        message: str = ""
    ) -> None:
        """Assert value is within bounds"""
        if inclusive:
            assert min_val <= value <= max_val, (
                f"Value {value} not in range [{min_val}, {max_val}]. {message}"
            )
        else:
            assert min_val < value < max_val, (
                f"Value {value} not in range ({min_val}, {max_val}). {message}"
            )


# ============= Trading-Specific Assertions =============
class TradingAssertions(AssertionHelpers):
    """Trading-specific assertion utilities"""
    
    @staticmethod
    def assert_valid_price(
        price: float,
        symbol: str = "BTCUSDT",
        message: str = ""
    ) -> None:
        """Assert price is valid for trading"""
        assert price > 0, f"Price must be positive: {price}. {message}"
        
        # Symbol-specific validations
        if symbol == "BTCUSDT":
            assert 1000 < price < 1000000, (
                f"BTC price {price} outside reasonable range. {message}"
            )
    
    @staticmethod
    def assert_valid_volume(
        volume: float,
        min_volume: float = 0.0001,
        max_volume: float = 10000,
        message: str = ""
    ) -> None:
        """Assert volume is valid for trading"""
        assert volume > 0, f"Volume must be positive: {volume}. {message}"
        assert min_volume <= volume <= max_volume, (
            f"Volume {volume} outside valid range [{min_volume}, {max_volume}]. {message}"
        )
    
    @staticmethod
    def assert_valid_order(
        order: Dict[str, Any],
        required_fields: Optional[List[str]] = None,
        message: str = ""
    ) -> None:
        """Assert order structure is valid"""
        if required_fields is None:
            required_fields = ['symbol', 'side', 'type', 'quantity', 'price']
        
        for field in required_fields:
            assert field in order, f"Order missing required field '{field}'. {message}"
        
        # Validate specific fields
        assert order.get('side') in ['buy', 'sell'], (
            f"Invalid order side: {order.get('side')}. {message}"
        )
        assert order.get('type') in ['limit', 'market', 'stop', 'stop_limit'], (
            f"Invalid order type: {order.get('type')}. {message}"
        )
        
        if 'quantity' in order:
            TradingAssertions.assert_valid_volume(
                order['quantity'], 
                message=f"Order quantity invalid. {message}"
            )
        
        if 'price' in order and order['type'] in ['limit', 'stop_limit']:
            TradingAssertions.assert_valid_price(
                order['price'],
                order.get('symbol', 'BTCUSDT'),
                message=f"Order price invalid. {message}"
            )
    
    @staticmethod
    def assert_position_limits(
        position_size: float,
        max_position: float,
        current_exposure: float = 0,
        max_exposure: float = 1.0,
        message: str = ""
    ) -> None:
        """Assert position respects risk limits"""
        assert abs(position_size) <= max_position, (
            f"Position size {position_size} exceeds limit {max_position}. {message}"
        )
        
        total_exposure = abs(current_exposure) + abs(position_size)
        assert total_exposure <= max_exposure, (
            f"Total exposure {total_exposure} exceeds limit {max_exposure}. {message}"
        )


# ============= Attention System Assertions =============
class AttentionAssertions(AssertionHelpers):
    """Attention system specific assertions"""
    
    @staticmethod
    def assert_valid_attention_weights(
        weights: Dict[str, float],
        tolerance: float = 1e-6,
        message: str = ""
    ) -> None:
        """Assert attention weights are valid"""
        # Check all weights are non-negative
        for feature, weight in weights.items():
            assert weight >= 0, (
                f"Negative attention weight for {feature}: {weight}. {message}"
            )
        
        # Check weights sum to 1 (normalized)
        total = sum(weights.values())
        assert abs(total - 1.0) < tolerance, (
            f"Attention weights sum to {total}, not 1.0. {message}"
        )
    
    @staticmethod
    def assert_phase_transition_valid(
        from_phase: AttentionPhase,
        to_phase: AttentionPhase,
        observation_count: int,
        message: str = ""
    ) -> None:
        """Assert attention phase transition is valid"""
        valid_transitions = {
            AttentionPhase.LEARNING: [AttentionPhase.SHADOW],
            AttentionPhase.SHADOW: [AttentionPhase.ACTIVE, AttentionPhase.LEARNING],
            AttentionPhase.ACTIVE: [AttentionPhase.SHADOW, AttentionPhase.LEARNING]
        }
        
        assert to_phase in valid_transitions.get(from_phase, []), (
            f"Invalid phase transition from {from_phase} to {to_phase}. {message}"
        )
        
        # Check observation requirements
        if from_phase == AttentionPhase.LEARNING and to_phase == AttentionPhase.SHADOW:
            assert observation_count >= 1000, (
                f"Insufficient observations ({observation_count}) for LEARNING→SHADOW. {message}"
            )
        elif from_phase == AttentionPhase.SHADOW and to_phase == AttentionPhase.ACTIVE:
            assert observation_count >= 200, (
                f"Insufficient observations ({observation_count}) for SHADOW→ACTIVE. {message}"
            )
    
    @staticmethod
    def assert_feature_importance_valid(
        importance: Dict[str, float],
        expected_features: Optional[List[str]] = None,
        message: str = ""
    ) -> None:
        """Assert feature importance scores are valid"""
        if expected_features:
            actual_features = set(importance.keys())
            expected_set = set(expected_features)
            assert actual_features == expected_set, (
                f"Feature mismatch. Expected: {expected_set}, Got: {actual_features}. {message}"
            )
        
        # Check importance scores are in valid range
        for feature, score in importance.items():
            assert 0 <= score <= 1, (
                f"Invalid importance score for {feature}: {score}. {message}"
            )


# ============= Market Regime Assertions =============
class RegimeAssertions(AssertionHelpers):
    """Market regime detection assertions"""
    
    @staticmethod
    def assert_valid_regime(
        regime: Union[str, MarketRegime],
        valid_regimes: Optional[List[MarketRegime]] = None,
        message: str = ""
    ) -> None:
        """Assert regime is valid"""
        if valid_regimes is None:
            valid_regimes = list(MarketRegime)
        
        if isinstance(regime, str):
            regime_enum = None
            for r in MarketRegime:
                if r.value == regime:
                    regime_enum = r
                    break
            assert regime_enum is not None, (
                f"Invalid regime string: {regime}. {message}"
            )
            regime = regime_enum
        
        assert regime in valid_regimes, (
            f"Regime {regime} not in valid regimes: {valid_regimes}. {message}"
        )
    
    @staticmethod
    def assert_regime_confidence(
        confidence: float,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
        message: str = ""
    ) -> None:
        """Assert regime confidence is valid"""
        assert min_confidence <= confidence <= max_confidence, (
            f"Confidence {confidence} outside range [{min_confidence}, {max_confidence}]. {message}"
        )
    
    @staticmethod
    def assert_regime_transition_valid(
        from_regime: MarketRegime,
        to_regime: MarketRegime,
        market_data: Dict[str, float],
        message: str = ""
    ) -> None:
        """Assert regime transition makes sense given market data"""
        volatility = market_data.get('volatility_5m', 0)
        trend = market_data.get('trend_strength', 0)
        
        # Example validation rules
        if from_regime == MarketRegime.RANGING and to_regime == MarketRegime.TRENDING:
            assert abs(trend) > 0.3, (
                f"Insufficient trend strength ({trend}) for RANGING→TRENDING. {message}"
            )
        
        if from_regime == MarketRegime.TRENDING and to_regime == MarketRegime.VOLATILE:
            assert volatility > 0.002, (
                f"Insufficient volatility ({volatility}) for TRENDING→VOLATILE. {message}"
            )


# ============= Grid Strategy Assertions =============
class GridAssertions(AssertionHelpers):
    """Grid trading strategy assertions"""
    
    @staticmethod
    def assert_grid_spacing_valid(
        levels: List[float],
        min_spacing_pct: float = 0.1,
        max_spacing_pct: float = 5.0,
        message: str = ""
    ) -> None:
        """Assert grid levels have valid spacing"""
        if len(levels) < 2:
            return
        
        sorted_levels = sorted(levels)
        for i in range(1, len(sorted_levels)):
            spacing_pct = ((sorted_levels[i] - sorted_levels[i-1]) / sorted_levels[i-1]) * 100
            assert min_spacing_pct <= spacing_pct <= max_spacing_pct, (
                f"Grid spacing {spacing_pct:.2f}% outside range "
                f"[{min_spacing_pct}, {max_spacing_pct}]% at level {i}. {message}"
            )
    
    @staticmethod
    def assert_grid_symmetry(
        buy_levels: List[float],
        sell_levels: List[float],
        center_price: float,
        tolerance_pct: float = 1.0,
        message: str = ""
    ) -> None:
        """Assert grid is symmetric around center price"""
        assert len(buy_levels) == len(sell_levels), (
            f"Asymmetric grid: {len(buy_levels)} buy levels vs "
            f"{len(sell_levels)} sell levels. {message}"
        )
        
        # Check distance from center
        for buy, sell in zip(sorted(buy_levels, reverse=True), sorted(sell_levels)):
            buy_dist = center_price - buy
            sell_dist = sell - center_price
            
            AssertionHelpers.assert_percentage_within(
                buy_dist, sell_dist, tolerance_pct,
                f"Grid asymmetry detected. {message}"
            )
    
    @staticmethod
    def assert_grid_coverage(
        grid_levels: List[float],
        current_price: float,
        min_coverage_pct: float = 2.0,
        max_coverage_pct: float = 20.0,
        message: str = ""
    ) -> None:
        """Assert grid covers appropriate price range"""
        if not grid_levels:
            return
        
        min_level = min(grid_levels)
        max_level = max(grid_levels)
        
        coverage_down = ((current_price - min_level) / current_price) * 100
        coverage_up = ((max_level - current_price) / current_price) * 100
        
        assert coverage_down >= min_coverage_pct, (
            f"Insufficient downside coverage: {coverage_down:.2f}% < {min_coverage_pct}%. {message}"
        )
        assert coverage_up >= min_coverage_pct, (
            f"Insufficient upside coverage: {coverage_up:.2f}% < {min_coverage_pct}%. {message}"
        )
        
        total_coverage = coverage_down + coverage_up
        assert total_coverage <= max_coverage_pct, (
            f"Excessive grid coverage: {total_coverage:.2f}% > {max_coverage_pct}%. {message}"
        )


# ============= Risk Management Assertions =============
class RiskAssertions(AssertionHelpers):
    """Risk management system assertions"""
    
    @staticmethod
    def assert_position_within_limits(
        position: Dict[str, Any],
        limits: Dict[str, float],
        message: str = ""
    ) -> None:
        """Assert position respects all risk limits"""
        # Position size limit
        if 'max_position_size' in limits:
            assert abs(position.get('size', 0)) <= limits['max_position_size'], (
                f"Position size {position.get('size')} exceeds limit "
                f"{limits['max_position_size']}. {message}"
            )
        
        # Exposure limit
        if 'max_exposure' in limits:
            exposure = abs(position.get('size', 0) * position.get('entry_price', 0))
            assert exposure <= limits['max_exposure'], (
                f"Position exposure {exposure} exceeds limit "
                f"{limits['max_exposure']}. {message}"
            )
        
        # Leverage limit
        if 'max_leverage' in limits and 'leverage' in position:
            assert position['leverage'] <= limits['max_leverage'], (
                f"Leverage {position['leverage']} exceeds limit "
                f"{limits['max_leverage']}. {message}"
            )
    
    @staticmethod
    def assert_drawdown_acceptable(
        current_equity: float,
        peak_equity: float,
        max_drawdown_pct: float = 20.0,
        message: str = ""
    ) -> None:
        """Assert drawdown is within acceptable limits"""
        if peak_equity <= 0:
            return
        
        drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100
        assert drawdown_pct <= max_drawdown_pct, (
            f"Drawdown {drawdown_pct:.2f}% exceeds limit {max_drawdown_pct}%. "
            f"Peak: {peak_equity}, Current: {current_equity}. {message}"
        )
    
    @staticmethod
    def assert_stop_loss_valid(
        entry_price: float,
        stop_loss: float,
        side: str,
        min_distance_pct: float = 0.1,
        max_distance_pct: float = 10.0,
        message: str = ""
    ) -> None:
        """Assert stop loss is properly set"""
        if side == 'buy':
            assert stop_loss < entry_price, (
                f"Buy stop loss {stop_loss} must be below entry {entry_price}. {message}"
            )
            distance_pct = ((entry_price - stop_loss) / entry_price) * 100
        else:  # sell
            assert stop_loss > entry_price, (
                f"Sell stop loss {stop_loss} must be above entry {entry_price}. {message}"
            )
            distance_pct = ((stop_loss - entry_price) / entry_price) * 100
        
        assert min_distance_pct <= distance_pct <= max_distance_pct, (
            f"Stop loss distance {distance_pct:.2f}% outside range "
            f"[{min_distance_pct}, {max_distance_pct}]%. {message}"
        )


# ============= Performance Assertions =============
class PerformanceAssertions(AssertionHelpers):
    """Performance monitoring assertions"""
    
    @staticmethod
    def assert_latency_acceptable(
        latency_ms: float,
        component: str,
        max_latency_ms: float = 100.0,
        warning_threshold_ms: float = 50.0,
        message: str = ""
    ) -> None:
        """Assert component latency is acceptable"""
        assert latency_ms >= 0, (
            f"Invalid negative latency: {latency_ms}ms. {message}"
        )
        
        if latency_ms > warning_threshold_ms:
            warnings.warn(
                f"{component} latency {latency_ms}ms exceeds warning threshold "
                f"{warning_threshold_ms}ms"
            )
        
        assert latency_ms <= max_latency_ms, (
            f"{component} latency {latency_ms}ms exceeds limit {max_latency_ms}ms. {message}"
        )
    
    @staticmethod
    def assert_memory_usage_acceptable(
        memory_mb: float,
        component: str,
        max_memory_mb: float = 1000.0,
        warning_threshold_mb: float = 800.0,
        message: str = ""
    ) -> None:
        """Assert memory usage is acceptable"""
        assert memory_mb >= 0, (
            f"Invalid negative memory usage: {memory_mb}MB. {message}"
        )
        
        if memory_mb > warning_threshold_mb:
            warnings.warn(
                f"{component} memory usage {memory_mb}MB exceeds warning threshold "
                f"{warning_threshold_mb}MB"
            )
        
        assert memory_mb <= max_memory_mb, (
            f"{component} memory usage {memory_mb}MB exceeds limit {max_memory_mb}MB. {message}"
        )
    
    @staticmethod
    def assert_metrics_consistency(
        metrics: Dict[str, Any],
        required_metrics: Optional[List[str]] = None,
        message: str = ""
    ) -> None:
        """Assert performance metrics are consistent"""
        if required_metrics is None:
            required_metrics = ['total_trades', 'win_rate', 'profit_factor', 'sharpe_ratio']
        
        # Check required metrics exist
        for metric in required_metrics:
            assert metric in metrics, (
                f"Missing required metric: {metric}. {message}"
            )
        
        # Validate metric ranges
        if 'win_rate' in metrics:
            assert 0 <= metrics['win_rate'] <= 1, (
                f"Invalid win rate: {metrics['win_rate']}. {message}"
            )
        
        if 'sharpe_ratio' in metrics:
            assert -5 <= metrics['sharpe_ratio'] <= 10, (
                f"Unrealistic Sharpe ratio: {metrics['sharpe_ratio']}. {message}"
            )
        
        if 'total_trades' in metrics and 'winning_trades' in metrics:
            assert metrics['winning_trades'] <= metrics['total_trades'], (
                f"Winning trades ({metrics['winning_trades']}) exceeds total trades "
                f"({metrics['total_trades']}). {message}"
            )


# ============= Async Assertions =============
class AsyncAssertions:
    """Assertions for async operations"""
    
    @staticmethod
    async def assert_completes_within(
        coro: Callable,
        timeout_seconds: float,
        message: str = ""
    ) -> Any:
        """Assert async operation completes within timeout"""
        try:
            result = await asyncio.wait_for(coro(), timeout=timeout_seconds)
            return result
        except asyncio.TimeoutError:
            raise AssertionError(
                f"Operation did not complete within {timeout_seconds}s. {message}"
            )
    
    @staticmethod
    async def assert_eventually_true(
        condition_func: Callable[[], bool],
        timeout_seconds: float = 5.0,
        check_interval: float = 0.1,
        message: str = ""
    ) -> None:
        """Assert condition becomes true eventually"""
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            if condition_func():
                return
            await asyncio.sleep(check_interval)
        
        raise AssertionError(
            f"Condition did not become true within {timeout_seconds}s. {message}"
        )
    
    @staticmethod
    async def assert_event_emitted(
        event_bus: Mock,
        event_type: str,
        timeout_seconds: float = 1.0,
        expected_data: Optional[Dict[str, Any]] = None,
        message: str = ""
    ) -> None:
        """Assert specific event is emitted"""
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            calls = event_bus.emit.call_args_list
            for call in calls:
                if call[0][0] == event_type:
                    if expected_data:
                        actual_data = call[0][1]
                        for key, value in expected_data.items():
                            assert key in actual_data, (
                                f"Event missing expected key: {key}. {message}"
                            )
                            assert actual_data[key] == value, (
                                f"Event data mismatch for {key}. "
                                f"Expected: {value}, Got: {actual_data[key]}. {message}"
                            )
                    return
            await asyncio.sleep(0.1)
        
        raise AssertionError(
            f"Event '{event_type}' not emitted within {timeout_seconds}s. {message}"
        )


# ============= Data Quality Assertions =============
class DataAssertions(AssertionHelpers):
    """Data quality and validation assertions"""
    
    @staticmethod
    def assert_no_missing_values(
        data: Dict[str, Any],
        required_fields: List[str],
        message: str = ""
    ) -> None:
        """Assert no missing values in required fields"""
        for field in required_fields:
            assert field in data, (
                f"Missing required field: {field}. {message}"
            )
            assert data[field] is not None, (
                f"Null value for required field: {field}. {message}"
            )
    
    @staticmethod
    def assert_data_freshness(
        timestamp: float,
        max_age_seconds: float = 60.0,
        message: str = ""
    ) -> None:
        """Assert data is fresh enough"""
        age = time.time() - timestamp
        assert age <= max_age_seconds, (
            f"Data is {age:.1f}s old, exceeds max age {max_age_seconds}s. {message}"
        )
    
    @staticmethod
    def assert_data_sequence_valid(
        timestamps: List[float],
        max_gap_seconds: float = 300.0,
        message: str = ""
    ) -> None:
        """Assert data sequence has no large gaps"""
        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i-1]
            assert gap <= max_gap_seconds, (
                f"Data gap of {gap:.1f}s exceeds max {max_gap_seconds}s "
                f"at index {i}. {message}"
            )


# ============= Overfitting Detection Assertions =============
class OverfittingAssertions(AssertionHelpers):
    """Overfitting detection and prevention assertions"""
    
    @staticmethod
    def assert_no_overfitting(
        train_performance: float,
        test_performance: float,
        max_gap: float = 0.1,
        message: str = ""
    ) -> None:
        """Assert model is not overfitting"""
        gap = train_performance - test_performance
        assert gap <= max_gap, (
            f"Overfitting detected: train ({train_performance:.3f}) - "
            f"test ({test_performance:.3f}) = {gap:.3f} > {max_gap}. {message}"
        )
    
    @staticmethod
    def assert_stability_over_time(
        performance_history: List[float],
        max_variance: float = 0.05,
        window_size: int = 10,
        message: str = ""
    ) -> None:
        """Assert performance is stable over time"""
        if len(performance_history) < window_size:
            return
        
        # Calculate rolling variance
        for i in range(window_size, len(performance_history)):
            window = performance_history[i-window_size:i]
            variance = np.var(window)
            assert variance <= max_variance, (
                f"Performance variance {variance:.4f} exceeds limit {max_variance} "
                f"at position {i}. {message}"
            )
    
    @staticmethod
    def assert_feature_stability(
        feature_importance_history: List[Dict[str, float]],
        max_change_rate: float = 0.3,
        message: str = ""
    ) -> None:
        """Assert feature importance is stable"""
        if len(feature_importance_history) < 2:
            return
        
        for i in range(1, len(feature_importance_history)):
            prev_importance = feature_importance_history[i-1]
            curr_importance = feature_importance_history[i]
            
            for feature in prev_importance:
                if feature in curr_importance:
                    prev_val = prev_importance[feature]
                    curr_val = curr_importance[feature]
                    
                    if prev_val > 0:
                        change_rate = abs(curr_val - prev_val) / prev_val
                        assert change_rate <= max_change_rate, (
                            f"Feature '{feature}' importance changed by {change_rate:.2%} "
                            f"(exceeds {max_change_rate:.2%}) at step {i}. {message}"
                        )


# ============= Integration Test Assertions =============
class IntegrationAssertions:
    """Assertions for integration testing"""
    
    @staticmethod
    def assert_components_connected(
        components: Dict[str, Any],
        expected_connections: List[Tuple[str, str]],
        message: str = ""
    ) -> None:
        """Assert components are properly connected"""
        for source, target in expected_connections:
            assert source in components, (
                f"Source component '{source}' not found. {message}"
            )
            assert target in components, (
                f"Target component '{target}' not found. {message}"
            )
            
            # Check if source has reference to target
            source_comp = components[source]
            has_connection = (
                hasattr(source_comp, target) or
                hasattr(source_comp, f'{target}_ref') or
                (hasattr(source_comp, 'connections') and target in source_comp.connections)
            )
            
            assert has_connection, (
                f"No connection found from '{source}' to '{target}'. {message}"
            )
    
    @staticmethod
    def assert_event_flow_valid(
        event_sequence: List[Dict[str, Any]],
        expected_pattern: List[str],
        message: str = ""
    ) -> None:
        """Assert events follow expected pattern"""
        event_types = [event['type'] for event in event_sequence]
        
        # Check if expected pattern is a subsequence
        pattern_idx = 0
        for event_type in event_types:
            if pattern_idx < len(expected_pattern) and event_type == expected_pattern[pattern_idx]:
                pattern_idx += 1
        
        assert pattern_idx == len(expected_pattern), (
            f"Event pattern mismatch. Expected: {expected_pattern}, "
            f"Got sequence: {event_types}. {message}"
        )
    
    @staticmethod
    def assert_system_state_consistent(
        system_state: Dict[str, Any],
        invariants: List[Callable[[Dict[str, Any]], bool]],
        message: str = ""
    ) -> None:
        """Assert system state satisfies invariants"""
        for i, invariant in enumerate(invariants):
            assert invariant(system_state), (
                f"System invariant {i} violated. {message}"
            )


# ============= Custom Test Markers =============
def mark_slow_test(func):
    """Mark test as slow"""
    return pytest.mark.slow(func)


def mark_integration_test(func):
    """Mark test as integration test"""
    return pytest.mark.integration(func)


def mark_requires_market_data(func):
    """Mark test as requiring market data"""
    return pytest.mark.requires_market_data(func)


# ============= Composite Assertions =============
class GridAttentionAssertions:
    """Composite assertions for GridAttention system"""
    
    def __init__(self):
        self.base = AssertionHelpers()
        self.trading = TradingAssertions()
        self.attention = AttentionAssertions()
        self.regime = RegimeAssertions()
        self.grid = GridAssertions()
        self.risk = RiskAssertions()
        self.performance = PerformanceAssertions()
        self.async_ops = AsyncAssertions()
        self.data = DataAssertions()
        self.overfitting = OverfittingAssertions()
        self.integration = IntegrationAssertions()
    
    def assert_system_ready(
        self,
        system_state: Dict[str, Any],
        message: str = ""
    ) -> None:
        """Assert entire system is ready for trading"""
        # Check all components initialized
        required_components = [
            'attention_layer', 'regime_detector', 'strategy_selector',
            'risk_manager', 'execution_engine', 'performance_monitor'
        ]
        
        for component in required_components:
            assert component in system_state, (
                f"Missing required component: {component}. {message}"
            )
            assert system_state[component] is not None, (
                f"Component {component} is None. {message}"
            )
        
        # Check attention system state
        if 'attention_state' in system_state:
            self.attention.assert_phase_transition_valid(
                AttentionPhase.LEARNING,
                system_state['attention_state'].get('phase', AttentionPhase.LEARNING),
                system_state['attention_state'].get('observations', 0),
                message=f"Attention system not ready. {message}"
            )
        
        # Check risk limits configured
        if 'risk_limits' in system_state:
            assert len(system_state['risk_limits']) > 0, (
                f"No risk limits configured. {message}"
            )


# ============= Factory Functions =============
def create_assertion_suite() -> GridAttentionAssertions:
    """Create complete assertion suite"""
    return GridAttentionAssertions()


def create_async_assertion_context():
    """Create context for async assertions"""
    return AsyncAssertions()


# ============= Usage Examples =============
"""
Example usage in tests:

# Basic usage
from tests.utils.assertion_helpers import (
    TradingAssertions, 
    AttentionAssertions,
    create_assertion_suite
)

# In test functions
def test_order_validation():
    order = {'symbol': 'BTCUSDT', 'side': 'buy', 'quantity': 0.01}
    TradingAssertions.assert_valid_order(order)

# Using composite assertions
def test_system_integration():
    assertions = create_assertion_suite()
    
    system_state = get_system_state()
    assertions.assert_system_ready(system_state)
    
    # Test specific components
    assertions.attention.assert_valid_attention_weights(weights)
    assertions.risk.assert_position_within_limits(position, limits)

# Async assertions
async def test_async_operation():
    from tests.utils.assertion_helpers import AsyncAssertions
    
    async def slow_operation():
        await asyncio.sleep(0.5)
        return "done"
    
    result = await AsyncAssertions.assert_completes_within(
        slow_operation, 
        timeout_seconds=1.0
    )
"""