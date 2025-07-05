"""
Comprehensive input validation for GridAttention system
"""
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of validation check"""
    def __init__(self, is_valid: bool, errors: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        
    def add_error(self, error: str):
        self.errors.append(error)
        self.is_valid = False
        
    def __bool__(self):
        return self.is_valid


@dataclass
class FeatureRange:
    """Valid range for a feature"""
    name: str
    min_value: float
    max_value: float
    allow_nan: bool = False
    allow_inf: bool = False
    
    def validate(self, value: float) -> ValidationResult:
        result = ValidationResult(True)
        
        if not isinstance(value, (int, float)):
            result.add_error(f"{self.name}: Invalid type {type(value)}")
            return result
            
        if not self.allow_nan and np.isnan(value):
            result.add_error(f"{self.name}: NaN not allowed")
            
        if not self.allow_inf and np.isinf(value):
            result.add_error(f"{self.name}: Inf not allowed")
            
        if result.is_valid and not (self.min_value <= value <= self.max_value):
            result.add_error(
                f"{self.name}: Value {value} outside range "
                f"[{self.min_value}, {self.max_value}]"
            )
            
        return result


class FeatureValidator:
    """Validator for feature dictionaries"""
    
    def __init__(self):
        self.feature_ranges = {
            'price_change': FeatureRange('price_change', -0.2, 0.2),
            'volume_ratio': FeatureRange('volume_ratio', 0, 100),
            'spread_bps': FeatureRange('spread_bps', 0, 1000),
            'volatility': FeatureRange('volatility', 0, 1),
            'rsi': FeatureRange('rsi', 0, 100),
            'momentum': FeatureRange('momentum', -1, 1),
            'macd': FeatureRange('macd', -10, 10),
            'bb_position': FeatureRange('bb_position', -5, 5),
            'volume_imbalance': FeatureRange('volume_imbalance', -1, 1),
            'tick_intensity': FeatureRange('tick_intensity', 0, 1000)
        }
        
        # Statistics for anomaly detection
        self.feature_stats = {}
        self.anomaly_threshold = 4  # Standard deviations
        
    def validate_features(self, features: Dict[str, float]) -> ValidationResult:
        """Validate feature dictionary"""
        result = ValidationResult(True)
        
        if not isinstance(features, dict):
            result.add_error(f"Features must be a dictionary, got {type(features)}")
            return result
            
        if len(features) == 0:
            result.add_error("Features dictionary is empty")
            return result
            
        # Check each feature
        for name, value in features.items():
            if name in self.feature_ranges:
                feature_result = self.feature_ranges[name].validate(value)
                if not feature_result:
                    result.errors.extend(feature_result.errors)
                    result.is_valid = False
            else:
                # Unknown feature - check basic validity
                if not isinstance(value, (int, float)):
                    result.add_error(f"{name}: Invalid type {type(value)}")
                elif np.isnan(value) or np.isinf(value):
                    result.add_error(f"{name}: Contains NaN or Inf")
                    
        # Check for anomalies
        anomalies = self._detect_anomalies(features)
        if anomalies:
            for anomaly in anomalies:
                result.add_error(f"Anomaly detected: {anomaly}")
                
        return result
        
    def _detect_anomalies(self, features: Dict[str, float]) -> List[str]:
        """Detect statistical anomalies in features"""
        anomalies = []
        
        for name, value in features.items():
            if name not in self.feature_stats:
                self.feature_stats[name] = {
                    'values': [],
                    'mean': 0,
                    'std': 1
                }
                
            stats = self.feature_stats[name]
            stats['values'].append(value)
            
            # Keep only recent values
            if len(stats['values']) > 1000:
                stats['values'] = stats['values'][-1000:]
                
            # Update statistics
            if len(stats['values']) > 30:
                stats['mean'] = np.mean(stats['values'])
                stats['std'] = np.std(stats['values'])
                
                # Check for anomaly
                if stats['std'] > 0:
                    z_score = abs((value - stats['mean']) / stats['std'])
                    if z_score > self.anomaly_threshold:
                        anomalies.append(
                            f"{name} = {value:.4f} "
                            f"(z-score: {z_score:.2f})"
                        )
                        
        return anomalies
        
    def add_custom_range(self, name: str, min_val: float, max_val: float):
        """Add custom feature range"""
        self.feature_ranges[name] = FeatureRange(name, min_val, max_val)


class OrderValidator:
    """Validator for order parameters"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_order_size = self.config.get('min_order_size', 0.001)
        self.max_order_size = self.config.get('max_order_size', 100.0)
        self.valid_order_types = {'market', 'limit', 'stop', 'stop_limit'}
        self.valid_sides = {'buy', 'sell'}
        
    def validate_order(self, order: Dict[str, Any]) -> ValidationResult:
        """Validate order parameters"""
        result = ValidationResult(True)
        
        # Required fields
        required = ['symbol', 'side', 'quantity', 'order_type']
        for field in required:
            if field not in order:
                result.add_error(f"Missing required field: {field}")
                
        if not result:
            return result
            
        # Validate side
        if order['side'] not in self.valid_sides:
            result.add_error(f"Invalid side: {order['side']}")
            
        # Validate order type
        if order['order_type'] not in self.valid_order_types:
            result.add_error(f"Invalid order type: {order['order_type']}")
            
        # Validate quantity
        qty = order['quantity']
        if not isinstance(qty, (int, float)) or qty <= 0:
            result.add_error(f"Invalid quantity: {qty}")
        elif qty < self.min_order_size:
            result.add_error(f"Quantity {qty} below minimum {self.min_order_size}")
        elif qty > self.max_order_size:
            result.add_error(f"Quantity {qty} above maximum {self.max_order_size}")
            
        # Validate price for limit orders
        if order['order_type'] in ['limit', 'stop_limit']:
            if 'price' not in order:
                result.add_error("Limit order requires price")
            else:
                price = order['price']
                if not isinstance(price, (int, float)) or price <= 0:
                    result.add_error(f"Invalid price: {price}")
                    
        return result


class DataIntegrityChecker:
    """Check data integrity and consistency"""
    
    def __init__(self):
        self.last_timestamps = {}
        self.sequence_numbers = {}
        
    def check_tick_integrity(self, tick: Dict[str, Any]) -> ValidationResult:
        """Check market tick data integrity"""
        result = ValidationResult(True)
        
        symbol = tick.get('symbol')
        if not symbol:
            result.add_error("Missing symbol")
            return result
            
        # Check timestamp
        timestamp = tick.get('timestamp')
        if not timestamp:
            result.add_error("Missing timestamp")
        else:
            # Check if timestamp is moving forward
            if symbol in self.last_timestamps:
                if timestamp <= self.last_timestamps[symbol]:
                    result.add_error(
                        f"Timestamp not increasing: {timestamp} <= "
                        f"{self.last_timestamps[symbol]}"
                    )
            self.last_timestamps[symbol] = timestamp
            
        # Check sequence number if present
        seq = tick.get('sequence')
        if seq is not None:
            if symbol in self.sequence_numbers:
                expected = self.sequence_numbers[symbol] + 1
                if seq != expected:
                    result.add_error(
                        f"Sequence gap: expected {expected}, got {seq}"
                    )
            self.sequence_numbers[symbol] = seq
            
        # Check price sanity
        price = tick.get('price')
        if price is not None:
            if price <= 0:
                result.add_error(f"Invalid price: {price}")
            elif price > 1e6:  # Sanity check
                result.add_error(f"Suspiciously high price: {price}")
                
        # Check volume
        volume = tick.get('volume')
        if volume is not None and volume < 0:
            result.add_error(f"Negative volume: {volume}")
            
        return result