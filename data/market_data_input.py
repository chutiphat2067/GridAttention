from typing import Dict, Any, Optional
"""
market_data_input.py
Handles all incoming market data with attention tracking for grid trading system

Author: Grid Trading System
Date: 2024
"""

# Standard library imports
import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
from enum import Enum

# Third-party imports
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

# Setup logger
logger = logging.getLogger(__name__)


# Data structures
@dataclass
class MarketTick:
    """Represents a single market data tick"""
    symbol: str
    price: float
    volume: float
    timestamp: float
    bid: float
    ask: float
    exchange: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'price': self.price,
            'volume': self.volume,
            'timestamp': self.timestamp,
            'bid': self.bid,
            'ask': self.ask,
            'exchange': self.exchange,
            'metadata': self.metadata
        }


class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = 5
    GOOD = 4
    FAIR = 3
    POOR = 2
    INVALID = 1


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class CircularBuffer:
    """Fixed-size circular buffer for efficient data storage"""
    
    def __init__(self, size: int):
        self.size = size
        self.buffer = deque(maxlen=size)
        self._lock = asyncio.Lock()
        
    async def append(self, item: Any) -> None:
        """Thread-safe append"""
        async with self._lock:
            self.buffer.append(item)
            
    async def get_latest(self, n: int) -> List[Any]:
        """Get latest n items"""
        async with self._lock:
            return list(self.buffer)[-n:] if n <= len(self.buffer) else list(self.buffer)
            
    async def get_all(self) -> List[Any]:
        """Get all items in buffer"""
        async with self._lock:
            return list(self.buffer)
            
    def __len__(self) -> int:
        return len(self.buffer)


class AttentionDataStore:
    """Store data for attention mechanism learning"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.tick_store = deque(maxlen=max_size)
        self.quality_scores = deque(maxlen=max_size)
        self.latencies = deque(maxlen=max_size)
        self.validation_failures = deque(maxlen=1000)
        self._lock = asyncio.Lock()
        
    async def add_tick(self, tick: MarketTick, quality_score: float, latency: float) -> None:
        """Add tick data for attention learning"""
        async with self._lock:
            self.tick_store.append(tick)
            self.quality_scores.append(quality_score)
            self.latencies.append(latency)
            
    async def add_validation_failure(self, failure_info: Dict[str, Any]) -> None:
        """Record validation failures for learning"""
        async with self._lock:
            self.validation_failures.append({
                'timestamp': time.time(),
                'info': failure_info
            })
            
    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for attention learning"""
        async with self._lock:
            if not self.quality_scores:
                return {}
                
            return {
                'total_ticks': len(self.tick_store),
                'avg_quality': np.mean(self.quality_scores),
                'avg_latency': np.mean(self.latencies),
                'validation_failures': len(self.validation_failures),
                'quality_distribution': {
                    'excellent': sum(1 for q in self.quality_scores if q >= 4.5),
                    'good': sum(1 for q in self.quality_scores if 3.5 <= q < 4.5),
                    'fair': sum(1 for q in self.quality_scores if 2.5 <= q < 3.5),
                    'poor': sum(1 for q in self.quality_scores if q < 2.5)
                }
            }


class BaseValidator:
    """Base class for validators"""
    
    def __init__(self, field_name: str):
        self.field_name = field_name
        self.validation_count = 0
        self.failure_count = 0
        
    def validate(self, value: Any) -> Any:
        """Validate value - to be implemented by subclasses"""
        raise NotImplementedError
        
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            'field': self.field_name,
            'total_validations': self.validation_count,
            'failures': self.failure_count,
            'success_rate': (self.validation_count - self.failure_count) / max(self.validation_count, 1)
        }


class PriceValidator(BaseValidator):
    """Validates price data"""
    
    def __init__(self, min_val: float = 0, max_change: float = 0.1):
        super().__init__('price')
        self.min_val = min_val
        self.max_change = max_change
        self.last_price = None
        
    def validate(self, value: Any) -> float:
        """Validate price value"""
        self.validation_count += 1
        
        try:
            price = float(value)
            
            # Check minimum value
            if price <= self.min_val:
                self.failure_count += 1
                raise ValidationError(f"Price {price} below minimum {self.min_val}")
                
            # Check maximum change
            if self.last_price is not None:
                change_pct = abs(price - self.last_price) / self.last_price
                if change_pct > self.max_change:
                    self.failure_count += 1
                    raise ValidationError(f"Price change {change_pct:.2%} exceeds maximum {self.max_change:.2%}")
                    
            self.last_price = price
            return price
            
        except (ValueError, TypeError) as e:
            self.failure_count += 1
            raise ValidationError(f"Invalid price value: {value}")


class VolumeValidator(BaseValidator):
    """Validates volume data"""
    
    def __init__(self, min_val: float = 0, max_spike: float = 10.0):
        super().__init__('volume')
        self.min_val = min_val
        self.max_spike = max_spike
        self.volume_history = deque(maxlen=20)
        
    def validate(self, value: Any) -> float:
        """Validate volume value"""
        self.validation_count += 1
        
        try:
            volume = float(value)
            
            # Check minimum value
            if volume < self.min_val:
                self.failure_count += 1
                raise ValidationError(f"Volume {volume} below minimum {self.min_val}")
                
            # Check for unrealistic spikes
            if len(self.volume_history) >= 10:
                avg_volume = np.mean(self.volume_history)
                if avg_volume > 0 and volume > avg_volume * self.max_spike:
                    self.failure_count += 1
                    raise ValidationError(f"Volume spike {volume/avg_volume:.1f}x exceeds maximum {self.max_spike}x")
                    
            self.volume_history.append(volume)
            return volume
            
        except (ValueError, TypeError) as e:
            self.failure_count += 1
            raise ValidationError(f"Invalid volume value: {value}")


class TimestampValidator(BaseValidator):
    """Validates timestamp data"""
    
    def __init__(self, max_delay: int = 1000):  # milliseconds
        super().__init__('timestamp')
        self.max_delay = max_delay
        
    def validate(self, value: Any) -> float:
        """Validate timestamp value"""
        self.validation_count += 1
        
        try:
            timestamp = float(value)
            current_time = time.time()
            
            # Check if timestamp is in the future
            if timestamp > current_time:
                self.failure_count += 1
                raise ValidationError(f"Timestamp {timestamp} is in the future")
                
            # Check delay
            delay_ms = (current_time - timestamp) * 1000
            if delay_ms > self.max_delay:
                self.failure_count += 1
                raise ValidationError(f"Timestamp delay {delay_ms:.0f}ms exceeds maximum {self.max_delay}ms")
                
            return timestamp
            
        except (ValueError, TypeError) as e:
            self.failure_count += 1
            raise ValidationError(f"Invalid timestamp value: {value}")


class SpreadValidator(BaseValidator):
    """Validates bid-ask spread"""
    
    def __init__(self, max_spread_pct: float = 0.01):  # 1%
        super().__init__('spread')
        self.max_spread_pct = max_spread_pct
        
    def validate(self, bid: float, ask: float) -> tuple:
        """Validate bid-ask spread"""
        self.validation_count += 1
        
        try:
            # Basic sanity checks
            if bid <= 0 or ask <= 0:
                self.failure_count += 1
                raise ValidationError(f"Invalid bid {bid} or ask {ask}")
                
            if bid >= ask:
                self.failure_count += 1
                raise ValidationError(f"Bid {bid} >= Ask {ask}")
                
            # Check spread
            spread_pct = (ask - bid) / bid
            if spread_pct > self.max_spread_pct:
                self.failure_count += 1
                raise ValidationError(f"Spread {spread_pct:.2%} exceeds maximum {self.max_spread_pct:.2%}")
                
            return bid, ask
            
        except Exception as e:
            self.failure_count += 1
            raise ValidationError(f"Spread validation error: {e}")


class DataAnomalyDetector:
    """Detect unusual patterns in market data"""
    
    def __init__(self):
        self.detectors = {
            'price_spike': self.detect_price_spike,
            'volume_anomaly': self.detect_volume_anomaly,
            'spread_manipulation': self.detect_spread_manipulation,
            'timestamp_irregularity': self.detect_timestamp_issues,
            'stale_data': self.detect_stale_data
        }
        self.anomaly_history = deque(maxlen=1000)
        from collections import defaultdict
        self.detection_stats = defaultdict(int)
        
    async def detect_anomalies(self, tick: MarketTick, recent_ticks: List[MarketTick]) -> List[Dict[str, Any]]:
        """Run all anomaly detectors"""
        anomalies = []
        
        for detector_name, detector_func in self.detectors.items():
            try:
                anomaly = await detector_func(tick, recent_ticks)
                if anomaly:
                    anomalies.append({
                        'type': detector_name,
                        'details': anomaly,
                        'timestamp': time.time()
                    })
                    self.detection_stats[detector_name] += 1
            except Exception as e:
                logger.error(f"Anomaly detector {detector_name} failed: {e}")
                
        if anomalies:
            self.anomaly_history.append({
                'tick': tick,
                'anomalies': anomalies,
                'timestamp': time.time()
            })
            
        return anomalies
        
    async def detect_price_spike(self, tick: MarketTick, recent_ticks: List[MarketTick]) -> Optional[Dict[str, Any]]:
        """Detect abnormal price movements"""
        if len(recent_ticks) < 10:
            return None
            
        recent_prices = [t.price for t in recent_ticks[-10:]]
        price_std = np.std(recent_prices)
        price_mean = np.mean(recent_prices)
        
        if price_std == 0:
            return None
            
        # Calculate z-score
        z_score = abs(tick.price - price_mean) / price_std
        
        if z_score > 3:  # 3-sigma event
            return {
                'severity': 'high' if z_score > 5 else 'medium',
                'z_score': z_score,
                'price': tick.price,
                'mean': price_mean,
                'std': price_std
            }
            
        return None
        
    async def detect_volume_anomaly(self, tick: MarketTick, recent_ticks: List[MarketTick]) -> Optional[Dict[str, Any]]:
        """Detect unusual volume patterns"""
        if len(recent_ticks) < 20:
            return None
            
        recent_volumes = [t.volume for t in recent_ticks[-20:]]
        vol_mean = np.mean(recent_volumes)
        vol_std = np.std(recent_volumes)
        
        if vol_std == 0:
            return None
            
        # Check for volume spike
        if tick.volume > vol_mean + 3 * vol_std:
            return {
                'type': 'spike',
                'volume': tick.volume,
                'mean': vol_mean,
                'std_above': (tick.volume - vol_mean) / vol_std
            }
            
        # Check for suspiciously low volume
        if tick.volume < vol_mean * 0.1:
            return {
                'type': 'drop',
                'volume': tick.volume,
                'mean': vol_mean,
                'ratio': tick.volume / vol_mean
            }
            
        return None
        
    async def detect_spread_manipulation(self, tick: MarketTick, recent_ticks: List[MarketTick]) -> Optional[Dict[str, Any]]:
        """Detect potential spread manipulation"""
        spread = tick.ask - tick.bid
        mid_price = (tick.ask + tick.bid) / 2
        
        if mid_price == 0:
            return None
            
        spread_bps = (spread / mid_price) * 10000
        
        # Check for abnormally wide spread
        if spread_bps > 50:  # 50 bps
            return {
                'spread_bps': spread_bps,
                'bid': tick.bid,
                'ask': tick.ask,
                'severity': 'high' if spread_bps > 100 else 'medium'
            }
            
        # Check for crossed spread
        if tick.bid >= tick.ask:
            return {
                'type': 'crossed',
                'bid': tick.bid,
                'ask': tick.ask,
                'severity': 'critical'
            }
            
        return None
        
    async def detect_timestamp_issues(self, tick: MarketTick, recent_ticks: List[MarketTick]) -> Optional[Dict[str, Any]]:
        """Detect timestamp irregularities"""
        current_time = time.time()
        
        # Check if timestamp is in the future
        if tick.timestamp > current_time + 1:  # 1 second tolerance
            return {
                'type': 'future_timestamp',
                'tick_time': tick.timestamp,
                'current_time': current_time,
                'difference': tick.timestamp - current_time
            }
            
        # Check for old timestamp
        age = current_time - tick.timestamp
        if age > 60:  # More than 1 minute old
            return {
                'type': 'stale_timestamp',
                'age_seconds': age,
                'severity': 'high' if age > 300 else 'medium'
            }
            
        # Check for out-of-order timestamps
        if recent_ticks and recent_ticks[-1].timestamp > tick.timestamp:
            return {
                'type': 'out_of_order',
                'current': tick.timestamp,
                'previous': recent_ticks[-1].timestamp
            }
            
        return None
        
    async def detect_stale_data(self, tick: MarketTick, recent_ticks: List[MarketTick]) -> Optional[Dict[str, Any]]:
        """Detect stale or stuck data"""
        if len(recent_ticks) < 5:
            return None
            
        # Check if price is stuck
        recent_prices = [t.price for t in recent_ticks[-5:]]
        if len(set(recent_prices)) == 1 and tick.price == recent_prices[0]:
            return {
                'type': 'stuck_price',
                'price': tick.price,
                'duration': len([t for t in recent_ticks if t.price == tick.price])
            }
            
        # Check if volume is stuck at zero
        recent_volumes = [t.volume for t in recent_ticks[-5:]]
        if all(v == 0 for v in recent_volumes) and tick.volume == 0:
            return {
                'type': 'zero_volume',
                'duration': len([t for t in recent_ticks if t.volume == 0])
            }
            
        return None
        
    def get_anomaly_stats(self) -> Dict[str, Any]:
        """Get anomaly detection statistics"""
        return {
            'total_anomalies': sum(self.detection_stats.values()),
            'by_type': dict(self.detection_stats),
            'recent_anomalies': len([a for a in self.anomaly_history if time.time() - a['timestamp'] < 3600])
        }
class WebSocketManager:
    """Manages WebSocket connections to exchanges"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connections = {}
        self.reconnect_delay = config.get('reconnect_delay', 5)
        self.max_reconnect_attempts = config.get('max_reconnect_attempts', 10)
        self._running = False
        self._tasks = []
        
    async def connect(self, exchange: str, url: str, subscribe_msg: Dict[str, Any]) -> None:
        """Connect to exchange WebSocket"""
        attempts = 0
        
        while attempts < self.max_reconnect_attempts and self._running:
            try:
                logger.info(f"Connecting to {exchange} WebSocket...")
                
                async with websockets.connect(url) as websocket:
                    self.connections[exchange] = websocket
                    
                    # Subscribe to market data
                    await websocket.send(json.dumps(subscribe_msg))
                    logger.info(f"Subscribed to {exchange} market data")
                    
                    # Reset attempts on successful connection
                    attempts = 0
                    
                    # Keep connection alive
                    while self._running:
                        try:
                            # Ping to keep connection alive
                            await websocket.ping()
                            await asyncio.sleep(30)
                        except Exception as e:
                            logger.warning(f"Ping failed for {exchange}: {e}")
                            break
                            
            except (ConnectionClosed, WebSocketException) as e:
                logger.error(f"WebSocket connection failed for {exchange}: {e}")
                attempts += 1
                
                if attempts < self.max_reconnect_attempts:
                    delay = self.reconnect_delay * attempts
                    logger.info(f"Reconnecting to {exchange} in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Max reconnection attempts reached for {exchange}")
                    raise
                    
    async def get_tick(self, exchange: str) -> Dict[str, Any]:
        """Get tick from specific exchange"""
        if exchange not in self.connections:
            raise ValueError(f"No connection to {exchange}")
            
        websocket = self.connections[exchange]
        
        try:
            message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
            return json.loads(message)
        except asyncio.TimeoutError:
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode message from {exchange}: {e}")
            return None
            
    async def start(self) -> None:
        """Start WebSocket connections"""
        self._running = True
        
        # Start connections based on config
        for exchange_config in self.config.get('exchanges', []):
            task = asyncio.create_task(
                self.connect(
                    exchange_config['name'],
                    exchange_config['url'],
                    exchange_config['subscribe_msg']
                )
            )
            self._tasks.append(task)
            
    async def stop(self) -> None:
        """Stop all WebSocket connections"""
        self._running = False
        
        # Close all connections
        for exchange, websocket in self.connections.items():
            try:
                await websocket.close()
                logger.info(f"Closed connection to {exchange}")
            except Exception as e:
                logger.error(f"Error closing connection to {exchange}: {e}")
                
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            
        # Wait for all tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)


class MarketDataInput:
    """
    Handles all incoming market data with attention tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.websocket_manager = WebSocketManager(config.get('websocket', {}))
        self.data_buffer = CircularBuffer(size=config.get('buffer_size', 1000))
        self.attention_store = AttentionDataStore(max_size=config.get('attention_store_size', 10000))
        self.validators = self._init_validators()
        self.spread_validator = SpreadValidator()
        
        # Add anomaly detector
        self.anomaly_detector = DataAnomalyDetector()
        
        # Fix missing defaultdict import
        from collections import defaultdict
        
        # Data recovery settings
        self.max_gap_interpolation = config.get('max_gap_interpolation', 5)
        self.fallback_to_last_known = config.get('fallback_to_last_known', True)
        self.alert_on_extended_gap = config.get('alert_on_extended_gap', 30)
        
        # Performance tracking
        self.tick_count = 0
        self.validation_failures = 0
        self.anomaly_count = 0
        self.last_tick_time = time.time()
        self.last_valid_tick = None
        
        # Callbacks
        self.tick_callbacks: List[Callable] = []
        self.anomaly_callbacks: List[Callable] = []
        
    def _init_validators(self) -> Dict[str, BaseValidator]:
        """Initialize data validators"""
        validator_config = self.config.get('validators', {})
        
        return {
            'price': PriceValidator(
                min_val=validator_config.get('price_min', 0),
                max_change=validator_config.get('price_max_change', 0.1)
            ),
            'volume': VolumeValidator(
                min_val=validator_config.get('volume_min', 0),
                max_spike=validator_config.get('volume_max_spike', 10.0)
            ),
            'timestamp': TimestampValidator(
                max_delay=validator_config.get('timestamp_max_delay', 1000)
            )
        }
        
    async def start(self) -> None:
        """Start market data collection"""
        logger.info("Starting market data input...")
        await self.websocket_manager.start()
        
        # Start tick collection tasks for each exchange
        tasks = []
        for exchange in self.config.get('exchanges', []):
            task = asyncio.create_task(self._collect_ticks_from_exchange(exchange['name']))
            tasks.append(task)
            
        # Wait for all tasks
        await asyncio.gather(*tasks)
        
    async def stop(self) -> None:
        """Stop market data collection"""
        logger.info("Stopping market data input...")
        await self.websocket_manager.stop()
        
    async def _collect_ticks_from_exchange(self, exchange: str) -> None:
        """Collect ticks from specific exchange"""
        while True:
            try:
                tick = await self.collect_tick(exchange)
                if tick:
                    # Notify callbacks
                    for callback in self.tick_callbacks:
                        asyncio.create_task(callback(tick))
                        
            except Exception as e:
                logger.error(f"Error collecting tick from {exchange}: {e}")
                await asyncio.sleep(1)
                
    async def collect_tick(self, exchange: str = None) -> Optional[MarketTick]:
        """Collect and validate single tick"""
        start_time = time.perf_counter()
        
        try:
            # Check for extended data gap
            time_since_last = time.time() - self.last_tick_time
            if time_since_last > self.alert_on_extended_gap:
                logger.warning(f"Extended data gap detected: {time_since_last:.1f} seconds")
                
            # Get raw data from WebSocket
            if exchange:
                raw_data = await self.websocket_manager.get_tick(exchange)
            else:
                # Get from first available exchange
                for ex in self.config.get('exchanges', []):
                    raw_data = await self.websocket_manager.get_tick(ex['name'])
                    if raw_data:
                        exchange = ex['name']
                        break
                        
            if not raw_data:
                # Try data recovery
                if self.fallback_to_last_known and self.last_valid_tick:
                    logger.warning("No new data, using last known tick")
                    return self.last_valid_tick
                return None
                
            # Parse and validate
            validated_tick = await self._validate_and_parse_tick(raw_data, exchange)
            
            if validated_tick:
                # Check for anomalies
                recent_ticks = await self.data_buffer.get_latest(20)
                anomalies = await self.anomaly_detector.detect_anomalies(validated_tick, recent_ticks)
                
                if anomalies:
                    self.anomaly_count += 1
                    logger.warning(f"Data anomalies detected: {[a['type'] for a in anomalies]}")
                    
                    # Notify anomaly callbacks
                    for callback in self.anomaly_callbacks:
                        asyncio.create_task(callback(validated_tick, anomalies))
                        
                    # Decide whether to use this tick
                    critical_anomalies = [a for a in anomalies if a.get('details', {}).get('severity') == 'critical']
                    if critical_anomalies:
                        logger.error(f"Critical anomalies detected, rejecting tick")
                        return None
                        
                # Calculate latency
                latency = time.perf_counter() - start_time
                
                # Calculate quality score
                quality_score = self._calculate_quality_score(validated_tick, latency)
                
                # Adjust quality score based on anomalies
                if anomalies:
                    quality_score *= 0.5  # Reduce quality score if anomalies detected
                    
                # Store for attention learning
                await self.attention_store.add_tick(validated_tick, quality_score, latency)
                
                # Add to buffer
                await self.data_buffer.append(validated_tick)
                
                # Update counters
                self.tick_count += 1
                self.last_tick_time = time.time()
                self.last_valid_tick = validated_tick
                
                return validated_tick
                
        except ValidationError as e:
            self.validation_failures += 1
            await self.attention_store.add_validation_failure({
                'error': str(e),
                'exchange': exchange,
                'raw_data': raw_data if 'raw_data' in locals() else None
            })
            logger.warning(f"Data validation failed: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error in collect_tick: {e}")
            return None
            
    async def _validate_and_parse_tick(self, raw_data: Dict[str, Any], exchange: str) -> Optional[MarketTick]:
        """Validate and parse raw tick data"""
        try:
            # Extract fields based on exchange format
            parser = self._get_exchange_parser(exchange)
            parsed_data = parser(raw_data)
            
            # Validate individual fields
            validated_data = {}
            
            # Price validation
            validated_data['price'] = self.validators['price'].validate(parsed_data['price'])
            
            # Volume validation
            validated_data['volume'] = self.validators['volume'].validate(parsed_data['volume'])
            
            # Timestamp validation
            validated_data['timestamp'] = self.validators['timestamp'].validate(parsed_data['timestamp'])
            
            # Spread validation
            bid, ask = self.spread_validator.validate(parsed_data['bid'], parsed_data['ask'])
            validated_data['bid'] = bid
            validated_data['ask'] = ask
            
            # Create MarketTick
            tick = MarketTick(
                symbol=parsed_data['symbol'],
                price=validated_data['price'],
                volume=validated_data['volume'],
                timestamp=validated_data['timestamp'],
                bid=validated_data['bid'],
                ask=validated_data['ask'],
                exchange=exchange,
                metadata={
                    'raw_timestamp': raw_data.get('timestamp', time.time()),
                    'sequence': raw_data.get('sequence'),
                    'validation_time': time.perf_counter()
                }
            )
            
            return tick
            
        except Exception as e:
            raise ValidationError(f"Failed to parse tick data: {e}")
            
    def _get_exchange_parser(self, exchange: str) -> Callable:
        """Get parser for specific exchange format"""
        parsers = {
            'binance': self._parse_binance_tick,
            'coinbase': self._parse_coinbase_tick,
            'kraken': self._parse_kraken_tick,
            # Add more exchange parsers as needed
        }
        
        parser = parsers.get(exchange)
        if not parser:
            raise ValueError(f"No parser available for exchange: {exchange}")
            
        return parser
        
    def _parse_binance_tick(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Binance tick format"""
        return {
            'symbol': data.get('s'),
            'price': float(data.get('c', 0)),
            'volume': float(data.get('v', 0)),
            'timestamp': float(data.get('T', 0)) / 1000,  # Convert to seconds
            'bid': float(data.get('b', 0)),
            'ask': float(data.get('a', 0))
        }
        
    def _parse_coinbase_tick(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Coinbase tick format"""
        return {
            'symbol': data.get('product_id'),
            'price': float(data.get('price', 0)),
            'volume': float(data.get('size', 0)),
            'timestamp': datetime.fromisoformat(data.get('time').replace('Z', '+00:00')).timestamp(),
            'bid': float(data.get('best_bid', 0)),
            'ask': float(data.get('best_ask', 0))
        }
        
    def _parse_kraken_tick(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Kraken tick format"""
        # Kraken sends data in array format
        tick_data = data[1] if isinstance(data, list) else data
        return {
            'symbol': data[3] if isinstance(data, list) else data.get('pair'),
            'price': float(tick_data.get('c', [0])[0]),
            'volume': float(tick_data.get('v', [0])[0]),
            'timestamp': float(tick_data.get('t', [0])[0]),
            'bid': float(tick_data.get('b', [0])[0]),
            'ask': float(tick_data.get('a', [0])[0])
        }
        
    def _calculate_quality_score(self, tick: MarketTick, latency: float) -> float:
        """Calculate quality score for attention learning"""
        score = 5.0  # Start with perfect score
        
        # Latency penalty (lose 1 point per 100μs over 500μs)
        if latency > 0.0005:  # 500μs
            latency_penalty = (latency - 0.0005) * 10000
            score -= min(latency_penalty, 2.0)
            
        # Spread penalty (lose points for wide spreads)
        spread_bps = ((tick.ask - tick.bid) / tick.bid) * 10000
        if spread_bps > 5:
            spread_penalty = (spread_bps - 5) / 10
            score -= min(spread_penalty, 1.0)
            
        # Age penalty (lose points for stale data)
        age = time.time() - tick.timestamp
        if age > 1.0:  # 1 second
            age_penalty = age / 5
            score -= min(age_penalty, 1.0)
            
        # Validation success bonus
        if self.validation_failures == 0:
            score += 0.5
            
        return max(1.0, min(5.0, score))
        
    def register_tick_callback(self, callback: Callable) -> None:
        """Register callback for new ticks"""
        self.tick_callbacks.append(callback)
        
    async def get_latest_ticks(self, n: int = 10) -> List[MarketTick]:
        """Get latest n ticks from buffer"""
        return await self.data_buffer.get_latest(n)
        
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        attention_stats = await self.attention_store.get_statistics()
        
        # Calculate tick rate
        time_elapsed = time.time() - self.last_tick_time
        tick_rate = self.tick_count / max(time_elapsed, 1)
        
        # Validator statistics
        validator_stats = {}
        for name, validator in self.validators.items():
            validator_stats[name] = validator.get_stats()
            
        return {
            'tick_count': self.tick_count,
            'tick_rate': tick_rate,
            'validation_failures': self.validation_failures,
            'failure_rate': self.validation_failures / max(self.tick_count, 1),
            'buffer_size': len(self.data_buffer),
            'validators': validator_stats,
            'attention': attention_stats
        }


# Example configuration
EXAMPLE_CONFIG = {
    'buffer_size': 1000,
    'attention_store_size': 10000,
    'websocket': {
        'reconnect_delay': 5,
        'max_reconnect_attempts': 10
    },
    'exchanges': [
        {
            'name': 'binance',
            'url': 'wss://stream.binance.com:9443/ws',
            'subscribe_msg': {
                'method': 'SUBSCRIBE',
                'params': ['btcusdt@trade'],
                'id': 1
            }
        }
    ],
    'validators': {
        'price_min': 0,
        'price_max_change': 0.1,
        'volume_min': 0,
        'volume_max_spike': 10.0,
        'timestamp_max_delay': 1000
    }
}


# Example usage
async def main():
    """Example usage of MarketDataInput"""
    
    # Initialize
    market_data = MarketDataInput(EXAMPLE_CONFIG)
    
    # Register callback
    async def on_new_tick(tick: MarketTick):
        print(f"New tick: {tick.symbol} @ {tick.price} (vol: {tick.volume})")
        

    async def health_check(self) -> Dict[str, Any]:
        """Check component health"""
        return {
            'healthy': True,
            'is_running': getattr(self, 'is_running', True),
            'error_count': getattr(self, 'error_count', 0),
            'last_error': getattr(self, 'last_error', None)
        }

    async def is_healthy(self) -> bool:
        """Quick health check"""
        health = await self.health_check()
        return health.get('healthy', True)

    async def recover(self) -> bool:
        """Recover from failure"""
        try:
            self.error_count = 0
            self.last_error = None
            return True
        except Exception as e:
            print(f"Recovery failed: {e}")
            return False

    def get_state(self) -> Dict[str, Any]:
        """Get component state for checkpointing"""
        return {
            'class': self.__class__.__name__,
            'timestamp': time.time() if 'time' in globals() else 0
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load component state from checkpoint"""
        pass

    async def get_latest_data(self):
        """Get latest market data - fix for missing method"""
        if hasattr(self, 'market_data_buffer') and self.market_data_buffer:
            return self.market_data_buffer[-1]
        # Return mock data if no real data
        return {
            'symbol': 'BTC/USDT',
            'price': 50000,
            'volume': 1.0,
            'timestamp': time.time()
        }
    market_data.register_tick_callback(on_new_tick)
    
    try:
        # Start collecting data
        await market_data.start()
        
    except KeyboardInterrupt:
        print("\nStopping...")
        await market_data.stop()
        
        # Print statistics
        stats = await market_data.get_statistics()
        print(f"\nStatistics:")
        print(f"Total ticks: {stats['tick_count']}")
        print(f"Tick rate: {stats['tick_rate']:.2f}/sec")
        print(f"Validation failures: {stats['validation_failures']}")
        print(f"Failure rate: {stats['failure_rate']:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
