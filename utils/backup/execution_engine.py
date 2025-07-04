"""
execution_engine.py
High-performance order execution with fee optimization for grid trading system

Author: Grid Trading System
Date: 2024
"""

# Standard library imports
import asyncio
import time
import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from datetime import datetime, timedelta
import numpy as np

# Third-party imports
import aiohttp
import websockets
from ccxt.async_support import Exchange
import ccxt.async_support as ccxt

# Local imports
from core.grid_strategy_selector import GridStrategyConfig, GridLevel
from core.risk_management_system import RiskViolationType

# Setup logger
logger = logging.getLogger(__name__)


# Constants
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_BASE = 1.0  # seconds
ORDER_TIMEOUT = 30  # seconds
EXECUTION_QUEUE_SIZE = 1000
RATE_LIMIT_BUFFER = 0.9  # Use 90% of rate limit
MIN_ORDER_INTERVAL = 0.1  # seconds between orders
MIN_ORDER_SIZE = 0.001  # Minimum order size
MAX_SLIPPAGE = 0.002  # 0.2% max slippage
PRICE_PRECISION = 8
QUANTITY_PRECISION = 8
LATENCY_TARGET = 0.005  # 5ms target
FEE_DISCOUNT_THRESHOLD = 0.0003  # 0.03% maker fee threshold


# Enums
class OrderStatus(Enum):
    """Order status types"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(Enum):
    """Time in force options"""
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTT = "GTT"  # Good Till Time
    POST_ONLY = "PO"  # Post Only (Maker only)


class ExecutionError(Exception):
    """Custom exception for execution errors"""
    pass


class ExecutionRejected(ExecutionError):
    """Order rejected by exchange"""
    pass


class ExecutionTimeout(ExecutionError):
    """Order execution timeout"""
    pass


@dataclass
class Order:
    """Represents a single order"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: OrderType
    price: float
    quantity: float
    time_in_force: TimeInForce
    status: OrderStatus = OrderStatus.PENDING
    exchange: str = ""
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    filled_quantity: float = 0.0
    average_price: float = 0.0
    fees: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'order_type': self.order_type.value,
            'price': self.price,
            'quantity': self.quantity,
            'time_in_force': self.time_in_force.value,
            'status': self.status.value,
            'exchange': self.exchange,
            'client_order_id': self.client_order_id,
            'created_at': self.created_at,
            'filled_quantity': self.filled_quantity,
            'average_price': self.average_price,
            'fees': self.fees
        }
        
    def is_complete(self) -> bool:
        """Check if order is in terminal state"""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.FAILED
        ]
        
    def is_active(self) -> bool:
        """Check if order is active"""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL
        ]


@dataclass
class GridOrder(Order):
    """Grid-specific order with additional metadata"""
    grid_id: str = ""
    level_index: int = 0
    strategy: str = ""
    distance_from_mid: float = 0.0
    

@dataclass
class ExecutionResult:
    """Result of order execution"""
    success: bool
    order: Order
    latency: float  # milliseconds
    error: Optional[str] = None
    retry_count: int = 0
    
    
@dataclass
class ExecutionMetrics:
    """Execution performance metrics"""
    total_orders: int = 0
    successful_orders: int = 0
    failed_orders: int = 0
    rejected_orders: int = 0
    total_latency: float = 0.0
    min_latency: float = float('inf')
    max_latency: float = 0.0
    fill_rates: Dict[str, float] = field(default_factory=dict)
    slippage: Dict[str, float] = field(default_factory=dict)
    
    def add_execution(self, result: ExecutionResult):
        """Add execution result to metrics"""
        self.total_orders += 1
        
        if result.success:
            self.successful_orders += 1
        else:
            self.failed_orders += 1
            
        self.total_latency += result.latency
        self.min_latency = min(self.min_latency, result.latency)
        self.max_latency = max(self.max_latency, result.latency)
        
    def get_average_latency(self) -> float:
        """Get average execution latency"""
        if self.total_orders == 0:
            return 0.0
        return self.total_latency / self.total_orders
        
    def get_success_rate(self) -> float:
        """Get execution success rate"""
        if self.total_orders == 0:
            return 0.0
        return self.successful_orders / self.total_orders


class ExchangeManager:
    """Manages connections to multiple exchanges"""
    
    def __init__(self, exchange_configs: Dict[str, Dict[str, Any]]):
        self.exchange_configs = exchange_configs
        self.exchanges = {}
        self.rate_limiters = {}
        self.websocket_connections = {}
        self._initialized = False
        self._price_cache = {}  # symbol -> price
        self._orderbook_cache = {}  # symbol -> orderbook
        self._cache_timestamps = {}
        
    async def initialize(self):
        """Initialize all exchange connections"""
        if self._initialized:
            return
            
        for exchange_name, config in self.exchange_configs.items():
            try:
                # Initialize exchange instance
                exchange_class = getattr(ccxt, config['class'])
                exchange = exchange_class({
                    'apiKey': config.get('api_key'),
                    'secret': config.get('secret'),
                    'enableRateLimit': True,
                    'rateLimit': config.get('rate_limit', 50),
                    'options': config.get('options', {})
                })
                
                # Load markets
                await exchange.load_markets()
                
                self.exchanges[exchange_name] = exchange
                
                # Initialize rate limiter
                self.rate_limiters[exchange_name] = RateLimiter(
                    config.get('rate_limit', 50),
                    config.get('rate_window', 1000)  # ms
                )
                
                # Initialize WebSocket if configured
                if config.get('websocket_url'):
                    await self._init_websocket(exchange_name, config['websocket_url'])
                    
                logger.info(f"Initialized exchange: {exchange_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize exchange {exchange_name}: {e}")
                
        self._initialized = True
        
    async def _init_websocket(self, exchange_name: str, url: str):
        """Initialize WebSocket connection for real-time data"""
        try:
            ws = await websockets.connect(url)
            self.websocket_connections[exchange_name] = ws
            
            # Start listening task
            asyncio.create_task(self._websocket_listener(exchange_name, ws))
            
        except Exception as e:
            logger.error(f"Failed to connect WebSocket for {exchange_name}: {e}")
            
    async def _websocket_listener(self, exchange_name: str, ws):
        """Listen to WebSocket messages"""
        try:
            async for message in ws:
                data = json.loads(message)
                await self._process_websocket_message(exchange_name, data)
        except Exception as e:
            logger.error(f"WebSocket error for {exchange_name}: {e}")
            # Attempt reconnection
            await asyncio.sleep(5)
            await self._init_websocket(
                exchange_name, 
                self.exchange_configs[exchange_name]['websocket_url']
            )
            
    async def _process_websocket_message(self, exchange_name: str, data: Dict[str, Any]):
        """Process WebSocket message"""
        # Update price cache from ticker updates
        if 'ticker' in data:
            symbol = data['ticker'].get('symbol')
            price = data['ticker'].get('last')
            if symbol and price:
                self._price_cache[symbol] = price
                self._cache_timestamps[f"price_{symbol}"] = time.time()
                
        # Update orderbook cache
        if 'orderbook' in data:
            symbol = data['orderbook'].get('symbol')
            self._orderbook_cache[symbol] = data['orderbook']
            self._cache_timestamps[f"orderbook_{symbol}"] = time.time()
            
    async def get_current_price(self, symbol: str, exchange: Optional[str] = None) -> float:
        """Get current price for symbol"""
        # Check cache first
        cache_key = f"price_{symbol}"
        if cache_key in self._cache_timestamps:
            age = time.time() - self._cache_timestamps[cache_key]
            if age < 1.0 and symbol in self._price_cache:  # 1 second cache
                return self._price_cache[symbol]
                
        # Get from exchange
        if exchange:
            exchanges_to_try = [exchange]
        else:
            exchanges_to_try = list(self.exchanges.keys())
            
        for ex_name in exchanges_to_try:
            try:
                exchange_obj = self.exchanges[ex_name]
                ticker = await exchange_obj.fetch_ticker(symbol)
                price = ticker['last']
                
                # Update cache
                self._price_cache[symbol] = price
                self._cache_timestamps[cache_key] = time.time()
                
                return price
                
            except Exception as e:
                logger.error(f"Failed to get price from {ex_name}: {e}")
                
        raise ExecutionError(f"Failed to get price for {symbol}")
        
    async def get_orderbook(self, symbol: str, exchange: str, limit: int = 10) -> Dict[str, Any]:
        """Get orderbook for symbol"""
        # Check cache
        cache_key = f"orderbook_{symbol}"
        if cache_key in self._cache_timestamps:
            age = time.time() - self._cache_timestamps[cache_key]
            if age < 0.5 and symbol in self._orderbook_cache:  # 500ms cache
                return self._orderbook_cache[symbol]
                
        # Get from exchange
        try:
            exchange_obj = self.exchanges[exchange]
            orderbook = await exchange_obj.fetch_order_book(symbol, limit)
            
            # Update cache
            self._orderbook_cache[symbol] = orderbook
            self._cache_timestamps[cache_key] = time.time()
            
            return orderbook
            
        except Exception as e:
            logger.error(f"Failed to get orderbook: {e}")
            raise ExecutionError(f"Failed to get orderbook for {symbol}")
            
    def get_exchange(self, name: str) -> Exchange:
        """Get exchange instance"""
        if name not in self.exchanges:
            raise ValueError(f"No connection to {name}")
        return self.exchanges[name]
        
    async def close_all(self):
        """Close all connections"""
        # Close WebSocket connections
        for ws in self.websocket_connections.values():
            await ws.close()
            
        # Close exchange connections
        for exchange in self.exchanges.values():
            await exchange.close()


class RateLimiter:
    """Rate limiter for exchange API calls"""
    
    def __init__(self, limit: int, window_ms: int):
        self.limit = int(limit * RATE_LIMIT_BUFFER)  # Use buffer
        self.window_ms = window_ms
        self.calls = deque()
        self._lock = asyncio.Lock()
        
    async def acquire(self):
        """Acquire permission to make API call"""
        async with self._lock:
            now = time.time() * 1000  # Convert to ms
            
            # Remove old calls outside window
            while self.calls and self.calls[0] < now - self.window_ms:
                self.calls.popleft()
                
            # Check if we can make call
            if len(self.calls) >= self.limit:
                # Calculate wait time
                oldest_call = self.calls[0]
                wait_time = (oldest_call + self.window_ms - now) / 1000
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    
            # Record call
            self.calls.append(now)
            
    def get_remaining_calls(self) -> int:
        """Get remaining calls in current window"""
        now = time.time() * 1000
        
        # Remove old calls
        while self.calls and self.calls[0] < now - self.window_ms:
            self.calls.popleft()
            
        return max(0, self.limit - len(self.calls))


class OrderManager:
    """Manages order lifecycle and tracking"""
    
    def __init__(self):
        self.active_orders = {}  # order_id -> Order
        self.completed_orders = deque(maxlen=10000)
        self.order_history = defaultdict(list)  # symbol -> List[Order]
        self._lock = asyncio.Lock()
        
    async def add_order(self, order: Order):
        """Add new order to tracking"""
        async with self._lock:
            self.active_orders[order.order_id] = order
            self.order_history[order.symbol].append(order)
            
    async def update_order(self, order_id: str, updates: Dict[str, Any]):
        """Update order status"""
        async with self._lock:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                
                # Update fields
                for key, value in updates.items():
                    if hasattr(order, key):
                        setattr(order, key, value)
                        
                order.updated_at = time.time()
                
                # Move to completed if done
                if order.is_complete():
                    self.completed_orders.append(order)
                    del self.active_orders[order_id]
                    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        async with self._lock:
            return self.active_orders.get(order_id)
            
    async def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get active orders"""
        async with self._lock:
            if symbol:
                return [
                    order for order in self.active_orders.values()
                    if order.symbol == symbol
                ]
            return list(self.active_orders.values())
            
    async def cancel_order(self, order_id: str):
        """Mark order as cancelled"""
        await self.update_order(order_id, {'status': OrderStatus.CANCELLED})
        
    def get_order_count(self, symbol: Optional[str] = None) -> int:
        """Get count of active orders"""
        if symbol:
            return sum(1 for order in self.active_orders.values() if order.symbol == symbol)
        return len(self.active_orders)
        
    def get_fill_rate(self, symbol: str, window: int = 100) -> float:
        """Get fill rate for recent orders"""
        recent_orders = list(self.order_history[symbol])[-window:]
        if not recent_orders:
            return 0.0
            
        filled = sum(1 for order in recent_orders if order.status == OrderStatus.FILLED)
        return filled / len(recent_orders)


class OrderBatcher:
    """Batches orders for efficient execution"""
    
    def __init__(self, batch_size: int = 10, batch_timeout: float = 0.1):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_orders = defaultdict(list)  # exchange -> List[Order]
        self._locks = defaultdict(asyncio.Lock)
        self._batch_tasks = {}
        
    async def add_order(self, order: Order, exchange: str) -> asyncio.Future:
        """Add order to batch"""
        async with self._locks[exchange]:
            # Create future for result
            future = asyncio.Future()
            
            # Add to pending
            self.pending_orders[exchange].append((order, future))
            
            # Start batch task if not running
            if exchange not in self._batch_tasks or self._batch_tasks[exchange].done():
                self._batch_tasks[exchange] = asyncio.create_task(
                    self._process_batch(exchange)
                )
                
            return future
            
    async def _process_batch(self, exchange: str):
        """Process batch of orders"""
        await asyncio.sleep(self.batch_timeout)
        
        async with self._locks[exchange]:
            if not self.pending_orders[exchange]:
                return
                
            # Get orders to process
            orders_to_process = self.pending_orders[exchange][:self.batch_size]
            self.pending_orders[exchange] = self.pending_orders[exchange][self.batch_size:]
            
        # Process orders (would be exchange-specific)
        for order, future in orders_to_process:
            try:
                # This would be actual batch execution
                result = ExecutionResult(
                    success=True,
                    order=order,
                    latency=1.0
                )
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)


class FeeOptimizer:
    """Optimize trading fees"""
    
    def __init__(self):
        self.fee_structures = {
            'binance': {
                'maker': 0.001,  # 0.1%
                'taker': 0.001,
                'discount_token': 'BNB',
                'discount_rate': 0.25,  # 25% discount with BNB
                'vip_tiers': {
                    0: {'maker': 0.001, 'taker': 0.001, 'volume': 0},
                    1: {'maker': 0.0009, 'taker': 0.001, 'volume': 50},
                    2: {'maker': 0.0008, 'taker': 0.001, 'volume': 200}
                }
            },
            'coinbase': {
                'maker': 0.005,
                'taker': 0.005,
                'discount_token': None,
                'discount_rate': 0,
                'vip_tiers': {}
            }
        }
        self.fee_tracking = defaultdict(float)
        self.volume_tracking = defaultdict(float)
        self.optimization_stats = {
            'maker_orders': 0,
            'taker_orders': 0,
            'fees_saved': 0.0,
            'discount_used': 0.0
        }
        
    def optimize_order_placement(self, order: Order, orderbook: Dict[str, Any], exchange: str) -> Dict[str, Any]:
        """Optimize order placement for fees"""
        fee_structure = self.fee_structures.get(exchange, {})
        
        optimization = {
            'prefer_post_only': False,
            'price_adjustment': 0,
            'use_discount_token': False,
            'estimated_fee': 0.0
        }
        
        # Check if we should use maker orders
        maker_fee = self.get_effective_fee(exchange, 'maker')
        taker_fee = self.get_effective_fee(exchange, 'taker')
        
        if maker_fee < taker_fee * 0.8:  # Significant maker advantage
            optimization['prefer_post_only'] = True
            
            # Adjust price to ensure maker order
            if order.side == 'buy' and orderbook.get('bids'):
                best_bid = orderbook['bids'][0][0]
                if order.price >= best_bid:
                    optimization['price_adjustment'] = -(order.price - best_bid + 0.0001)
                    
            elif order.side == 'sell' and orderbook.get('asks'):
                best_ask = orderbook['asks'][0][0]
                if order.price <= best_ask:
                    optimization['price_adjustment'] = best_ask - order.price + 0.0001
                    
        # Check for fee discount tokens
        if fee_structure.get('discount_token') and fee_structure.get('discount_rate') > 0:
            optimization['use_discount_token'] = True
            
        # Calculate estimated fee
        is_maker = optimization['prefer_post_only']
        fee_rate = maker_fee if is_maker else taker_fee
        optimization['estimated_fee'] = order.quantity * order.price * fee_rate
            
        return optimization
        
    def get_effective_fee(self, exchange: str, order_type: str) -> float:
        """Get effective fee rate including discounts and VIP tiers"""
        fee_structure = self.fee_structures.get(exchange, {})
        base_fee = fee_structure.get(order_type, 0.001)
        
        # Apply VIP tier discount
        volume = self.volume_tracking.get(exchange, 0)
        vip_tiers = fee_structure.get('vip_tiers', {})
        
        for tier in sorted(vip_tiers.keys(), reverse=True):
            if volume >= vip_tiers[tier]['volume']:
                base_fee = vip_tiers[tier][order_type]
                break
                
        # Apply discount token if available
        if fee_structure.get('discount_token'):
            base_fee *= (1 - fee_structure.get('discount_rate', 0))
            
        return base_fee
        
    def calculate_fee_impact(self, order: Order, exchange: str, is_maker: bool) -> float:
        """Calculate fee impact of order"""
        fee_rate = self.get_effective_fee(exchange, 'maker' if is_maker else 'taker')
        fee_amount = order.quantity * order.price * fee_rate
        
        # Track fees
        self.fee_tracking[exchange] += fee_amount
        self.volume_tracking[exchange] += order.quantity * order.price
        
        if is_maker:
            self.optimization_stats['maker_orders'] += 1
        else:
            self.optimization_stats['taker_orders'] += 1
            
        return fee_amount
        
    def get_cheapest_exchange(self, order_type: str = 'maker') -> str:
        """Get exchange with lowest fees"""
        cheapest = None
        lowest_fee = float('inf')
        
        for exchange, fees in self.fee_structures.items():
            effective_fee = self.get_effective_fee(exchange, order_type)
                
            if effective_fee < lowest_fee:
                lowest_fee = effective_fee
                cheapest = exchange
                
        return cheapest or 'binance'
        
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get fee optimization summary"""
        total_fees = sum(self.fee_tracking.values())
        total_volume = sum(self.volume_tracking.values())
        
        return {
            'total_fees_paid': total_fees,
            'total_volume': total_volume,
            'effective_fee_rate': total_fees / total_volume if total_volume > 0 else 0,
            'maker_percentage': self.optimization_stats['maker_orders'] / 
                               max(self.optimization_stats['maker_orders'] + 
                                   self.optimization_stats['taker_orders'], 1),
            'fees_by_exchange': dict(self.fee_tracking),
            'volume_by_exchange': dict(self.volume_tracking)
        }


class OrderValidator:
    """Enhanced order validation"""
    
    def __init__(self):
        self.validation_rules = {
            'min_size': self.validate_min_size,
            'max_slippage': self.validate_max_slippage,
            'price_sanity': self.validate_price_sanity,
            'risk_limits': self.validate_risk_limits,
            'market_conditions': self.validate_market_conditions,
            'duplicate_check': self.validate_no_duplicates,
            'balance_check': self.validate_sufficient_balance
        }
        self.validation_stats = defaultdict(int)
        self.recent_orders = deque(maxlen=100)
        
    async def validate_order(self, order: Order, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Comprehensive order validation"""
        failures = []
        
        # Track order for duplicate detection
        self.recent_orders.append({
            'symbol': order.symbol,
            'side': order.side,
            'price': order.price,
            'time': time.time()
        })
        
        for rule_name, validator in self.validation_rules.items():
            try:
                is_valid, message = await validator(order, context)
                if not is_valid:
                    failures.append(f"{rule_name}: {message}")
                    self.validation_stats[f"{rule_name}_failed"] += 1
                else:
                    self.validation_stats[f"{rule_name}_passed"] += 1
            except Exception as e:
                logger.error(f"Validation rule {rule_name} failed: {e}")
                failures.append(f"{rule_name}: error")
                
        return len(failures) == 0, failures
        
    async def validate_min_size(self, order: Order, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate minimum order size"""
        min_size = context.get('min_order_size', MIN_ORDER_SIZE)
        
        if order.quantity < min_size:
            return False, f"Order size {order.quantity} below minimum {min_size}"
            
        # Check notional value
        notional = order.quantity * order.price
        min_notional = context.get('min_notional', 10)  # $10 minimum
        
        if notional < min_notional:
            return False, f"Notional value ${notional:.2f} below minimum ${min_notional}"
            
        return True, "OK"
        
    async def validate_max_slippage(self, order: Order, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate maximum slippage"""
        if order.order_type != OrderType.LIMIT:
            return True, "OK"  # Only check for limit orders
            
        current_price = context.get('current_price', order.price)
        slippage = abs(order.price - current_price) / current_price
        
        max_allowed_slippage = context.get('max_slippage', MAX_SLIPPAGE)
        
        if slippage > max_allowed_slippage:
            return False, f"Slippage {slippage:.2%} exceeds maximum {max_allowed_slippage:.2%}"
            
        return True, "OK"
        
    async def validate_price_sanity(self, order: Order, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate price is reasonable"""
        current_price = context.get('current_price', order.price)
        
        # Check for zero or negative price
        if order.price <= 0:
            return False, f"Invalid price: {order.price}"
            
        # Check price deviation
        deviation = abs(order.price - current_price) / current_price
        max_deviation = context.get('max_price_deviation', 0.1)  # 10%
        
        if deviation > max_deviation:
            return False, f"Price deviation {deviation:.2%} exceeds maximum {max_deviation:.2%}"
            
        # Check for price manipulation attempts
        if 'orderbook' in context:
            orderbook = context['orderbook']
            if order.side == 'buy' and orderbook.get('asks'):
                best_ask = orderbook['asks'][0][0]
                if order.price > best_ask * 1.5:  # 50% above best ask
                    return False, f"Buy price significantly above market"
                    
            elif order.side == 'sell' and orderbook.get('bids'):
                best_bid = orderbook['bids'][0][0]
                if order.price < best_bid * 0.5:  # 50% below best bid
                    return False, f"Sell price significantly below market"
                    
        return True, "OK"
        
    async def validate_risk_limits(self, order: Order, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate against risk limits"""
        risk_limits = context.get('risk_limits', {})
        
        # Check position size
        max_position = risk_limits.get('max_position_size', float('inf'))
        if order.quantity * order.price > max_position:
            return False, f"Position size exceeds risk limit: ${max_position:.2f}"
            
        # Check daily volume
        daily_volume = context.get('daily_volume', 0)
        max_daily_volume = risk_limits.get('max_daily_volume', float('inf'))
        
        if daily_volume + (order.quantity * order.price) > max_daily_volume:
            return False, f"Would exceed daily volume limit: ${max_daily_volume:.2f}"
            
        # Check leverage
        if 'leverage' in context:
            max_leverage = risk_limits.get('max_leverage', 3.0)
            if context['leverage'] > max_leverage:
                return False, f"Leverage {context['leverage']:.1f}x exceeds limit {max_leverage:.1f}x"
                
        return True, "OK"
        
    async def validate_market_conditions(self, order: Order, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate market conditions are suitable"""
        market_conditions = context.get('market_conditions', {})
        
        # Check spread
        spread_bps = market_conditions.get('spread_bps', 0)
        max_spread = context.get('max_spread_bps', 50)  # 50 bps
        
        if spread_bps > max_spread:
            return False, f"Spread {spread_bps} bps exceeds maximum {max_spread} bps"
            
        # Check volatility
        volatility = market_conditions.get('volatility', 0)
        max_volatility = context.get('max_volatility', 0.1)  # 10%
        
        if volatility > max_volatility:
            return False, f"Volatility {volatility:.2%} exceeds maximum {max_volatility:.2%}"
            
        # Check liquidity
        if 'orderbook' in context:
            orderbook = context['orderbook']
            total_bid_volume = sum(bid[1] for bid in orderbook.get('bids', [])[:5])
            total_ask_volume = sum(ask[1] for ask in orderbook.get('asks', [])[:5])
            
            min_liquidity = context.get('min_liquidity', 100)  # $100 minimum
            
            if order.side == 'sell' and total_bid_volume * order.price < min_liquidity:
                return False, f"Insufficient bid liquidity: ${total_bid_volume * order.price:.2f}"
            elif order.side == 'buy' and total_ask_volume * order.price < min_liquidity:
                return False, f"Insufficient ask liquidity: ${total_ask_volume * order.price:.2f}"
                
        return True, "OK"
        
    async def validate_no_duplicates(self, order: Order, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check for duplicate orders"""
        # Look for similar recent orders
        duplicate_window = 5  # seconds
        current_time = time.time()
        
        for recent in self.recent_orders:
            if (recent['symbol'] == order.symbol and
                recent['side'] == order.side and
                abs(recent['price'] - order.price) / order.price < 0.001 and  # 0.1% price difference
                current_time - recent['time'] < duplicate_window):
                return False, f"Duplicate order detected within {duplicate_window}s"
                
        return True, "OK"
        
    async def validate_sufficient_balance(self, order: Order, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate sufficient balance for order"""
        if 'balance' not in context:
            return True, "OK"  # Skip if balance not provided
            
        balance = context['balance']
        required = order.quantity * order.price
        
        # Add buffer for fees
        fee_buffer = 1.005  # 0.5% buffer
        required_with_fees = required * fee_buffer
        
        if required_with_fees > balance:
            return False, f"Insufficient balance: ${balance:.2f} < ${required_with_fees:.2f}"
            
        return True, "OK"
        
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        total_validations = sum(v for k, v in self.validation_stats.items() if k.endswith('_passed'))
        total_failures = sum(v for k, v in self.validation_stats.items() if k.endswith('_failed'))
        
        rule_stats = {}
        for rule in self.validation_rules:
            passed = self.validation_stats.get(f"{rule}_passed", 0)
            failed = self.validation_stats.get(f"{rule}_failed", 0)
            total = passed + failed
            
            if total > 0:
                rule_stats[rule] = {
                    'pass_rate': passed / total,
                    'total_checks': total
                }
                
        return {
            'total_validations': total_validations + total_failures,
            'overall_pass_rate': total_validations / max(total_validations + total_failures, 1),
            'rule_statistics': rule_stats
        }


class ExecutionEngine:
    """
    High-performance order execution with fee optimization
    """
    
    def __init__(self, exchange_configs: Dict[str, Dict[str, Any]]):
        self.exchange_manager = ExchangeManager(exchange_configs)
        self.order_manager = OrderManager()
        self.order_batcher = OrderBatcher()
        self.fee_optimizer = FeeOptimizer()
        self.order_validator = OrderValidator()
        self.execution_queue = asyncio.Queue(maxsize=EXECUTION_QUEUE_SIZE)
        self.active_orders = {}  # order_id -> Order
        
        # Performance tracking
        self.metrics = ExecutionMetrics()
        self.latency_monitor = LatencyMonitor()
        
        # Execution strategies
        self.execution_strategies = {
            'aggressive': AggressiveExecutionStrategy(),
            'passive': PassiveExecutionStrategy(),
            'smart': SmartExecutionStrategy()
        }
        
        # State
        self._running = False
        self._workers = []
        self._lock = asyncio.Lock()
        
        logger.info("Initialized Execution Engine with Fee Optimization")
        
    async def start(self):
        """Start execution engine"""
        if self._running:
            return
            
        # Initialize exchanges
        await self.exchange_manager.initialize()
        
        # Start worker tasks
        self._running = True
        for i in range(4):  # 4 worker tasks
            worker = asyncio.create_task(self._execution_worker(i))
            self._workers.append(worker)
            
        logger.info("Started execution engine")
        
    async def stop(self):
        """Stop execution engine"""
        self._running = False
        
        # Cancel workers
        for worker in self._workers:
            worker.cancel()
            
        # Wait for completion
        await asyncio.gather(*self._workers, return_exceptions=True)
        
        # Close exchange connections
        await self.exchange_manager.close_all()
        
        logger.info("Stopped execution engine")
        
    async def execute_grid_strategy(
        self, 
        strategy_config: GridStrategyConfig, 
        risk_params: Dict[str, Any]
    ) -> List[ExecutionResult]:
        """Execute grid trading strategy"""
        start_time = time.perf_counter()
        
        try:
            # Get current price
            symbol = risk_params.get('symbol', 'BTCUSDT')
            current_price = await self.exchange_manager.get_current_price(symbol)
            
            # Get orderbook for fee optimization
            exchange = list(self.exchange_manager.exchanges.keys())[0]
            orderbook = await self.exchange_manager.get_orderbook(symbol, exchange)
            
            # Prepare grid orders
            grid_orders = await self._prepare_grid_orders(
                strategy_config, 
                risk_params, 
                current_price
            )
            
            # Create validation context
            validation_context = {
                'current_price': current_price,
                'orderbook': orderbook,
                'min_order_size': MIN_ORDER_SIZE,
                'risk_limits': risk_params.get('risk_limits', {}),
                'market_conditions': {
                    'spread_bps': self._calculate_spread_bps(orderbook),
                    'volatility': risk_params.get('volatility', 0.001)
                },
                'balance': risk_params.get('account_balance', 10000),
                'daily_volume': risk_params.get('daily_volume', 0)
            }
            
            # Pre-validate all orders
            validated_orders = await self._validate_orders(grid_orders, validation_context)
            
            # Apply fee optimization
            optimized_orders = await self._optimize_orders_for_fees(
                validated_orders, orderbook, exchange
            )
            
            # Choose execution strategy
            execution_strategy = self._select_execution_strategy(strategy_config)
            
            # Batch execution
            execution_results = await self._batch_execute(
                optimized_orders, 
                execution_strategy
            )
            
            # Track active orders
            for order, result in zip(optimized_orders, execution_results):
                if result.success:
                    self.active_orders[result.order.order_id] = result.order
                    await self.order_manager.add_order(result.order)
                    
            # Log performance
            total_latency = (time.perf_counter() - start_time) * 1000
            logger.info(f"Grid execution completed in {total_latency:.2f}ms")
            
            # Log fee optimization results
            fee_summary = self.fee_optimizer.get_optimization_summary()
            logger.info(f"Fee optimization - Maker rate: {fee_summary['maker_percentage']:.2%}")
            
            return execution_results
            
        except ExecutionError as e:
            logger.error(f"Execution failed: {e}")
            await self._handle_execution_failure(e, strategy_config)
            raise
            
    async def _prepare_grid_orders(
        self, 
        strategy_config: GridStrategyConfig, 
        risk_params: Dict[str, Any],
        current_price: float
    ) -> List[GridOrder]:
        """Prepare grid orders for execution"""
        orders = []
        
        # Calculate grid levels
        grid_levels = self._calculate_grid_levels(
            current_price,
            strategy_config.spacing,
            strategy_config.levels,
            strategy_config.grid_type.value
        )
        
        # Create orders for each level
        symbol = risk_params.get('symbol', 'BTCUSDT')
        position_size = risk_params.get('position_size', 0.01)
        
        for i, level in enumerate(grid_levels):
            # Determine order side
            if level['price'] < current_price:
                side = 'buy'
            else:
                side = 'sell'
                
            # Calculate order size
            if strategy_config.order_distribution.value == 'uniform':
                size = position_size / len(grid_levels)
            else:
                size = position_size * level['weight']
                
            # Create grid order
            order = GridOrder(
                order_id='',  # Will be assigned by exchange
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                price=self._round_price(level['price']),
                quantity=self._round_quantity(size / level['price']),
                time_in_force=self._get_time_in_force(strategy_config),
                grid_id=f"{strategy_config.regime.value}_{int(time.time())}",
                level_index=i,
                strategy=strategy_config.regime.value,
                distance_from_mid=level['distance'],
                metadata={
                    'strategy_config': strategy_config.to_dict(),
                    'risk_params': risk_params
                }
            )
            
            orders.append(order)
            
        return orders
        
    def _calculate_grid_levels(
        self, 
        current_price: float, 
        spacing: float, 
        levels: int,
        grid_type: str
    ) -> List[Dict[str, Any]]:
        """Calculate grid price levels"""
        grid_levels = []
        
        if grid_type == 'symmetric':
            # Equal spacing above and below
            for i in range(1, levels + 1):
                # Buy levels
                buy_price = current_price * (1 - spacing * i)
                grid_levels.append({
                    'price': buy_price,
                    'distance': spacing * i,
                    'weight': 1.0 / (levels * 2)
                })
                
                # Sell levels
                sell_price = current_price * (1 + spacing * i)
                grid_levels.append({
                    'price': sell_price,
                    'distance': spacing * i,
                    'weight': 1.0 / (levels * 2)
                })
                
        elif grid_type == 'geometric':
            # Geometric progression
            multiplier = 1.2
            current_spacing = spacing
            
            for i in range(1, levels + 1):
                # Buy levels
                buy_price = current_price * (1 - current_spacing)
                grid_levels.append({
                    'price': buy_price,
                    'distance': current_spacing,
                    'weight': 1.0 / (levels * 2)
                })
                
                # Sell levels
                sell_price = current_price * (1 + current_spacing)
                grid_levels.append({
                    'price': sell_price,
                    'distance': current_spacing,
                    'weight': 1.0 / (levels * 2)
                })
                
                current_spacing *= multiplier
                
        # Add more grid types as needed
        
        return grid_levels
        
    def _calculate_spread_bps(self, orderbook: Dict[str, Any]) -> float:
        """Calculate spread in basis points"""
        if not orderbook.get('bids') or not orderbook.get('asks'):
            return 0.0
            
        best_bid = orderbook['bids'][0][0]
        best_ask = orderbook['asks'][0][0]
        
        if best_bid <= 0:
            return 0.0
            
        spread = best_ask - best_bid
        spread_bps = (spread / best_bid) * 10000
        
        return spread_bps
        
    def _round_price(self, price: float) -> float:
        """Round price to exchange precision"""
        return round(price, PRICE_PRECISION)
        
    def _round_quantity(self, quantity: float) -> float:
        """Round quantity to exchange precision"""
        return round(quantity, QUANTITY_PRECISION)
        
    def _get_time_in_force(self, strategy_config: GridStrategyConfig) -> TimeInForce:
        """Get time in force based on strategy"""
        execution_rules = strategy_config.execution_rules
        
        if execution_rules.get('post_only'):
            return TimeInForce.POST_ONLY
        elif execution_rules.get('time_in_force') == 'IOC':
            return TimeInForce.IOC
        else:
            return TimeInForce.GTC
            
    async def _validate_orders(
        self, 
        orders: List[GridOrder], 
        context: Dict[str, Any]
    ) -> List[GridOrder]:
        """Enhanced validation with comprehensive checks"""
        validated = []
        
        for order in orders:
            try:
                # Enhanced validation
                is_valid, failures = await self.order_validator.validate_order(order, context)
                
                if not is_valid:
                    logger.warning(f"Order validation failed: {failures}")
                    continue
                    
                validated.append(order)
                
            except Exception as e:
                logger.error(f"Order validation failed: {e}")
                
        logger.info(f"Validated {len(validated)}/{len(orders)} orders")
        
        # Log validation statistics
        val_stats = self.order_validator.get_validation_stats()
        logger.info(f"Validation pass rate: {val_stats['overall_pass_rate']:.2%}")
        
        return validated
        
    async def _optimize_orders_for_fees(
        self, 
        orders: List[GridOrder], 
        orderbook: Dict[str, Any],
        exchange: str
    ) -> List[GridOrder]:
        """Apply fee optimization to orders"""
        optimized = []
        
        for order in orders:
            # Get fee optimization recommendations
            optimization = self.fee_optimizer.optimize_order_placement(
                order, orderbook, exchange
            )
            
            # Apply optimizations
            if optimization['prefer_post_only']:
                order.time_in_force = TimeInForce.POST_ONLY
                
            if optimization['price_adjustment'] != 0:
                order.price += optimization['price_adjustment']
                order.price = self._round_price(order.price)
                
            # Add fee estimate to metadata
            order.metadata['estimated_fee'] = optimization['estimated_fee']
            order.metadata['fee_optimized'] = True
            
            optimized.append(order)
            
        return optimized
        
    def _select_execution_strategy(self, strategy_config: GridStrategyConfig) -> Any:
        """Select execution strategy based on config"""
        if strategy_config.regime.value == 'VOLATILE':
            return self.execution_strategies['aggressive']
        elif strategy_config.execution_rules.get('post_only'):
            return self.execution_strategies['passive']
        else:
            return self.execution_strategies['smart']
            
    async def _batch_execute(
        self, 
        orders: List[GridOrder], 
        execution_strategy: Any
    ) -> List[ExecutionResult]:
        """Execute orders in optimized batches"""
        # Group by exchange
        exchange_groups = defaultdict(list)
        
        # For this example, assume single exchange
        exchange = list(self.exchange_manager.exchanges.keys())[0]
        exchange_groups[exchange] = orders
        
        # Execute in parallel per exchange
        tasks = []
        for exchange, exchange_orders in exchange_groups.items():
            task = self._execute_on_exchange(exchange, exchange_orders, execution_strategy)
            tasks.append(task)
            
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_results = []
        for exchange_results in results:
            all_results.extend(exchange_results)
            
        return all_results
        
    async def _execute_on_exchange(
        self, 
        exchange_name: str, 
        orders: List[GridOrder],
        execution_strategy: Any
    ) -> List[ExecutionResult]:
        """Execute orders on specific exchange"""
        results = []
        exchange = self.exchange_manager.get_exchange(exchange_name)
        rate_limiter = self.exchange_manager.rate_limiters[exchange_name]
        
        # Execute orders with rate limiting
        for order in orders:
            try:
                # Rate limit
                await rate_limiter.acquire()
                
                # Execute based on strategy
                result = await execution_strategy.execute(order, exchange, self)
                
                # Calculate actual fees
                if result.success:
                    is_maker = order.time_in_force == TimeInForce.POST_ONLY
                    actual_fee = self.fee_optimizer.calculate_fee_impact(
                        order, exchange_name, is_maker
                    )
                    result.order.fees = actual_fee
                
                # Update metrics
                self.metrics.add_execution(result)
                results.append(result)
                
                # Add small delay between orders
                await asyncio.sleep(MIN_ORDER_INTERVAL)
                
            except Exception as e:
                logger.error(f"Order execution failed: {e}")
                result = ExecutionResult(
                    success=False,
                    order=order,
                    latency=0.0,
                    error=str(e)
                )
                results.append(result)
                
        return results
        
    async def _execution_worker(self, worker_id: int):
        """Worker task for processing execution queue"""
        logger.info(f"Execution worker {worker_id} started")
        
        while self._running:
            try:
                # Get order from queue
                order = await asyncio.wait_for(
                    self.execution_queue.get(),
                    timeout=1.0
                )
                
                # Process order
                await self._process_queued_order(order)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                
        logger.info(f"Execution worker {worker_id} stopped")
        
    async def _process_queued_order(self, order: Order):
        """Process order from queue"""
        try:
            # Determine exchange
            exchange_name = order.metadata.get('exchange', list(self.exchange_manager.exchanges.keys())[0])
            exchange = self.exchange_manager.get_exchange(exchange_name)
            
            # Execute order
            start_time = time.perf_counter()
            
            if order.order_type == OrderType.MARKET:
                response = await exchange.create_market_order(
                    order.symbol,
                    order.side,
                    order.quantity
                )
            else:  # LIMIT
                response = await exchange.create_limit_order(
                    order.symbol,
                    order.side,
                    order.quantity,
                    order.price
                )
                
            # Update order
            order.order_id = response['id']
            order.status = self._map_exchange_status(response['status'])
            
            # Calculate latency
            latency = (time.perf_counter() - start_time) * 1000
            self.latency_monitor.add_latency(latency)
            
            # Update order manager
            await self.order_manager.update_order(order.order_id, {
                'status': order.status,
                'filled_quantity': response.get('filled', 0),
                'average_price': response.get('average', order.price)
            })
            
        except Exception as e:
            logger.error(f"Failed to process order: {e}")
            order.status = OrderStatus.FAILED
            
    def _map_exchange_status(self, exchange_status: str) -> OrderStatus:
        """Map exchange status to internal status"""
        status_map = {
            'open': OrderStatus.SUBMITTED,
            'partially_filled': OrderStatus.PARTIAL,
            'filled': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'rejected': OrderStatus.REJECTED,
            'expired': OrderStatus.EXPIRED
        }
        
        return status_map.get(exchange_status.lower(), OrderStatus.PENDING)
        
    async def cancel_order(self, order_id: str, exchange_name: Optional[str] = None):
        """Cancel specific order"""
        try:
            # Get order
            order = await self.order_manager.get_order(order_id)
            if not order:
                logger.warning(f"Order {order_id} not found")
                return
                
            # Determine exchange
            if not exchange_name:
                exchange_name = order.metadata.get('exchange', list(self.exchange_manager.exchanges.keys())[0])
                
            exchange = self.exchange_manager.get_exchange(exchange_name)
            
            # Cancel on exchange
            await exchange.cancel_order(order_id, order.symbol)
            
            # Update status
            await self.order_manager.cancel_order(order_id)
            
            logger.info(f"Cancelled order {order_id}")
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            
    async def cancel_all_orders(self, symbol: Optional[str] = None):
        """Cancel all active orders"""
        active_orders = await self.order_manager.get_active_orders(symbol)
        
        cancel_tasks = []
        for order in active_orders:
            task = self.cancel_order(order.order_id)
            cancel_tasks.append(task)
            
        await asyncio.gather(*cancel_tasks, return_exceptions=True)
        
        logger.info(f"Cancelled {len(active_orders)} orders")
        
    async def update_order_status(self, order_id: str):
        """Update order status from exchange"""
        try:
            order = await self.order_manager.get_order(order_id)
            if not order or order.is_complete():
                return
                
            # Get from exchange
            exchange_name = order.metadata.get('exchange', list(self.exchange_manager.exchanges.keys())[0])
            exchange = self.exchange_manager.get_exchange(exchange_name)
            
            response = await exchange.fetch_order(order_id, order.symbol)
            
            # Update order
            updates = {
                'status': self._map_exchange_status(response['status']),
                'filled_quantity': response.get('filled', 0),
                'average_price': response.get('average', order.price),
                'fees': response.get('fee', {}).get('cost', 0)
            }
            
            await self.order_manager.update_order(order_id, updates)
            
        except Exception as e:
            logger.error(f"Failed to update order status: {e}")
            
    async def _handle_execution_failure(self, error: ExecutionError, strategy_config: GridStrategyConfig):
        """Handle execution failure"""
        logger.error(f"Execution failure for {strategy_config.regime.value}: {error}")
        
        # Cancel any partial orders
        await self.cancel_all_orders()
        
        # Notify risk system
        # This would integrate with risk management
        
    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        fee_summary = self.fee_optimizer.get_optimization_summary()
        validation_stats = self.order_validator.get_validation_stats()
        
        return {
            'metrics': {
                'total_orders': self.metrics.total_orders,
                'success_rate': self.metrics.get_success_rate(),
                'average_latency': self.metrics.get_average_latency(),
                'min_latency': self.metrics.min_latency,
                'max_latency': self.metrics.max_latency
            },
            'active_orders': len(self.active_orders),
            'order_manager': {
                'active': self.order_manager.get_order_count(),
                'completed': len(self.order_manager.completed_orders)
            },
            'rate_limits': {
                exchange: limiter.get_remaining_calls()
                for exchange, limiter in self.exchange_manager.rate_limiters.items()
            },
            'fee_optimization': fee_summary,
            'validation': validation_stats
        }


class ExecutionStrategy:
    """Base class for execution strategies"""
    
    async def execute(self, order: Order, exchange: Exchange, engine: ExecutionEngine) -> ExecutionResult:
        """Execute order using strategy"""
        raise NotImplementedError


class AggressiveExecutionStrategy(ExecutionStrategy):
    """Aggressive execution - prioritize speed"""
    
    async def execute(self, order: Order, exchange: Exchange, engine: ExecutionEngine) -> ExecutionResult:
        """Execute aggressively"""
        start_time = time.perf_counter()
        
        try:
            # Use market order for immediate execution
            if order.order_type == OrderType.LIMIT:
                # Convert to aggressive limit
                orderbook = await engine.exchange_manager.get_orderbook(order.symbol, exchange.name)
                
                if order.side == 'buy':
                    # Place at best ask
                    order.price = orderbook['asks'][0][0] if orderbook['asks'] else order.price
                else:
                    # Place at best bid
                    order.price = orderbook['bids'][0][0] if orderbook['bids'] else order.price
                    
            # Execute
            response = await exchange.create_order(
                order.symbol,
                order.order_type.value,
                order.side,
                order.quantity,
                order.price if order.order_type == OrderType.LIMIT else None
            )
            
            # Update order
            order.order_id = response['id']
            order.status = OrderStatus.SUBMITTED
            
            latency = (time.perf_counter() - start_time) * 1000
            
            return ExecutionResult(
                success=True,
                order=order,
                latency=latency
            )
            
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                success=False,
                order=order,
                latency=latency,
                error=str(e)
            )


class PassiveExecutionStrategy(ExecutionStrategy):
    """Passive execution - prioritize price"""
    
    async def execute(self, order: Order, exchange: Exchange, engine: ExecutionEngine) -> ExecutionResult:
        """Execute passively"""
        start_time = time.perf_counter()
        
        try:
            # Ensure post-only
            order.time_in_force = TimeInForce.POST_ONLY
            
            # Improve price for maker rebate
            orderbook = await engine.exchange_manager.get_orderbook(order.symbol, exchange.name)
            
            if order.side == 'buy':
                # Place below best bid
                best_bid = orderbook['bids'][0][0] if orderbook['bids'] else order.price
                order.price = min(order.price, best_bid - 0.0001)
            else:
                # Place above best ask
                best_ask = orderbook['asks'][0][0] if orderbook['asks'] else order.price
                order.price = max(order.price, best_ask + 0.0001)
                
            # Execute
            response = await exchange.create_order(
                order.symbol,
                'limit',
                order.side,
                order.quantity,
                order.price,
                {'postOnly': True}
            )
            
            order.order_id = response['id']
            order.status = OrderStatus.SUBMITTED
            
            latency = (time.perf_counter() - start_time) * 1000
            
            return ExecutionResult(
                success=True,
                order=order,
                latency=latency
            )
            
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                success=False,
                order=order,
                latency=latency,
                error=str(e)
            )


class SmartExecutionStrategy(ExecutionStrategy):
    """Smart execution - balance speed and price"""
    
    async def execute(self, order: Order, exchange: Exchange, engine: ExecutionEngine) -> ExecutionResult:
        """Execute with smart routing"""
        start_time = time.perf_counter()
        
        try:
            # Get market conditions
            orderbook = await engine.exchange_manager.get_orderbook(order.symbol, exchange.name)
            spread = (orderbook['asks'][0][0] - orderbook['bids'][0][0]) if orderbook['asks'] and orderbook['bids'] else 0
            
            # Decide execution method based on spread
            if spread < 0.0001:  # Tight spread
                # Use aggressive execution
                strategy = AggressiveExecutionStrategy()
            else:  # Wide spread
                # Use passive execution
                strategy = PassiveExecutionStrategy()
                
            return await strategy.execute(order, exchange, engine)
            
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                success=False,
                order=order,
                latency=latency,
                error=str(e)
            )


class LatencyMonitor:
    """Monitor execution latency"""
    
    def __init__(self, window_size: int = 1000):
        self.latencies = deque(maxlen=window_size)
        self.target_latency = LATENCY_TARGET * 1000  # Convert to ms
        
    def add_latency(self, latency: float):
        """Add latency measurement"""
        self.latencies.append(latency)
        
    def get_percentile(self, percentile: float) -> float:
        """Get latency percentile"""
        if not self.latencies:
            return 0.0
        return np.percentile(list(self.latencies), percentile)
        
    def is_meeting_target(self) -> bool:
       """Check if meeting latency target"""
       if not self.latencies:
           return True
       p99 = self.get_percentile(99)
       return p99 <= self.target_latency
       
   def get_statistics(self) -> Dict[str, float]:
       """Get latency statistics"""
       if not self.latencies:
           return {
               'mean': 0.0,
               'p50': 0.0,
               'p95': 0.0,
               'p99': 0.0,
               'min': 0.0,
               'max': 0.0
           }
           
       latency_list = list(self.latencies)
       return {
           'mean': np.mean(latency_list),
           'p50': np.percentile(latency_list, 50),
           'p95': np.percentile(latency_list, 95),
           'p99': np.percentile(latency_list, 99),
           'min': min(latency_list),
           'max': max(latency_list)
       }


# Example usage
async def main():
   """Example usage of ExecutionEngine"""
   
   # Exchange configuration
   exchange_configs = {
       'binance': {
           'class': 'binance',
           'api_key': 'your_api_key',
           'secret': 'your_secret',
           'rate_limit': 50,
           'rate_window': 1000,
           'websocket_url': 'wss://stream.binance.com:9443/ws',
           'options': {
               'defaultType': 'spot'
           }
       }
   }
   
   # Initialize engine
   engine = ExecutionEngine(exchange_configs)
   await engine.start()
   
   try:
       # Create sample grid strategy config
       from core.grid_strategy_selector import GridStrategyConfig, GridType, OrderDistribution
       from core.market_regime_detector import MarketRegime
       
       strategy_config = GridStrategyConfig(
           regime=MarketRegime.RANGING,
           grid_type=GridType.SYMMETRIC,
           spacing=0.001,
           levels=5,
           position_size=0.01,
           order_distribution=OrderDistribution.UNIFORM,
           risk_limits={'max_position': 1000},
           execution_rules={'post_only': True}
       )
       
       risk_params = {
           'symbol': 'BTC/USDT',
           'position_size': 1000,
           'account_balance': 10000,
           'volatility': 0.001,
           'risk_limits': {
               'max_position_size': 2000,
               'max_daily_volume': 10000,
               'max_leverage': 3.0
           }
       }
       
       # Execute grid
       print("Executing grid strategy...")
       results = await engine.execute_grid_strategy(strategy_config, risk_params)
       
       # Print results
       successful = sum(1 for r in results if r.success)
       print(f"\nExecution Results:")
       print(f"  Total orders: {len(results)}")
       print(f"  Successful: {successful}")
       print(f"  Failed: {len(results) - successful}")
       
       if results:
           avg_latency = sum(r.latency for r in results) / len(results)
           print(f"  Average latency: {avg_latency:.2f}ms")
           
       # Get execution statistics
       stats = await engine.get_execution_stats()
       print(f"\nExecution Statistics:")
       print(f"  Success rate: {stats['metrics']['success_rate']:.2%}")
       print(f"  Average latency: {stats['metrics']['average_latency']:.2f}ms")
       print(f"  Active orders: {stats['active_orders']}")
       
       # Fee optimization statistics
       fee_stats = stats['fee_optimization']
       print(f"\nFee Optimization:")
       print(f"  Effective fee rate: {fee_stats['effective_fee_rate']:.4%}")
       print(f"  Maker order percentage: {fee_stats['maker_percentage']:.2%}")
       print(f"  Total fees paid: ${fee_stats['total_fees_paid']:.2f}")
       
       # Validation statistics
       val_stats = stats['validation']
       print(f"\nValidation Statistics:")
       print(f"  Overall pass rate: {val_stats['overall_pass_rate']:.2%}")
       print(f"  Total validations: {val_stats['total_validations']}")
       
       # Latency monitoring
       latency_stats = engine.latency_monitor.get_statistics()
       print(f"\nLatency Statistics:")
       print(f"  Mean: {latency_stats['mean']:.2f}ms")
       print(f"  P95: {latency_stats['p95']:.2f}ms")
       print(f"  P99: {latency_stats['p99']:.2f}ms")
       print(f"  Meeting target: {engine.latency_monitor.is_meeting_target()}")
       
       # Wait a bit for orders to process
       await asyncio.sleep(5)
       
       # Update order statuses
       print("\nUpdating order statuses...")
       for order_id in list(engine.active_orders.keys())[:5]:  # Update first 5
           await engine.update_order_status(order_id)
           
       # Cancel remaining orders
       print("\nCancelling all orders...")
       await engine.cancel_all_orders()
       
   finally:
       # Stop engine
       await engine.stop()
       print("\nExecution engine stopped")


if __name__ == "__main__":
   asyncio.run(main())