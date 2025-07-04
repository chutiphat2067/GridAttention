#!/usr/bin/env python3
"""
Mock Exchange for GridAttention Trading System
Provides a realistic exchange simulation for testing trading strategies
"""

import asyncio
import time
import random
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_DOWN
import threading
from concurrent.futures import ThreadPoolExecutor


logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status types"""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    PENDING_CANCEL = "PENDING_CANCEL"


class OrderType(Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"
    LIMIT_MAKER = "LIMIT_MAKER"


class TimeInForce(Enum):
    """Time in force types"""
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTX = "GTX"  # Good Till Crossing


@dataclass
class Order:
    """Order data structure"""
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    price: Decimal
    quantity: Decimal
    time_in_force: str = "GTC"
    
    # Status tracking
    status: str = OrderStatus.NEW.value
    executed_quantity: Decimal = Decimal("0")
    cumulative_quote_quantity: Decimal = Decimal("0")
    
    # Timestamps
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: datetime = field(default_factory=datetime.now)
    
    # Optional fields
    stop_price: Optional[Decimal] = None
    iceberg_quantity: Optional[Decimal] = None
    
    # Execution details
    fills: List[Dict] = field(default_factory=list)
    commission_paid: Decimal = Decimal("0")
    
    @property
    def remaining_quantity(self) -> Decimal:
        return self.quantity - self.executed_quantity
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.NEW.value, OrderStatus.PARTIALLY_FILLED.value]
    
    @property
    def average_price(self) -> Decimal:
        if self.executed_quantity > 0:
            return self.cumulative_quote_quantity / self.executed_quantity
        return Decimal("0")


@dataclass
class Trade:
    """Trade/Fill data structure"""
    trade_id: str
    order_id: str
    symbol: str
    price: Decimal
    quantity: Decimal
    side: str
    maker: bool
    commission: Decimal
    commission_asset: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def quote_quantity(self) -> Decimal:
        return self.price * self.quantity


@dataclass
class Balance:
    """Asset balance"""
    asset: str
    free: Decimal = Decimal("0")
    locked: Decimal = Decimal("0")
    
    @property
    def total(self) -> Decimal:
        return self.free + self.locked
    
    def lock(self, amount: Decimal):
        """Lock amount for order"""
        if amount > self.free:
            raise ValueError(f"Insufficient balance: need {amount}, have {self.free}")
        self.free -= amount
        self.locked += amount
    
    def unlock(self, amount: Decimal):
        """Unlock amount from order"""
        if amount > self.locked:
            amount = self.locked
        self.locked -= amount
        self.free += amount
    
    def deduct(self, amount: Decimal):
        """Deduct amount (for fees)"""
        if amount > self.free:
            raise ValueError(f"Insufficient balance for fee: need {amount}, have {self.free}")
        self.free -= amount


class OrderBook:
    """Order book implementation"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: Dict[Decimal, Decimal] = {}  # price -> quantity
        self.asks: Dict[Decimal, Decimal] = {}  # price -> quantity
        self.last_update_id = 0
        self.lock = threading.Lock()
    
    def update(self, bids: List[Tuple[Decimal, Decimal]], asks: List[Tuple[Decimal, Decimal]]):
        """Update order book"""
        with self.lock:
            # Update bids
            for price, quantity in bids:
                if quantity == 0:
                    self.bids.pop(price, None)
                else:
                    self.bids[price] = quantity
            
            # Update asks
            for price, quantity in asks:
                if quantity == 0:
                    self.asks.pop(price, None)
                else:
                    self.asks[price] = quantity
            
            self.last_update_id += 1
    
    def get_best_bid(self) -> Optional[Decimal]:
        """Get best bid price"""
        with self.lock:
            return max(self.bids.keys()) if self.bids else None
    
    def get_best_ask(self) -> Optional[Decimal]:
        """Get best ask price"""
        with self.lock:
            return min(self.asks.keys()) if self.asks else None
    
    def get_spread(self) -> Optional[Decimal]:
        """Get bid-ask spread"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return best_ask - best_bid
        return None
    
    def get_mid_price(self) -> Optional[Decimal]:
        """Get mid price"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        return None
    
    def get_depth(self, levels: int = 10) -> Dict[str, List[List[str]]]:
        """Get order book depth"""
        with self.lock:
            sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:levels]
            sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:levels]
            
            return {
                "bids": [[str(price), str(qty)] for price, qty in sorted_bids],
                "asks": [[str(price), str(qty)] for price, qty in sorted_asks],
                "lastUpdateId": self.last_update_id
            }


class MockExchange:
    """Mock exchange implementation"""
    
    def __init__(self, name: str = "MockExchange"):
        self.name = name
        self.exchange_id = uuid.uuid4().hex[:8]
        
        # Order management
        self.orders: Dict[str, Order] = {}
        self.orders_by_client_id: Dict[str, Order] = {}
        self.active_orders: Dict[str, Set[str]] = defaultdict(set)  # symbol -> order_ids
        
        # Balances
        self.balances: Dict[str, Balance] = defaultdict(lambda: Balance(""))
        
        # Order books
        self.order_books: Dict[str, OrderBook] = {}
        
        # Trade history
        self.trades: List[Trade] = []
        self.trades_by_symbol: Dict[str, List[Trade]] = defaultdict(list)
        
        # Market data
        self.tickers: Dict[str, Dict[str, Any]] = {}
        self.klines: Dict[str, pd.DataFrame] = {}
        
        # Configuration
        self.maker_fee = Decimal("0.001")  # 0.1%
        self.taker_fee = Decimal("0.0015")  # 0.15%
        self.min_notional = Decimal("10")  # Minimum order value
        
        # Rate limiting
        self.rate_limiter = RateLimiter(
            requests_per_minute=1200,
            weight_per_minute=1200
        )
        
        # Execution engine
        self.matching_engine = MatchingEngine(self)
        self.price_engine = PriceEngine()
        
        # WebSocket connections
        self.ws_connections: Set[str] = set()
        self.ws_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = True
        
        # Initialize with some default assets
        self._initialize_balances()
        self._initialize_markets()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _initialize_balances(self):
        """Initialize default balances"""
        default_balances = {
            "BTC": Decimal("1.0"),
            "ETH": Decimal("10.0"),
            "USDT": Decimal("50000.0"),
            "BNB": Decimal("50.0")
        }
        
        for asset, amount in default_balances.items():
            self.balances[asset] = Balance(asset, free=amount)
    
    def _initialize_markets(self):
        """Initialize market data"""
        markets = [
            ("BTCUSDT", 45000, 0.02),
            ("ETHUSDT", 3000, 0.025),
            ("BNBUSDT", 450, 0.02),
            ("SOLUSDT", 120, 0.03)
        ]
        
        for symbol, base_price, volatility in markets:
            # Initialize order book
            self.order_books[symbol] = OrderBook(symbol)
            self._generate_orderbook(symbol, Decimal(str(base_price)))
            
            # Initialize ticker
            self.tickers[symbol] = self._generate_ticker(symbol, base_price)
            
            # Initialize price engine
            self.price_engine.add_market(symbol, base_price, volatility)
    
    def _start_background_tasks(self):
        """Start background tasks"""
        # Price updates
        self.executor.submit(self._price_update_loop)
        
        # Order matching
        self.executor.submit(self._order_matching_loop)
        
        # WebSocket broadcasts
        self.executor.submit(self._websocket_broadcast_loop)
    
    async def _price_update_loop(self):
        """Update prices periodically"""
        while self.running:
            try:
                for symbol in self.order_books:
                    # Update price
                    new_price = self.price_engine.get_next_price(symbol)
                    
                    # Update order book
                    self._generate_orderbook(symbol, new_price)
                    
                    # Update ticker
                    self.tickers[symbol] = self._generate_ticker(symbol, float(new_price))
                    
                    # Trigger order matching
                    await self._match_orders(symbol)
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Price update error: {e}")
    
    async def _order_matching_loop(self):
        """Match orders periodically"""
        while self.running:
            try:
                for symbol in self.active_orders:
                    await self._match_orders(symbol)
                
                await asyncio.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                logger.error(f"Order matching error: {e}")
    
    async def _websocket_broadcast_loop(self):
        """Broadcast updates to WebSocket connections"""
        while self.running:
            try:
                # Broadcast ticker updates
                for symbol, ticker in self.tickers.items():
                    await self._broadcast_ticker(symbol, ticker)
                
                # Broadcast order book updates
                for symbol, book in self.order_books.items():
                    await self._broadcast_orderbook(symbol, book.get_depth())
                
                await asyncio.sleep(1)  # Broadcast every second
                
            except Exception as e:
                logger.error(f"WebSocket broadcast error: {e}")
    
    # Exchange API Methods
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information"""
        symbols = []
        
        for symbol in self.order_books:
            base_asset = symbol[:-4]  # Assume USDT pairs
            quote_asset = "USDT"
            
            symbols.append({
                "symbol": symbol,
                "status": "TRADING",
                "baseAsset": base_asset,
                "baseAssetPrecision": 8,
                "quoteAsset": quote_asset,
                "quotePrecision": 8,
                "baseCommissionPrecision": 8,
                "quoteCommissionPrecision": 8,
                "orderTypes": [t.value for t in OrderType],
                "icebergAllowed": True,
                "ocoAllowed": True,
                "quoteOrderQtyMarketAllowed": True,
                "isSpotTradingAllowed": True,
                "isMarginTradingAllowed": False,
                "filters": [
                    {
                        "filterType": "PRICE_FILTER",
                        "minPrice": "0.01",
                        "maxPrice": "1000000.00",
                        "tickSize": "0.01"
                    },
                    {
                        "filterType": "LOT_SIZE",
                        "minQty": "0.00001",
                        "maxQty": "9000.00",
                        "stepSize": "0.00001"
                    },
                    {
                        "filterType": "MIN_NOTIONAL",
                        "minNotional": str(self.min_notional)
                    }
                ]
            })
        
        return {
            "timezone": "UTC",
            "serverTime": int(datetime.now().timestamp() * 1000),
            "rateLimits": [
                {
                    "rateLimitType": "REQUEST_WEIGHT",
                    "interval": "MINUTE",
                    "intervalNum": 1,
                    "limit": 1200
                }
            ],
            "symbols": symbols
        }
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get 24hr ticker"""
        if symbol not in self.tickers:
            raise ValueError(f"Invalid symbol: {symbol}")
        
        self.rate_limiter.consume(1)
        return self.tickers[symbol]
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get order book"""
        if symbol not in self.order_books:
            raise ValueError(f"Invalid symbol: {symbol}")
        
        self.rate_limiter.consume(1 if limit <= 100 else 5)
        return self.order_books[symbol].get_depth(limit)
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100
    ) -> List[List[Any]]:
        """Get kline/candlestick data"""
        if symbol not in self.order_books:
            raise ValueError(f"Invalid symbol: {symbol}")
        
        self.rate_limiter.consume(1)
        
        # Generate klines if not exists
        if symbol not in self.klines:
            self.klines[symbol] = self._generate_klines(symbol, interval, limit)
        
        df = self.klines[symbol].tail(limit)
        
        # Convert to API format
        klines = []
        for _, row in df.iterrows():
            klines.append([
                int(row['timestamp'].timestamp() * 1000),
                str(row['open']),
                str(row['high']),
                str(row['low']),
                str(row['close']),
                str(row['volume']),
                int(row['close_time'].timestamp() * 1000),
                str(row['quote_volume']),
                int(row['trades']),
                str(row['taker_buy_volume']),
                str(row['taker_buy_quote_volume']),
                "0"
            ])
        
        return klines
    
    async def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        self.rate_limiter.consume(10)
        
        balances = []
        for asset, balance in self.balances.items():
            if balance.total > 0:
                balances.append({
                    "asset": asset,
                    "free": str(balance.free),
                    "locked": str(balance.locked)
                })
        
        return {
            "makerCommission": int(self.maker_fee * 10000),
            "takerCommission": int(self.taker_fee * 10000),
            "buyerCommission": 0,
            "sellerCommission": 0,
            "canTrade": True,
            "canWithdraw": True,
            "canDeposit": True,
            "updateTime": int(datetime.now().timestamp() * 1000),
            "accountType": "SPOT",
            "balances": balances,
            "permissions": ["SPOT"]
        }
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Place new order"""
        self.rate_limiter.consume(1)
        
        # Validate symbol
        if symbol not in self.order_books:
            raise ValueError(f"Invalid symbol: {symbol}")
        
        # Create order
        order_id = str(uuid.uuid4().hex[:16])
        client_order_id = kwargs.get('newClientOrderId', f"client_{order_id}")
        
        # Parse parameters
        quantity = Decimal(str(kwargs.get('quantity', 0)))
        price = Decimal(str(kwargs.get('price', 0))) if order_type != OrderType.MARKET.value else None
        
        # Validate order
        validation_error = self._validate_order(symbol, side, order_type, quantity, price)
        if validation_error:
            raise ValueError(validation_error)
        
        # Create order object
        order = Order(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side.upper(),
            order_type=order_type.upper(),
            price=price or Decimal("0"),
            quantity=quantity,
            time_in_force=kwargs.get('timeInForce', 'GTC')
        )
        
        # Lock balance
        if side.upper() == "BUY":
            # Lock quote asset (USDT)
            amount_to_lock = quantity * price if price else quantity * self._get_market_price(symbol)
            self.balances["USDT"].lock(amount_to_lock)
        else:
            # Lock base asset
            base_asset = symbol[:-4]
            self.balances[base_asset].lock(quantity)
        
        # Store order
        self.orders[order_id] = order
        self.orders_by_client_id[client_order_id] = order
        self.active_orders[symbol].add(order_id)
        
        # Try immediate execution for market orders
        if order_type == OrderType.MARKET.value:
            await self._execute_market_order(order)
        
        # Return order response
        return {
            "symbol": order.symbol,
            "orderId": order.order_id,
            "orderListId": -1,
            "clientOrderId": order.client_order_id,
            "transactTime": int(order.created_time.timestamp() * 1000),
            "price": str(order.price),
            "origQty": str(order.quantity),
            "executedQty": str(order.executed_quantity),
            "cummulativeQuoteQty": str(order.cumulative_quote_quantity),
            "status": order.status,
            "timeInForce": order.time_in_force,
            "type": order.order_type,
            "side": order.side,
            "fills": order.fills
        }
    
    async def cancel_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Cancel order"""
        self.rate_limiter.consume(1)
        
        # Find order
        order = None
        if order_id and order_id in self.orders:
            order = self.orders[order_id]
        elif client_order_id and client_order_id in self.orders_by_client_id:
            order = self.orders_by_client_id[client_order_id]
        
        if not order:
            raise ValueError("Order not found")
        
        if order.symbol != symbol:
            raise ValueError("Order symbol mismatch")
        
        if not order.is_active:
            raise ValueError("Order is not active")
        
        # Cancel order
        order.status = OrderStatus.CANCELED.value
        order.updated_time = datetime.now()
        
        # Unlock balance
        if order.side == "BUY":
            # Unlock remaining USDT
            remaining_value = order.remaining_quantity * order.price
            self.balances["USDT"].unlock(remaining_value)
        else:
            # Unlock remaining base asset
            base_asset = order.symbol[:-4]
            self.balances[base_asset].unlock(order.remaining_quantity)
        
        # Remove from active orders
        self.active_orders[symbol].discard(order.order_id)
        
        return {
            "symbol": order.symbol,
            "origClientOrderId": order.client_order_id,
            "orderId": order.order_id,
            "orderListId": -1,
            "clientOrderId": order.client_order_id,
            "price": str(order.price),
            "origQty": str(order.quantity),
            "executedQty": str(order.executed_quantity),
            "cummulativeQuoteQty": str(order.cumulative_quote_quantity),
            "status": order.status,
            "timeInForce": order.time_in_force,
            "type": order.order_type,
            "side": order.side
        }
    
    async def get_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Query order"""
        self.rate_limiter.consume(2)
        
        # Find order
        order = None
        if order_id and order_id in self.orders:
            order = self.orders[order_id]
        elif client_order_id and client_order_id in self.orders_by_client_id:
            order = self.orders_by_client_id[client_order_id]
        
        if not order:
            raise ValueError("Order not found")
        
        if order.symbol != symbol:
            raise ValueError("Order symbol mismatch")
        
        return {
            "symbol": order.symbol,
            "orderId": order.order_id,
            "orderListId": -1,
            "clientOrderId": order.client_order_id,
            "price": str(order.price),
            "origQty": str(order.quantity),
            "executedQty": str(order.executed_quantity),
            "cummulativeQuoteQty": str(order.cumulative_quote_quantity),
            "status": order.status,
            "timeInForce": order.time_in_force,
            "type": order.order_type,
            "side": order.side,
            "stopPrice": str(order.stop_price) if order.stop_price else "0",
            "icebergQty": str(order.iceberg_quantity) if order.iceberg_quantity else "0",
            "time": int(order.created_time.timestamp() * 1000),
            "updateTime": int(order.updated_time.timestamp() * 1000),
            "isWorking": order.is_active,
            "origQuoteOrderQty": "0"
        }
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders"""
        self.rate_limiter.consume(3 if symbol else 40)
        
        open_orders = []
        
        # Get orders for specific symbol or all symbols
        symbols = [symbol] if symbol else list(self.active_orders.keys())
        
        for sym in symbols:
            for order_id in self.active_orders[sym]:
                order = self.orders.get(order_id)
                if order and order.is_active:
                    open_orders.append({
                        "symbol": order.symbol,
                        "orderId": order.order_id,
                        "orderListId": -1,
                        "clientOrderId": order.client_order_id,
                        "price": str(order.price),
                        "origQty": str(order.quantity),
                        "executedQty": str(order.executed_quantity),
                        "cummulativeQuoteQty": str(order.cumulative_quote_quantity),
                        "status": order.status,
                        "timeInForce": order.time_in_force,
                        "type": order.order_type,
                        "side": order.side,
                        "stopPrice": str(order.stop_price) if order.stop_price else "0",
                        "icebergQty": str(order.iceberg_quantity) if order.iceberg_quantity else "0",
                        "time": int(order.created_time.timestamp() * 1000),
                        "updateTime": int(order.updated_time.timestamp() * 1000),
                        "isWorking": True,
                        "origQuoteOrderQty": "0"
                    })
        
        return open_orders
    
    async def get_all_orders(
        self,
        symbol: str,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Get all orders"""
        self.rate_limiter.consume(10)
        
        all_orders = []
        
        for order in self.orders.values():
            if order.symbol == symbol:
                all_orders.append({
                    "symbol": order.symbol,
                    "orderId": order.order_id,
                    "orderListId": -1,
                    "clientOrderId": order.client_order_id,
                    "price": str(order.price),
                    "origQty": str(order.quantity),
                    "executedQty": str(order.executed_quantity),
                    "cummulativeQuoteQty": str(order.cumulative_quote_quantity),
                    "status": order.status,
                    "timeInForce": order.time_in_force,
                    "type": order.order_type,
                    "side": order.side,
                    "stopPrice": str(order.stop_price) if order.stop_price else "0",
                    "icebergQty": str(order.iceberg_quantity) if order.iceberg_quantity else "0",
                    "time": int(order.created_time.timestamp() * 1000),
                    "updateTime": int(order.updated_time.timestamp() * 1000),
                    "isWorking": order.is_active,
                    "origQuoteOrderQty": "0"
                })
        
        # Sort by time descending and limit
        all_orders.sort(key=lambda x: x["time"], reverse=True)
        return all_orders[:limit]
    
    async def get_my_trades(
        self,
        symbol: str,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Get trades"""
        self.rate_limiter.consume(10)
        
        trades = []
        
        for trade in self.trades_by_symbol.get(symbol, []):
            trades.append({
                "symbol": trade.symbol,
                "id": trade.trade_id,
                "orderId": trade.order_id,
                "orderListId": -1,
                "price": str(trade.price),
                "qty": str(trade.quantity),
                "quoteQty": str(trade.quote_quantity),
                "commission": str(trade.commission),
                "commissionAsset": trade.commission_asset,
                "time": int(trade.timestamp.timestamp() * 1000),
                "isBuyer": trade.side == "BUY",
                "isMaker": trade.maker,
                "isBestMatch": True
            })
        
        # Sort by time descending and limit
        trades.sort(key=lambda x: x["time"], reverse=True)
        return trades[:limit]
    
    # WebSocket Methods
    
    async def subscribe_ticker(self, symbol: str, callback: Callable):
        """Subscribe to ticker updates"""
        stream_name = f"{symbol.lower()}@ticker"
        self.ws_callbacks[stream_name].append(callback)
        self.ws_connections.add(stream_name)
    
    async def subscribe_orderbook(self, symbol: str, callback: Callable):
        """Subscribe to order book updates"""
        stream_name = f"{symbol.lower()}@depth"
        self.ws_callbacks[stream_name].append(callback)
        self.ws_connections.add(stream_name)
    
    async def subscribe_trades(self, symbol: str, callback: Callable):
        """Subscribe to trade updates"""
        stream_name = f"{symbol.lower()}@trade"
        self.ws_callbacks[stream_name].append(callback)
        self.ws_connections.add(stream_name)
    
    async def subscribe_klines(self, symbol: str, interval: str, callback: Callable):
        """Subscribe to kline updates"""
        stream_name = f"{symbol.lower()}@kline_{interval}"
        self.ws_callbacks[stream_name].append(callback)
        self.ws_connections.add(stream_name)
    
    async def unsubscribe(self, stream_name: str):
        """Unsubscribe from stream"""
        self.ws_connections.discard(stream_name)
        self.ws_callbacks.pop(stream_name, None)
    
    # Helper Methods
    
    def _validate_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal]
    ) -> Optional[str]:
        """Validate order parameters"""
        
        # Check quantity
        if quantity <= 0:
            return "Invalid quantity"
        
        # Check price for limit orders
        if order_type in [OrderType.LIMIT.value, OrderType.LIMIT_MAKER.value]:
            if not price or price <= 0:
                return "Invalid price"
        
        # Check notional value
        if order_type != OrderType.MARKET.value:
            notional = quantity * price
            if notional < self.min_notional:
                return f"Order value below minimum: {notional} < {self.min_notional}"
        
        # Check balance
        if side.upper() == "BUY":
            # Check USDT balance
            required = quantity * price if price else quantity * self._get_market_price(symbol)
            if self.balances["USDT"].free < required:
                return "Insufficient balance"
        else:
            # Check base asset balance
            base_asset = symbol[:-4]
            if base_asset not in self.balances or self.balances[base_asset].free < quantity:
                return "Insufficient balance"
        
        return None
    
    def _get_market_price(self, symbol: str) -> Decimal:
        """Get current market price"""
        book = self.order_books[symbol]
        mid_price = book.get_mid_price()
        return mid_price or Decimal("0")
    
    def _generate_orderbook(self, symbol: str, mid_price: Decimal):
        """Generate realistic order book"""
        bids = []
        asks = []
        
        # Generate bids
        for i in range(20):
            price = mid_price * (Decimal("1") - Decimal(str(0.0001 * (i + 1))))
            quantity = Decimal(str(random.uniform(0.1, 5.0) * (1 + i * 0.1)))
            bids.append((price.quantize(Decimal("0.01")), quantity.quantize(Decimal("0.00001"))))
        
        # Generate asks
        for i in range(20):
            price = mid_price * (Decimal("1") + Decimal(str(0.0001 * (i + 1))))
            quantity = Decimal(str(random.uniform(0.1, 5.0) * (1 + i * 0.1)))
            asks.append((price.quantize(Decimal("0.01")), quantity.quantize(Decimal("0.00001"))))
        
        self.order_books[symbol].update(bids, asks)
    
    def _generate_ticker(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """Generate ticker data"""
        price_change_percent = random.uniform(-5, 5)
        open_price = current_price / (1 + price_change_percent / 100)
        high_price = current_price * (1 + random.uniform(0, 0.02))
        low_price = current_price * (1 - random.uniform(0, 0.02))
        volume = random.uniform(1000, 10000)
        
        return {
            "symbol": symbol,
            "priceChange": f"{current_price - open_price:.2f}",
            "priceChangePercent": f"{price_change_percent:.2f}",
            "weightedAvgPrice": f"{(open_price + current_price) / 2:.2f}",
            "prevClosePrice": f"{open_price:.2f}",
            "lastPrice": f"{current_price:.2f}",
            "lastQty": f"{random.uniform(0.01, 1):.5f}",
            "bidPrice": f"{current_price - 0.01:.2f}",
            "bidQty": f"{random.uniform(1, 10):.5f}",
            "askPrice": f"{current_price + 0.01:.2f}",
            "askQty": f"{random.uniform(1, 10):.5f}",
            "openPrice": f"{open_price:.2f}",
            "highPrice": f"{high_price:.2f}",
            "lowPrice": f"{low_price:.2f}",
            "volume": f"{volume:.8f}",
            "quoteVolume": f"{volume * current_price:.2f}",
            "openTime": int((datetime.now() - timedelta(hours=24)).timestamp() * 1000),
            "closeTime": int(datetime.now().timestamp() * 1000),
            "firstId": random.randint(100000, 999999),
            "lastId": random.randint(1000000, 9999999),
            "count": random.randint(10000, 100000)
        }
    
    def _generate_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Generate kline data"""
        # Interval to minutes
        interval_minutes = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440
        }
        minutes = interval_minutes.get(interval, 5)
        
        # Generate data
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=minutes * limit)
        
        timestamps = pd.date_range(start=start_time, end=end_time, freq=f"{minutes}min")
        
        # Get base price
        base_price = float(self._get_market_price(symbol))
        
        # Generate OHLCV
        data = []
        current_price = base_price
        
        for timestamp in timestamps:
            # Random walk
            change = random.uniform(-0.002, 0.002)
            current_price *= (1 + change)
            
            open_price = current_price
            high_price = current_price * (1 + random.uniform(0, 0.001))
            low_price = current_price * (1 - random.uniform(0, 0.001))
            close_price = current_price * (1 + random.uniform(-0.001, 0.001))
            
            volume = random.uniform(10, 100)
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'close_time': timestamp + timedelta(minutes=minutes) - timedelta(seconds=1),
                'quote_volume': volume * close_price,
                'trades': random.randint(100, 1000),
                'taker_buy_volume': volume * 0.5,
                'taker_buy_quote_volume': volume * close_price * 0.5
            })
            
            current_price = close_price
        
        return pd.DataFrame(data)
    
    async def _execute_market_order(self, order: Order):
        """Execute market order immediately"""
        book = self.order_books[order.symbol]
        
        if order.side == "BUY":
            # Buy from asks
            asks = sorted(book.asks.items(), key=lambda x: x[0])
            remaining = order.quantity
            
            for price, available in asks:
                if remaining <= 0:
                    break
                
                fill_quantity = min(remaining, available)
                await self._execute_trade(order, price, fill_quantity)
                remaining -= fill_quantity
        else:
            # Sell to bids
            bids = sorted(book.bids.items(), key=lambda x: x[0], reverse=True)
            remaining = order.quantity
            
            for price, available in bids:
                if remaining <= 0:
                    break
                
                fill_quantity = min(remaining, available)
                await self._execute_trade(order, price, fill_quantity)
                remaining -= fill_quantity
        
        # Mark as filled or partially filled
        if order.executed_quantity >= order.quantity:
            order.status = OrderStatus.FILLED.value
            self.active_orders[order.symbol].discard(order.order_id)
        else:
            order.status = OrderStatus.PARTIALLY_FILLED.value
    
    async def _match_orders(self, symbol: str):
        """Match orders for a symbol"""
        book = self.order_books[symbol]
        
        # Get active orders for symbol
        for order_id in list(self.active_orders[symbol]):
            order = self.orders.get(order_id)
            if not order or not order.is_active:
                continue
            
            # Skip market orders (already executed)
            if order.order_type == OrderType.MARKET.value:
                continue
            
            # Check if order can be filled
            if order.side == "BUY":
                best_ask = book.get_best_ask()
                if best_ask and order.price >= best_ask:
                    # Can fill at best ask
                    available = book.asks.get(best_ask, Decimal("0"))
                    fill_quantity = min(order.remaining_quantity, available)
                    if fill_quantity > 0:
                        await self._execute_trade(order, best_ask, fill_quantity)
            else:
                best_bid = book.get_best_bid()
                if best_bid and order.price <= best_bid:
                    # Can fill at best bid
                    available = book.bids.get(best_bid, Decimal("0"))
                    fill_quantity = min(order.remaining_quantity, available)
                    if fill_quantity > 0:
                        await self._execute_trade(order, best_bid, fill_quantity)
    
    async def _execute_trade(self, order: Order, price: Decimal, quantity: Decimal):
        """Execute a trade"""
        # Create trade
        trade_id = str(uuid.uuid4().hex[:16])
        
        # Determine if maker or taker
        is_maker = order.order_type == OrderType.LIMIT_MAKER.value
        commission_rate = self.maker_fee if is_maker else self.taker_fee
        
        # Calculate commission
        if order.side == "BUY":
            # Commission in base asset
            commission = quantity * commission_rate
            commission_asset = order.symbol[:-4]
        else:
            # Commission in quote asset
            commission = quantity * price * commission_rate
            commission_asset = "USDT"
        
        trade = Trade(
            trade_id=trade_id,
            order_id=order.order_id,
            symbol=order.symbol,
            price=price,
            quantity=quantity,
            side=order.side,
            maker=is_maker,
            commission=commission,
            commission_asset=commission_asset
        )
        
        # Update order
        order.executed_quantity += quantity
        order.cumulative_quote_quantity += quantity * price
        order.commission_paid += commission
        order.updated_time = datetime.now()
        
        # Add fill
        order.fills.append({
            "price": str(price),
            "qty": str(quantity),
            "commission": str(commission),
            "commissionAsset": commission_asset,
            "tradeId": trade_id
        })
        
        # Update balances
        if order.side == "BUY":
            # Receive base asset (minus commission)
            base_asset = order.symbol[:-4]
            self.balances[base_asset].free += quantity - commission
            
            # Deduct from locked USDT
            quote_value = quantity * price
            self.balances["USDT"].locked -= quote_value
        else:
            # Receive USDT (minus commission)
            quote_value = quantity * price
            self.balances["USDT"].free += quote_value - commission
            
            # Deduct from locked base asset
            base_asset = order.symbol[:-4]
            self.balances[base_asset].locked -= quantity
        
        # Store trade
        self.trades.append(trade)
        self.trades_by_symbol[order.symbol].append(trade)
        
        # Check if order is filled
        if order.executed_quantity >= order.quantity * Decimal("0.999"):  # Allow small rounding
            order.status = OrderStatus.FILLED.value
            self.active_orders[order.symbol].discard(order.order_id)
            
            # Unlock any remaining locked balance (due to rounding)
            if order.side == "BUY":
                remaining_locked = order.remaining_quantity * order.price
                if remaining_locked > 0:
                    self.balances["USDT"].unlock(remaining_locked)
            else:
                base_asset = order.symbol[:-4]
                if order.remaining_quantity > 0:
                    self.balances[base_asset].unlock(order.remaining_quantity)
        
        # Broadcast trade
        await self._broadcast_trade(trade)
        await self._broadcast_order_update(order)
    
    async def _broadcast_ticker(self, symbol: str, ticker: Dict[str, Any]):
        """Broadcast ticker update"""
        stream_name = f"{symbol.lower()}@ticker"
        callbacks = self.ws_callbacks.get(stream_name, [])
        
        message = {
            "e": "24hrTicker",
            "E": int(datetime.now().timestamp() * 1000),
            "s": symbol,
            "p": ticker["priceChange"],
            "P": ticker["priceChangePercent"],
            "w": ticker["weightedAvgPrice"],
            "x": ticker["prevClosePrice"],
            "c": ticker["lastPrice"],
            "Q": ticker["lastQty"],
            "b": ticker["bidPrice"],
            "B": ticker["bidQty"],
            "a": ticker["askPrice"],
            "A": ticker["askQty"],
            "o": ticker["openPrice"],
            "h": ticker["highPrice"],
            "l": ticker["lowPrice"],
            "v": ticker["volume"],
            "q": ticker["quoteVolume"],
            "O": ticker["openTime"],
            "C": ticker["closeTime"],
            "F": ticker["firstId"],
            "L": ticker["lastId"],
            "n": ticker["count"]
        }
        
        for callback in callbacks:
            await callback(message)
    
    async def _broadcast_orderbook(self, symbol: str, depth: Dict[str, Any]):
        """Broadcast order book update"""
        stream_name = f"{symbol.lower()}@depth"
        callbacks = self.ws_callbacks.get(stream_name, [])
        
        message = {
            "e": "depthUpdate",
            "E": int(datetime.now().timestamp() * 1000),
            "s": symbol,
            "U": depth["lastUpdateId"] - 1,
            "u": depth["lastUpdateId"],
            "b": depth["bids"][:5],  # Top 5 bids
            "a": depth["asks"][:5]   # Top 5 asks
        }
        
        for callback in callbacks:
            await callback(message)
    
    async def _broadcast_trade(self, trade: Trade):
        """Broadcast trade"""
        stream_name = f"{trade.symbol.lower()}@trade"
        callbacks = self.ws_callbacks.get(stream_name, [])
        
        message = {
            "e": "trade",
            "E": int(trade.timestamp.timestamp() * 1000),
            "s": trade.symbol,
            "t": trade.trade_id,
            "p": str(trade.price),
            "q": str(trade.quantity),
            "b": trade.order_id,
            "a": trade.order_id,
            "T": int(trade.timestamp.timestamp() * 1000),
            "m": trade.maker,
            "M": True
        }
        
        for callback in callbacks:
            await callback(message)
    
    async def _broadcast_order_update(self, order: Order):
        """Broadcast order update"""
        # This would typically be sent to user data stream
        # For testing, we'll just log it
        logger.info(f"Order update: {order.order_id} - {order.status}")
    
    def shutdown(self):
        """Shutdown exchange"""
        self.running = False
        self.executor.shutdown(wait=True)


class MatchingEngine:
    """Order matching engine"""
    
    def __init__(self, exchange: MockExchange):
        self.exchange = exchange
    
    async def match_orders(self, symbol: str):
        """Match orders for a symbol"""
        # Implemented in exchange._match_orders
        pass


class PriceEngine:
    """Price simulation engine"""
    
    def __init__(self):
        self.markets = {}
        self.last_prices = {}
    
    def add_market(self, symbol: str, base_price: float, volatility: float):
        """Add market to price engine"""
        self.markets[symbol] = {
            "base_price": base_price,
            "volatility": volatility,
            "trend": 0.0,
            "momentum": 0.0
        }
        self.last_prices[symbol] = Decimal(str(base_price))
    
    def get_next_price(self, symbol: str) -> Decimal:
        """Generate next price"""
        if symbol not in self.markets:
            return Decimal("0")
        
        market = self.markets[symbol]
        last_price = float(self.last_prices[symbol])
        
        # Random walk with momentum
        market["momentum"] = 0.9 * market["momentum"] + 0.1 * random.uniform(-1, 1)
        
        # Add trend component
        if random.random() < 0.01:  # 1% chance to change trend
            market["trend"] = random.uniform(-0.001, 0.001)
        
        # Calculate price change
        random_component = random.gauss(0, market["volatility"])
        trend_component = market["trend"]
        momentum_component = market["momentum"] * 0.0001
        
        price_change = random_component + trend_component + momentum_component
        
        # Apply change
        new_price = last_price * (1 + price_change)
        
        # Keep within reasonable bounds
        min_price = market["base_price"] * 0.5
        max_price = market["base_price"] * 2.0
        new_price = max(min_price, min(max_price, new_price))
        
        self.last_prices[symbol] = Decimal(str(new_price))
        return self.last_prices[symbol]


class RateLimiter:
    """Rate limiter implementation"""
    
    def __init__(self, requests_per_minute: int, weight_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.weight_per_minute = weight_per_minute
        self.requests = deque()
        self.weights = deque()
        self.lock = threading.Lock()
    
    def consume(self, weight: int = 1):
        """Consume rate limit"""
        with self.lock:
            now = time.time()
            minute_ago = now - 60
            
            # Clean old requests
            while self.requests and self.requests[0] < minute_ago:
                self.requests.popleft()
            
            while self.weights and self.weights[0][0] < minute_ago:
                self.weights.popleft()
            
            # Check limits
            if len(self.requests) >= self.requests_per_minute:
                raise Exception("Rate limit exceeded: too many requests")
            
            total_weight = sum(w[1] for w in self.weights)
            if total_weight + weight > self.weight_per_minute:
                raise Exception("Rate limit exceeded: too much weight")
            
            # Record request
            self.requests.append(now)
            self.weights.append((now, weight))
    
    def get_remaining(self) -> Dict[str, int]:
        """Get remaining limits"""
        with self.lock:
            now = time.time()
            minute_ago = now - 60
            
            # Count recent requests
            recent_requests = sum(1 for r in self.requests if r > minute_ago)
            recent_weight = sum(w[1] for w in self.weights if w[0] > minute_ago)
            
            return {
                "requests": self.requests_per_minute - recent_requests,
                "weight": self.weight_per_minute - recent_weight
            }


# Convenience functions

def create_mock_exchange(name: str = "MockExchange") -> MockExchange:
    """Create mock exchange instance"""
    return MockExchange(name)


def create_test_exchange() -> MockExchange:
    """Create test exchange with pre-configured state"""
    exchange = MockExchange("TestExchange")
    
    # Add some test balances
    exchange.balances["BTC"].free = Decimal("2.0")
    exchange.balances["ETH"].free = Decimal("20.0")
    exchange.balances["USDT"].free = Decimal("100000.0")
    
    return exchange


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Create exchange
        exchange = create_mock_exchange()
        
        # Get exchange info
        info = await exchange.get_exchange_info()
        print(f"Exchange: {exchange.name}")
        print(f"Symbols: {len(info['symbols'])}")
        
        # Get ticker
        ticker = await exchange.get_ticker("BTCUSDT")
        print(f"\nBTCUSDT Price: {ticker['lastPrice']}")
        
        # Place order
        order = await exchange.place_order(
            symbol="BTCUSDT",
            side="buy",
            order_type="limit",
            quantity="0.1",
            price="44000",
            timeInForce="GTC"
        )
        print(f"\nPlaced order: {order['orderId']}")
        
        # Get account
        account = await exchange.get_account()
        print(f"\nBalances:")
        for balance in account['balances']:
            print(f"  {balance['asset']}: {balance['free']} (free) + {balance['locked']} (locked)")
        
        # Subscribe to updates
        async def on_ticker(data):
            print(f"Ticker update: {data['s']} @ {data['c']}")
        
        await exchange.subscribe_ticker("BTCUSDT", on_ticker)
        
        # Wait for some updates
        await asyncio.sleep(5)
        
        # Shutdown
        exchange.shutdown()
    
    asyncio.run(main())