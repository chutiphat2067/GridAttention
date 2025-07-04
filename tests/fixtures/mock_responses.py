#!/usr/bin/env python3
"""
Mock API Responses for GridAttention Trading System
Provides realistic mock responses for exchange APIs, websockets, and external services
"""

import json
import time
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np


class ResponseType(Enum):
    """Types of API responses"""
    SUCCESS = "success"
    ERROR = "error"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    PARTIAL = "partial"


class ExchangeType(Enum):
    """Supported exchanges"""
    BINANCE = "binance"
    BYBIT = "bybit"
    OKX = "okx"
    GENERIC = "generic"


@dataclass
class MockResponse:
    """Base mock response structure"""
    status_code: int
    data: Any
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    latency_ms: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "status_code": self.status_code,
            "data": self.data,
            "headers": self.headers,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class ExchangeMockResponses:
    """Mock responses for exchange APIs"""
    
    def __init__(self, exchange: ExchangeType = ExchangeType.BINANCE):
        self.exchange = exchange
        self.request_counter = 0
        self.rate_limit_remaining = 1200
        self.weight_remaining = 1000
        
    def _update_rate_limits(self, weight: int = 1):
        """Update rate limit counters"""
        self.request_counter += 1
        self.rate_limit_remaining -= 1
        self.weight_remaining -= weight
        
        # Reset limits periodically
        if self.request_counter % 100 == 0:
            self.rate_limit_remaining = 1200
            self.weight_remaining = 1000
    
    def _get_common_headers(self) -> Dict[str, str]:
        """Get common response headers"""
        return {
            "Content-Type": "application/json",
            "X-MBX-USED-WEIGHT": str(1200 - self.weight_remaining),
            "X-MBX-ORDER-COUNT-1M": str(self.request_counter),
            "X-RATE-LIMIT-REMAINING": str(self.rate_limit_remaining),
            "X-RESPONSE-TIME": str(random.randint(10, 100))
        }
    
    # Market Data Endpoints
    
    def get_ticker_24hr(self, symbol: str = "BTCUSDT") -> MockResponse:
        """Mock 24hr ticker response"""
        self._update_rate_limits(weight=1)
        
        base_price = 45000 if "BTC" in symbol else 3000
        price_change = random.uniform(-0.05, 0.05)
        
        data = {
            "symbol": symbol,
            "priceChange": str(base_price * price_change),
            "priceChangePercent": f"{price_change * 100:.2f}",
            "weightedAvgPrice": str(base_price * (1 + price_change/2)),
            "lastPrice": str(base_price * (1 + price_change)),
            "lastQty": "0.00100000",
            "bidPrice": str(base_price * (1 + price_change - 0.0001)),
            "bidQty": "4.00000000",
            "askPrice": str(base_price * (1 + price_change + 0.0001)),
            "askQty": "4.00000000",
            "openPrice": str(base_price),
            "highPrice": str(base_price * (1 + abs(price_change) + 0.01)),
            "lowPrice": str(base_price * (1 - abs(price_change) - 0.01)),
            "volume": "12614.42000000",
            "quoteVolume": f"{base_price * 12614.42:.2f}",
            "openTime": int((datetime.now() - timedelta(hours=24)).timestamp() * 1000),
            "closeTime": int(datetime.now().timestamp() * 1000),
            "firstId": 100000000,
            "lastId": 100100000,
            "count": 100000
        }
        
        return MockResponse(
            status_code=200,
            data=data,
            headers=self._get_common_headers(),
            latency_ms=random.randint(20, 50)
        )
    
    def get_orderbook(self, symbol: str = "BTCUSDT", limit: int = 100) -> MockResponse:
        """Mock orderbook response"""
        self._update_rate_limits(weight=5 if limit > 100 else 1)
        
        base_price = 45000 if "BTC" in symbol else 3000
        spread = base_price * 0.0001  # 1 basis point spread
        
        # Generate bids
        bids = []
        current_bid = base_price - spread/2
        for i in range(limit):
            price = current_bid - (i * spread * 0.1)
            amount = random.uniform(0.1, 2.0) * (1 + i * 0.1)  # Increasing liquidity
            bids.append([f"{price:.2f}", f"{amount:.8f}"])
        
        # Generate asks
        asks = []
        current_ask = base_price + spread/2
        for i in range(limit):
            price = current_ask + (i * spread * 0.1)
            amount = random.uniform(0.1, 2.0) * (1 + i * 0.1)
            asks.append([f"{price:.2f}", f"{amount:.8f}"])
        
        data = {
            "lastUpdateId": random.randint(1000000, 9999999),
            "bids": bids,
            "asks": asks
        }
        
        return MockResponse(
            status_code=200,
            data=data,
            headers=self._get_common_headers(),
            latency_ms=random.randint(15, 40)
        )
    
    def get_klines(
        self, 
        symbol: str = "BTCUSDT", 
        interval: str = "5m",
        limit: int = 100
    ) -> MockResponse:
        """Mock klines/candlestick response"""
        self._update_rate_limits(weight=1)
        
        base_price = 45000 if "BTC" in symbol else 3000
        klines = []
        
        # Time intervals
        interval_ms = {
            "1m": 60000,
            "5m": 300000,
            "15m": 900000,
            "1h": 3600000,
            "4h": 14400000,
            "1d": 86400000
        }
        
        time_step = interval_ms.get(interval, 300000)
        current_time = int(datetime.now().timestamp() * 1000)
        current_price = base_price
        
        for i in range(limit):
            # Generate OHLCV data
            open_time = current_time - (limit - i) * time_step
            close_time = open_time + time_step - 1
            
            # Random price movement
            price_change = random.uniform(-0.001, 0.001)
            current_price *= (1 + price_change)
            
            open_price = current_price
            high_price = current_price * (1 + abs(random.uniform(0, 0.002)))
            low_price = current_price * (1 - abs(random.uniform(0, 0.002)))
            close_price = current_price * (1 + random.uniform(-0.001, 0.001))
            
            volume = random.uniform(10, 100)
            quote_volume = volume * current_price
            trades = random.randint(50, 500)
            
            kline = [
                open_time,                    # Open time
                f"{open_price:.2f}",         # Open
                f"{high_price:.2f}",         # High
                f"{low_price:.2f}",          # Low
                f"{close_price:.2f}",        # Close
                f"{volume:.8f}",             # Volume
                close_time,                   # Close time
                f"{quote_volume:.2f}",       # Quote asset volume
                trades,                       # Number of trades
                f"{volume * 0.4:.8f}",       # Taker buy base asset volume
                f"{quote_volume * 0.4:.2f}", # Taker buy quote asset volume
                "0"                          # Ignore
            ]
            
            klines.append(kline)
            current_price = float(close_price)
        
        return MockResponse(
            status_code=200,
            data=klines,
            headers=self._get_common_headers(),
            latency_ms=random.randint(30, 60)
        )
    
    # Account Endpoints
    
    def get_account(self) -> MockResponse:
        """Mock account information response"""
        self._update_rate_limits(weight=10)
        
        data = {
            "makerCommission": 10,
            "takerCommission": 10,
            "buyerCommission": 0,
            "sellerCommission": 0,
            "canTrade": True,
            "canWithdraw": True,
            "canDeposit": True,
            "updateTime": int(datetime.now().timestamp() * 1000),
            "accountType": "SPOT",
            "balances": [
                {
                    "asset": "BTC",
                    "free": "0.50000000",
                    "locked": "0.00000000"
                },
                {
                    "asset": "ETH",
                    "free": "5.00000000",
                    "locked": "0.00000000"
                },
                {
                    "asset": "USDT",
                    "free": "10000.00000000",
                    "locked": "5000.00000000"
                }
            ],
            "permissions": ["SPOT", "MARGIN"]
        }
        
        return MockResponse(
            status_code=200,
            data=data,
            headers=self._get_common_headers(),
            latency_ms=random.randint(50, 100)
        )
    
    # Order Endpoints
    
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        **kwargs
    ) -> MockResponse:
        """Mock order placement response"""
        self._update_rate_limits(weight=1)
        
        # Simulate order rejection occasionally
        if random.random() < 0.05:  # 5% rejection rate
            return MockResponse(
                status_code=400,
                data={
                    "code": -2010,
                    "msg": "Account has insufficient balance for requested action."
                },
                headers=self._get_common_headers(),
                latency_ms=random.randint(20, 40)
            )
        
        order_id = random.randint(10000000, 99999999)
        client_order_id = kwargs.get('newClientOrderId', f"client_{uuid.uuid4().hex[:8]}")
        
        data = {
            "symbol": symbol,
            "orderId": order_id,
            "orderListId": -1,
            "clientOrderId": client_order_id,
            "transactTime": int(datetime.now().timestamp() * 1000),
            "price": kwargs.get('price', "0.00000000"),
            "origQty": kwargs.get('quantity', "0.00000000"),
            "executedQty": "0.00000000",
            "cummulativeQuoteQty": "0.00000000",
            "status": "NEW",
            "timeInForce": kwargs.get('timeInForce', 'GTC'),
            "type": order_type.upper(),
            "side": side.upper(),
            "fills": []
        }
        
        return MockResponse(
            status_code=200,
            data=data,
            headers=self._get_common_headers(),
            latency_ms=random.randint(50, 150)
        )
    
    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None
    ) -> MockResponse:
        """Mock order cancellation response"""
        self._update_rate_limits(weight=1)
        
        # Simulate order not found occasionally
        if random.random() < 0.1:  # 10% not found rate
            return MockResponse(
                status_code=400,
                data={
                    "code": -2011,
                    "msg": "Unknown order sent."
                },
                headers=self._get_common_headers(),
                latency_ms=random.randint(20, 40)
            )
        
        data = {
            "symbol": symbol,
            "origClientOrderId": client_order_id or f"client_{uuid.uuid4().hex[:8]}",
            "orderId": order_id or random.randint(10000000, 99999999),
            "orderListId": -1,
            "clientOrderId": f"cancel_{uuid.uuid4().hex[:8]}",
            "price": "45000.00",
            "origQty": "0.10000000",
            "executedQty": "0.00000000",
            "cummulativeQuoteQty": "0.00000000",
            "status": "CANCELED",
            "timeInForce": "GTC",
            "type": "LIMIT",
            "side": "BUY"
        }
        
        return MockResponse(
            status_code=200,
            data=data,
            headers=self._get_common_headers(),
            latency_ms=random.randint(30, 80)
        )
    
    def get_order(
        self,
        symbol: str,
        order_id: Optional[int] = None
    ) -> MockResponse:
        """Mock order query response"""
        self._update_rate_limits(weight=2)
        
        # Generate order status
        statuses = ["NEW", "PARTIALLY_FILLED", "FILLED", "CANCELED"]
        weights = [0.2, 0.1, 0.6, 0.1]
        status = np.random.choice(statuses, p=weights)
        
        executed_qty = "0.00000000"
        if status == "PARTIALLY_FILLED":
            executed_qty = "0.05000000"
        elif status == "FILLED":
            executed_qty = "0.10000000"
        
        data = {
            "symbol": symbol,
            "orderId": order_id or random.randint(10000000, 99999999),
            "orderListId": -1,
            "clientOrderId": f"client_{uuid.uuid4().hex[:8]}",
            "price": "45000.00",
            "origQty": "0.10000000",
            "executedQty": executed_qty,
            "cummulativeQuoteQty": str(float(executed_qty) * 45000),
            "status": status,
            "timeInForce": "GTC",
            "type": "LIMIT",
            "side": "BUY",
            "stopPrice": "0.00000000",
            "icebergQty": "0.00000000",
            "time": int((datetime.now() - timedelta(minutes=random.randint(1, 60))).timestamp() * 1000),
            "updateTime": int(datetime.now().timestamp() * 1000),
            "isWorking": True,
            "origQuoteOrderQty": "0.00000000"
        }
        
        return MockResponse(
            status_code=200,
            data=data,
            headers=self._get_common_headers(),
            latency_ms=random.randint(20, 50)
        )
    
    def get_open_orders(self, symbol: Optional[str] = None) -> MockResponse:
        """Mock open orders response"""
        self._update_rate_limits(weight=3 if symbol else 40)
        
        orders = []
        num_orders = random.randint(0, 5)
        
        for i in range(num_orders):
            order = {
                "symbol": symbol or "BTCUSDT",
                "orderId": random.randint(10000000, 99999999),
                "orderListId": -1,
                "clientOrderId": f"client_{uuid.uuid4().hex[:8]}",
                "price": f"{45000 - i * 100:.2f}",
                "origQty": "0.10000000",
                "executedQty": "0.00000000",
                "cummulativeQuoteQty": "0.00000000",
                "status": "NEW",
                "timeInForce": "GTC",
                "type": "LIMIT",
                "side": "BUY" if i % 2 == 0 else "SELL",
                "stopPrice": "0.00000000",
                "icebergQty": "0.00000000",
                "time": int((datetime.now() - timedelta(minutes=random.randint(1, 120))).timestamp() * 1000),
                "updateTime": int(datetime.now().timestamp() * 1000),
                "isWorking": True,
                "origQuoteOrderQty": "0.00000000"
            }
            orders.append(order)
        
        return MockResponse(
            status_code=200,
            data=orders,
            headers=self._get_common_headers(),
            latency_ms=random.randint(30, 70)
        )
    
    # Error Responses
    
    def get_rate_limit_error(self) -> MockResponse:
        """Mock rate limit error response"""
        return MockResponse(
            status_code=429,
            data={
                "code": -1003,
                "msg": "Too many requests; please use the websocket for live updates."
            },
            headers={
                **self._get_common_headers(),
                "Retry-After": "60"
            },
            latency_ms=random.randint(10, 20)
        )
    
    def get_server_error(self) -> MockResponse:
        """Mock server error response"""
        return MockResponse(
            status_code=500,
            data={
                "code": -1000,
                "msg": "An unknown error occurred while processing the request."
            },
            headers=self._get_common_headers(),
            latency_ms=random.randint(100, 500)
        )
    
    def get_timeout_error(self) -> MockResponse:
        """Mock timeout error"""
        return MockResponse(
            status_code=504,
            data={
                "code": -1007,
                "msg": "Timeout waiting for response from backend server."
            },
            headers=self._get_common_headers(),
            latency_ms=30000  # 30 second timeout
        )


class WebSocketMockResponses:
    """Mock WebSocket responses"""
    
    def __init__(self):
        self.stream_counter = 0
        self.base_prices = {
            "btcusdt": 45000,
            "ethusdt": 3000,
            "solusdt": 120,
            "bnbusdt": 450
        }
    
    def get_trade_stream(self, symbol: str = "btcusdt") -> Dict:
        """Mock trade stream message"""
        self.stream_counter += 1
        base_price = self.base_prices.get(symbol.lower(), 100)
        
        return {
            "e": "trade",
            "E": int(datetime.now().timestamp() * 1000),
            "s": symbol.upper(),
            "t": self.stream_counter,
            "p": f"{base_price * (1 + random.uniform(-0.001, 0.001)):.2f}",
            "q": f"{random.uniform(0.001, 1.0):.8f}",
            "b": random.randint(10000000, 99999999),
            "a": random.randint(10000000, 99999999),
            "T": int(datetime.now().timestamp() * 1000),
            "m": random.choice([True, False]),
            "M": True
        }
    
    def get_depth_update(self, symbol: str = "btcusdt") -> Dict:
        """Mock depth/orderbook update"""
        base_price = self.base_prices.get(symbol.lower(), 100)
        
        # Generate bid/ask updates
        num_updates = random.randint(1, 5)
        bids = []
        asks = []
        
        for i in range(num_updates):
            bid_price = base_price * (1 - 0.0001 * (i + 1))
            ask_price = base_price * (1 + 0.0001 * (i + 1))
            
            bids.append([
                f"{bid_price:.2f}",
                f"{random.uniform(0.1, 5.0):.8f}"
            ])
            
            asks.append([
                f"{ask_price:.2f}",
                f"{random.uniform(0.1, 5.0):.8f}"
            ])
        
        return {
            "e": "depthUpdate",
            "E": int(datetime.now().timestamp() * 1000),
            "s": symbol.upper(),
            "U": random.randint(100000000, 999999999),
            "u": random.randint(100000000, 999999999),
            "b": bids,
            "a": asks
        }
    
    def get_kline_stream(self, symbol: str = "btcusdt", interval: str = "5m") -> Dict:
        """Mock kline/candlestick stream"""
        base_price = self.base_prices.get(symbol.lower(), 100)
        current_price = base_price * (1 + random.uniform(-0.01, 0.01))
        
        return {
            "e": "kline",
            "E": int(datetime.now().timestamp() * 1000),
            "s": symbol.upper(),
            "k": {
                "t": int(datetime.now().timestamp() * 1000),
                "T": int((datetime.now() + timedelta(minutes=5)).timestamp() * 1000),
                "s": symbol.upper(),
                "i": interval,
                "f": random.randint(100000000, 999999999),
                "L": random.randint(100000000, 999999999),
                "o": f"{current_price:.2f}",
                "c": f"{current_price * (1 + random.uniform(-0.001, 0.001)):.2f}",
                "h": f"{current_price * (1 + random.uniform(0, 0.002)):.2f}",
                "l": f"{current_price * (1 - random.uniform(0, 0.002)):.2f}",
                "v": f"{random.uniform(10, 100):.8f}",
                "n": random.randint(100, 1000),
                "x": False,
                "q": f"{random.uniform(450000, 550000):.2f}",
                "V": f"{random.uniform(5, 50):.8f}",
                "Q": f"{random.uniform(225000, 275000):.2f}",
                "B": "0"
            }
        }
    
    def get_account_update(self) -> Dict:
        """Mock account update message"""
        return {
            "e": "outboundAccountPosition",
            "E": int(datetime.now().timestamp() * 1000),
            "u": int(datetime.now().timestamp() * 1000),
            "B": [
                {
                    "a": "BTC",
                    "f": "0.50000000",
                    "l": "0.00000000"
                },
                {
                    "a": "USDT",
                    "f": "10000.00000000",
                    "l": "5000.00000000"
                }
            ]
        }
    
    def get_order_update(self, symbol: str = "BTCUSDT") -> Dict:
        """Mock order update message"""
        statuses = ["NEW", "PARTIALLY_FILLED", "FILLED", "CANCELED"]
        status = random.choice(statuses)
        
        return {
            "e": "executionReport",
            "E": int(datetime.now().timestamp() * 1000),
            "s": symbol,
            "c": f"client_{uuid.uuid4().hex[:8]}",
            "S": random.choice(["BUY", "SELL"]),
            "o": "LIMIT",
            "f": "GTC",
            "q": "0.10000000",
            "p": "45000.00",
            "P": "0.00000000",
            "F": "0.00000000",
            "g": -1,
            "C": "",
            "x": "NEW" if status == "NEW" else "TRADE",
            "X": status,
            "r": "NONE",
            "i": random.randint(10000000, 99999999),
            "l": "0.05000000" if status in ["PARTIALLY_FILLED", "FILLED"] else "0.00000000",
            "z": "0.05000000" if status in ["PARTIALLY_FILLED", "FILLED"] else "0.00000000",
            "L": "45000.00" if status in ["PARTIALLY_FILLED", "FILLED"] else "0.00",
            "n": "0.04500000" if status in ["PARTIALLY_FILLED", "FILLED"] else "0",
            "N": "USDT",
            "T": int(datetime.now().timestamp() * 1000),
            "t": random.randint(10000000, 99999999) if status in ["PARTIALLY_FILLED", "FILLED"] else -1,
            "I": random.randint(100000000, 999999999),
            "w": True,
            "m": False,
            "M": False,
            "O": int((datetime.now() - timedelta(minutes=random.randint(1, 60))).timestamp() * 1000),
            "Z": "2250.00000000" if status in ["PARTIALLY_FILLED", "FILLED"] else "0.00000000",
            "Y": "2250.00000000" if status in ["PARTIALLY_FILLED", "FILLED"] else "0.00000000",
            "Q": "0.00000000"
        }
    
    def get_error_message(self, error_type: str = "connection") -> Dict:
        """Mock WebSocket error message"""
        errors = {
            "connection": {
                "e": "error",
                "m": "WebSocket connection failed",
                "c": 1006,
                "E": int(datetime.now().timestamp() * 1000)
            },
            "auth": {
                "e": "error",
                "m": "Invalid API key",
                "c": 2001,
                "E": int(datetime.now().timestamp() * 1000)
            },
            "subscription": {
                "e": "error",
                "m": "Invalid subscription",
                "c": 2002,
                "E": int(datetime.now().timestamp() * 1000)
            }
        }
        
        return errors.get(error_type, errors["connection"])


class ExternalServiceMocks:
    """Mock responses for external services"""
    
    @staticmethod
    def get_sentiment_analysis(symbol: str = "BTC") -> MockResponse:
        """Mock sentiment analysis API response"""
        sentiment_score = random.uniform(-1, 1)
        
        data = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "sentiment": {
                "score": sentiment_score,
                "label": "bullish" if sentiment_score > 0.2 else "bearish" if sentiment_score < -0.2 else "neutral",
                "confidence": random.uniform(0.6, 0.95)
            },
            "sources": {
                "twitter": random.uniform(-1, 1),
                "reddit": random.uniform(-1, 1),
                "news": random.uniform(-1, 1)
            },
            "volume": {
                "mentions": random.randint(1000, 50000),
                "trend": random.choice(["increasing", "decreasing", "stable"])
            }
        }
        
        return MockResponse(
            status_code=200,
            data=data,
            headers={"Content-Type": "application/json"},
            latency_ms=random.randint(100, 300)
        )
    
    @staticmethod
    def get_economic_calendar() -> MockResponse:
        """Mock economic calendar API response"""
        events = []
        
        for i in range(5):
            event_time = datetime.now() + timedelta(days=random.randint(0, 7))
            events.append({
                "id": f"event_{i}",
                "title": random.choice([
                    "Fed Interest Rate Decision",
                    "CPI Data Release",
                    "NFP Report",
                    "GDP Growth Rate",
                    "Unemployment Rate"
                ]),
                "timestamp": event_time.isoformat(),
                "impact": random.choice(["high", "medium", "low"]),
                "forecast": f"{random.uniform(-2, 5):.1f}%",
                "previous": f"{random.uniform(-2, 5):.1f}%",
                "country": random.choice(["US", "EU", "UK", "JP", "CN"])
            })
        
        return MockResponse(
            status_code=200,
            data={"events": events},
            headers={"Content-Type": "application/json"},
            latency_ms=random.randint(50, 150)
        )
    
    @staticmethod
    def get_news_feed(query: str = "bitcoin") -> MockResponse:
        """Mock news API response"""
        articles = []
        
        for i in range(10):
            publish_time = datetime.now() - timedelta(hours=random.randint(0, 48))
            articles.append({
                "id": f"article_{uuid.uuid4().hex[:8]}",
                "title": f"Breaking: {query.title()} {random.choice(['Surges', 'Drops', 'Stabilizes', 'Reaches New High', 'Faces Resistance'])}",
                "summary": f"Market analysis shows {query} experiencing significant movement...",
                "source": random.choice(["Reuters", "Bloomberg", "CoinDesk", "CryptoNews", "MarketWatch"]),
                "published_at": publish_time.isoformat(),
                "url": f"https://example.com/article/{i}",
                "sentiment": random.choice(["positive", "negative", "neutral"]),
                "relevance_score": random.uniform(0.5, 1.0)
            })
        
        return MockResponse(
            status_code=200,
            data={
                "query": query,
                "total_results": len(articles),
                "articles": articles
            },
            headers={"Content-Type": "application/json"},
            latency_ms=random.randint(200, 500)
        )


class MockResponseFactory:
    """Factory for creating mock responses"""
    
    def __init__(self):
        self.exchange_mocks = {
            ExchangeType.BINANCE: ExchangeMockResponses(ExchangeType.BINANCE),
            ExchangeType.BYBIT: ExchangeMockResponses(ExchangeType.BYBIT),
            ExchangeType.OKX: ExchangeMockResponses(ExchangeType.OKX),
            ExchangeType.GENERIC: ExchangeMockResponses(ExchangeType.GENERIC)
        }
        self.websocket_mocks = WebSocketMockResponses()
        self.external_mocks = ExternalServiceMocks()
        
        # Response scenarios
        self.scenarios = {
            "normal": self._normal_scenario,
            "high_latency": self._high_latency_scenario,
            "errors": self._error_scenario,
            "rate_limited": self._rate_limited_scenario,
            "partial_outage": self._partial_outage_scenario
        }
        
        self.current_scenario = "normal"
    
    def set_scenario(self, scenario: str):
        """Set response scenario"""
        if scenario in self.scenarios:
            self.current_scenario = scenario
    
    def get_exchange_response(
        self,
        exchange: ExchangeType,
        endpoint: str,
        **kwargs
    ) -> MockResponse:
        """Get mock exchange response based on current scenario"""
        scenario_handler = self.scenarios[self.current_scenario]
        return scenario_handler(exchange, endpoint, **kwargs)
    
    def _normal_scenario(self, exchange: ExchangeType, endpoint: str, **kwargs) -> MockResponse:
        """Normal response scenario"""
        exchange_mock = self.exchange_mocks[exchange]
        
        if hasattr(exchange_mock, endpoint):
            method = getattr(exchange_mock, endpoint)
            return method(**kwargs)
        
        # Default response
        return MockResponse(
            status_code=200,
            data={"message": "OK"},
            headers={"Content-Type": "application/json"}
        )
    
    def _high_latency_scenario(self, exchange: ExchangeType, endpoint: str, **kwargs) -> MockResponse:
        """High latency response scenario"""
        response = self._normal_scenario(exchange, endpoint, **kwargs)
        response.latency_ms = random.randint(1000, 5000)
        return response
    
    def _error_scenario(self, exchange: ExchangeType, endpoint: str, **kwargs) -> MockResponse:
        """Random error scenario"""
        if random.random() < 0.3:  # 30% error rate
            exchange_mock = self.exchange_mocks[exchange]
            error_type = random.choice(["rate_limit", "server", "timeout"])
            
            if error_type == "rate_limit":
                return exchange_mock.get_rate_limit_error()
            elif error_type == "server":
                return exchange_mock.get_server_error()
            else:
                return exchange_mock.get_timeout_error()
        
        return self._normal_scenario(exchange, endpoint, **kwargs)
    
    def _rate_limited_scenario(self, exchange: ExchangeType, endpoint: str, **kwargs) -> MockResponse:
        """Rate limited scenario"""
        exchange_mock = self.exchange_mocks[exchange]
        
        # Always return rate limit error
        return exchange_mock.get_rate_limit_error()
    
    def _partial_outage_scenario(self, exchange: ExchangeType, endpoint: str, **kwargs) -> MockResponse:
        """Partial outage scenario"""
        # Some endpoints work, others timeout
        working_endpoints = ["get_ticker_24hr", "get_orderbook"]
        
        if endpoint in working_endpoints:
            return self._normal_scenario(exchange, endpoint, **kwargs)
        else:
            exchange_mock = self.exchange_mocks[exchange]
            return exchange_mock.get_timeout_error()


# Convenience functions

def create_mock_exchange_client(exchange: ExchangeType = ExchangeType.BINANCE) -> ExchangeMockResponses:
    """Create mock exchange client"""
    return ExchangeMockResponses(exchange)


def create_mock_websocket_client() -> WebSocketMockResponses:
    """Create mock WebSocket client"""
    return WebSocketMockResponses()


def create_mock_response_factory(scenario: str = "normal") -> MockResponseFactory:
    """Create mock response factory with scenario"""
    factory = MockResponseFactory()
    factory.set_scenario(scenario)
    return factory


# Test response data
MOCK_TICKER = {
    "symbol": "BTCUSDT",
    "lastPrice": "45000.00",
    "bidPrice": "44999.50",
    "askPrice": "45000.50",
    "volume": "12614.42000000"
}

MOCK_ORDER = {
    "orderId": 12345678,
    "symbol": "BTCUSDT",
    "status": "NEW",
    "side": "BUY",
    "price": "45000.00",
    "origQty": "0.10000000"
}

MOCK_BALANCE = {
    "asset": "USDT",
    "free": "10000.00000000",
    "locked": "5000.00000000"
}


if __name__ == "__main__":
    # Example usage
    
    # Create exchange mock
    exchange = create_mock_exchange_client()
    
    # Get ticker
    ticker_response = exchange.get_ticker_24hr("BTCUSDT")
    print(f"Ticker Response: {ticker_response.data['lastPrice']}")
    
    # Place order
    order_response = exchange.place_order(
        "BTCUSDT",
        "buy",
        "limit",
        price="45000",
        quantity="0.1"
    )
    print(f"Order Response: {order_response.data['orderId']}")
    
    # Create WebSocket mock
    ws = create_mock_websocket_client()
    
    # Get trade stream
    trade = ws.get_trade_stream("btcusdt")
    print(f"Trade Stream: {trade['p']} @ {trade['q']}")
    
    # Test scenarios
    factory = create_mock_response_factory("high_latency")
    response = factory.get_exchange_response(
        ExchangeType.BINANCE,
        "get_ticker_24hr",
        symbol="BTCUSDT"
    )
    print(f"High Latency Response: {response.latency_ms}ms")