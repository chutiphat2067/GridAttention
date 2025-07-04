#!/usr/bin/env python3
"""
Mock API Client for GridAttention Trading System
Provides realistic API client simulation for various external services
"""

import asyncio
import aiohttp
import json
import time
import random
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from decimal import Decimal
import hashlib
import hmac
import base64
from urllib.parse import urlencode
import jwt


logger = logging.getLogger(__name__)


class APIProvider(Enum):
    """API provider types"""
    EXCHANGE = "exchange"
    MARKET_DATA = "market_data"
    SENTIMENT = "sentiment"
    NEWS = "news"
    ECONOMIC = "economic"
    BLOCKCHAIN = "blockchain"
    SOCIAL = "social"


class RequestMethod(Enum):
    """HTTP request methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass
class APIResponse:
    """API response wrapper"""
    status_code: int
    data: Any
    headers: Dict[str, str] = field(default_factory=dict)
    elapsed_ms: float = 0
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300
    
    @property
    def is_error(self) -> bool:
        return self.status_code >= 400
    
    def json(self) -> Any:
        """Get JSON data"""
        if isinstance(self.data, str):
            return json.loads(self.data)
        return self.data
    
    def raise_for_status(self):
        """Raise exception for error status"""
        if self.is_error:
            raise APIError(f"API error {self.status_code}: {self.data}")


class APIError(Exception):
    """API error exception"""
    pass


class RateLimitError(APIError):
    """Rate limit error"""
    pass


class AuthenticationError(APIError):
    """Authentication error"""
    pass


class MockAPIClient:
    """Base mock API client"""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        timeout: int = 30
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout
        
        # Request tracking
        self.request_count = 0
        self.request_history: deque = deque(maxlen=1000)
        
        # Rate limiting
        self.rate_limiter = RateLimiter()
        
        # Response delays
        self.min_delay = 0.01
        self.max_delay = 0.1
        
        # Error simulation
        self.error_rate = 0.0
        self.timeout_rate = 0.0
        
        # Mock responses
        self.mock_responses: Dict[str, Callable] = {}
        self._setup_mock_responses()
    
    def _setup_mock_responses(self):
        """Setup mock response handlers"""
        # Override in subclasses
        pass
    
    async def request(
        self,
        method: Union[str, RequestMethod],
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        authenticated: bool = False
    ) -> APIResponse:
        """Make API request"""
        start_time = time.time()
        
        # Convert method to string
        if isinstance(method, RequestMethod):
            method = method.value
        
        # Build URL
        url = f"{self.base_url}{endpoint}"
        
        # Track request
        self.request_count += 1
        request_data = {
            "method": method,
            "url": url,
            "params": params,
            "data": data,
            "timestamp": datetime.now()
        }
        self.request_history.append(request_data)
        
        # Check rate limit
        if not self.rate_limiter.check_limit(endpoint):
            elapsed = (time.time() - start_time) * 1000
            return APIResponse(
                status_code=429,
                data={"error": "Rate limit exceeded"},
                headers={"Retry-After": "60"},
                elapsed_ms=elapsed
            )
        
        # Simulate network delay
        delay = random.uniform(self.min_delay, self.max_delay)
        await asyncio.sleep(delay)
        
        # Simulate timeout
        if random.random() < self.timeout_rate:
            raise asyncio.TimeoutError(f"Request timeout after {self.timeout}s")
        
        # Simulate random errors
        if random.random() < self.error_rate:
            elapsed = (time.time() - start_time) * 1000
            return APIResponse(
                status_code=500,
                data={"error": "Internal server error"},
                elapsed_ms=elapsed
            )
        
        # Build headers
        request_headers = self._build_headers(headers or {})
        
        # Add authentication if required
        if authenticated:
            auth_headers = self._build_auth_headers(method, endpoint, params, data)
            request_headers.update(auth_headers)
        
        # Get mock response
        response = await self._get_mock_response(method, endpoint, params, data)
        
        # Calculate elapsed time
        elapsed = (time.time() - start_time) * 1000
        response.elapsed_ms = elapsed
        
        return response
    
    async def get(self, endpoint: str, params: Optional[Dict] = None, **kwargs) -> APIResponse:
        """GET request"""
        return await self.request(RequestMethod.GET, endpoint, params=params, **kwargs)
    
    async def post(self, endpoint: str, data: Optional[Dict] = None, **kwargs) -> APIResponse:
        """POST request"""
        return await self.request(RequestMethod.POST, endpoint, data=data, **kwargs)
    
    async def put(self, endpoint: str, data: Optional[Dict] = None, **kwargs) -> APIResponse:
        """PUT request"""
        return await self.request(RequestMethod.PUT, endpoint, data=data, **kwargs)
    
    async def delete(self, endpoint: str, **kwargs) -> APIResponse:
        """DELETE request"""
        return await self.request(RequestMethod.DELETE, endpoint, **kwargs)
    
    def _build_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Build request headers"""
        default_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "GridAttention/1.0"
        }
        default_headers.update(headers)
        return default_headers
    
    def _build_auth_headers(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict],
        data: Optional[Dict]
    ) -> Dict[str, str]:
        """Build authentication headers"""
        if self.api_key:
            return {"X-API-Key": self.api_key}
        return {}
    
    async def _get_mock_response(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict],
        data: Optional[Dict]
    ) -> APIResponse:
        """Get mock response for request"""
        # Look for specific handler
        handler_key = f"{method}:{endpoint}"
        if handler_key in self.mock_responses:
            handler = self.mock_responses[handler_key]
            return await handler(params, data)
        
        # Default response
        return APIResponse(
            status_code=200,
            data={"message": "OK"},
            headers={"Content-Type": "application/json"}
        )
    
    def set_error_rate(self, rate: float):
        """Set random error rate"""
        self.error_rate = max(0, min(1, rate))
    
    def set_timeout_rate(self, rate: float):
        """Set timeout rate"""
        self.timeout_rate = max(0, min(1, rate))
    
    def set_delay(self, min_delay: float, max_delay: float):
        """Set response delay range"""
        self.min_delay = max(0, min_delay)
        self.max_delay = max(self.min_delay, max_delay)
    
    def get_request_stats(self) -> Dict[str, Any]:
        """Get request statistics"""
        if not self.request_history:
            return {}
        
        # Count by method
        method_counts = defaultdict(int)
        endpoint_counts = defaultdict(int)
        
        for req in self.request_history:
            method_counts[req["method"]] += 1
            url_parts = req["url"].split("/")
            if len(url_parts) > 3:
                endpoint = "/" + "/".join(url_parts[3:])
                endpoint_counts[endpoint] += 1
        
        return {
            "total_requests": self.request_count,
            "by_method": dict(method_counts),
            "by_endpoint": dict(endpoint_counts),
            "rate_limit_remaining": self.rate_limiter.get_remaining()
        }


class MockExchangeAPIClient(MockAPIClient):
    """Mock exchange API client"""
    
    def __init__(self, exchange_name: str = "binance", **kwargs):
        self.exchange_name = exchange_name
        base_urls = {
            "binance": "https://api.binance.com",
            "bybit": "https://api.bybit.com",
            "okx": "https://www.okx.com"
        }
        super().__init__(base_urls.get(exchange_name, base_urls["binance"]), **kwargs)
        
        # Exchange-specific data
        self.symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
        self.balances = {
            "BTC": {"free": 0.5, "locked": 0.0},
            "ETH": {"free": 5.0, "locked": 0.0},
            "USDT": {"free": 10000.0, "locked": 5000.0}
        }
        self.open_orders = []
        self.order_id_counter = 1000000
    
    def _setup_mock_responses(self):
        """Setup exchange-specific mock responses"""
        self.mock_responses = {
            "GET:/api/v3/exchangeInfo": self._mock_exchange_info,
            "GET:/api/v3/ticker/24hr": self._mock_ticker_24hr,
            "GET:/api/v3/depth": self._mock_orderbook,
            "GET:/api/v3/klines": self._mock_klines,
            "GET:/api/v3/account": self._mock_account,
            "POST:/api/v3/order": self._mock_place_order,
            "DELETE:/api/v3/order": self._mock_cancel_order,
            "GET:/api/v3/order": self._mock_query_order,
            "GET:/api/v3/openOrders": self._mock_open_orders,
            "GET:/api/v3/myTrades": self._mock_my_trades
        }
    
    async def _mock_exchange_info(self, params: Optional[Dict], data: Optional[Dict]) -> APIResponse:
        """Mock exchange info response"""
        symbols_info = []
        
        for symbol in self.symbols:
            symbols_info.append({
                "symbol": symbol,
                "status": "TRADING",
                "baseAsset": symbol[:-4],
                "quoteAsset": "USDT",
                "baseAssetPrecision": 8,
                "quotePrecision": 8,
                "orderTypes": ["LIMIT", "MARKET", "STOP_LOSS", "TAKE_PROFIT"],
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
                    }
                ]
            })
        
        return APIResponse(
            status_code=200,
            data={
                "timezone": "UTC",
                "serverTime": int(datetime.now().timestamp() * 1000),
                "symbols": symbols_info
            }
        )
    
    async def _mock_ticker_24hr(self, params: Optional[Dict], data: Optional[Dict]) -> APIResponse:
        """Mock 24hr ticker response"""
        symbol = params.get("symbol", "BTCUSDT") if params else "BTCUSDT"
        
        if symbol not in self.symbols:
            return APIResponse(
                status_code=400,
                data={"code": -1121, "msg": "Invalid symbol."}
            )
        
        base_prices = {
            "BTCUSDT": 45000,
            "ETHUSDT": 3000,
            "BNBUSDT": 450,
            "SOLUSDT": 120
        }
        
        base_price = base_prices.get(symbol, 100)
        change_percent = random.uniform(-5, 5)
        
        return APIResponse(
            status_code=200,
            data={
                "symbol": symbol,
                "priceChange": f"{base_price * change_percent / 100:.2f}",
                "priceChangePercent": f"{change_percent:.2f}",
                "weightedAvgPrice": f"{base_price:.2f}",
                "prevClosePrice": f"{base_price:.2f}",
                "lastPrice": f"{base_price * (1 + change_percent / 100):.2f}",
                "lastQty": "1.00000000",
                "bidPrice": f"{base_price * (1 + change_percent / 100 - 0.0001):.2f}",
                "bidQty": "4.00000000",
                "askPrice": f"{base_price * (1 + change_percent / 100 + 0.0001):.2f}",
                "askQty": "4.00000000",
                "openPrice": f"{base_price:.2f}",
                "highPrice": f"{base_price * 1.02:.2f}",
                "lowPrice": f"{base_price * 0.98:.2f}",
                "volume": "12345.67890000",
                "quoteVolume": f"{base_price * 12345.67890:.2f}",
                "openTime": int((datetime.now() - timedelta(hours=24)).timestamp() * 1000),
                "closeTime": int(datetime.now().timestamp() * 1000),
                "firstId": 100000,
                "lastId": 200000,
                "count": 100000
            }
        )
    
    async def _mock_orderbook(self, params: Optional[Dict], data: Optional[Dict]) -> APIResponse:
        """Mock orderbook response"""
        symbol = params.get("symbol", "BTCUSDT") if params else "BTCUSDT"
        limit = int(params.get("limit", 100)) if params else 100
        
        base_prices = {"BTCUSDT": 45000, "ETHUSDT": 3000, "BNBUSDT": 450, "SOLUSDT": 120}
        base_price = base_prices.get(symbol, 100)
        
        # Generate bids and asks
        bids = []
        asks = []
        
        for i in range(min(limit, 100)):
            bid_price = base_price * (1 - 0.0001 * (i + 1))
            ask_price = base_price * (1 + 0.0001 * (i + 1))
            
            bid_qty = random.uniform(0.1, 2.0) * (1 + i * 0.1)
            ask_qty = random.uniform(0.1, 2.0) * (1 + i * 0.1)
            
            bids.append([f"{bid_price:.2f}", f"{bid_qty:.8f}"])
            asks.append([f"{ask_price:.2f}", f"{ask_qty:.8f}"])
        
        return APIResponse(
            status_code=200,
            data={
                "lastUpdateId": random.randint(1000000, 9999999),
                "bids": bids,
                "asks": asks
            }
        )
    
    async def _mock_klines(self, params: Optional[Dict], data: Optional[Dict]) -> APIResponse:
        """Mock klines response"""
        if not params:
            return APIResponse(
                status_code=400,
                data={"code": -1102, "msg": "Mandatory parameter 'symbol' was not sent"}
            )
        
        symbol = params.get("symbol", "BTCUSDT")
        interval = params.get("interval", "5m")
        limit = int(params.get("limit", 100))
        
        base_prices = {"BTCUSDT": 45000, "ETHUSDT": 3000, "BNBUSDT": 450, "SOLUSDT": 120}
        base_price = base_prices.get(symbol, 100)
        
        # Generate klines
        klines = []
        current_price = base_price
        
        interval_ms = {
            "1m": 60000, "5m": 300000, "15m": 900000,
            "1h": 3600000, "4h": 14400000, "1d": 86400000
        }
        time_step = interval_ms.get(interval, 300000)
        
        current_time = int(datetime.now().timestamp() * 1000)
        
        for i in range(limit):
            open_time = current_time - (limit - i) * time_step
            close_time = open_time + time_step - 1
            
            # Random price movement
            change = random.uniform(-0.002, 0.002)
            current_price *= (1 + change)
            
            open_price = current_price
            high_price = current_price * (1 + random.uniform(0, 0.001))
            low_price = current_price * (1 - random.uniform(0, 0.001))
            close_price = current_price * (1 + random.uniform(-0.001, 0.001))
            
            volume = random.uniform(10, 100)
            
            klines.append([
                open_time,
                f"{open_price:.2f}",
                f"{high_price:.2f}",
                f"{low_price:.2f}",
                f"{close_price:.2f}",
                f"{volume:.8f}",
                close_time,
                f"{volume * close_price:.2f}",
                random.randint(100, 1000),
                f"{volume * 0.5:.8f}",
                f"{volume * close_price * 0.5:.2f}",
                "0"
            ])
            
            current_price = close_price
        
        return APIResponse(
            status_code=200,
            data=klines
        )
    
    async def _mock_account(self, params: Optional[Dict], data: Optional[Dict]) -> APIResponse:
        """Mock account response"""
        balances_list = []
        
        for asset, balance in self.balances.items():
            balances_list.append({
                "asset": asset,
                "free": str(balance["free"]),
                "locked": str(balance["locked"])
            })
        
        return APIResponse(
            status_code=200,
            data={
                "makerCommission": 10,
                "takerCommission": 10,
                "buyerCommission": 0,
                "sellerCommission": 0,
                "canTrade": True,
                "canWithdraw": True,
                "canDeposit": True,
                "updateTime": int(datetime.now().timestamp() * 1000),
                "accountType": "SPOT",
                "balances": balances_list,
                "permissions": ["SPOT"]
            }
        )
    
    async def _mock_place_order(self, params: Optional[Dict], data: Optional[Dict]) -> APIResponse:
        """Mock place order response"""
        if not data:
            return APIResponse(
                status_code=400,
                data={"code": -1102, "msg": "Mandatory parameter was not sent"}
            )
        
        # Validate required parameters
        required = ["symbol", "side", "type", "quantity"]
        for param in required:
            if param not in data:
                return APIResponse(
                    status_code=400,
                    data={"code": -1102, "msg": f"Mandatory parameter '{param}' was not sent"}
                )
        
        # Generate order ID
        order_id = self.order_id_counter
        self.order_id_counter += 1
        
        # Create order
        order = {
            "symbol": data["symbol"],
            "orderId": order_id,
            "orderListId": -1,
            "clientOrderId": data.get("newClientOrderId", f"auto_{order_id}"),
            "transactTime": int(datetime.now().timestamp() * 1000),
            "price": data.get("price", "0.00000000"),
            "origQty": data["quantity"],
            "executedQty": "0.00000000",
            "cummulativeQuoteQty": "0.00000000",
            "status": "NEW",
            "timeInForce": data.get("timeInForce", "GTC"),
            "type": data["type"],
            "side": data["side"],
            "fills": []
        }
        
        # Add to open orders
        self.open_orders.append(order)
        
        return APIResponse(
            status_code=200,
            data=order
        )
    
    async def _mock_cancel_order(self, params: Optional[Dict], data: Optional[Dict]) -> APIResponse:
        """Mock cancel order response"""
        if not params or "symbol" not in params:
            return APIResponse(
                status_code=400,
                data={"code": -1102, "msg": "Mandatory parameter 'symbol' was not sent"}
            )
        
        order_id = params.get("orderId")
        if not order_id:
            return APIResponse(
                status_code=400,
                data={"code": -1102, "msg": "Either orderId or origClientOrderId must be sent"}
            )
        
        # Find and remove order
        for i, order in enumerate(self.open_orders):
            if str(order["orderId"]) == str(order_id):
                cancelled_order = self.open_orders.pop(i)
                cancelled_order["status"] = "CANCELED"
                return APIResponse(
                    status_code=200,
                    data=cancelled_order
                )
        
        return APIResponse(
            status_code=400,
            data={"code": -2011, "msg": "Unknown order sent."}
        )
    
    async def _mock_query_order(self, params: Optional[Dict], data: Optional[Dict]) -> APIResponse:
        """Mock query order response"""
        if not params or "symbol" not in params:
            return APIResponse(
                status_code=400,
                data={"code": -1102, "msg": "Mandatory parameter 'symbol' was not sent"}
            )
        
        order_id = params.get("orderId")
        if not order_id:
            return APIResponse(
                status_code=400,
                data={"code": -1102, "msg": "Either orderId or origClientOrderId must be sent"}
            )
        
        # Find order
        for order in self.open_orders:
            if str(order["orderId"]) == str(order_id):
                return APIResponse(
                    status_code=200,
                    data=order
                )
        
        # Mock a filled order
        return APIResponse(
            status_code=200,
            data={
                "symbol": params["symbol"],
                "orderId": order_id,
                "orderListId": -1,
                "clientOrderId": f"auto_{order_id}",
                "price": "45000.00",
                "origQty": "0.10000000",
                "executedQty": "0.10000000",
                "cummulativeQuoteQty": "4500.00000000",
                "status": "FILLED",
                "timeInForce": "GTC",
                "type": "LIMIT",
                "side": "BUY",
                "stopPrice": "0.00000000",
                "icebergQty": "0.00000000",
                "time": int((datetime.now() - timedelta(minutes=30)).timestamp() * 1000),
                "updateTime": int(datetime.now().timestamp() * 1000),
                "isWorking": False,
                "origQuoteOrderQty": "0.00000000"
            }
        )
    
    async def _mock_open_orders(self, params: Optional[Dict], data: Optional[Dict]) -> APIResponse:
        """Mock open orders response"""
        symbol = params.get("symbol") if params else None
        
        if symbol:
            orders = [o for o in self.open_orders if o["symbol"] == symbol]
        else:
            orders = self.open_orders.copy()
        
        return APIResponse(
            status_code=200,
            data=orders
        )
    
    async def _mock_my_trades(self, params: Optional[Dict], data: Optional[Dict]) -> APIResponse:
        """Mock my trades response"""
        if not params or "symbol" not in params:
            return APIResponse(
                status_code=400,
                data={"code": -1102, "msg": "Mandatory parameter 'symbol' was not sent"}
            )
        
        symbol = params["symbol"]
        limit = int(params.get("limit", 500))
        
        # Generate mock trades
        trades = []
        for i in range(min(5, limit)):
            trades.append({
                "symbol": symbol,
                "id": random.randint(1000000, 9999999),
                "orderId": random.randint(1000000, 9999999),
                "orderListId": -1,
                "price": "45000.00",
                "qty": "0.10000000",
                "quoteQty": "4500.00000000",
                "commission": "0.00010000",
                "commissionAsset": "BTC",
                "time": int((datetime.now() - timedelta(hours=i)).timestamp() * 1000),
                "isBuyer": i % 2 == 0,
                "isMaker": i % 3 == 0,
                "isBestMatch": True
            })
        
        return APIResponse(
            status_code=200,
            data=trades
        )
    
    def _build_auth_headers(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict],
        data: Optional[Dict]
    ) -> Dict[str, str]:
        """Build exchange authentication headers"""
        if not self.api_key or not self.api_secret:
            return {}
        
        # Add timestamp
        timestamp = int(datetime.now().timestamp() * 1000)
        
        # Build query string
        query_params = params or {}
        query_params["timestamp"] = timestamp
        
        if data:
            query_params.update(data)
        
        # Sort parameters
        sorted_params = sorted(query_params.items())
        query_string = urlencode(sorted_params)
        
        # Create signature
        signature = hmac.new(
            self.api_secret.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return {
            "X-MBX-APIKEY": self.api_key
        }


class MockMarketDataAPIClient(MockAPIClient):
    """Mock market data API client"""
    
    def __init__(self, provider: str = "cryptocompare", **kwargs):
        self.provider = provider
        base_urls = {
            "cryptocompare": "https://min-api.cryptocompare.com",
            "coingecko": "https://api.coingecko.com/api/v3",
            "messari": "https://data.messari.io/api"
        }
        super().__init__(base_urls.get(provider, base_urls["cryptocompare"]), **kwargs)
    
    def _setup_mock_responses(self):
        """Setup market data mock responses"""
        self.mock_responses = {
            "GET:/data/v2/histoday": self._mock_historical_daily,
            "GET:/data/v2/histohour": self._mock_historical_hourly,
            "GET:/data/pricemultifull": self._mock_price_multi,
            "GET:/data/v2/news/": self._mock_news,
            "GET:/data/social/coin/latest": self._mock_social_stats
        }
    
    async def _mock_historical_daily(self, params: Optional[Dict], data: Optional[Dict]) -> APIResponse:
        """Mock historical daily data"""
        fsym = params.get("fsym", "BTC") if params else "BTC"
        tsym = params.get("tsym", "USD") if params else "USD"
        limit = int(params.get("limit", 30)) if params else 30
        
        # Generate historical data
        data_points = []
        base_price = 45000 if fsym == "BTC" else 3000
        current_time = int(datetime.now().timestamp())
        
        for i in range(limit):
            timestamp = current_time - (limit - i) * 86400
            price = base_price * (1 + random.uniform(-0.02, 0.02))
            
            data_points.append({
                "time": timestamp,
                "high": price * 1.01,
                "low": price * 0.99,
                "open": price,
                "volumefrom": random.uniform(1000, 5000),
                "volumeto": random.uniform(45000000, 55000000),
                "close": price * (1 + random.uniform(-0.01, 0.01)),
                "conversionType": "direct",
                "conversionSymbol": ""
            })
        
        return APIResponse(
            status_code=200,
            data={
                "Response": "Success",
                "Data": {
                    "Aggregated": False,
                    "TimeFrom": data_points[0]["time"],
                    "TimeTo": data_points[-1]["time"],
                    "Data": data_points
                }
            }
        )
    
    async def _mock_historical_hourly(self, params: Optional[Dict], data: Optional[Dict]) -> APIResponse:
        """Mock historical hourly data"""
        # Similar to daily but with hourly intervals
        fsym = params.get("fsym", "BTC") if params else "BTC"
        limit = int(params.get("limit", 24)) if params else 24
        
        data_points = []
        base_price = 45000 if fsym == "BTC" else 3000
        current_time = int(datetime.now().timestamp())
        
        for i in range(limit):
            timestamp = current_time - (limit - i) * 3600
            price = base_price * (1 + random.uniform(-0.005, 0.005))
            
            data_points.append({
                "time": timestamp,
                "high": price * 1.002,
                "low": price * 0.998,
                "open": price,
                "volumefrom": random.uniform(100, 500),
                "volumeto": random.uniform(4500000, 5500000),
                "close": price * (1 + random.uniform(-0.002, 0.002))
            })
        
        return APIResponse(
            status_code=200,
            data={
                "Response": "Success",
                "Data": {
                    "Data": data_points
                }
            }
        )
    
    async def _mock_price_multi(self, params: Optional[Dict], data: Optional[Dict]) -> APIResponse:
        """Mock multi-price data"""
        fsyms = params.get("fsyms", "BTC,ETH").split(",") if params else ["BTC", "ETH"]
        tsyms = params.get("tsyms", "USD,EUR").split(",") if params else ["USD", "EUR"]
        
        result = {}
        base_prices = {"BTC": 45000, "ETH": 3000, "BNB": 450}
        
        for fsym in fsyms:
            result[fsym] = {}
            base_price = base_prices.get(fsym, 100)
            
            for tsym in tsyms:
                exchange_rate = 1.0 if tsym == "USD" else 0.85
                current_price = base_price * exchange_rate
                
                result[fsym][tsym] = {
                    "PRICE": current_price,
                    "CHANGE24HOUR": random.uniform(-1000, 1000),
                    "CHANGEPCT24HOUR": random.uniform(-5, 5),
                    "VOLUME24HOUR": random.uniform(10000, 50000),
                    "VOLUME24HOURTO": random.uniform(450000000, 550000000),
                    "OPEN24HOUR": current_price * 0.98,
                    "HIGH24HOUR": current_price * 1.02,
                    "LOW24HOUR": current_price * 0.97,
                    "MKTCAP": current_price * 19000000
                }
        
        return APIResponse(
            status_code=200,
            data={
                "DISPLAY": result
            }
        )
    
    async def _mock_news(self, params: Optional[Dict], data: Optional[Dict]) -> APIResponse:
        """Mock news data"""
        categories = params.get("categories", "").split(",") if params else []
        
        news_items = []
        titles = [
            "Bitcoin Reaches New High Amid Institutional Interest",
            "Ethereum 2.0 Staking Surpasses Expectations",
            "Regulatory Clarity Boosts Crypto Market Confidence",
            "DeFi Protocol Launches Revolutionary Feature",
            "Major Bank Announces Crypto Custody Service"
        ]
        
        for i, title in enumerate(titles[:5]):
            news_items.append({
                "id": str(1000 + i),
                "guid": f"https://example.com/news/{1000 + i}",
                "published_on": int((datetime.now() - timedelta(hours=i)).timestamp()),
                "imageurl": f"https://example.com/images/{i}.jpg",
                "title": title,
                "url": f"https://example.com/news/{1000 + i}",
                "source": random.choice(["CoinDesk", "CoinTelegraph", "Reuters", "Bloomberg"]),
                "body": f"Article body for {title}...",
                "tags": "bitcoin|cryptocurrency|blockchain",
                "categories": "Market|Analysis",
                "lang": "EN",
                "source_info": {
                    "name": "Example News",
                    "lang": "EN",
                    "img": "https://example.com/logo.png"
                }
            })
        
        return APIResponse(
            status_code=200,
            data={
                "Data": news_items
            }
        )
    
    async def _mock_social_stats(self, params: Optional[Dict], data: Optional[Dict]) -> APIResponse:
        """Mock social statistics"""
        coin_id = params.get("coinId", "1182") if params else "1182"  # BTC
        
        return APIResponse(
            status_code=200,
            data={
                "Response": "Success",
                "Data": {
                    "General": {
                        "Points": random.randint(1000000, 2000000),
                        "Name": "Bitcoin",
                        "CoinName": "BTC",
                        "Type": "Webpagecoinp"
                    },
                    "CryptoCompare": {
                        "SimilarItems": [],
                        "Followers": random.randint(100000, 200000),
                        "Posts": random.randint(10000, 20000),
                        "Comments": random.randint(50000, 100000),
                        "Points": random.randint(500000, 1000000)
                    },
                    "Twitter": {
                        "followers": random.randint(1000000, 2000000),
                        "statuses": random.randint(10000, 20000),
                        "lists": random.randint(1000, 2000),
                        "Points": random.randint(500000, 1000000)
                    },
                    "Reddit": {
                        "subscribers": random.randint(3000000, 4000000),
                        "active_users": random.randint(10000, 20000),
                        "posts_per_hour": random.uniform(10, 50),
                        "posts_per_day": random.uniform(200, 500),
                        "comments_per_hour": random.uniform(100, 500),
                        "comments_per_day": random.uniform(2000, 10000),
                        "Points": random.randint(500000, 1000000)
                    }
                }
            }
        )


class MockSentimentAPIClient(MockAPIClient):
    """Mock sentiment analysis API client"""
    
    def __init__(self, **kwargs):
        super().__init__("https://api.sentiment.example.com", **kwargs)
    
    def _setup_mock_responses(self):
        """Setup sentiment mock responses"""
        self.mock_responses = {
            "POST:/analyze": self._mock_analyze,
            "GET:/sentiment/crypto": self._mock_crypto_sentiment,
            "GET:/sentiment/social": self._mock_social_sentiment
        }
    
    async def _mock_analyze(self, params: Optional[Dict], data: Optional[Dict]) -> APIResponse:
        """Mock sentiment analysis"""
        if not data or "text" not in data:
            return APIResponse(
                status_code=400,
                data={"error": "Missing text parameter"}
            )
        
        # Simple sentiment based on keywords
        text = data["text"].lower()
        positive_words = ["bullish", "moon", "pump", "buy", "growth", "positive"]
        negative_words = ["bearish", "dump", "sell", "crash", "negative", "fear"]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            sentiment = "positive"
            score = min(1.0, 0.5 + positive_count * 0.1)
        elif negative_count > positive_count:
            sentiment = "negative"
            score = max(-1.0, -0.5 - negative_count * 0.1)
        else:
            sentiment = "neutral"
            score = random.uniform(-0.2, 0.2)
        
        return APIResponse(
            status_code=200,
            data={
                "text": data["text"],
                "sentiment": sentiment,
                "score": score,
                "confidence": random.uniform(0.7, 0.95),
                "entities": [],
                "keywords": positive_words[:3] if sentiment == "positive" else negative_words[:3]
            }
        )
    
    async def _mock_crypto_sentiment(self, params: Optional[Dict], data: Optional[Dict]) -> APIResponse:
        """Mock crypto sentiment data"""
        symbol = params.get("symbol", "BTC") if params else "BTC"
        
        return APIResponse(
            status_code=200,
            data={
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "sentiment": {
                    "overall": random.uniform(-1, 1),
                    "twitter": random.uniform(-1, 1),
                    "reddit": random.uniform(-1, 1),
                    "news": random.uniform(-1, 1)
                },
                "volume": {
                    "mentions": random.randint(1000, 50000),
                    "unique_users": random.randint(500, 10000)
                },
                "trending": random.choice([True, False]),
                "momentum": random.choice(["increasing", "decreasing", "stable"])
            }
        )
    
    async def _mock_social_sentiment(self, params: Optional[Dict], data: Optional[Dict]) -> APIResponse:
        """Mock social sentiment data"""
        platform = params.get("platform", "twitter") if params else "twitter"
        
        sentiments = []
        for i in range(10):
            sentiments.append({
                "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
                "sentiment_score": random.uniform(-1, 1),
                "volume": random.randint(100, 1000),
                "engagement": random.randint(1000, 10000),
                "reach": random.randint(10000, 100000)
            })
        
        return APIResponse(
            status_code=200,
            data={
                "platform": platform,
                "data": sentiments,
                "aggregated": {
                    "average_sentiment": sum(s["sentiment_score"] for s in sentiments) / len(sentiments),
                    "total_volume": sum(s["volume"] for s in sentiments),
                    "total_engagement": sum(s["engagement"] for s in sentiments)
                }
            }
        )


class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self):
        self.limits = {
            "default": {"requests": 1200, "window": 60},
            "/api/v3/order": {"requests": 100, "window": 60},
            "/api/v3/openOrders": {"requests": 40, "window": 60}
        }
        self.requests = defaultdict(deque)
    
    def check_limit(self, endpoint: str) -> bool:
        """Check if request is within rate limit"""
        limit_key = endpoint if endpoint in self.limits else "default"
        limit = self.limits[limit_key]
        
        now = time.time()
        window_start = now - limit["window"]
        
        # Clean old requests
        while self.requests[limit_key] and self.requests[limit_key][0] < window_start:
            self.requests[limit_key].popleft()
        
        # Check limit
        if len(self.requests[limit_key]) >= limit["requests"]:
            return False
        
        # Record request
        self.requests[limit_key].append(now)
        return True
    
    def get_remaining(self) -> Dict[str, int]:
        """Get remaining requests for each endpoint"""
        remaining = {}
        now = time.time()
        
        for endpoint, limit in self.limits.items():
            window_start = now - limit["window"]
            recent_requests = sum(1 for r in self.requests[endpoint] if r > window_start)
            remaining[endpoint] = limit["requests"] - recent_requests
        
        return remaining


# Helper functions

def create_mock_exchange_client(
    exchange: str = "binance",
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None
) -> MockExchangeAPIClient:
    """Create mock exchange API client"""
    return MockExchangeAPIClient(
        exchange_name=exchange,
        api_key=api_key or "mock_api_key",
        api_secret=api_secret or "mock_api_secret"
    )


def create_mock_market_data_client(provider: str = "cryptocompare") -> MockMarketDataAPIClient:
    """Create mock market data API client"""
    return MockMarketDataAPIClient(
        provider=provider,
        api_key="mock_api_key"
    )


def create_mock_sentiment_client() -> MockSentimentAPIClient:
    """Create mock sentiment API client"""
    return MockSentimentAPIClient(api_key="mock_api_key")


# WebSocket mock client

class MockWebSocketClient:
    """Mock WebSocket client"""
    
    def __init__(self, url: str):
        self.url = url
        self.connected = False
        self.callbacks = {}
        self.running = False
        
    async def connect(self):
        """Connect to WebSocket"""
        await asyncio.sleep(0.1)  # Simulate connection delay
        self.connected = True
        self.running = True
        
        # Start message loop
        asyncio.create_task(self._message_loop())
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        self.running = False
        self.connected = False
    
    async def subscribe(self, channel: str, callback: Callable):
        """Subscribe to channel"""
        self.callbacks[channel] = callback
    
    async def unsubscribe(self, channel: str):
        """Unsubscribe from channel"""
        self.callbacks.pop(channel, None)
    
    async def _message_loop(self):
        """Simulate incoming messages"""
        while self.running:
            for channel, callback in self.callbacks.items():
                if "ticker" in channel:
                    message = self._generate_ticker_message(channel)
                elif "depth" in channel:
                    message = self._generate_depth_message(channel)
                elif "trade" in channel:
                    message = self._generate_trade_message(channel)
                else:
                    message = {"channel": channel, "data": {}}
                
                await callback(message)
            
            await asyncio.sleep(1)  # Send updates every second
    
    def _generate_ticker_message(self, channel: str) -> Dict:
        """Generate ticker message"""
        symbol = channel.split("@")[0].upper()
        return {
            "e": "24hrTicker",
            "E": int(datetime.now().timestamp() * 1000),
            "s": symbol,
            "c": f"{random.uniform(44000, 46000):.2f}",
            "p": f"{random.uniform(-100, 100):.2f}",
            "P": f"{random.uniform(-2, 2):.2f}",
            "v": f"{random.uniform(1000, 2000):.8f}"
        }
    
    def _generate_depth_message(self, channel: str) -> Dict:
        """Generate depth update message"""
        symbol = channel.split("@")[0].upper()
        return {
            "e": "depthUpdate",
            "E": int(datetime.now().timestamp() * 1000),
            "s": symbol,
            "b": [[f"{random.uniform(44000, 45000):.2f}", f"{random.uniform(0.1, 1):.8f}"]],
            "a": [[f"{random.uniform(45000, 46000):.2f}", f"{random.uniform(0.1, 1):.8f}"]]
        }
    
    def _generate_trade_message(self, channel: str) -> Dict:
        """Generate trade message"""
        symbol = channel.split("@")[0].upper()
        return {
            "e": "trade",
            "E": int(datetime.now().timestamp() * 1000),
            "s": symbol,
            "t": random.randint(1000000, 9999999),
            "p": f"{random.uniform(44000, 46000):.2f}",
            "q": f"{random.uniform(0.01, 1):.8f}",
            "m": random.choice([True, False])
        }


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create exchange client
        exchange = create_mock_exchange_client("binance")
        
        # Get ticker
        response = await exchange.get("/api/v3/ticker/24hr", params={"symbol": "BTCUSDT"})
        print(f"Ticker: {response.json()}")
        
        # Place order
        order_response = await exchange.post(
            "/api/v3/order",
            data={
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "quantity": "0.1",
                "price": "45000",
                "timeInForce": "GTC"
            },
            authenticated=True
        )
        print(f"Order: {order_response.json()}")
        
        # Get market data
        market_data = create_mock_market_data_client()
        history = await market_data.get(
            "/data/v2/histoday",
            params={"fsym": "BTC", "tsym": "USD", "limit": 10}
        )
        print(f"Historical data: {len(history.json()['Data']['Data'])} days")
        
        # Get sentiment
        sentiment = create_mock_sentiment_client()
        analysis = await sentiment.post(
            "/analyze",
            data={"text": "Bitcoin is looking very bullish today!"}
        )
        print(f"Sentiment: {analysis.json()}")
        
        # Test WebSocket
        ws = MockWebSocketClient("wss://stream.binance.com:9443/ws")
        await ws.connect()
        
        async def on_ticker(msg):
            print(f"WS Ticker: {msg['s']} @ {msg['c']}")
        
        await ws.subscribe("btcusdt@ticker", on_ticker)
        await asyncio.sleep(3)
        await ws.disconnect()
    
    asyncio.run(main())