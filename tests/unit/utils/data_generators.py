#!/usr/bin/env python3
"""
Data Generators for GridAttention Trading System
Provides comprehensive data generation utilities for testing all system components
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Generator, Iterator
from dataclasses import dataclass, field
from decimal import Decimal
import json
import uuid
import string
from enum import Enum
import math
from collections import defaultdict
import hashlib


# Market Regime Types for realistic data generation
class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"


# Volatility Profiles
class VolatilityProfile(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class GeneratorConfig:
    """Configuration for data generators"""
    seed: Optional[int] = None
    deterministic: bool = False
    realistic_mode: bool = True
    include_anomalies: bool = False
    anomaly_rate: float = 0.01
    
    def __post_init__(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)


class MarketDataGenerator:
    """Generate realistic market data for testing"""
    
    # Realistic price ranges for different assets
    PRICE_RANGES = {
        "BTC": (30000, 70000),
        "ETH": (2000, 5000),
        "BNB": (300, 700),
        "SOL": (50, 200),
        "ADA": (0.3, 1.5),
        "DOT": (5, 50),
        "MATIC": (0.5, 3),
        "LINK": (5, 50),
        "AVAX": (10, 150),
        "ATOM": (5, 45)
    }
    
    # Volatility parameters
    VOLATILITY_PARAMS = {
        VolatilityProfile.LOW: {"mean": 0.0001, "std": 0.0002},
        VolatilityProfile.NORMAL: {"mean": 0.0005, "std": 0.001},
        VolatilityProfile.HIGH: {"mean": 0.002, "std": 0.003},
        VolatilityProfile.EXTREME: {"mean": 0.005, "std": 0.01}
    }
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.price_memory = {}  # Store last prices for continuity
        
    def generate_symbol(self, base: Optional[str] = None, quote: Optional[str] = None) -> str:
        """Generate trading symbol"""
        if not base:
            base = random.choice(list(self.PRICE_RANGES.keys()))
        if not quote:
            quote = random.choice(["USDT", "BUSD", "USDC", "BTC", "ETH"])
        
        # Avoid same base and quote
        if base == quote:
            quote = "USDT"
            
        return f"{base}/{quote}"
    
    def generate_price(
        self,
        symbol: str,
        base_price: Optional[float] = None,
        volatility: VolatilityProfile = VolatilityProfile.NORMAL
    ) -> float:
        """Generate realistic price based on symbol and volatility"""
        # Extract base asset
        base = symbol.split("/")[0] if "/" in symbol else symbol
        
        # Get base price
        if base_price is None:
            if base in self.PRICE_RANGES:
                price_range = self.PRICE_RANGES[base]
                base_price = self.price_memory.get(
                    symbol,
                    random.uniform(price_range[0], price_range[1])
                )
            else:
                base_price = self.price_memory.get(symbol, 100.0)
        
        # Apply volatility
        vol_params = self.VOLATILITY_PARAMS[volatility]
        change = np.random.normal(vol_params["mean"], vol_params["std"])
        
        # Add momentum
        if symbol in self.price_memory:
            momentum = (self.price_memory[symbol] - base_price) / base_price
            change += momentum * 0.1  # Momentum factor
        
        # Calculate new price
        new_price = base_price * (1 + change)
        
        # Keep within realistic bounds
        if base in self.PRICE_RANGES:
            min_price, max_price = self.PRICE_RANGES[base]
            new_price = max(min_price * 0.5, min(max_price * 2, new_price))
        
        # Store for continuity
        self.price_memory[symbol] = new_price
        
        # Add anomaly if configured
        if self.config.include_anomalies and random.random() < self.config.anomaly_rate:
            anomaly_factor = random.choice([0.5, 0.8, 1.2, 2.0])
            new_price *= anomaly_factor
        
        return round(new_price, 8)
    
    def generate_ohlcv(
        self,
        symbol: str,
        periods: int,
        interval: timedelta,
        start_time: Optional[datetime] = None,
        regime: MarketRegime = MarketRegime.RANGING,
        volatility: VolatilityProfile = VolatilityProfile.NORMAL
    ) -> pd.DataFrame:
        """Generate OHLCV data with specified market regime"""
        if start_time is None:
            start_time = datetime.now() - (interval * periods)
        
        data = []
        current_price = self.generate_price(symbol, volatility=volatility)
        
        for i in range(periods):
            timestamp = start_time + (interval * i)
            
            # Generate candle based on regime
            candle = self._generate_candle(
                current_price,
                regime,
                volatility
            )
            
            # Update current price
            current_price = candle["close"]
            
            # Create OHLCV record
            volume = self._generate_volume(symbol, candle["close"], volatility)
            
            data.append({
                "timestamp": timestamp,
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
                "volume": volume
            })
        
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df
    
    def generate_orderbook(
        self,
        symbol: str,
        mid_price: Optional[float] = None,
        depth: int = 20,
        spread_bps: float = 10,
        liquidity_profile: str = "normal"
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Generate realistic order book"""
        if mid_price is None:
            mid_price = self.generate_price(symbol)
        
        spread = mid_price * spread_bps / 10000
        
        # Liquidity profiles
        liquidity_profiles = {
            "thin": {"base": 0.1, "growth": 1.05},
            "normal": {"base": 1.0, "growth": 1.1},
            "deep": {"base": 10.0, "growth": 1.2}
        }
        
        profile = liquidity_profiles.get(liquidity_profile, liquidity_profiles["normal"])
        
        bids = []
        asks = []
        
        # Generate bids
        bid_price = mid_price - spread / 2
        for i in range(depth):
            price = bid_price * (1 - 0.0001 * i)
            amount = profile["base"] * (profile["growth"] ** i) * random.uniform(0.8, 1.2)
            bids.append((round(price, 2), round(amount, 8)))
        
        # Generate asks
        ask_price = mid_price + spread / 2
        for i in range(depth):
            price = ask_price * (1 + 0.0001 * i)
            amount = profile["base"] * (profile["growth"] ** i) * random.uniform(0.8, 1.2)
            asks.append((round(price, 2), round(amount, 8)))
        
        return {
            "bids": bids,
            "asks": asks,
            "mid_price": mid_price,
            "spread": spread,
            "timestamp": datetime.now()
        }
    
    def generate_trades(
        self,
        symbol: str,
        count: int,
        time_window: timedelta,
        start_time: Optional[datetime] = None,
        size_distribution: str = "normal"
    ) -> List[Dict[str, Any]]:
        """Generate trade history"""
        if start_time is None:
            start_time = datetime.now() - time_window
        
        trades = []
        current_price = self.generate_price(symbol)
        
        # Size distributions
        size_generators = {
            "uniform": lambda: random.uniform(0.01, 1.0),
            "normal": lambda: abs(np.random.normal(0.1, 0.05)),
            "pareto": lambda: np.random.pareto(2) * 0.01,
            "whale": lambda: random.choice([0.01, 0.01, 0.01, 10.0])
        }
        
        size_gen = size_generators.get(size_distribution, size_generators["normal"])
        
        for i in range(count):
            # Random time within window
            time_offset = random.uniform(0, time_window.total_seconds())
            timestamp = start_time + timedelta(seconds=time_offset)
            
            # Price movement
            current_price *= (1 + np.random.normal(0, 0.0005))
            
            # Trade details
            is_buy = random.random() > 0.5
            size = size_gen()
            
            trade = {
                "id": self._generate_trade_id(),
                "symbol": symbol,
                "timestamp": timestamp,
                "price": round(current_price, 2),
                "amount": round(size, 8),
                "side": "buy" if is_buy else "sell",
                "is_maker": random.random() > 0.3
            }
            
            trades.append(trade)
        
        # Sort by timestamp
        trades.sort(key=lambda x: x["timestamp"])
        return trades
    
    def generate_ticker(
        self,
        symbol: str,
        base_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate 24hr ticker data"""
        current_price = self.generate_price(symbol, base_price)
        change_pct = random.uniform(-10, 10)
        
        open_price = current_price / (1 + change_pct / 100)
        high_price = max(current_price, open_price) * (1 + random.uniform(0, 0.05))
        low_price = min(current_price, open_price) * (1 - random.uniform(0, 0.05))
        
        volume = random.uniform(1000, 100000)
        quote_volume = volume * (open_price + current_price) / 2
        
        return {
            "symbol": symbol,
            "priceChange": current_price - open_price,
            "priceChangePercent": change_pct,
            "weightedAvgPrice": (open_price + current_price) / 2,
            "prevClosePrice": open_price,
            "lastPrice": current_price,
            "lastQty": random.uniform(0.01, 1.0),
            "bidPrice": current_price - 0.01,
            "bidQty": random.uniform(1, 10),
            "askPrice": current_price + 0.01,
            "askQty": random.uniform(1, 10),
            "openPrice": open_price,
            "highPrice": high_price,
            "lowPrice": low_price,
            "volume": volume,
            "quoteVolume": quote_volume,
            "openTime": int((datetime.now() - timedelta(hours=24)).timestamp() * 1000),
            "closeTime": int(datetime.now().timestamp() * 1000),
            "count": random.randint(10000, 100000)
        }
    
    def _generate_candle(
        self,
        base_price: float,
        regime: MarketRegime,
        volatility: VolatilityProfile
    ) -> Dict[str, float]:
        """Generate single candle based on regime"""
        vol_factor = {
            VolatilityProfile.LOW: 0.001,
            VolatilityProfile.NORMAL: 0.003,
            VolatilityProfile.HIGH: 0.01,
            VolatilityProfile.EXTREME: 0.03
        }[volatility]
        
        if regime == MarketRegime.TRENDING_UP:
            # Upward bias
            open_price = base_price
            close_price = base_price * (1 + random.uniform(0, vol_factor * 2))
            high_price = close_price * (1 + random.uniform(0, vol_factor * 0.5))
            low_price = open_price * (1 - random.uniform(0, vol_factor * 0.3))
            
        elif regime == MarketRegime.TRENDING_DOWN:
            # Downward bias
            open_price = base_price
            close_price = base_price * (1 - random.uniform(0, vol_factor * 2))
            high_price = open_price * (1 + random.uniform(0, vol_factor * 0.3))
            low_price = close_price * (1 - random.uniform(0, vol_factor * 0.5))
            
        elif regime == MarketRegime.VOLATILE:
            # High volatility, random direction
            open_price = base_price
            direction = random.choice([-1, 1])
            close_price = base_price * (1 + direction * random.uniform(vol_factor, vol_factor * 3))
            high_price = max(open_price, close_price) * (1 + random.uniform(vol_factor, vol_factor * 2))
            low_price = min(open_price, close_price) * (1 - random.uniform(vol_factor, vol_factor * 2))
            
        elif regime == MarketRegime.BREAKOUT:
            # Sudden move
            open_price = base_price
            breakout_direction = random.choice([-1, 1])
            close_price = base_price * (1 + breakout_direction * vol_factor * 5)
            if breakout_direction > 0:
                high_price = close_price * (1 + vol_factor)
                low_price = open_price * (1 - vol_factor * 0.2)
            else:
                high_price = open_price * (1 + vol_factor * 0.2)
                low_price = close_price * (1 - vol_factor)
                
        else:  # RANGING, ACCUMULATION, DISTRIBUTION
            # Mean reverting
            open_price = base_price * (1 + random.uniform(-vol_factor, vol_factor))
            close_price = base_price * (1 + random.uniform(-vol_factor, vol_factor))
            high_price = max(open_price, close_price) * (1 + random.uniform(0, vol_factor * 0.5))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, vol_factor * 0.5))
        
        return {
            "open": round(open_price, 8),
            "high": round(high_price, 8),
            "low": round(low_price, 8),
            "close": round(close_price, 8)
        }
    
    def _generate_volume(
        self,
        symbol: str,
        price: float,
        volatility: VolatilityProfile
    ) -> float:
        """Generate realistic volume"""
        # Base volume depends on asset
        base = symbol.split("/")[0]
        base_volumes = {
            "BTC": 10000,
            "ETH": 50000,
            "BNB": 20000,
            "SOL": 100000,
            "ADA": 500000,
            "DOT": 30000,
            "MATIC": 200000,
            "LINK": 40000,
            "AVAX": 25000,
            "ATOM": 35000
        }
        
        base_volume = base_volumes.get(base, 10000)
        
        # Volatility affects volume
        vol_multiplier = {
            VolatilityProfile.LOW: 0.5,
            VolatilityProfile.NORMAL: 1.0,
            VolatilityProfile.HIGH: 2.0,
            VolatilityProfile.EXTREME: 5.0
        }[volatility]
        
        volume = base_volume * vol_multiplier * random.uniform(0.5, 1.5)
        return round(volume, 2)
    
    def _generate_trade_id(self) -> str:
        """Generate unique trade ID"""
        return f"T{int(datetime.now().timestamp() * 1000000)}_{random.randint(1000, 9999)}"


class OrderDataGenerator:
    """Generate order-related test data"""
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.order_counter = 0
        
    def generate_order(
        self,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
        order_type: Optional[str] = None,
        price: Optional[float] = None,
        amount: Optional[float] = None,
        status: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate single order"""
        market_gen = MarketDataGenerator(self.config)
        
        if not symbol:
            symbol = market_gen.generate_symbol()
        
        if not side:
            side = random.choice(["buy", "sell"])
            
        if not order_type:
            order_type = random.choice(["limit", "market", "stop_loss", "take_profit"])
        
        if not price and order_type != "market":
            price = market_gen.generate_price(symbol)
        
        if not amount:
            # Realistic amounts based on price
            if price and price > 10000:
                amount = random.uniform(0.001, 0.1)
            elif price and price > 100:
                amount = random.uniform(0.1, 10)
            else:
                amount = random.uniform(10, 1000)
        
        if not status:
            status = random.choice(["new", "partially_filled", "filled", "cancelled"])
        
        self.order_counter += 1
        order_id = f"ORD{datetime.now().strftime('%Y%m%d')}_{self.order_counter:06d}"
        
        order = {
            "id": order_id,
            "client_order_id": f"CLIENT_{uuid.uuid4().hex[:8]}",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "status": status,
            "amount": round(amount, 8),
            "filled": round(amount * random.uniform(0, 1) if status == "partially_filled" else 
                          (amount if status == "filled" else 0), 8),
            "remaining": round(amount if status == "new" else 
                             (amount * random.uniform(0.1, 0.9) if status == "partially_filled" else 0), 8),
            "timestamp": datetime.now(),
            "updated_at": datetime.now()
        }
        
        if order_type != "market":
            order["price"] = round(price, 2)
        
        if order_type in ["stop_loss", "take_profit"]:
            if side == "buy":
                order["stop_price"] = round(price * (1.02 if order_type == "stop_loss" else 0.98), 2)
            else:
                order["stop_price"] = round(price * (0.98 if order_type == "stop_loss" else 1.02), 2)
        
        # Add execution details for filled orders
        if status in ["filled", "partially_filled"]:
            order["average_price"] = round(price * random.uniform(0.999, 1.001) if price else 0, 2)
            order["commission"] = round(order["filled"] * order.get("average_price", 0) * 0.001, 4)
            order["commission_asset"] = symbol.split("/")[1] if "/" in symbol else "USDT"
        
        return order
    
    def generate_order_batch(
        self,
        count: int,
        symbol: Optional[str] = None,
        order_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate batch of orders"""
        orders = []
        for _ in range(count):
            orders.append(self.generate_order(symbol=symbol, order_type=order_type))
        return orders
    
    def generate_order_book_orders(
        self,
        symbol: str,
        mid_price: float,
        levels: int = 10,
        amount_range: Tuple[float, float] = (0.1, 5.0)
    ) -> Tuple[List[Dict], List[Dict]]:
        """Generate orders for order book"""
        buy_orders = []
        sell_orders = []
        
        # Generate buy orders
        for i in range(levels):
            price = mid_price * (1 - 0.001 * (i + 1))
            amount = random.uniform(*amount_range) * (1 + i * 0.1)
            
            order = self.generate_order(
                symbol=symbol,
                side="buy",
                order_type="limit",
                price=price,
                amount=amount,
                status="new"
            )
            buy_orders.append(order)
        
        # Generate sell orders
        for i in range(levels):
            price = mid_price * (1 + 0.001 * (i + 1))
            amount = random.uniform(*amount_range) * (1 + i * 0.1)
            
            order = self.generate_order(
                symbol=symbol,
                side="sell",
                order_type="limit",
                price=price,
                amount=amount,
                status="new"
            )
            sell_orders.append(order)
        
        return buy_orders, sell_orders


class PositionDataGenerator:
    """Generate position-related test data"""
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.position_counter = 0
        
    def generate_position(
        self,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
        size: Optional[float] = None,
        entry_price: Optional[float] = None,
        current_price: Optional[float] = None,
        leverage: float = 1.0
    ) -> Dict[str, Any]:
        """Generate single position"""
        market_gen = MarketDataGenerator(self.config)
        
        if not symbol:
            symbol = market_gen.generate_symbol()
        
        if not side:
            side = random.choice(["long", "short"])
        
        if not entry_price:
            entry_price = market_gen.generate_price(symbol)
        
        if not current_price:
            # Generate current price with some P&L
            pnl_factor = random.uniform(-0.05, 0.05)  # ±5% P&L
            if side == "long":
                current_price = entry_price * (1 + pnl_factor)
            else:
                current_price = entry_price * (1 - pnl_factor)
        
        if not size:
            # Size based on typical position sizes
            position_value = random.uniform(100, 10000)  # USD value
            size = position_value / entry_price
        
        self.position_counter += 1
        position_id = f"POS{datetime.now().strftime('%Y%m%d')}_{self.position_counter:06d}"
        
        # Calculate P&L
        if side == "long":
            unrealized_pnl = (current_price - entry_price) * size
            pnl_percentage = ((current_price - entry_price) / entry_price) * 100
        else:
            unrealized_pnl = (entry_price - current_price) * size
            pnl_percentage = ((entry_price - current_price) / entry_price) * 100
        
        # Calculate margin and liquidation
        position_value = size * entry_price
        margin_used = position_value / leverage
        
        if side == "long":
            liquidation_price = entry_price * (1 - 0.8 / leverage)
        else:
            liquidation_price = entry_price * (1 + 0.8 / leverage)
        
        position = {
            "id": position_id,
            "symbol": symbol,
            "side": side,
            "size": round(size, 8),
            "entry_price": round(entry_price, 2),
            "current_price": round(current_price, 2),
            "mark_price": round(current_price * random.uniform(0.999, 1.001), 2),
            "liquidation_price": round(liquidation_price, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "realized_pnl": round(random.uniform(-100, 100), 2),
            "pnl_percentage": round(pnl_percentage, 2),
            "margin_used": round(margin_used, 2),
            "leverage": leverage,
            "position_value": round(position_value, 2),
            "opened_at": datetime.now() - timedelta(hours=random.randint(1, 168)),
            "updated_at": datetime.now()
        }
        
        # Add stop loss and take profit
        if random.random() > 0.5:
            if side == "long":
                position["stop_loss"] = round(entry_price * random.uniform(0.95, 0.98), 2)
                position["take_profit"] = round(entry_price * random.uniform(1.02, 1.10), 2)
            else:
                position["stop_loss"] = round(entry_price * random.uniform(1.02, 1.05), 2)
                position["take_profit"] = round(entry_price * random.uniform(0.90, 0.98), 2)
        
        return position
    
    def generate_position_batch(
        self,
        count: int,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate batch of positions"""
        positions = []
        for _ in range(count):
            positions.append(self.generate_position(symbol=symbol))
        return positions
    
    def generate_position_history(
        self,
        position: Dict[str, Any],
        periods: int = 100,
        interval: timedelta = timedelta(minutes=5)
    ) -> pd.DataFrame:
        """Generate historical data for a position"""
        data = []
        current_price = position["entry_price"]
        current_size = position["size"]
        
        for i in range(periods):
            timestamp = position["opened_at"] + (interval * i)
            
            # Price movement
            current_price *= (1 + np.random.normal(0, 0.001))
            
            # Calculate metrics
            if position["side"] == "long":
                unrealized_pnl = (current_price - position["entry_price"]) * current_size
            else:
                unrealized_pnl = (position["entry_price"] - current_price) * current_size
            
            data.append({
                "timestamp": timestamp,
                "price": current_price,
                "size": current_size,
                "unrealized_pnl": unrealized_pnl,
                "margin_ratio": random.uniform(0.1, 0.5)
            })
        
        return pd.DataFrame(data).set_index("timestamp")


class GridDataGenerator:
    """Generate grid trading specific test data"""
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.grid_counter = 0
        
    def generate_grid_levels(
        self,
        symbol: str,
        grid_type: str = "symmetric",
        num_levels: int = 10,
        price_range: Optional[Tuple[float, float]] = None,
        current_price: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Generate grid levels"""
        market_gen = MarketDataGenerator(self.config)
        
        if not current_price:
            current_price = market_gen.generate_price(symbol)
        
        if not price_range:
            # Default range ±5% from current price
            price_range = (
                current_price * 0.95,
                current_price * 1.05
            )
        
        levels = []
        
        if grid_type == "symmetric":
            # Equal spacing
            prices = np.linspace(price_range[0], price_range[1], num_levels)
            
        elif grid_type == "asymmetric":
            # More levels near current price
            lower_levels = np.linspace(price_range[0], current_price, num_levels // 2)
            upper_levels = np.linspace(current_price, price_range[1], num_levels // 2)
            prices = np.concatenate([lower_levels[:-1], upper_levels])
            
        elif grid_type == "geometric":
            # Geometric progression
            log_prices = np.linspace(
                np.log(price_range[0]),
                np.log(price_range[1]),
                num_levels
            )
            prices = np.exp(log_prices)
            
        elif grid_type == "dynamic":
            # Based on volatility zones
            volatility = abs(current_price - price_range[0]) / current_price
            prices = []
            for i in range(num_levels):
                if i < num_levels // 3:
                    # Dense near current price
                    offset = (i - num_levels // 6) * volatility * 0.001
                elif i < 2 * num_levels // 3:
                    # Medium density
                    offset = (i - num_levels // 2) * volatility * 0.002
                else:
                    # Sparse at extremes
                    offset = (i - 5 * num_levels // 6) * volatility * 0.004
                
                prices.append(current_price * (1 + offset))
            
            prices = sorted(prices)
        else:
            raise ValueError(f"Unknown grid type: {grid_type}")
        
        # Create level details
        for i, price in enumerate(prices):
            side = "buy" if price < current_price else "sell"
            
            level = {
                "level": i,
                "price": round(price, 2),
                "side": side,
                "amount": round(random.uniform(0.01, 0.1), 8),
                "status": "pending",
                "distance_from_current": abs(price - current_price),
                "distance_percentage": abs(price - current_price) / current_price * 100
            }
            
            levels.append(level)
        
        return levels
    
    def generate_grid_config(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate grid configuration"""
        market_gen = MarketDataGenerator(self.config)
        
        if not symbol:
            symbol = market_gen.generate_symbol()
        
        if not strategy:
            strategy = random.choice([
                "conservative",
                "balanced",
                "aggressive",
                "scalping",
                "swing"
            ])
        
        self.grid_counter += 1
        grid_id = f"GRID{datetime.now().strftime('%Y%m%d')}_{self.grid_counter:06d}"
        
        # Strategy parameters
        strategy_params = {
            "conservative": {
                "num_levels": 5,
                "grid_spacing": 0.01,  # 1%
                "take_profit": 0.005,  # 0.5%
                "stop_loss": 0.02,     # 2%
                "amount_per_level": 0.01
            },
            "balanced": {
                "num_levels": 10,
                "grid_spacing": 0.005,
                "take_profit": 0.003,
                "stop_loss": 0.015,
                "amount_per_level": 0.05
            },
            "aggressive": {
                "num_levels": 20,
                "grid_spacing": 0.003,
                "take_profit": 0.002,
                "stop_loss": 0.01,
                "amount_per_level": 0.1
            },
            "scalping": {
                "num_levels": 30,
                "grid_spacing": 0.001,
                "take_profit": 0.001,
                "stop_loss": 0.005,
                "amount_per_level": 0.2
            },
            "swing": {
                "num_levels": 7,
                "grid_spacing": 0.02,
                "take_profit": 0.01,
                "stop_loss": 0.03,
                "amount_per_level": 0.03
            }
        }
        
        params = strategy_params[strategy]
        current_price = market_gen.generate_price(symbol)
        
        config = {
            "id": grid_id,
            "symbol": symbol,
            "strategy": strategy,
            "grid_type": random.choice(["symmetric", "asymmetric", "geometric"]),
            "num_levels": params["num_levels"],
            "upper_price": round(current_price * (1 + params["grid_spacing"] * params["num_levels"] / 2), 2),
            "lower_price": round(current_price * (1 - params["grid_spacing"] * params["num_levels"] / 2), 2),
            "grid_spacing": params["grid_spacing"],
            "amount_per_level": params["amount_per_level"],
            "take_profit": params["take_profit"],
            "stop_loss": params["stop_loss"],
            "total_investment": round(params["amount_per_level"] * params["num_levels"] * current_price, 2),
            "created_at": datetime.now(),
            "status": "active"
        }
        
        return config
    
    def generate_grid_performance(
        self,
        grid_config: Dict[str, Any],
        duration_hours: int = 24
    ) -> Dict[str, Any]:
        """Generate grid performance metrics"""
        # Simulate grid trading results
        num_trades = random.randint(10, 100)
        win_rate = random.uniform(0.4, 0.8)
        
        winning_trades = int(num_trades * win_rate)
        losing_trades = num_trades - winning_trades
        
        avg_profit_per_trade = grid_config["take_profit"] * grid_config["amount_per_level"] * \
                              (grid_config["upper_price"] + grid_config["lower_price"]) / 2
        
        avg_loss_per_trade = grid_config["stop_loss"] * grid_config["amount_per_level"] * \
                            (grid_config["upper_price"] + grid_config["lower_price"]) / 2 * 0.3
        
        total_profit = winning_trades * avg_profit_per_trade
        total_loss = losing_trades * avg_loss_per_trade
        net_profit = total_profit - total_loss
        
        performance = {
            "grid_id": grid_config["id"],
            "duration_hours": duration_hours,
            "total_trades": num_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 2),
            "total_profit": round(total_profit, 2),
            "total_loss": round(total_loss, 2),
            "net_profit": round(net_profit, 2),
            "roi": round(net_profit / grid_config["total_investment"] * 100, 2),
            "profit_factor": round(total_profit / total_loss if total_loss > 0 else float('inf'), 2),
            "avg_profit_per_trade": round(avg_profit_per_trade, 2),
            "avg_loss_per_trade": round(avg_loss_per_trade, 2),
            "max_drawdown": round(random.uniform(0.02, 0.10) * grid_config["total_investment"], 2),
            "sharpe_ratio": round(random.uniform(0.5, 2.5), 2),
            "grid_utilization": round(random.uniform(0.3, 0.9), 2),
            "timestamp": datetime.now()
        }
        
        return performance


class PerformanceDataGenerator:
    """Generate performance metrics test data"""
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        
    def generate_metrics_snapshot(
        self,
        portfolio_value: float = 10000,
        time_period: str = "24h"
    ) -> Dict[str, Any]:
        """Generate performance metrics snapshot"""
        # Base metrics
        metrics = {
            "timestamp": datetime.now(),
            "period": time_period,
            "portfolio_value": portfolio_value,
            "total_balance": portfolio_value * random.uniform(0.9, 1.1),
            "free_balance": portfolio_value * random.uniform(0.3, 0.7),
            "locked_balance": portfolio_value * random.uniform(0.2, 0.5),
            
            # Returns
            "total_return": random.uniform(-0.1, 0.3),
            "daily_return": random.uniform(-0.05, 0.05),
            "weekly_return": random.uniform(-0.1, 0.1),
            "monthly_return": random.uniform(-0.2, 0.2),
            
            # Risk metrics
            "sharpe_ratio": random.uniform(-1, 3),
            "sortino_ratio": random.uniform(-1, 3.5),
            "calmar_ratio": random.uniform(0, 2),
            "max_drawdown": random.uniform(0.02, 0.3),
            "current_drawdown": random.uniform(0, 0.1),
            "volatility": random.uniform(0.1, 0.5),
            "beta": random.uniform(0.5, 1.5),
            
            # Trading metrics
            "total_trades": random.randint(10, 1000),
            "winning_trades": random.randint(5, 600),
            "losing_trades": random.randint(5, 400),
            "win_rate": random.uniform(0.3, 0.7),
            "profit_factor": random.uniform(0.8, 2.5),
            "average_win": random.uniform(10, 100),
            "average_loss": random.uniform(5, 80),
            "largest_win": random.uniform(100, 1000),
            "largest_loss": random.uniform(-1000, -50),
            
            # Execution metrics
            "avg_slippage": random.uniform(0.0001, 0.001),
            "total_commission": random.uniform(10, 500),
            "avg_execution_time": random.uniform(0.01, 0.5),
            
            # Grid-specific metrics
            "active_grids": random.randint(0, 10),
            "grid_profit": random.uniform(-100, 500),
            "grid_efficiency": random.uniform(0.3, 0.9)
        }
        
        # Calculate some dependent metrics
        metrics["risk_adjusted_return"] = metrics["total_return"] / metrics["volatility"] \
                                         if metrics["volatility"] > 0 else 0
        metrics["expectancy"] = (metrics["win_rate"] * metrics["average_win"]) - \
                               ((1 - metrics["win_rate"]) * abs(metrics["average_loss"]))
        
        return metrics
    
    def generate_metrics_history(
        self,
        periods: int = 100,
        interval: timedelta = timedelta(hours=1),
        start_value: float = 10000
    ) -> pd.DataFrame:
        """Generate historical performance metrics"""
        data = []
        current_value = start_value
        
        for i in range(periods):
            timestamp = datetime.now() - (interval * (periods - i))
            
            # Random walk for portfolio value
            returns = np.random.normal(0.0001, 0.01)
            current_value *= (1 + returns)
            
            # Generate correlated metrics
            snapshot = self.generate_metrics_snapshot(current_value)
            snapshot["timestamp"] = timestamp
            
            data.append(snapshot)
        
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        
        # Calculate rolling metrics
        df["rolling_sharpe"] = df["daily_return"].rolling(window=30).mean() / \
                               df["daily_return"].rolling(window=30).std() * np.sqrt(252)
        
        df["rolling_max"] = df["portfolio_value"].rolling(window=30, min_periods=1).max()
        df["rolling_drawdown"] = (df["portfolio_value"] - df["rolling_max"]) / df["rolling_max"]
        
        return df
    
    def generate_benchmark_comparison(
        self,
        strategy_metrics: Dict[str, Any],
        benchmark_name: str = "BTC Buy & Hold"
    ) -> Dict[str, Any]:
        """Generate benchmark comparison data"""
        # Generate benchmark performance
        benchmark_return = random.uniform(-0.1, 0.4)
        benchmark_volatility = random.uniform(0.2, 0.6)
        
        comparison = {
            "benchmark_name": benchmark_name,
            "period": strategy_metrics.get("period", "24h"),
            
            # Strategy metrics
            "strategy_return": strategy_metrics["total_return"],
            "strategy_volatility": strategy_metrics["volatility"],
            "strategy_sharpe": strategy_metrics["sharpe_ratio"],
            "strategy_drawdown": strategy_metrics["max_drawdown"],
            
            # Benchmark metrics
            "benchmark_return": benchmark_return,
            "benchmark_volatility": benchmark_volatility,
            "benchmark_sharpe": benchmark_return / benchmark_volatility * np.sqrt(252),
            "benchmark_drawdown": random.uniform(0.1, 0.4),
            
            # Relative metrics
            "alpha": strategy_metrics["total_return"] - benchmark_return,
            "tracking_error": random.uniform(0.05, 0.2),
            "information_ratio": (strategy_metrics["total_return"] - benchmark_return) / \
                               random.uniform(0.05, 0.2),
            "correlation": random.uniform(0.3, 0.9),
            
            # Win/Loss
            "outperformance": strategy_metrics["total_return"] > benchmark_return,
            "outperformance_margin": strategy_metrics["total_return"] - benchmark_return
        }
        
        return comparison


class EventDataGenerator:
    """Generate event and alert test data"""
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.event_counter = 0
        
    def generate_event(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        component: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate single event"""
        event_types = [
            "trade_executed",
            "order_placed",
            "order_cancelled",
            "position_opened",
            "position_closed",
            "grid_created",
            "grid_completed",
            "stop_loss_triggered",
            "take_profit_triggered",
            "risk_alert",
            "system_error",
            "api_error",
            "connection_lost",
            "phase_changed",
            "regime_detected"
        ]
        
        severities = ["low", "medium", "high", "critical"]
        
        components = [
            "TradingEngine",
            "RiskManager",
            "GridStrategy",
            "AttentionLayer",
            "MarketRegimeDetector",
            "ExecutionEngine",
            "PerformanceMonitor",
            "DataFeed",
            "WebSocket",
            "Database"
        ]
        
        if not event_type:
            event_type = random.choice(event_types)
        
        if not severity:
            # Severity based on event type
            if "error" in event_type or "lost" in event_type:
                severity = random.choice(["high", "critical"])
            elif "alert" in event_type or "triggered" in event_type:
                severity = random.choice(["medium", "high"])
            else:
                severity = random.choice(["low", "medium"])
        
        if not component:
            component = random.choice(components)
        
        self.event_counter += 1
        event_id = f"EVT{datetime.now().strftime('%Y%m%d')}_{self.event_counter:06d}"
        
        # Generate event data based on type
        event_data = self._generate_event_data(event_type)
        
        event = {
            "id": event_id,
            "type": event_type,
            "severity": severity,
            "component": component,
            "timestamp": datetime.now(),
            "data": event_data,
            "message": self._generate_event_message(event_type, event_data),
            "handled": random.choice([True, False]),
            "handler_response": "processed" if random.random() > 0.2 else "failed"
        }
        
        return event
    
    def _generate_event_data(self, event_type: str) -> Dict[str, Any]:
        """Generate event-specific data"""
        market_gen = MarketDataGenerator(self.config)
        
        if event_type == "trade_executed":
            return {
                "symbol": market_gen.generate_symbol(),
                "side": random.choice(["buy", "sell"]),
                "price": round(random.uniform(100, 50000), 2),
                "amount": round(random.uniform(0.01, 1), 8),
                "order_id": f"ORD_{uuid.uuid4().hex[:8]}"
            }
        
        elif event_type == "risk_alert":
            return {
                "metric": random.choice(["drawdown", "exposure", "volatility", "correlation"]),
                "current_value": round(random.uniform(0.1, 0.5), 3),
                "threshold": round(random.uniform(0.05, 0.3), 3),
                "action": random.choice(["reduce_position", "stop_trading", "alert_only"])
            }
        
        elif event_type == "phase_changed":
            return {
                "old_phase": random.choice(["learning", "shadow", "live"]),
                "new_phase": random.choice(["learning", "shadow", "live"]),
                "reason": random.choice(["performance", "time_based", "manual", "risk"])
            }
        
        elif event_type == "regime_detected":
            return {
                "regime": random.choice(["trending_up", "trending_down", "ranging", "volatile"]),
                "confidence": round(random.uniform(0.6, 0.95), 2),
                "indicators": {
                    "trend_strength": round(random.uniform(-1, 1), 3),
                    "volatility": round(random.uniform(0.1, 0.5), 3),
                    "volume_ratio": round(random.uniform(0.5, 2), 2)
                }
            }
        
        else:
            return {"details": f"Event data for {event_type}"}
    
    def _generate_event_message(self, event_type: str, data: Dict[str, Any]) -> str:
        """Generate human-readable event message"""
        messages = {
            "trade_executed": f"Trade executed: {data.get('side', 'buy')} {data.get('amount', 0)} @ {data.get('price', 0)}",
            "risk_alert": f"Risk alert: {data.get('metric', 'unknown')} exceeded threshold",
            "phase_changed": f"Phase changed from {data.get('old_phase', 'unknown')} to {data.get('new_phase', 'unknown')}",
            "regime_detected": f"Market regime detected: {data.get('regime', 'unknown')} (confidence: {data.get('confidence', 0):.0%})"
        }
        
        return messages.get(event_type, f"Event occurred: {event_type}")


# Utility functions

def create_test_dataset(
    symbols: List[str] = None,
    periods: int = 1000,
    interval: str = "5m",
    include_features: bool = True
) -> Dict[str, pd.DataFrame]:
    """Create comprehensive test dataset"""
    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    
    market_gen = MarketDataGenerator()
    datasets = {}
    
    interval_map = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1)
    }
    
    for symbol in symbols:
        # Generate OHLCV data
        df = market_gen.generate_ohlcv(
            symbol=symbol,
            periods=periods,
            interval=interval_map[interval],
            regime=random.choice(list(MarketRegime)),
            volatility=random.choice(list(VolatilityProfile))
        )
        
        if include_features:
            # Add technical indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['rsi'] = calculate_rsi(df['close'])
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volatility'] = df['close'].pct_change().rolling(window=20).std()
            
            # Add market microstructure features
            df['spread'] = np.random.uniform(0.0001, 0.0005, len(df))
            df['order_flow_imbalance'] = np.random.normal(0, 0.1, len(df))
            
        datasets[symbol] = df
    
    return datasets


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def generate_random_portfolio(
    num_positions: int = 5,
    total_value: float = 10000
) -> List[Dict[str, Any]]:
    """Generate random portfolio"""
    position_gen = PositionDataGenerator()
    positions = []
    
    # Allocate capital
    allocations = np.random.dirichlet(np.ones(num_positions)) * total_value
    
    for i, allocation in enumerate(allocations):
        position = position_gen.generate_position()
        
        # Adjust size based on allocation
        position["position_value"] = round(allocation, 2)
        position["size"] = round(allocation / position["entry_price"], 8)
        position["margin_used"] = round(allocation / position["leverage"], 2)
        
        positions.append(position)
    
    return positions


# Example usage
if __name__ == "__main__":
    # Initialize generators
    config = GeneratorConfig(seed=42, deterministic=True)
    
    market_gen = MarketDataGenerator(config)
    order_gen = OrderDataGenerator(config)
    position_gen = PositionDataGenerator(config)
    grid_gen = GridDataGenerator(config)
    perf_gen = PerformanceDataGenerator(config)
    event_gen = EventDataGenerator(config)
    
    # Generate sample data
    print("=== Market Data ===")
    ticker = market_gen.generate_ticker("BTC/USDT")
    print(f"Ticker: {ticker['symbol']} @ ${ticker['lastPrice']}")
    
    print("\n=== Orders ===")
    orders = order_gen.generate_order_batch(3)
    for order in orders:
        print(f"Order {order['id']}: {order['side']} {order['amount']} {order['symbol']}")
    
    print("\n=== Positions ===")
    position = position_gen.generate_position()
    print(f"Position: {position['side']} {position['size']} {position['symbol']} "
          f"P&L: ${position['unrealized_pnl']:.2f} ({position['pnl_percentage']:.2f}%)")
    
    print("\n=== Grid Configuration ===")
    grid_config = grid_gen.generate_grid_config()
    print(f"Grid {grid_config['id']}: {grid_config['strategy']} strategy with "
          f"{grid_config['num_levels']} levels")
    
    print("\n=== Performance Metrics ===")
    metrics = perf_gen.generate_metrics_snapshot()
    print(f"Portfolio Value: ${metrics['portfolio_value']:.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    
    print("\n=== Events ===")
    for _ in range(3):
        event = event_gen.generate_event()
        print(f"[{event['severity'].upper()}] {event['message']}")