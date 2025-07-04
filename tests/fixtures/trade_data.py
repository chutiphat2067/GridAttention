#!/usr/bin/env python3
"""
Trade Data Test Fixtures for GridAttention Trading System
Provides comprehensive trade-related data for testing execution, performance, and analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import random
import uuid
import json
from collections import defaultdict, deque


class OrderStatus(Enum):
    """Order status types"""
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


class PositionSide(Enum):
    """Position sides"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class TradeResult(Enum):
    """Trade result types"""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


@dataclass
class Order:
    """Order data structure"""
    order_id: str
    symbol: str
    side: str
    order_type: str
    price: float
    amount: float
    status: str
    timestamp: datetime
    
    # Optional fields
    stop_price: Optional[float] = None
    filled_amount: float = 0.0
    average_fill_price: Optional[float] = None
    commission: float = 0.0
    commission_asset: str = "USDT"
    time_in_force: str = "GTC"
    post_only: bool = False
    reduce_only: bool = False
    
    # Grid-specific
    grid_id: Optional[str] = None
    grid_level: Optional[int] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED.value
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.PENDING.value, OrderStatus.PARTIAL.value]
    
    @property
    def fill_rate(self) -> float:
        return self.filled_amount / self.amount if self.amount > 0 else 0.0


@dataclass
class Trade:
    """Executed trade data structure"""
    trade_id: str
    order_id: str
    symbol: str
    side: str
    price: float
    amount: float
    timestamp: datetime
    
    # Costs
    commission: float = 0.0
    commission_asset: str = "USDT"
    realized_pnl: float = 0.0
    
    # Position tracking
    position_side: Optional[str] = None
    position_amount_before: float = 0.0
    position_amount_after: float = 0.0
    
    # Performance
    slippage: float = 0.0
    execution_time_ms: int = 0
    
    # Grid-specific
    grid_id: Optional[str] = None
    grid_level: Optional[int] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    side: str
    amount: float
    entry_price: float
    current_price: float
    timestamp: datetime
    
    # P&L tracking
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    liquidation_price: Optional[float] = None
    
    # Position details
    leverage: float = 1.0
    margin_used: float = 0.0
    position_value: float = 0.0
    
    # History
    trades: List[Trade] = field(default_factory=list)
    peak_pnl: float = 0.0
    trough_pnl: float = 0.0
    
    # Grid-specific
    grid_id: Optional[str] = None
    grid_positions: Dict[int, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['trades'] = [t.to_dict() for t in self.trades]
        return data
    
    @property
    def pnl_percentage(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100
    
    @property
    def is_profitable(self) -> bool:
        return self.unrealized_pnl > 0


@dataclass
class GridTrade:
    """Grid-specific trade data"""
    grid_id: str
    symbol: str
    grid_type: str  # symmetric, asymmetric, dynamic
    
    # Grid parameters
    upper_price: float
    lower_price: float
    grid_levels: int
    amount_per_level: float
    
    # Grid state
    active_orders: Dict[int, Order] = field(default_factory=dict)
    filled_orders: Dict[int, List[Order]] = field(default_factory=dict)
    grid_positions: Dict[int, float] = field(default_factory=dict)
    
    # Performance
    total_trades: int = 0
    profitable_trades: int = 0
    total_pnl: float = 0.0
    grid_profit: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    
    def get_grid_prices(self) -> List[float]:
        """Get all grid level prices"""
        if self.grid_type == "symmetric":
            return list(np.linspace(self.lower_price, self.upper_price, self.grid_levels))
        elif self.grid_type == "exponential":
            log_prices = np.linspace(np.log(self.lower_price), np.log(self.upper_price), self.grid_levels)
            return list(np.exp(log_prices))
        else:
            # Default to symmetric
            return list(np.linspace(self.lower_price, self.upper_price, self.grid_levels))


class TradeDataGenerator:
    """Generate realistic trade data for testing"""
    
    def __init__(self, seed: Optional[int] = None):
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        
        self.order_counter = 0
        self.trade_counter = 0
        
        # Realistic commission rates
        self.commission_rates = {
            "maker": 0.001,  # 0.1%
            "taker": 0.0015  # 0.15%
        }
        
        # Slippage parameters
        self.slippage_params = {
            "market": {"mean": 0.001, "std": 0.0005},
            "limit": {"mean": 0.0, "std": 0.0001},
            "stop": {"mean": 0.002, "std": 0.001}
        }
    
    def generate_order(
        self,
        symbol: str,
        side: str,
        order_type: str = "limit",
        price: Optional[float] = None,
        amount: Optional[float] = None,
        status: str = "pending",
        grid_id: Optional[str] = None,
        grid_level: Optional[int] = None
    ) -> Order:
        """Generate a single order"""
        
        self.order_counter += 1
        order_id = f"ORD_{self.order_counter:06d}_{uuid.uuid4().hex[:8]}"
        
        # Generate price if not provided
        if price is None:
            base_price = 45000 if "BTC" in symbol else 3000
            price = base_price * (1 + np.random.normal(0, 0.01))
        
        # Generate amount if not provided
        if amount is None:
            amount = np.random.uniform(0.01, 1.0)
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=round(price, 2),
            amount=round(amount, 8),
            status=status,
            timestamp=datetime.now(),
            grid_id=grid_id,
            grid_level=grid_level
        )
        
        # Add stop price for stop orders
        if "stop" in order_type:
            if side == "buy":
                order.stop_price = price * 1.01
            else:
                order.stop_price = price * 0.99
        
        return order
    
    def generate_trade_from_order(
        self,
        order: Order,
        fill_percentage: float = 1.0,
        slippage_override: Optional[float] = None
    ) -> Trade:
        """Generate trade from order"""
        
        self.trade_counter += 1
        trade_id = f"TRD_{self.trade_counter:06d}_{uuid.uuid4().hex[:8]}"
        
        # Calculate fill amount
        fill_amount = order.amount * fill_percentage
        
        # Calculate slippage
        if slippage_override is not None:
            slippage = slippage_override
        else:
            slippage_params = self.slippage_params.get(order.order_type, self.slippage_params["limit"])
            slippage = np.random.normal(slippage_params["mean"], slippage_params["std"])
        
        # Calculate execution price
        if order.side == "buy":
            execution_price = order.price * (1 + abs(slippage))
        else:
            execution_price = order.price * (1 - abs(slippage))
        
        # Calculate commission
        is_maker = order.order_type == "limit" and order.post_only
        commission_rate = self.commission_rates["maker"] if is_maker else self.commission_rates["taker"]
        commission = fill_amount * execution_price * commission_rate
        
        # Create trade
        trade = Trade(
            trade_id=trade_id,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            price=round(execution_price, 2),
            amount=round(fill_amount, 8),
            timestamp=datetime.now(),
            commission=round(commission, 4),
            commission_asset="USDT",
            slippage=slippage,
            execution_time_ms=np.random.randint(10, 100),
            grid_id=order.grid_id,
            grid_level=order.grid_level
        )
        
        # Update order
        order.filled_amount += fill_amount
        order.average_fill_price = execution_price
        order.commission += commission
        
        if order.filled_amount >= order.amount * 0.99:
            order.status = OrderStatus.FILLED.value
        else:
            order.status = OrderStatus.PARTIAL.value
        
        return trade
    
    def generate_position(
        self,
        symbol: str,
        side: str,
        trades: List[Trade],
        current_price: Optional[float] = None
    ) -> Position:
        """Generate position from trades"""
        
        if not trades:
            raise ValueError("Need at least one trade to create position")
        
        # Calculate position metrics
        total_amount = sum(t.amount for t in trades)
        total_value = sum(t.amount * t.price for t in trades)
        entry_price = total_value / total_amount if total_amount > 0 else 0
        
        # Current price
        if current_price is None:
            # Simulate some price movement
            price_change = np.random.normal(0, 0.02)
            current_price = entry_price * (1 + price_change)
        
        # Calculate P&L
        if side == PositionSide.LONG.value:
            unrealized_pnl = (current_price - entry_price) * total_amount
        else:
            unrealized_pnl = (entry_price - current_price) * total_amount
        
        # Create position
        position = Position(
            symbol=symbol,
            side=side,
            amount=total_amount,
            entry_price=round(entry_price, 2),
            current_price=round(current_price, 2),
            timestamp=trades[0].timestamp,
            unrealized_pnl=round(unrealized_pnl, 2),
            position_value=round(total_amount * current_price, 2),
            trades=trades
        )
        
        # Set risk levels
        if side == PositionSide.LONG.value:
            position.stop_loss = entry_price * 0.98
            position.take_profit = entry_price * 1.05
            position.liquidation_price = entry_price * 0.75
        else:
            position.stop_loss = entry_price * 1.02
            position.take_profit = entry_price * 0.95
            position.liquidation_price = entry_price * 1.25
        
        return position
    
    def generate_trade_sequence(
        self,
        symbol: str,
        num_trades: int,
        win_rate: float = 0.6,
        avg_win: float = 0.02,
        avg_loss: float = 0.01,
        start_time: Optional[datetime] = None
    ) -> List[Trade]:
        """Generate sequence of trades with specified statistics"""
        
        trades = []
        current_time = start_time or datetime.now()
        position = 0
        entry_price = 0
        
        for i in range(num_trades):
            # Determine if winner
            is_winner = random.random() < win_rate
            
            # Alternate between entry and exit
            if position == 0:
                # Entry trade
                side = random.choice(["buy", "sell"])
                order = self.generate_order(symbol, side)
                trade = self.generate_trade_from_order(order)
                entry_price = trade.price
                position = 1 if side == "buy" else -1
                
            else:
                # Exit trade
                exit_side = "sell" if position > 0 else "buy"
                
                # Calculate exit price based on win/loss
                if is_winner:
                    pnl_pct = avg_win * (1 + random.uniform(-0.3, 0.3))
                else:
                    pnl_pct = -avg_loss * (1 + random.uniform(-0.3, 0.3))
                
                if position > 0:
                    exit_price = entry_price * (1 + pnl_pct)
                else:
                    exit_price = entry_price * (1 - pnl_pct)
                
                order = self.generate_order(symbol, exit_side, price=exit_price)
                trade = self.generate_trade_from_order(order)
                
                # Calculate realized P&L
                if position > 0:
                    trade.realized_pnl = (exit_price - entry_price) * trade.amount
                else:
                    trade.realized_pnl = (entry_price - exit_price) * trade.amount
                
                position = 0
            
            # Update timestamp
            current_time += timedelta(minutes=random.randint(5, 60))
            trade.timestamp = current_time
            
            trades.append(trade)
        
        return trades
    
    def generate_grid_trades(
        self,
        symbol: str,
        grid_type: str,
        num_levels: int,
        price_range: Tuple[float, float],
        duration_hours: int = 24,
        volatility: float = 0.01
    ) -> GridTrade:
        """Generate grid trading history"""
        
        grid_id = f"GRID_{uuid.uuid4().hex[:8]}"
        
        # Create grid
        grid = GridTrade(
            grid_id=grid_id,
            symbol=symbol,
            grid_type=grid_type,
            upper_price=price_range[1],
            lower_price=price_range[0],
            grid_levels=num_levels,
            amount_per_level=0.1
        )
        
        # Get grid prices
        grid_prices = grid.get_grid_prices()
        
        # Place initial orders
        mid_price = (price_range[0] + price_range[1]) / 2
        for i, price in enumerate(grid_prices):
            if price < mid_price:
                order = self.generate_order(
                    symbol, "buy", "limit", price, 
                    grid.amount_per_level, "pending", grid_id, i
                )
            else:
                order = self.generate_order(
                    symbol, "sell", "limit", price,
                    grid.amount_per_level, "pending", grid_id, i
                )
            grid.active_orders[i] = order
        
        # Simulate price movement and grid executions
        current_time = datetime.now()
        end_time = current_time + timedelta(hours=duration_hours)
        current_price = mid_price
        
        while current_time < end_time:
            # Random price movement
            price_change = np.random.normal(0, volatility)
            current_price *= (1 + price_change)
            
            # Check for order executions
            for level, order in list(grid.active_orders.items()):
                if order.status != "pending":
                    continue
                
                # Check if price crossed order level
                if (order.side == "buy" and current_price <= order.price) or \
                   (order.side == "sell" and current_price >= order.price):
                    
                    # Execute order
                    trade = self.generate_trade_from_order(order)
                    trade.timestamp = current_time
                    
                    # Update grid state
                    if level not in grid.filled_orders:
                        grid.filled_orders[level] = []
                    grid.filled_orders[level].append(order)
                    
                    # Track positions
                    if order.side == "buy":
                        grid.grid_positions[level] = grid.amount_per_level
                    else:
                        if level in grid.grid_positions:
                            # Calculate profit
                            buy_price = grid_prices[level - 1] if level > 0 else grid_prices[0]
                            profit = (order.price - buy_price) * grid.amount_per_level
                            grid.grid_profit += profit
                            grid.total_pnl += profit
                            if profit > 0:
                                grid.profitable_trades += 1
                            del grid.grid_positions[level]
                    
                    grid.total_trades += 1
                    
                    # Place new order
                    if order.side == "buy" and level < num_levels - 1:
                        # Place sell order one level up
                        new_order = self.generate_order(
                            symbol, "sell", "limit", grid_prices[level + 1],
                            grid.amount_per_level, "pending", grid_id, level + 1
                        )
                        grid.active_orders[level + 1] = new_order
                    
                    # Remove executed order
                    del grid.active_orders[level]
            
            # Advance time
            current_time += timedelta(minutes=random.randint(1, 10))
        
        grid.last_update = current_time
        return grid
    
    def generate_performance_data(
        self,
        trades: List[Trade],
        initial_balance: float = 10000
    ) -> Dict[str, Any]:
        """Generate performance metrics from trades"""
        
        if not trades:
            return {}
        
        # Calculate metrics
        balance = initial_balance
        peak_balance = initial_balance
        balances = [initial_balance]
        returns = []
        
        winning_trades = 0
        losing_trades = 0
        total_profit = 0
        total_loss = 0
        
        for trade in trades:
            # Update balance
            pnl = getattr(trade, 'realized_pnl', 0)
            balance += pnl - trade.commission
            balances.append(balance)
            
            # Track peak
            peak_balance = max(peak_balance, balance)
            
            # Calculate return
            if balances[-2] > 0:
                ret = (balance - balances[-2]) / balances[-2]
                returns.append(ret)
            
            # Track wins/losses
            if pnl > 0:
                winning_trades += 1
                total_profit += pnl
            elif pnl < 0:
                losing_trades += 1
                total_loss += abs(pnl)
        
        # Calculate advanced metrics
        returns_array = np.array(returns)
        
        # Sharpe ratio (assuming daily returns)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)
        else:
            sharpe = 0
        
        # Sortino ratio
        downside_returns = returns_array[returns_array < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            sortino = np.sqrt(252) * np.mean(returns) / downside_std if downside_std > 0 else 0
        else:
            sortino = sharpe
        
        # Max drawdown
        drawdowns = [(peak_balance - b) / peak_balance for b in balances]
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        # Profit factor
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Win rate
        total_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Average win/loss
        avg_win = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        
        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        return {
            "initial_balance": initial_balance,
            "final_balance": balance,
            "total_return": (balance - initial_balance) / initial_balance,
            "total_trades": len(trades),
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "average_win": avg_win,
            "average_loss": avg_loss,
            "largest_win": max([t.realized_pnl for t in trades if hasattr(t, 'realized_pnl')], default=0),
            "largest_loss": min([t.realized_pnl for t in trades if hasattr(t, 'realized_pnl')], default=0),
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "total_commission": sum(t.commission for t in trades),
            "net_profit": balance - initial_balance,
            "roi": ((balance - initial_balance) / initial_balance) * 100
        }


def create_sample_orders(
    symbol: str = "BTC/USDT",
    num_orders: int = 10,
    order_type: str = "limit"
) -> List[Order]:
    """Create sample orders for testing"""
    generator = TradeDataGenerator(seed=42)
    orders = []
    
    for _ in range(num_orders):
        side = random.choice(["buy", "sell"])
        order = generator.generate_order(symbol, side, order_type)
        orders.append(order)
    
    return orders


def create_sample_trades(
    symbol: str = "BTC/USDT",
    num_trades: int = 20,
    win_rate: float = 0.6
) -> List[Trade]:
    """Create sample trades for testing"""
    generator = TradeDataGenerator(seed=42)
    return generator.generate_trade_sequence(symbol, num_trades, win_rate)


def create_sample_position(
    symbol: str = "BTC/USDT",
    side: str = "long",
    num_trades: int = 5
) -> Position:
    """Create sample position for testing"""
    generator = TradeDataGenerator(seed=42)
    
    # Generate entry trades
    trades = []
    for _ in range(num_trades):
        order = generator.generate_order(symbol, "buy" if side == "long" else "sell")
        trade = generator.generate_trade_from_order(order)
        trades.append(trade)
    
    return generator.generate_position(symbol, side, trades)


def create_grid_scenario(
    symbol: str = "BTC/USDT",
    scenario_type: str = "ranging"
) -> GridTrade:
    """Create grid trading scenario"""
    generator = TradeDataGenerator(seed=42)
    
    scenarios = {
        "ranging": {
            "price_range": (44000, 46000),
            "volatility": 0.005,
            "duration_hours": 24
        },
        "trending": {
            "price_range": (44000, 48000),
            "volatility": 0.01,
            "duration_hours": 48
        },
        "volatile": {
            "price_range": (43000, 47000),
            "volatility": 0.02,
            "duration_hours": 12
        }
    }
    
    config = scenarios.get(scenario_type, scenarios["ranging"])
    
    return generator.generate_grid_trades(
        symbol=symbol,
        grid_type="symmetric",
        num_levels=10,
        price_range=config["price_range"],
        duration_hours=config["duration_hours"],
        volatility=config["volatility"]
    )


def create_performance_report(trades: List[Trade]) -> Dict[str, Any]:
    """Create performance report from trades"""
    generator = TradeDataGenerator()
    return generator.generate_performance_data(trades)


# Test data constants
SAMPLE_ORDER = {
    "order_id": "ORD_000001",
    "symbol": "BTC/USDT",
    "side": "buy",
    "order_type": "limit",
    "price": 45000,
    "amount": 0.1,
    "status": "pending"
}

SAMPLE_TRADE = {
    "trade_id": "TRD_000001",
    "order_id": "ORD_000001",
    "symbol": "BTC/USDT",
    "side": "buy",
    "price": 45050,
    "amount": 0.1,
    "commission": 4.505,
    "realized_pnl": 0
}

SAMPLE_POSITION = {
    "symbol": "BTC/USDT",
    "side": "long",
    "amount": 0.5,
    "entry_price": 45000,
    "current_price": 46000,
    "unrealized_pnl": 500,
    "realized_pnl": 0
}


class TradingScenarios:
    """Pre-defined trading scenarios for testing"""
    
    @staticmethod
    def create_winning_streak(num_trades: int = 10) -> List[Trade]:
        """Create a winning streak scenario"""
        generator = TradeDataGenerator(seed=42)
        return generator.generate_trade_sequence(
            "BTC/USDT", num_trades, win_rate=0.9, avg_win=0.02, avg_loss=0.005
        )
    
    @staticmethod
    def create_losing_streak(num_trades: int = 10) -> List[Trade]:
        """Create a losing streak scenario"""
        generator = TradeDataGenerator(seed=42)
        return generator.generate_trade_sequence(
            "BTC/USDT", num_trades, win_rate=0.1, avg_win=0.01, avg_loss=0.02
        )
    
    @staticmethod
    def create_choppy_market(num_trades: int = 20) -> List[Trade]:
        """Create choppy market scenario"""
        generator = TradeDataGenerator(seed=42)
        return generator.generate_trade_sequence(
            "BTC/USDT", num_trades, win_rate=0.5, avg_win=0.01, avg_loss=0.01
        )
    
    @staticmethod
    def create_trending_market(num_trades: int = 15) -> List[Trade]:
        """Create trending market scenario"""
        generator = TradeDataGenerator(seed=42)
        trades = []
        
        # Generate trades with increasing wins
        for i in range(num_trades):
            win_rate = 0.3 + (i / num_trades) * 0.4  # Increasing win rate
            avg_win = 0.015 + (i / num_trades) * 0.01  # Increasing win size
            
            order = generator.generate_order("BTC/USDT", "buy")
            trade = generator.generate_trade_from_order(order)
            
            # Set P&L based on trend
            if random.random() < win_rate:
                trade.realized_pnl = order.amount * order.price * avg_win
            else:
                trade.realized_pnl = -order.amount * order.price * 0.005
            
            trades.append(trade)
        
        return trades


if __name__ == "__main__":
    # Example usage
    generator = TradeDataGenerator()
    
    # Generate single order
    order = generator.generate_order("BTC/USDT", "buy", "limit", 45000, 0.1)
    print(f"Generated order: {order.order_id}")
    
    # Generate trade from order
    trade = generator.generate_trade_from_order(order)
    print(f"Generated trade: {trade.trade_id}, fill price: {trade.price}")
    
    # Generate trade sequence
    trades = generator.generate_trade_sequence("BTC/USDT", 20, win_rate=0.6)
    print(f"\nGenerated {len(trades)} trades")
    
    # Calculate performance
    performance = generator.generate_performance_data(trades)
    print(f"\nPerformance Report:")
    print(f"Win Rate: {performance['win_rate']:.2%}")
    print(f"Profit Factor: {performance['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
    
    # Generate grid trades
    grid = generator.generate_grid_trades(
        "BTC/USDT", "symmetric", 10, (44000, 46000), 24
    )
    print(f"\nGrid Trading Results:")
    print(f"Total Trades: {grid.total_trades}")
    print(f"Grid Profit: ${grid.grid_profit:.2f}")