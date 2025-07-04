#!/usr/bin/env python3
"""
Market Data Test Fixtures for GridAttention Trading System
Provides realistic market data generators for testing all components
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import random
import json
from collections import defaultdict

# Constants for market simulation
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']

# Price ranges for different assets
PRICE_RANGES = {
    'BTC/USDT': (40000, 50000),
    'ETH/USDT': (2800, 3500),
    'SOL/USDT': (100, 150),
    'BNB/USDT': (400, 500),
    'XRP/USDT': (0.4, 0.6)
}

# Volatility parameters
VOLATILITY_PARAMS = {
    'BTC/USDT': 0.02,
    'ETH/USDT': 0.025,
    'SOL/USDT': 0.035,
    'BNB/USDT': 0.02,
    'XRP/USDT': 0.03
}


class MarketRegime(Enum):
    """Market regime types"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"


@dataclass
class MarketDataPoint:
    """Single market data point"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'symbol': self.symbol
        }


@dataclass
class OrderBookSnapshot:
    """Order book snapshot"""
    timestamp: datetime
    symbol: str
    bids: List[Tuple[float, float]]  # [(price, amount), ...]
    asks: List[Tuple[float, float]]  # [(price, amount), ...]
    
    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None
    
    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None


class MarketDataGenerator:
    """Generate realistic market data for testing"""
    
    def __init__(self, seed: Optional[int] = None):
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        self.regime_probabilities = {
            MarketRegime.TRENDING_UP: 0.25,
            MarketRegime.TRENDING_DOWN: 0.25,
            MarketRegime.RANGING: 0.30,
            MarketRegime.VOLATILE: 0.10,
            MarketRegime.BREAKOUT: 0.05,
            MarketRegime.ACCUMULATION: 0.025,
            MarketRegime.DISTRIBUTION: 0.025
        }
    
    def generate_ohlcv(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        timeframe: str = '5m',
        regime: Optional[MarketRegime] = None
    ) -> pd.DataFrame:
        """Generate OHLCV data for specified period"""
        
        # Parse timeframe
        minutes = self._parse_timeframe(timeframe)
        
        # Generate timestamps
        timestamps = pd.date_range(start=start_time, end=end_time, freq=f'{minutes}min')
        
        # Get price range and volatility
        price_range = PRICE_RANGES.get(symbol, (100, 200))
        volatility = VOLATILITY_PARAMS.get(symbol, 0.02)
        
        # Choose regime if not specified
        if regime is None:
            regime = self._choose_regime()
        
        # Generate price data based on regime
        prices = self._generate_prices(
            len(timestamps),
            price_range[0],
            price_range[1],
            volatility,
            regime
        )
        
        # Generate OHLCV
        data = []
        for i, (timestamp, close_price) in enumerate(zip(timestamps, prices)):
            # Generate candle data
            candle = self._generate_candle(close_price, volatility)
            
            # Generate volume (higher during trends and breakouts)
            base_volume = np.random.uniform(1000, 5000)
            if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                base_volume *= 1.5
            elif regime == MarketRegime.BREAKOUT:
                base_volume *= 2.0
            
            volume = base_volume * (1 + np.random.normal(0, 0.2))
            
            data.append({
                'timestamp': timestamp,
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': max(0, volume),
                'symbol': symbol
            })
        
        return pd.DataFrame(data)
    
    def generate_orderbook(
        self,
        symbol: str,
        mid_price: float,
        depth: int = 20,
        spread_bps: float = 10,
        liquidity_factor: float = 1.0
    ) -> OrderBookSnapshot:
        """Generate realistic order book snapshot"""
        
        spread = mid_price * spread_bps / 10000
        
        bids = []
        asks = []
        
        # Generate bid side
        current_price = mid_price - spread / 2
        for i in range(depth):
            # Price decreases as we go deeper
            price = current_price - (i * spread * 0.1)
            
            # Amount increases as we go deeper (liquidity pyramid)
            base_amount = np.random.uniform(0.1, 1.0) * liquidity_factor
            amount = base_amount * (1 + i * 0.1)
            
            bids.append((round(price, 2), round(amount, 8)))
        
        # Generate ask side
        current_price = mid_price + spread / 2
        for i in range(depth):
            # Price increases as we go deeper
            price = current_price + (i * spread * 0.1)
            
            # Amount increases as we go deeper
            base_amount = np.random.uniform(0.1, 1.0) * liquidity_factor
            amount = base_amount * (1 + i * 0.1)
            
            asks.append((round(price, 2), round(amount, 8)))
        
        return OrderBookSnapshot(
            timestamp=datetime.now(),
            symbol=symbol,
            bids=bids,
            asks=asks
        )
    
    def generate_trades(
        self,
        symbol: str,
        start_time: datetime,
        num_trades: int,
        price_range: Tuple[float, float],
        aggressive_ratio: float = 0.5
    ) -> List[Dict]:
        """Generate trade history"""
        
        trades = []
        current_time = start_time
        
        for i in range(num_trades):
            # Random time increment
            current_time += timedelta(seconds=np.random.exponential(5))
            
            # Generate price within range
            price = np.random.uniform(price_range[0], price_range[1])
            
            # Generate amount (power law distribution)
            amount = np.random.pareto(2) * 0.1
            
            # Determine side (aggressive trades)
            is_aggressive = np.random.random() < aggressive_ratio
            side = 'buy' if is_aggressive else 'sell'
            
            trades.append({
                'timestamp': current_time,
                'symbol': symbol,
                'price': round(price, 2),
                'amount': round(amount, 8),
                'side': side,
                'trade_id': f"trade_{i}",
                'is_maker': not is_aggressive
            })
        
        return trades
    
    def generate_market_event(
        self,
        event_type: str,
        symbol: str,
        severity: str = 'medium'
    ) -> Dict:
        """Generate market events for testing"""
        
        severity_impacts = {
            'low': 0.01,
            'medium': 0.03,
            'high': 0.05,
            'extreme': 0.10
        }
        
        events = {
            'flash_crash': {
                'type': 'flash_crash',
                'symbol': symbol,
                'impact': -severity_impacts[severity],
                'duration_minutes': np.random.randint(1, 10),
                'recovery_time_minutes': np.random.randint(10, 60)
            },
            'pump': {
                'type': 'pump',
                'symbol': symbol,
                'impact': severity_impacts[severity],
                'duration_minutes': np.random.randint(5, 30),
                'volume_multiplier': np.random.uniform(2, 5)
            },
            'news_event': {
                'type': 'news_event',
                'symbol': symbol,
                'sentiment': np.random.choice(['positive', 'negative', 'neutral']),
                'impact': np.random.uniform(-0.05, 0.05),
                'confidence': np.random.uniform(0.5, 1.0)
            },
            'liquidity_crisis': {
                'type': 'liquidity_crisis',
                'symbol': symbol,
                'spread_multiplier': np.random.uniform(2, 10),
                'volume_reduction': np.random.uniform(0.5, 0.9),
                'duration_minutes': np.random.randint(10, 120)
            },
            'exchange_outage': {
                'type': 'exchange_outage',
                'affected_symbols': [symbol],
                'duration_minutes': np.random.randint(5, 60),
                'partial': np.random.choice([True, False])
            }
        }
        
        event = events.get(event_type, events['news_event'])
        event['timestamp'] = datetime.now()
        event['event_id'] = f"event_{np.random.randint(10000, 99999)}"
        
        return event
    
    def generate_feature_data(
        self,
        ohlcv_data: pd.DataFrame,
        include_technical: bool = True,
        include_microstructure: bool = True,
        include_sentiment: bool = False
    ) -> pd.DataFrame:
        """Generate feature data for testing feature engineering"""
        
        features = ohlcv_data.copy()
        
        if include_technical:
            # Moving averages
            features['sma_5'] = features['close'].rolling(5).mean()
            features['sma_20'] = features['close'].rolling(20).mean()
            features['ema_12'] = features['close'].ewm(span=12).mean()
            features['ema_26'] = features['close'].ewm(span=26).mean()
            
            # Volatility
            features['volatility_5'] = features['close'].pct_change().rolling(5).std()
            features['volatility_20'] = features['close'].pct_change().rolling(20).std()
            
            # RSI
            features['rsi_14'] = self._calculate_rsi(features['close'], 14)
            
            # MACD
            features['macd'] = features['ema_12'] - features['ema_26']
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            
            # Bollinger Bands
            bb_std = features['close'].rolling(20).std()
            features['bb_upper'] = features['sma_20'] + 2 * bb_std
            features['bb_lower'] = features['sma_20'] - 2 * bb_std
            features['bb_width'] = features['bb_upper'] - features['bb_lower']
            
            # Volume indicators
            features['volume_sma'] = features['volume'].rolling(20).mean()
            features['volume_ratio'] = features['volume'] / features['volume_sma']
        
        if include_microstructure:
            # Spread simulation
            features['spread'] = features['close'] * np.random.uniform(0.0001, 0.0005, len(features))
            features['spread_bps'] = (features['spread'] / features['close']) * 10000
            
            # Order flow imbalance
            features['order_flow_imbalance'] = np.random.normal(0, 0.1, len(features))
            
            # Trade intensity
            features['trade_count'] = np.random.poisson(100, len(features))
            features['trade_intensity'] = features['trade_count'] / features['trade_count'].rolling(20).mean()
        
        if include_sentiment:
            # Simulated sentiment scores
            features['sentiment_score'] = np.random.normal(0, 0.5, len(features))
            features['sentiment_confidence'] = np.random.uniform(0.5, 1.0, len(features))
            features['social_volume'] = np.random.poisson(1000, len(features))
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def generate_grid_test_scenario(
        self,
        symbol: str,
        scenario_type: str,
        duration_hours: int = 24
    ) -> Dict:
        """Generate specific test scenarios for grid trading"""
        
        scenarios = {
            'perfect_ranging': {
                'description': 'Perfect ranging market for grid trading',
                'regime': MarketRegime.RANGING,
                'volatility_multiplier': 0.5,
                'trend_strength': 0.0,
                'expected_grid_performance': 'excellent'
            },
            'trending_challenge': {
                'description': 'Strong trend that challenges grid strategy',
                'regime': MarketRegime.TRENDING_UP,
                'volatility_multiplier': 1.0,
                'trend_strength': 0.8,
                'expected_grid_performance': 'poor'
            },
            'whipsaw': {
                'description': 'High volatility whipsaw conditions',
                'regime': MarketRegime.VOLATILE,
                'volatility_multiplier': 2.0,
                'trend_strength': 0.0,
                'expected_grid_performance': 'moderate'
            },
            'breakout_recovery': {
                'description': 'Breakout followed by return to range',
                'regime': MarketRegime.BREAKOUT,
                'volatility_multiplier': 1.5,
                'trend_strength': 0.5,
                'expected_grid_performance': 'challenging'
            },
            'low_volatility': {
                'description': 'Low volatility accumulation phase',
                'regime': MarketRegime.ACCUMULATION,
                'volatility_multiplier': 0.3,
                'trend_strength': 0.1,
                'expected_grid_performance': 'poor'
            }
        }
        
        scenario = scenarios.get(scenario_type, scenarios['perfect_ranging'])
        
        # Generate market data for scenario
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        # Adjust volatility
        original_volatility = VOLATILITY_PARAMS.get(symbol, 0.02)
        adjusted_volatility = original_volatility * scenario['volatility_multiplier']
        
        # Generate OHLCV data
        ohlcv = self.generate_ohlcv(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            timeframe='5m',
            regime=scenario['regime']
        )
        
        # Generate feature data
        features = self.generate_feature_data(ohlcv)
        
        # Generate events for scenario
        events = []
        if scenario_type == 'breakout_recovery':
            events.append(self.generate_market_event('pump', symbol, 'high'))
        elif scenario_type == 'whipsaw':
            for _ in range(5):
                events.append(self.generate_market_event('flash_crash', symbol, 'medium'))
        
        return {
            'scenario_type': scenario_type,
            'scenario_config': scenario,
            'market_data': ohlcv,
            'feature_data': features,
            'events': events,
            'metadata': {
                'symbol': symbol,
                'start_time': start_time,
                'end_time': end_time,
                'total_candles': len(ohlcv),
                'volatility_used': adjusted_volatility
            }
        }
    
    def _parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe string to minutes"""
        mapping = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        return mapping.get(timeframe, 5)
    
    def _choose_regime(self) -> MarketRegime:
        """Choose market regime based on probabilities"""
        regimes = list(self.regime_probabilities.keys())
        probs = list(self.regime_probabilities.values())
        return np.random.choice(regimes, p=probs)
    
    def _generate_prices(
        self,
        num_points: int,
        min_price: float,
        max_price: float,
        volatility: float,
        regime: MarketRegime
    ) -> np.ndarray:
        """Generate price series based on regime"""
        
        # Initialize price
        prices = np.zeros(num_points)
        prices[0] = (min_price + max_price) / 2
        
        # Generate based on regime
        if regime == MarketRegime.TRENDING_UP:
            trend = np.linspace(0, 0.0001, num_points)
            noise = np.random.normal(0, volatility, num_points)
            for i in range(1, num_points):
                prices[i] = prices[i-1] * (1 + trend[i] + noise[i])
        
        elif regime == MarketRegime.TRENDING_DOWN:
            trend = np.linspace(0, -0.0001, num_points)
            noise = np.random.normal(0, volatility, num_points)
            for i in range(1, num_points):
                prices[i] = prices[i-1] * (1 + trend[i] + noise[i])
        
        elif regime == MarketRegime.RANGING:
            # Mean reverting process
            mean_price = (min_price + max_price) / 2
            reversion_speed = 0.1
            for i in range(1, num_points):
                pull = reversion_speed * (mean_price - prices[i-1]) / mean_price
                noise = np.random.normal(0, volatility)
                prices[i] = prices[i-1] * (1 + pull + noise)
        
        elif regime == MarketRegime.VOLATILE:
            # High volatility random walk
            high_vol = volatility * 2
            for i in range(1, num_points):
                prices[i] = prices[i-1] * (1 + np.random.normal(0, high_vol))
        
        elif regime == MarketRegime.BREAKOUT:
            # Sudden move then stabilization
            breakout_point = num_points // 3
            direction = np.random.choice([-1, 1])
            
            # Pre-breakout ranging
            for i in range(1, breakout_point):
                prices[i] = prices[i-1] * (1 + np.random.normal(0, volatility * 0.5))
            
            # Breakout
            prices[breakout_point] = prices[breakout_point-1] * (1 + direction * volatility * 5)
            
            # Post-breakout trend
            for i in range(breakout_point + 1, num_points):
                prices[i] = prices[i-1] * (1 + direction * 0.00005 + np.random.normal(0, volatility))
        
        else:  # ACCUMULATION or DISTRIBUTION
            # Low volatility sideways
            low_vol = volatility * 0.3
            for i in range(1, num_points):
                prices[i] = prices[i-1] * (1 + np.random.normal(0, low_vol))
        
        # Ensure prices stay within bounds
        prices = np.clip(prices, min_price * 0.8, max_price * 1.2)
        
        return prices
    
    def _generate_candle(self, close_price: float, volatility: float) -> Dict:
        """Generate OHLC from close price"""
        
        # Generate intra-candle movement
        candle_range = close_price * volatility * np.random.uniform(0.5, 2.0)
        
        # Open price
        open_price = close_price + np.random.normal(0, candle_range * 0.3)
        
        # High and low
        high = max(open_price, close_price) + abs(np.random.normal(0, candle_range * 0.5))
        low = min(open_price, close_price) - abs(np.random.normal(0, candle_range * 0.5))
        
        return {
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class GridTestDataGenerator:
    """Generate specific test data for grid trading strategies"""
    
    def __init__(self, market_generator: Optional[MarketDataGenerator] = None):
        self.market_generator = market_generator or MarketDataGenerator()
    
    def generate_grid_levels(
        self,
        symbol: str,
        grid_type: str,
        num_levels: int,
        price_range: Optional[Tuple[float, float]] = None
    ) -> List[float]:
        """Generate grid levels for testing"""
        
        if price_range is None:
            price_range = PRICE_RANGES.get(symbol, (100, 200))
        
        if grid_type == 'symmetric':
            # Equal spacing
            return list(np.linspace(price_range[0], price_range[1], num_levels))
        
        elif grid_type == 'asymmetric':
            # More levels near current price
            mid_price = (price_range[0] + price_range[1]) / 2
            lower_levels = np.linspace(price_range[0], mid_price, num_levels // 2)
            upper_levels = np.linspace(mid_price, price_range[1], num_levels // 2)
            return list(lower_levels) + list(upper_levels)[1:]
        
        elif grid_type == 'exponential':
            # Exponential spacing
            log_range = (np.log(price_range[0]), np.log(price_range[1]))
            log_levels = np.linspace(log_range[0], log_range[1], num_levels)
            return list(np.exp(log_levels))
        
        else:
            # Default to symmetric
            return list(np.linspace(price_range[0], price_range[1], num_levels))
    
    def generate_grid_execution_history(
        self,
        grid_levels: List[float],
        market_data: pd.DataFrame,
        execution_delay_ms: int = 100
    ) -> List[Dict]:
        """Generate simulated grid execution history"""
        
        executions = []
        active_orders = {level: None for level in grid_levels}
        position = 0
        
        for _, candle in market_data.iterrows():
            current_price = candle['close']
            
            # Check each grid level
            for level in grid_levels:
                # Buy levels below price
                if level < current_price and active_orders[level] is None and position < len(grid_levels) // 2:
                    execution = {
                        'timestamp': candle['timestamp'] + timedelta(milliseconds=execution_delay_ms),
                        'grid_level': level,
                        'side': 'buy',
                        'price': level,
                        'amount': 0.1,
                        'status': 'filled',
                        'order_id': f"grid_buy_{len(executions)}"
                    }
                    executions.append(execution)
                    active_orders[level] = execution
                    position += 1
                
                # Sell levels above price
                elif level > current_price and active_orders[level] is not None:
                    execution = {
                        'timestamp': candle['timestamp'] + timedelta(milliseconds=execution_delay_ms),
                        'grid_level': level,
                        'side': 'sell',
                        'price': level,
                        'amount': 0.1,
                        'status': 'filled',
                        'order_id': f"grid_sell_{len(executions)}",
                        'pnl': (level - active_orders[level]['price']) * 0.1
                    }
                    executions.append(execution)
                    active_orders[level] = None
                    position -= 1
        
        return executions


def create_test_market_data(
    symbols: List[str] = None,
    duration_hours: int = 24,
    timeframe: str = '5m',
    include_features: bool = True
) -> Dict[str, pd.DataFrame]:
    """Convenience function to create test market data"""
    
    if symbols is None:
        symbols = ['BTC/USDT', 'ETH/USDT']
    
    generator = MarketDataGenerator(seed=42)
    market_data = {}
    
    start_time = datetime.now() - timedelta(hours=duration_hours)
    end_time = datetime.now()
    
    for symbol in symbols:
        # Generate OHLCV
        ohlcv = generator.generate_ohlcv(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            timeframe=timeframe
        )
        
        # Add features if requested
        if include_features:
            ohlcv = generator.generate_feature_data(ohlcv)
        
        market_data[symbol] = ohlcv
    
    return market_data


def create_test_orderbook(
    symbol: str = 'BTC/USDT',
    mid_price: Optional[float] = None,
    depth: int = 20
) -> OrderBookSnapshot:
    """Convenience function to create test orderbook"""
    
    if mid_price is None:
        price_range = PRICE_RANGES.get(symbol, (100, 200))
        mid_price = (price_range[0] + price_range[1]) / 2
    
    generator = MarketDataGenerator()
    return generator.generate_orderbook(symbol, mid_price, depth)


def create_test_scenario(
    scenario_type: str,
    symbol: str = 'BTC/USDT',
    duration_hours: int = 24
) -> Dict:
    """Create a complete test scenario"""
    
    generator = MarketDataGenerator(seed=42)
    return generator.generate_grid_test_scenario(
        symbol=symbol,
        scenario_type=scenario_type,
        duration_hours=duration_hours
    )


# Fixtures for pytest
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "market_data: mark test as needing market data"
    )


# Example test data constants
SAMPLE_CANDLE = {
    'timestamp': datetime.now(),
    'open': 45000,
    'high': 45500,
    'low': 44800,
    'close': 45200,
    'volume': 1234.56,
    'symbol': 'BTC/USDT'
}

SAMPLE_ORDERBOOK = {
    'bids': [(44990, 0.5), (44980, 1.0), (44970, 1.5)],
    'asks': [(45010, 0.5), (45020, 1.0), (45030, 1.5)]
}

SAMPLE_TRADE = {
    'timestamp': datetime.now(),
    'symbol': 'BTC/USDT',
    'price': 45000,
    'amount': 0.1,
    'side': 'buy',
    'trade_id': 'trade_12345'
}


if __name__ == "__main__":
    # Example usage
    generator = MarketDataGenerator()
    
    # Generate market data
    data = generator.generate_ohlcv(
        'BTC/USDT',
        datetime.now() - timedelta(hours=24),
        datetime.now(),
        '5m'
    )
    
    print(f"Generated {len(data)} candles")
    print(data.head())
    
    # Generate orderbook
    orderbook = generator.generate_orderbook('BTC/USDT', 45000)
    print(f"\nOrderbook spread: {orderbook.spread}")
    
    # Generate test scenario
    scenario = generator.generate_grid_test_scenario('BTC/USDT', 'perfect_ranging')
    print(f"\nScenario: {scenario['scenario_config']['description']}")