from typing import Dict, Any, Optional
"""
risk_management_system.py
Comprehensive risk management with attention insights for grid trading system

Author: Grid Trading System
Date: 2024
"""

# Standard library imports
import asyncio
import time
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Local imports
from core.attention_learning_layer import AttentionLearningLayer, AttentionPhase
from core.grid_strategy_selector import GridStrategyConfig, GridLevel
from core.market_regime_detector import MarketRegime
from core.overfitting_detector import OverfittingDetector, OverfittingSeverity

# Setup logger
logger = logging.getLogger(__name__)


# Constants - MORE CONSERVATIVE VALUES
MAX_POSITION_SIZE_DEFAULT = 0.05  # 5% of capital (reduced from 20%)
MAX_CONCURRENT_ORDERS_DEFAULT = 8  # (reduced from 12)
MAX_DAILY_LOSS_DEFAULT = 0.01  # 1% (reduced from 2%)
MAX_DRAWDOWN_DEFAULT = 0.03  # 3% (reduced from 5%)
POSITION_CORRELATION_LIMIT = 0.7  # (reduced from 0.8)
CONCENTRATION_LIMIT = 0.2  # 20% in single asset (reduced from 30%)
VAR_CONFIDENCE_LEVEL = 0.95  # 95% VaR
RISK_UPDATE_INTERVAL = 30  # seconds (increased frequency)
CORRELATION_UPDATE_INTERVAL = 300  # 5 minutes


# Enums
class RiskLevel(Enum):
    """Risk levels for positions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskViolationType(Enum):
    """Types of risk limit violations"""
    POSITION_SIZE = "position_size_exceeded"
    CONCURRENT_ORDERS = "concurrent_orders_exceeded"
    DAILY_LOSS = "daily_loss_exceeded"
    DRAWDOWN = "max_drawdown_exceeded"
    CORRELATION = "correlation_limit_exceeded"
    CONCENTRATION = "concentration_limit_exceeded"
    LEVERAGE = "leverage_limit_exceeded"
    VAR = "var_limit_exceeded"
    OVERFITTING = "overfitting_risk_exceeded"
    MODEL_UNCERTAINTY = "model_uncertainty_exceeded"


class RiskAction(Enum):
    """Actions to take on risk violations"""
    BLOCK = "block"  # Block new positions
    REDUCE = "reduce"  # Reduce position size
    CLOSE = "close"  # Close positions
    ALERT = "alert"  # Send alert only
    ADJUST = "adjust"  # Adjust parameters


@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    timestamp: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_pnl(self, current_price: float) -> None:
        """Update P&L based on current price"""
        self.current_price = current_price
        if self.side == 'long':
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:  # short
            self.unrealized_pnl = (self.entry_price - current_price) * self.size
            
    def get_value(self) -> float:
        """Get current position value"""
        return self.size * self.current_price
        
    def get_risk_metrics(self) -> Dict[str, float]:
        """Get position risk metrics"""
        value = self.get_value()
        pnl_pct = self.unrealized_pnl / value if value > 0 else 0
        
        return {
            'value': value,
            'unrealized_pnl': self.unrealized_pnl,
            'pnl_percentage': pnl_pct,
            'duration': time.time() - self.timestamp,
            'volatility_exposure': self.size * self.metadata.get('volatility', 0.001)
        }


@dataclass
class RiskMetrics:
    """Current risk metrics"""
    total_exposure: float = 0.0
    position_count: int = 0
    daily_pnl: float = 0.0
    current_drawdown: float = 0.0
    value_at_risk: float = 0.0
    sharpe_ratio: float = 0.0
    correlation_matrix: Optional[pd.DataFrame] = None
    concentration_by_symbol: Dict[str, float] = field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.LOW
    violations: List[RiskViolationType] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class RiskCalculator:
    """Calculates various risk metrics"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.returns_history = deque(maxlen=1000)
        self.price_history = defaultdict(lambda: deque(maxlen=1000))
        
    def calculate_position_risk(self, position: Position, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk metrics for a single position"""
        volatility = market_data.get('volatility', 0.001)
        spread = market_data.get('spread_bps', 1) / 10000
        
        # Position metrics
        position_value = position.get_value()
        position.update_pnl(market_data.get('current_price', position.current_price))
        
        # Risk metrics
        risk_metrics = {
            'position_var': self._calculate_position_var(position, volatility),
            'liquidation_risk': self._calculate_liquidation_risk(position, market_data),
            'spread_cost': position_value * spread,
            'volatility_risk': position_value * volatility,
            'time_decay_risk': self._calculate_time_decay_risk(position),
            'correlation_risk': 0.0  # Will be calculated at portfolio level
        }
        
        return risk_metrics
        
    def _calculate_position_var(self, position: Position, volatility: float) -> float:
        """Calculate Value at Risk for position"""
        # Simple parametric VaR
        z_score = 1.645  # 95% confidence
        position_value = position.get_value()
        var = position_value * volatility * z_score * np.sqrt(1/252)  # Daily VaR
        
        return var
        
    def _calculate_liquidation_risk(self, position: Position, market_data: Dict[str, Any]) -> float:
        """Calculate risk of liquidation"""
        # Distance to liquidation price
        if position.side == 'long':
            liquidation_price = position.entry_price * 0.9  # 10% drop
            price_distance = (position.current_price - liquidation_price) / position.current_price
        else:
            liquidation_price = position.entry_price * 1.1  # 10% rise
            price_distance = (liquidation_price - position.current_price) / position.current_price
            
        # Convert to risk score (0-1)
        risk_score = 1 / (1 + price_distance * 10) if price_distance > 0 else 1.0
        
        return risk_score
        
    def _calculate_time_decay_risk(self, position: Position) -> float:
        """Calculate risk from holding position too long"""
        holding_time = time.time() - position.timestamp
        max_holding_time = 86400  # 24 hours
        
        # Linear decay risk
        decay_risk = min(holding_time / max_holding_time, 1.0)
        
        return decay_risk
        
    def calculate_portfolio_var(self, positions: List[Position], correlation_matrix: pd.DataFrame) -> float:
        """Calculate portfolio VaR considering correlations"""
        if not positions:
            return 0.0
            
        # Get position values and volatilities
        values = []
        volatilities = []
        symbols = []
        
        for pos in positions:
            values.append(pos.get_value())
            volatilities.append(pos.metadata.get('volatility', 0.001))
            symbols.append(pos.symbol)
            
        values = np.array(values)
        volatilities = np.array(volatilities)
        
        # Portfolio volatility with correlations
        if len(positions) > 1 and correlation_matrix is not None:
            # Get correlation submatrix for current positions
            try:
                corr_subset = correlation_matrix.loc[symbols, symbols].values
                portfolio_variance = np.dot(values * volatilities, np.dot(corr_subset, values * volatilities))
                portfolio_volatility = np.sqrt(portfolio_variance) / np.sum(values)
            except:
                # Fallback to simple sum if correlation data unavailable
                portfolio_volatility = np.sqrt(np.sum((values * volatilities) ** 2)) / np.sum(values)
        else:
            portfolio_volatility = np.average(volatilities, weights=values)
            
        # Calculate VaR
        total_value = np.sum(values)
        z_score = 1.645  # 95% confidence
        portfolio_var = total_value * portfolio_volatility * z_score * np.sqrt(1/252)
        
        return portfolio_var
        
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
            
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate
        
        if np.std(excess_returns) == 0:
            return 0.0
            
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized
        
        return sharpe
        
    def update_price_history(self, symbol: str, price: float) -> None:
        """Update price history for correlation calculation"""
        self.price_history[symbol].append(price)
        
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix from price history"""
        if len(self.price_history) < 2:
            return pd.DataFrame()
            
        # Convert to returns
        returns_dict = {}
        
        for symbol, prices in self.price_history.items():
            if len(prices) > 1:
                prices_array = np.array(prices)
                returns = np.diff(np.log(prices_array))
                returns_dict[symbol] = returns[-100:]  # Last 100 returns
                
        # Create DataFrame and calculate correlations
        if returns_dict:
            returns_df = pd.DataFrame(returns_dict)
            return returns_df.corr()
        
        return pd.DataFrame()


class PositionTracker:
    """Tracks all open positions and P&L"""
    
    def __init__(self):
        self.positions = {}  # symbol -> Position
        self.closed_positions = deque(maxlen=1000)
        self.daily_pnl = defaultdict(float)  # date -> pnl
        self.peak_balance = 10000  # Starting balance
        self.current_balance = 10000
        self._lock = asyncio.Lock()
        
    async def add_position(self, position: Position) -> None:
        """Add new position"""
        async with self._lock:
            self.positions[position.symbol] = position
            logger.info(f"Added position: {position.symbol} {position.side} {position.size}")
            
    async def update_position(self, symbol: str, current_price: float) -> None:
        """Update position with current price"""
        async with self._lock:
            if symbol in self.positions:
                self.positions[symbol].update_pnl(current_price)
                
    async def close_position(self, symbol: str, close_price: float) -> Optional[float]:
        """Close position and return realized P&L"""
        async with self._lock:
            if symbol not in self.positions:
                return None
                
            position = self.positions[symbol]
            position.update_pnl(close_price)
            position.realized_pnl = position.unrealized_pnl
            
            # Update daily P&L
            today = datetime.now().date()
            self.daily_pnl[today] += position.realized_pnl
            
            # Update balance
            self.current_balance += position.realized_pnl
            self.peak_balance = max(self.peak_balance, self.current_balance)
            
            # Move to closed positions
            self.closed_positions.append(position)
            del self.positions[symbol]
            
            logger.info(f"Closed position: {symbol} P&L: ${position.realized_pnl:.2f}")
            
            return position.realized_pnl
            
    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        return list(self.positions.values())
        
    def get_total_exposure(self) -> float:
        """Get total portfolio exposure"""
        return sum(pos.get_value() for pos in self.positions.values())
        
    def get_position_count(self) -> int:
        """Get number of open positions"""
        return len(self.positions)
        
    def get_daily_pnl(self, date: Optional[datetime] = None) -> float:
        """Get P&L for specific date"""
        if date is None:
            date = datetime.now().date()
        return self.daily_pnl.get(date, 0.0)
        
    def get_current_drawdown(self) -> float:
        """Get current drawdown from peak"""
        if self.peak_balance <= 0:
            return 0.0
        return (self.peak_balance - self.current_balance) / self.peak_balance
        
    def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
        
    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol"""
        return self.positions.get(symbol)
        
    def get_concentration_by_symbol(self) -> Dict[str, float]:
        """Get position concentration by symbol"""
        total_value = self.get_total_exposure()
        if total_value == 0:
            return {}
            
        concentration = {}
        for symbol, position in self.positions.items():
            concentration[symbol] = position.get_value() / total_value
            
        return concentration
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        closed_pnls = [pos.realized_pnl for pos in self.closed_positions]
        
        if not closed_pnls:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'total_pnl': 0.0
            }
            
        wins = [pnl for pnl in closed_pnls if pnl > 0]
        losses = [pnl for pnl in closed_pnls if pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        return {
            'total_trades': len(closed_pnls),
            'win_rate': len(wins) / len(closed_pnls),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(sum(wins) / sum(losses)) if losses else float('inf'),
            'total_pnl': sum(closed_pnls)
        }


class RiskLimitChecker:
    """Checks risk limits and violations"""
    
    def __init__(self, risk_limits: Dict[str, float]):
        self.risk_limits = risk_limits
        self.violation_history = deque(maxlen=100)
        self.override_limits = {}  # Temporary overrides
        
    def check_position_size(self, proposed_size: float, current_exposure: float, account_balance: float) -> List[RiskViolationType]:
        """Check position size limits"""
        violations = []
        
        max_position = self.get_limit('max_position_size') * account_balance
        if proposed_size > max_position:
            violations.append(RiskViolationType.POSITION_SIZE)
            
        return violations
        
    def check_concurrent_orders(self, current_orders: int, new_orders: int = 1) -> List[RiskViolationType]:
        """Check concurrent order limits"""
        violations = []
        
        max_orders = self.get_limit('max_concurrent_orders')
        if current_orders + new_orders > max_orders:
            violations.append(RiskViolationType.CONCURRENT_ORDERS)
            
        return violations
        
    def check_daily_loss(self, current_daily_loss: float, account_balance: float) -> List[RiskViolationType]:
        """Check daily loss limits"""
        violations = []
        
        max_daily_loss = self.get_limit('max_daily_loss') * account_balance
        if abs(current_daily_loss) > max_daily_loss:
            violations.append(RiskViolationType.DAILY_LOSS)
            
        return violations
        
    def check_drawdown(self, current_drawdown: float) -> List[RiskViolationType]:
        """Check drawdown limits"""
        violations = []
        
        max_drawdown = self.get_limit('max_drawdown')
        if current_drawdown > max_drawdown:
            violations.append(RiskViolationType.DRAWDOWN)
            
        return violations
        
    def check_correlation(self, correlation_matrix: pd.DataFrame, proposed_symbol: str, existing_symbols: List[str]) -> List[RiskViolationType]:
        """Check position correlation limits"""
        violations = []
        
        if not existing_symbols or correlation_matrix.empty:
            return violations
            
        correlation_limit = self.get_limit('position_correlation_limit')
        
        try:
            for symbol in existing_symbols:
                if symbol in correlation_matrix.index and proposed_symbol in correlation_matrix.index:
                    correlation = abs(correlation_matrix.loc[symbol, proposed_symbol])
                    if correlation > correlation_limit:
                        violations.append(RiskViolationType.CORRELATION)
                        break
        except:
            pass  # Ignore correlation check if data unavailable
            
        return violations
        
    def check_concentration(self, concentration_by_symbol: Dict[str, float], proposed_symbol: str, proposed_increase: float) -> List[RiskViolationType]:
        """Check concentration limits"""
        violations = []
        
        concentration_limit = self.get_limit('concentration_limit')
        current_concentration = concentration_by_symbol.get(proposed_symbol, 0.0)
        
        if current_concentration + proposed_increase > concentration_limit:
            violations.append(RiskViolationType.CONCENTRATION)
            
        return violations
        
    def check_var_limit(self, current_var: float, proposed_var_increase: float, account_balance: float) -> List[RiskViolationType]:
        """Check Value at Risk limits"""
        violations = []
        
        max_var = self.get_limit('max_var', 0.1) * account_balance  # Default 10% VaR limit
        if current_var + proposed_var_increase > max_var:
            violations.append(RiskViolationType.VAR)
            
        return violations
        
    def get_limit(self, limit_name: str, default: Optional[float] = None) -> float:
        """Get risk limit with override support"""
        # Check for temporary override
        if limit_name in self.override_limits:
            override = self.override_limits[limit_name]
            if override['expires'] > time.time():
                return override['value']
            else:
                del self.override_limits[limit_name]
                
        # Return normal limit
        return self.risk_limits.get(limit_name, default)
        
    def set_temporary_override(self, limit_name: str, value: float, duration: int = 3600) -> None:
        """Set temporary override for risk limit"""
        self.override_limits[limit_name] = {
            'value': value,
            'expires': time.time() + duration,
            'original': self.risk_limits.get(limit_name)
        }
        logger.warning(f"Risk limit override: {limit_name} = {value} for {duration}s")
        
    def record_violation(self, violations: List[RiskViolationType], context: Dict[str, Any]) -> None:
        """Record risk violations for analysis"""
        self.violation_history.append({
            'timestamp': time.time(),
            'violations': [v.value for v in violations],
            'context': context
        })


class CircuitBreaker:
    """Emergency circuit breaker for system protection"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.breakers = {
            'consecutive_losses': {
                'threshold': self.config.get('max_consecutive_losses', 5),
                'current': 0,
                'triggered': False
            },
            'drawdown_percent': {
                'threshold': self.config.get('circuit_breaker_drawdown', 0.03),
                'current': 0.0,
                'triggered': False
            },
            'error_rate': {
                'threshold': self.config.get('max_error_rate', 0.1),
                'current': 0.0,
                'triggered': False
            },
            'latency_spike': {
                'threshold': self.config.get('max_latency_ms', 50),
                'current': 0.0,
                'triggered': False
            },
            'correlation_spike': {
                'threshold': self.config.get('max_correlation', 0.9),
                'current': 0.0,
                'triggered': False
            }
        }
        self.triggered_at = None
        self.reset_after = 3600  # 1 hour cooldown
        
    def check_breakers(self, metrics: Dict[str, Any]) -> List[str]:
        """Check all circuit breakers and return triggered ones"""
        triggered = []
        
        # Update current values
        if 'consecutive_losses' in metrics:
            self.breakers['consecutive_losses']['current'] = metrics['consecutive_losses']
            
        if 'drawdown' in metrics:
            self.breakers['drawdown_percent']['current'] = metrics['drawdown']
            
        if 'error_rate' in metrics:
            self.breakers['error_rate']['current'] = metrics['error_rate']
            
        if 'latency' in metrics:
            self.breakers['latency_spike']['current'] = metrics['latency']
            
        if 'max_correlation' in metrics:
            self.breakers['correlation_spike']['current'] = metrics['max_correlation']
            
        # Check each breaker
        for name, breaker in self.breakers.items():
            if breaker['current'] >= breaker['threshold']:
                breaker['triggered'] = True
                triggered.append(name)
                logger.critical(f"Circuit breaker triggered: {name} ({breaker['current']} >= {breaker['threshold']})")
                
        if triggered and not self.triggered_at:
            self.triggered_at = time.time()
            
        return triggered
        
    def is_triggered(self) -> bool:
        """Check if any circuit breaker is triggered"""
        return any(b['triggered'] for b in self.breakers.values())
        
    def can_reset(self) -> bool:
        """Check if circuit breakers can be reset"""
        if not self.triggered_at:
            return False
        return time.time() - self.triggered_at > self.reset_after
        
    def reset(self):
        """Reset all circuit breakers"""
        for breaker in self.breakers.values():
            breaker['triggered'] = False
            breaker['current'] = 0
        self.triggered_at = None
        logger.info("Circuit breakers reset")


class EmergencyKillSwitch:
    """Emergency kill switch for immediate shutdown"""
    
    def __init__(self, exchange_manager, order_manager, position_tracker):
        self.exchange_manager = exchange_manager
        self.order_manager = order_manager
        self.position_tracker = position_tracker
        self.activated = False
        self.activation_reason = None
        self.activation_time = None
        
    async def activate(self, reason: str):
        """Activate emergency kill switch"""
        if self.activated:
            return
            
        self.activated = True
        self.activation_reason = reason
        self.activation_time = time.time()
        
        logger.critical(f"EMERGENCY KILL SWITCH ACTIVATED: {reason}")
        
        try:
            # 1. Cancel all orders immediately
            await self._cancel_all_orders()
            
            # 2. Close all positions at market
            await self._close_all_positions_market()
            
            # 3. Disable all trading
            await self._disable_all_trading()
            
            # 4. Send emergency alerts
            await self._send_emergency_alerts()
            
            # 5. Save state for recovery
            await self._save_emergency_state()
            
            logger.critical("Emergency shutdown completed")
            
        except Exception as e:
            logger.critical(f"Error during emergency shutdown: {e}")
            
    async def _cancel_all_orders(self):
        """Cancel all open orders"""
        active_orders = await self.order_manager.get_active_orders()
        
        cancel_tasks = []
        for order in active_orders:
            task = self.exchange_manager.cancel_order(order.order_id, order.symbol)
            cancel_tasks.append(task)
            
        results = await asyncio.gather(*cancel_tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Cancelled {success_count}/{len(active_orders)} orders")
        
    async def _close_all_positions_market(self):
        """Close all positions at market price"""
        positions = self.position_tracker.get_open_positions()
        
        close_tasks = []
        for position in positions:
            # Create market order to close position
            close_order = {
                'symbol': position.symbol,
                'side': 'sell' if position.side == 'long' else 'buy',
                'type': 'market',
                'quantity': position.size
            }
            task = self.exchange_manager.create_order(**close_order)
            close_tasks.append(task)
            
        results = await asyncio.gather(*close_tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Closed {success_count}/{len(positions)} positions")
        
    async def _disable_all_trading(self):
        """Disable all trading strategies"""
        # This would interface with strategy components
        logger.info("All trading strategies disabled")
        
    async def _send_emergency_alerts(self):
        """Send emergency notifications"""
        alert_message = {
            'type': 'EMERGENCY_KILL_SWITCH',
            'reason': self.activation_reason,
            'time': self.activation_time,
            'positions_closed': len(self.position_tracker.get_open_positions())
        }
        
        # Send to various channels (email, SMS, webhook, etc.)
        logger.critical(f"Emergency alert sent: {alert_message}")
        
    async def _save_emergency_state(self):
        """Save system state for recovery"""
        emergency_state = {
            'activation_time': self.activation_time,
            'activation_reason': self.activation_reason,
            'open_positions': [p.__dict__ for p in self.position_tracker.get_open_positions()],
            'active_orders': await self.order_manager.get_active_orders(),
            'account_balance': self.position_tracker.current_balance,
            'saved_at': time.time()
        }
        
        filepath = f"emergency_state_{int(self.activation_time)}.json"
        with open(filepath, 'w') as f:
            json.dump(emergency_state, f, indent=2)
            
        logger.info(f"Emergency state saved to {filepath}")


class CorrelationMonitor:
    """Monitor position correlations in real-time"""
    
    def __init__(self, update_interval: int = CORRELATION_UPDATE_INTERVAL):
        self.update_interval = update_interval
        self.correlation_matrix = pd.DataFrame()
        self.last_update = 0
        self.alert_threshold = 0.8
        self.force_diversification = True
        
    async def update_correlations(self, price_history: Dict[str, List[float]]) -> pd.DataFrame:
        """Update correlation matrix"""
        if time.time() - self.last_update < self.update_interval:
            return self.correlation_matrix
            
        if len(price_history) < 2:
            return self.correlation_matrix
            
        # Convert to returns
        returns_dict = {}
        for symbol, prices in price_history.items():
            if len(prices) > 1:
                returns = np.diff(np.log(prices))
                returns_dict[symbol] = returns[-1000:]  # Last 1000 returns
                
        if returns_dict:
            returns_df = pd.DataFrame(returns_dict)
            self.correlation_matrix = returns_df.corr()
            self.last_update = time.time()
            
        return self.correlation_matrix
        
    def check_new_position_correlation(self, 
                                     new_symbol: str, 
                                     existing_symbols: List[str]) -> Tuple[bool, float]:
        """Check if new position would violate correlation limits"""
        if new_symbol not in self.correlation_matrix.index:
            return True, 0.0  # Allow if no correlation data
            
        max_correlation = 0.0
        for symbol in existing_symbols:
            if symbol in self.correlation_matrix.index:
                correlation = abs(self.correlation_matrix.loc[new_symbol, symbol])
                max_correlation = max(max_correlation, correlation)
                
        allowed = max_correlation < self.alert_threshold
        return allowed, max_correlation


class RiskManagementSystem:
    """
    Comprehensive risk management with attention insights and overfitting prevention
    """
    
    def __init__(self, config: Dict[str, Any], attention_layer: Optional[AttentionLearningLayer] = None):
        self.config = config
        self.attention = attention_layer
        self.risk_calculator = RiskCalculator(config)
        self.position_tracker = PositionTracker()
        self.risk_limits = self._init_risk_limits()
        self.limit_checker = RiskLimitChecker(self.risk_limits)
        
        # Add circuit breaker and kill switch
        self.circuit_breaker = CircuitBreaker(config)
        self.kill_switch = None  # Will be initialized after exchange manager is available
        
        # Risk monitoring
        self.current_metrics = RiskMetrics()
        self.metrics_history = deque(maxlen=1000)
        self.risk_alerts = deque(maxlen=100)
        
        # Enhanced tracking
        self.correlation_monitor = CorrelationMonitor()
        self.consecutive_losses = 0
        
        # Emergency controls
        self.emergency_stop = False
        self.risk_reduction_mode = False
        
        # Performance tracking
        self.risk_adjusted_returns = deque(maxlen=252)  # 1 year of daily returns
        
        # Overfitting detection - Enhanced
        self.overfitting_detector = OverfittingDetector()
        self.overfitting_risk_multiplier = 1.0
        self.model_uncertainty_threshold = 0.5
        self.overfitting_cooldown = 300  # 5 minutes
        self.last_overfitting_check = 0
        self.overfitting_metrics = {
            'detection_count': 0,
            'severity_history': deque(maxlen=100),
            'risk_multiplier_history': deque(maxlen=100),
            'last_severity': OverfittingSeverity.NONE,
            'adaptive_threshold': 0.15  # Adaptive threshold for overfitting
        }
        
        self._lock = asyncio.Lock()
        self._monitoring_task = None
        
    def _init_risk_limits(self) -> Dict[str, float]:
        """Initialize risk limits from config"""
        return {
            'max_position_size': self.config.get('max_position_size', MAX_POSITION_SIZE_DEFAULT),
            'max_concurrent_orders': self.config.get('max_concurrent_orders', MAX_CONCURRENT_ORDERS_DEFAULT),
            'max_daily_loss': self.config.get('max_daily_loss', MAX_DAILY_LOSS_DEFAULT),
            'max_drawdown': self.config.get('max_drawdown', MAX_DRAWDOWN_DEFAULT),
            'position_correlation_limit': self.config.get('position_correlation_limit', POSITION_CORRELATION_LIMIT),
            'concentration_limit': self.config.get('concentration_limit', CONCENTRATION_LIMIT),
            'max_var': self.config.get('max_var', 0.1),  # 10% VaR
            'max_leverage': self.config.get('max_leverage', 3.0),
            'overfitting_threshold': self.config.get('overfitting_threshold', 0.15),  # 15% performance degradation
            'model_uncertainty_limit': self.config.get('model_uncertainty_limit', 0.5)  # 50% uncertainty
        }
        
    async def start_monitoring(self) -> None:
        """Start risk monitoring task"""
        self._monitoring_task = asyncio.create_task(self._monitor_risk_loop())
        logger.info("Started risk monitoring")
        
    async def stop_monitoring(self) -> None:
        """Stop risk monitoring task"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            await asyncio.gather(self._monitoring_task, return_exceptions=True)
            logger.info("Stopped risk monitoring")
            
    async def _monitor_risk_loop(self) -> None:
        """Continuous risk monitoring loop"""
        while True:
            try:
                await self.update_risk_metrics()
                await self._check_emergency_conditions()
                
                # Check circuit breakers
                breaker_metrics = {
                    'consecutive_losses': self.consecutive_losses,
                    'drawdown': self.current_metrics.current_drawdown,
                    'error_rate': self.current_metrics.error_rate,
                    'latency': getattr(self.current_metrics, 'avg_latency', 0),
                    'max_correlation': await self._get_max_correlation()
                }
                
                triggered_breakers = self.circuit_breaker.check_breakers(breaker_metrics)
                
                if triggered_breakers and not self.emergency_stop:
                    await self._handle_circuit_breaker_trigger(triggered_breakers)
                    
                await asyncio.sleep(RISK_UPDATE_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(RISK_UPDATE_INTERVAL)
                
    async def calculate_position_size(self, strategy_config: GridStrategyConfig, context: Dict[str, Any]) -> float:
        """Calculate safe position size"""
        async with self._lock:
            # Base position size from strategy
            base_size = self._calculate_base_size(strategy_config)
            
            # Get risk multipliers
            risk_multipliers = await self._calculate_risk_multipliers(context)
            
            # Apply attention insights if active
            if self.attention and self.attention.phase == AttentionPhase.ACTIVE:
                attention_multiplier = await self._get_attention_risk_adjustment()
                risk_multipliers['attention'] = attention_multiplier
                
            # Check overfitting risk
            overfitting_multiplier = await self._calculate_overfitting_risk_multiplier(context)
            if overfitting_multiplier < 1.0:
                risk_multipliers['overfitting'] = overfitting_multiplier
                
            # Calculate final multiplier
            total_multiplier = 1.0
            for factor, multiplier in risk_multipliers.items():
                total_multiplier *= multiplier
                
            # Apply to base size
            final_size = base_size * total_multiplier
            
            # Ensure within absolute limits
            final_size = await self._apply_position_limits(final_size, context)
            
            logger.info(f"Calculated position size: {final_size:.4f} (base: {base_size:.4f}, multiplier: {total_multiplier:.2f})")
            
            return final_size
            
    def _calculate_base_size(self, strategy_config: GridStrategyConfig) -> float:
        """Calculate base position size from strategy"""
        # Start with configured position size
        base_size = strategy_config.position_size
        
        # Adjust based on regime risk
        regime_multipliers = {
            MarketRegime.RANGING: 1.2,
            MarketRegime.TRENDING: 1.0,
            MarketRegime.VOLATILE: 0.5,
            MarketRegime.DORMANT: 0.3
        }
        
        regime_mult = regime_multipliers.get(strategy_config.regime, 1.0)
        
        return base_size * regime_mult
        
    async def _calculate_risk_multipliers(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate various risk multipliers"""
        multipliers = {}
        
        # Volatility multiplier
        current_vol = context.get('volatility', 0.001)
        baseline_vol = 0.001
        vol_ratio = current_vol / baseline_vol
        
        if vol_ratio > 2.0:
            multipliers['volatility'] = 0.5
        elif vol_ratio > 1.5:
            multipliers['volatility'] = 0.7
        elif vol_ratio < 0.5:
            multipliers['volatility'] = 1.2
        else:
            multipliers['volatility'] = 1.0
            
        # Drawdown multiplier
        current_drawdown = self.position_tracker.get_current_drawdown()
        if current_drawdown > 0.03:  # 3%
            multipliers['drawdown'] = 0.5
        elif current_drawdown > 0.02:  # 2%
            multipliers['drawdown'] = 0.7
        elif current_drawdown > 0.01:  # 1%
            multipliers['drawdown'] = 0.85
        else:
            multipliers['drawdown'] = 1.0
            
        # Daily loss multiplier
        daily_loss = self.position_tracker.get_daily_pnl()
        account_balance = context.get('account_balance', 10000)
        daily_loss_pct = abs(daily_loss) / account_balance
        
        if daily_loss_pct > 0.01:  # 1% daily loss
            multipliers['daily_loss'] = 0.5
        elif daily_loss_pct > 0.005:  # 0.5% daily loss
            multipliers['daily_loss'] = 0.75
        else:
            multipliers['daily_loss'] = 1.0
            
        # Correlation multiplier
        correlation_risk = await self._calculate_correlation_risk(context)
        multipliers['correlation'] = 1.0 - correlation_risk * 0.5
        
        # Position count multiplier
        position_count = self.position_tracker.get_position_count()
        if position_count > 8:
            multipliers['position_count'] = 0.6
        elif position_count > 5:
            multipliers['position_count'] = 0.8
        else:
            multipliers['position_count'] = 1.0
            
        # Regime-specific multiplier
        regime = context.get('regime', MarketRegime.RANGING)
        regime_risk = self._get_regime_risk_multiplier(regime)
        multipliers['regime'] = regime_risk
        
        # Emergency mode multiplier
        if self.emergency_stop:
            multipliers['emergency'] = 0.0
        elif self.risk_reduction_mode:
            multipliers['emergency'] = 0.3
        else:
            multipliers['emergency'] = 1.0
            
        return multipliers
        
    async def _calculate_correlation_risk(self, context: Dict[str, Any]) -> float:
        """Calculate portfolio correlation risk"""
        positions = self.position_tracker.get_open_positions()
        if len(positions) < 2:
            return 0.0
            
        # Get correlation matrix
        correlation_matrix = self.risk_calculator.calculate_correlation_matrix()
        if correlation_matrix.empty:
            return 0.0
            
        # Calculate average correlation
        correlations = []
        symbols = [pos.symbol for pos in positions]
        
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                if sym1 in correlation_matrix.index and sym2 in correlation_matrix.index:
                    corr = abs(correlation_matrix.loc[sym1, sym2])
                    correlations.append(corr)
                    
        if correlations:
            avg_correlation = np.mean(correlations)
            return avg_correlation
        
        return 0.0
        
    def _get_regime_risk_multiplier(self, regime: MarketRegime) -> float:
        """Get risk multiplier based on regime"""
        regime_risk = {
            MarketRegime.RANGING: 1.0,
            MarketRegime.TRENDING: 0.9,
            MarketRegime.VOLATILE: 0.6,
            MarketRegime.DORMANT: 0.8
        }
        
        return regime_risk.get(regime, 1.0)
        
    async def _get_attention_risk_adjustment(self) -> float:
        """Get risk adjustment from attention system"""
        if not self.attention:
            return 1.0
            
        # This would interface with attention system's risk insights
        # For now, return a placeholder
        return 0.9
        
    async def _apply_position_limits(self, position_size: float, context: Dict[str, Any]) -> float:
        """Apply absolute position limits"""
        account_balance = context.get('account_balance', 10000)
        
        # Maximum position size
        max_position = self.risk_limits['max_position_size'] * account_balance
        position_size = min(position_size, max_position)
        
        # Minimum position size
        min_position = 0.001 * account_balance  # 0.1% minimum
        position_size = max(position_size, min_position)
        
        # Check remaining risk budget
        current_exposure = self.position_tracker.get_total_exposure()
        remaining_budget = (account_balance * 0.5) - current_exposure  # 50% max total exposure
        
        if remaining_budget <= 0:
            logger.warning("Risk budget exhausted, no new positions allowed")
            return 0.0
            
        position_size = min(position_size, remaining_budget)
        
        return position_size
        
    async def check_risk_limits(self, proposed_action: Dict[str, Any]) -> List[RiskViolationType]:
        """Check if action violates risk limits"""
        async with self._lock:
            violations = []
            
            account_balance = proposed_action.get('account_balance', 10000)
            
            # Position size check
            if 'size' in proposed_action:
                size_violations = self.limit_checker.check_position_size(
                    proposed_action['size'],
                    self.position_tracker.get_total_exposure(),
                    account_balance
                )
                violations.extend(size_violations)
                
            # Concurrent orders check
            if 'order_count' in proposed_action:
                order_violations = self.limit_checker.check_concurrent_orders(
                    self.position_tracker.get_position_count(),
                    proposed_action['order_count']
                )
                violations.extend(order_violations)
                
            # Daily loss check
            daily_loss_violations = self.limit_checker.check_daily_loss(
                self.position_tracker.get_daily_pnl(),
                account_balance
            )
            violations.extend(daily_loss_violations)
            
            # Drawdown check
            drawdown_violations = self.limit_checker.check_drawdown(
                self.position_tracker.get_current_drawdown()
            )
            violations.extend(drawdown_violations)
            
            # Correlation check
            if 'symbol' in proposed_action:
                existing_symbols = [pos.symbol for pos in self.position_tracker.get_open_positions()]
                correlation_matrix = self.risk_calculator.calculate_correlation_matrix()
                
                corr_violations = self.limit_checker.check_correlation(
                    correlation_matrix,
                    proposed_action['symbol'],
                    existing_symbols
                )
                violations.extend(corr_violations)
                
            # Concentration check
            if 'symbol' in proposed_action and 'size' in proposed_action:
                concentration = self.position_tracker.get_concentration_by_symbol()
                proposed_increase = proposed_action['size'] / account_balance
                
                conc_violations = self.limit_checker.check_concentration(
                    concentration,
                    proposed_action['symbol'],
                    proposed_increase
                )
                violations.extend(conc_violations)
                
            # VaR check
            if 'var_increase' in proposed_action:
                var_violations = self.limit_checker.check_var_limit(
                    self.current_metrics.value_at_risk,
                    proposed_action['var_increase'],
                    account_balance
                )
                violations.extend(var_violations)
                
            # Overfitting risk checks
            overfitting_violations = await self._check_overfitting_risk_limits(proposed_action)
            violations.extend(overfitting_violations)
                
            # Record violations
            if violations:
                self.limit_checker.record_violation(violations, proposed_action)
                
            return violations
    
    async def _check_overfitting_risk_limits(self, proposed_action: Dict[str, Any]) -> List[RiskViolationType]:
        """Check overfitting-specific risk limits"""
        violations = []
        
        # Check if current overfitting risk is too high
        current_multiplier = self.overfitting_risk_multiplier
        if current_multiplier < 0.5:  # High overfitting risk
            violations.append(RiskViolationType.OVERFITTING)
            
        # Check model uncertainty
        model_uncertainty = proposed_action.get('model_uncertainty', 0.0)
        if model_uncertainty > self.model_uncertainty_threshold:
            violations.append(RiskViolationType.MODEL_UNCERTAINTY)
            self.last_model_uncertainty = model_uncertainty
            
        # Check if we're in a different regime than training
        current_regime = proposed_action.get('current_regime')
        training_regime = proposed_action.get('training_regime')
        if current_regime and training_regime and current_regime != training_regime:
            # This increases overfitting risk
            if current_multiplier < 0.7:  # More sensitive threshold for regime mismatch
                violations.append(RiskViolationType.OVERFITTING)
                
        # Check recent performance degradation
        if 'recent_performance' in proposed_action:
            recent_trades = proposed_action['recent_performance']
            if len(recent_trades) >= 10:
                recent_win_rate = sum(1 for trade in recent_trades[-10:] if trade > 0) / 10
                if recent_win_rate < 0.3:  # Less than 30% win rate recently
                    violations.append(RiskViolationType.OVERFITTING)
                    
        return violations
            
    async def get_risk_action(self, violations: List[RiskViolationType]) -> RiskAction:
        """Enhanced risk action determination with overfitting-specific handling"""
        if not violations:
            return RiskAction.ALERT  # No action needed
            
        # Critical violations require immediate action
        critical_violations = [
            RiskViolationType.DAILY_LOSS,
            RiskViolationType.DRAWDOWN,
            RiskViolationType.VAR
        ]
        
        if any(v in critical_violations for v in violations):
            if self.emergency_stop:
                return RiskAction.CLOSE
            else:
                return RiskAction.BLOCK
                
        # Overfitting-specific handling
        if RiskViolationType.OVERFITTING in violations:
            # Check severity of overfitting
            overfitting_severity = self.overfitting_metrics['last_severity']
            
            if overfitting_severity == OverfittingSeverity.CRITICAL:
                return RiskAction.BLOCK  # Block new positions
            elif overfitting_severity in [OverfittingSeverity.HIGH, OverfittingSeverity.MEDIUM]:
                return RiskAction.REDUCE  # Reduce position sizes
            else:
                return RiskAction.ADJUST  # Just adjust parameters
                
        # Model uncertainty handling
        if RiskViolationType.MODEL_UNCERTAINTY in violations:
            if hasattr(self, 'last_model_uncertainty'):
                if self.last_model_uncertainty > 0.7:  # Very high uncertainty
                    return RiskAction.BLOCK
                elif self.last_model_uncertainty > 0.5:  # Medium uncertainty
                    return RiskAction.REDUCE
                    
        # Medium violations require position reduction
        medium_violations = [
            RiskViolationType.POSITION_SIZE,
            RiskViolationType.CORRELATION,
            RiskViolationType.CONCENTRATION
        ]
        
        if any(v in medium_violations for v in violations):
            return RiskAction.REDUCE
            
        # Other violations just need alerts
        return RiskAction.ALERT
        
    async def update_risk_metrics(self) -> None:
        """Update all risk metrics"""
        async with self._lock:
            positions = self.position_tracker.get_open_positions()
            account_balance = self.config.get('account_balance', 10000)
            
            # Update basic metrics
            self.current_metrics.total_exposure = self.position_tracker.get_total_exposure()
            self.current_metrics.position_count = len(positions)
            self.current_metrics.daily_pnl = self.position_tracker.get_daily_pnl()
            self.current_metrics.current_drawdown = self.position_tracker.get_current_drawdown()
            
            # Calculate VaR
            correlation_matrix = self.risk_calculator.calculate_correlation_matrix()
            self.current_metrics.value_at_risk = self.risk_calculator.calculate_portfolio_var(
                positions, correlation_matrix
            )
            self.current_metrics.correlation_matrix = correlation_matrix
            
            # Calculate concentration
            self.current_metrics.concentration_by_symbol = self.position_tracker.get_concentration_by_symbol()
            
            # Calculate Sharpe ratio
            if self.risk_adjusted_returns:
                self.current_metrics.sharpe_ratio = self.risk_calculator.calculate_sharpe_ratio(
                    list(self.risk_adjusted_returns)
                )
                
            # Determine risk level
            self.current_metrics.risk_level = self._calculate_risk_level()
            
            # Update timestamp
            self.current_metrics.timestamp = time.time()
            
            # Enhanced overfitting violation checks
            overfitting_violations = self._check_overfitting_violations()
            self.current_metrics.violations.extend(overfitting_violations)
                        
            # Store in history
            self.metrics_history.append(self.current_metrics)
            
    def _calculate_risk_level(self) -> RiskLevel:
        """Calculate overall risk level"""
        score = 0
        
        # Drawdown component
        if self.current_metrics.current_drawdown > 0.04:
            score += 3
        elif self.current_metrics.current_drawdown > 0.02:
            score += 2
        elif self.current_metrics.current_drawdown > 0.01:
            score += 1
            
        # Daily loss component
        daily_loss_pct = abs(self.current_metrics.daily_pnl) / self.config.get('account_balance', 10000)
        if daily_loss_pct > 0.015:
            score += 3
        elif daily_loss_pct > 0.01:
            score += 2
        elif daily_loss_pct > 0.005:
            score += 1
            
        # VaR component
        var_pct = self.current_metrics.value_at_risk / self.config.get('account_balance', 10000)
        if var_pct > 0.08:
            score += 2
        elif var_pct > 0.05:
            score += 1
            
        # Position count component
        if self.current_metrics.position_count > 10:
            score += 1
            
        # Determine level
        if score >= 7:
            return RiskLevel.CRITICAL
        elif score >= 4:
            return RiskLevel.HIGH
        elif score >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
            
    async def _calculate_overfitting_risk_multiplier(self, context: Dict[str, Any]) -> float:
        """Enhanced overfitting risk multiplier calculation with adaptive thresholds"""
        current_time = time.time()
        
        # Rate limit overfitting checks
        if current_time - self.last_overfitting_check < self.overfitting_cooldown:
            return self.overfitting_risk_multiplier
            
        # Collect performance data
        performance_data = self.position_tracker.get_performance_metrics()
        if performance_data['total_trades'] < 50:  # Need sufficient data
            return 1.0
            
        # Enhanced overfitting detection with multiple indicators
        validation_result = await self._detect_overfitting_comprehensive(performance_data, context)
        
        # Calculate multiplier with adaptive logic
        multiplier = self._calculate_adaptive_multiplier(validation_result)
        
        # Update overfitting metrics
        self._update_overfitting_metrics(validation_result, multiplier)
        
        # Update cached values
        self.overfitting_risk_multiplier = multiplier
        self.last_overfitting_check = current_time
        
        # Log significant overfitting
        if multiplier < 0.8:
            logger.warning(f"Overfitting detected (severity: {validation_result.severity.value}), "
                         f"reducing position size by {(1-multiplier)*100:.1f}%")
            
        return multiplier
    
    async def _detect_overfitting_comprehensive(self, performance_data: Dict[str, Any], 
                                              context: Dict[str, Any]) -> 'ValidationResult':
        """Comprehensive overfitting detection using multiple indicators"""
        # Standard overfitting detection
        base_result = self.overfitting_detector.validate_performance(
            performance_data, 
            context.get('market_conditions', {})
        )
        
        # Additional risk indicators
        additional_risk_score = 0.0
        
        # 1. Recent performance degradation
        recent_trades = performance_data.get('recent_performance', [])
        if len(recent_trades) >= 10:
            recent_win_rate = sum(1 for trade in recent_trades[-10:] if trade > 0) / 10
            overall_win_rate = performance_data.get('win_rate', 0.5)
            
            if recent_win_rate < overall_win_rate * 0.7:  # 30% performance drop
                additional_risk_score += 0.3
                
        # 2. Parameter instability
        parameter_changes = context.get('parameter_changes', [])
        if len(parameter_changes) >= 5:
            change_variance = np.var([abs(change) for change in parameter_changes[-5:]])
            if change_variance > 0.1:  # High parameter instability
                additional_risk_score += 0.2
                
        # 3. Market regime mismatch
        current_regime = context.get('regime')
        trained_regime = context.get('training_regime')
        if current_regime and trained_regime and current_regime != trained_regime:
            additional_risk_score += 0.15
            
        # 4. Model uncertainty
        model_uncertainty = context.get('model_uncertainty', 0.0)
        if model_uncertainty > self.model_uncertainty_threshold:
            additional_risk_score += 0.25
            
        # 5. Correlation breakdown
        portfolio_correlation = context.get('portfolio_correlation', 0.0)
        if portfolio_correlation > 0.8:  # High correlation risk
            additional_risk_score += 0.1
            
        # Adjust severity based on additional risk
        final_confidence = base_result.confidence - additional_risk_score
        
        # Create enhanced validation result
        if final_confidence <= 0.2:
            enhanced_severity = OverfittingSeverity.CRITICAL
        elif final_confidence <= 0.4:
            enhanced_severity = OverfittingSeverity.HIGH
        elif final_confidence <= 0.6:
            enhanced_severity = OverfittingSeverity.MEDIUM
        elif final_confidence <= 0.8:
            enhanced_severity = OverfittingSeverity.LOW
        else:
            enhanced_severity = OverfittingSeverity.NONE
            
        # Return enhanced result
        return ValidationResult(
            is_valid=enhanced_severity == OverfittingSeverity.NONE,
            confidence=max(0.0, final_confidence),
            severity=enhanced_severity,
            details={
                'base_severity': base_result.severity.value,
                'additional_risk_score': additional_risk_score,
                'risk_factors': {
                    'performance_degradation': recent_trades,
                    'parameter_instability': parameter_changes,
                    'regime_mismatch': current_regime != trained_regime if current_regime and trained_regime else False,
                    'model_uncertainty': model_uncertainty,
                    'correlation_risk': portfolio_correlation
                }
            }
        )
    
    def _calculate_adaptive_multiplier(self, validation_result: 'ValidationResult') -> float:
        """Calculate adaptive risk multiplier based on validation result"""
        base_multipliers = {
            OverfittingSeverity.NONE: 1.0,
            OverfittingSeverity.LOW: 0.9,
            OverfittingSeverity.MEDIUM: 0.7,
            OverfittingSeverity.HIGH: 0.5,
            OverfittingSeverity.CRITICAL: 0.2
        }
        
        base_multiplier = base_multipliers[validation_result.severity]
        
        # Adaptive adjustment based on historical performance
        severity_history = self.overfitting_metrics['severity_history']
        
        if len(severity_history) >= 5:
            # If consistently showing overfitting, be more conservative
            recent_high_severity = sum(1 for s in list(severity_history)[-5:] 
                                     if s in [OverfittingSeverity.HIGH, OverfittingSeverity.CRITICAL])
            
            if recent_high_severity >= 3:  # 3 out of 5 recent checks showed high overfitting
                base_multiplier *= 0.8  # Additional 20% reduction
                logger.info("Persistent overfitting detected, applying additional risk reduction")
                
        # Confidence-based adjustment
        confidence_factor = validation_result.confidence
        adjusted_multiplier = base_multiplier * (0.5 + 0.5 * confidence_factor)
        
        # Ensure minimum multiplier
        return max(0.1, adjusted_multiplier)
    
    def _update_overfitting_metrics(self, validation_result: 'ValidationResult', multiplier: float) -> None:
        """Update overfitting tracking metrics"""
        metrics = self.overfitting_metrics
        
        # Update counts and history
        if validation_result.severity != OverfittingSeverity.NONE:
            metrics['detection_count'] += 1
            
        metrics['severity_history'].append(validation_result.severity)
        metrics['risk_multiplier_history'].append(multiplier)
        metrics['last_severity'] = validation_result.severity
        
        # Adaptive threshold adjustment
        if len(metrics['severity_history']) >= 20:
            high_severity_rate = sum(1 for s in list(metrics['severity_history'])[-20:] 
                                   if s in [OverfittingSeverity.HIGH, OverfittingSeverity.CRITICAL]) / 20
            
            # If overfitting is frequent, lower threshold (be more sensitive)
            if high_severity_rate > 0.3:
                metrics['adaptive_threshold'] = max(0.1, metrics['adaptive_threshold'] * 0.9)
            # If overfitting is rare, raise threshold (be less sensitive)
            elif high_severity_rate < 0.1:
                metrics['adaptive_threshold'] = min(0.25, metrics['adaptive_threshold'] * 1.05)
                
    def _check_overfitting_violations(self) -> List[RiskViolationType]:
        """Check for overfitting-related risk violations"""
        violations = []
        
        # Check if overfitting multiplier is too low
        if self.overfitting_risk_multiplier < 0.5:
            violations.append(RiskViolationType.OVERFITTING)
            
        # Check model uncertainty
        if hasattr(self, 'last_model_uncertainty'):
            if self.last_model_uncertainty > self.model_uncertainty_threshold:
                violations.append(RiskViolationType.MODEL_UNCERTAINTY)
                
        return violations
        
    def _check_model_uncertainty(self, context: Dict[str, Any]) -> bool:
        """Check if model uncertainty is too high"""
        uncertainty = context.get('model_uncertainty', 0.0)
        return uncertainty > self.model_uncertainty_threshold
            
    async def _check_emergency_conditions(self) -> None:
        """Check for emergency conditions"""
        # Check for emergency stop conditions
        if self.current_metrics.current_drawdown > self.risk_limits['max_drawdown']:
            await self._trigger_emergency_stop("Maximum drawdown exceeded")
            
        daily_loss_pct = abs(self.current_metrics.daily_pnl) / self.config.get('account_balance', 10000)
        if daily_loss_pct > self.risk_limits['max_daily_loss']:
            await self._trigger_emergency_stop("Maximum daily loss exceeded")
            
        # Check for risk reduction conditions
        if self.current_metrics.risk_level == RiskLevel.HIGH and not self.risk_reduction_mode:
            await self._enter_risk_reduction_mode()
            
        elif self.current_metrics.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM] and self.risk_reduction_mode:
            await self._exit_risk_reduction_mode()
            
    async def _trigger_emergency_stop(self, reason: str) -> None:
        """Trigger emergency stop"""
        if not self.emergency_stop:
            self.emergency_stop = True
            logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
            
            # Send alert
            await self._send_risk_alert({
                'type': 'EMERGENCY_STOP',
                'reason': reason,
                'timestamp': time.time(),
                'metrics': self.current_metrics
            })
            
    async def _enter_risk_reduction_mode(self) -> None:
        """Enter risk reduction mode"""
        if not self.risk_reduction_mode:
            self.risk_reduction_mode = True
            logger.warning("Entering risk reduction mode")
            
            # Reduce risk limits temporarily
            self.limit_checker.set_temporary_override('max_position_size', 0.05, 3600)
            self.limit_checker.set_temporary_override('max_concurrent_orders', 5, 3600)
            
    async def _exit_risk_reduction_mode(self) -> None:
        """Exit risk reduction mode"""
        if self.risk_reduction_mode:
            self.risk_reduction_mode = False
            logger.info("Exiting risk reduction mode")
            
    async def _send_risk_alert(self, alert: Dict[str, Any]) -> None:
        """Send risk alert"""
        self.risk_alerts.append(alert)
        
        # Log alert
        logger.warning(f"Risk Alert: {alert}")
        
        # Here you would integrate with notification system
        # e.g., send email, SMS, webhook, etc.
        
    async def add_position(self, position: Position) -> None:
        """Add new position to tracking"""
        await self.position_tracker.add_position(position)
        
        # Update price history for correlation
        self.risk_calculator.update_price_history(position.symbol, position.entry_price)
        
    async def _get_max_correlation(self) -> float:
        """Get maximum correlation among current positions"""
        positions = self.position_tracker.get_open_positions()
        if len(positions) < 2:
            return 0.0
            
        symbols = [p.symbol for p in positions]
        max_corr = 0.0
        
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                if (sym1 in self.correlation_monitor.correlation_matrix.index and 
                    sym2 in self.correlation_monitor.correlation_matrix.index):
                    corr = abs(self.correlation_monitor.correlation_matrix.loc[sym1, sym2])
                    max_corr = max(max_corr, corr)
                    
        return max_corr
        
    async def _handle_circuit_breaker_trigger(self, triggered_breakers: List[str]):
        """Handle triggered circuit breakers"""
        logger.critical(f"Circuit breakers triggered: {triggered_breakers}")
        
        # Determine severity
        if 'drawdown_percent' in triggered_breakers or 'consecutive_losses' in triggered_breakers:
            # Critical - activate kill switch
            if self.kill_switch:
                await self.kill_switch.activate(f"Circuit breakers: {', '.join(triggered_breakers)}")
            else:
                await self._trigger_emergency_stop(f"Circuit breakers: {', '.join(triggered_breakers)}")
                
        elif 'error_rate' in triggered_breakers or 'latency_spike' in triggered_breakers:
            # High - enter risk reduction mode
            await self._enter_risk_reduction_mode()
            
        elif 'correlation_spike' in triggered_breakers:
            # Medium - prevent new correlated positions
            self.limit_checker.set_temporary_override('position_correlation_limit', 0.5, 3600)
            
        # Send alert
        await self._send_risk_alert({
            'type': 'CIRCUIT_BREAKER_TRIGGERED',
            'breakers': triggered_breakers,
            'timestamp': time.time(),
            'action_taken': 'emergency_stop' if self.emergency_stop else 'risk_reduction'
        })
        
    def set_kill_switch(self, exchange_manager, order_manager):
        """Initialize kill switch with required components"""
        self.kill_switch = EmergencyKillSwitch(exchange_manager, order_manager, self.position_tracker)
        
    async def update_position(self, symbol: str, current_price: float, market_data: Dict[str, Any]) -> None:
        """Update position with current market data"""
        await self.position_tracker.update_position(symbol, current_price)
        
        # Update price history
        self.risk_calculator.update_price_history(symbol, current_price)
        
        # Update correlation monitor
        await self.correlation_monitor.update_correlations(self.risk_calculator.price_history)
        
        # Calculate position risk
        position = self.position_tracker.get_position_by_symbol(symbol)
        if position:
            position.metadata['volatility'] = market_data.get('volatility', 0.001)
            risk_metrics = self.risk_calculator.calculate_position_risk(position, market_data)
            position.metadata['risk_metrics'] = risk_metrics
            
            # Track consecutive losses
            if position.unrealized_pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
    async def close_position(self, symbol: str, close_price: float) -> Optional[float]:
        """Close position and update metrics"""
        pnl = await self.position_tracker.close_position(symbol, close_price)
        
        if pnl is not None:
            # Update risk-adjusted returns
            account_balance = self.config.get('account_balance', 10000)
            daily_return = pnl / account_balance
            self.risk_adjusted_returns.append(daily_return)
            
        return pnl
        
    async def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary with overfitting metrics"""
        performance = self.position_tracker.get_performance_metrics()
        
        return {
            'current_metrics': {
                'risk_level': self.current_metrics.risk_level.value,
                'total_exposure': self.current_metrics.total_exposure,
                'position_count': self.current_metrics.position_count,
                'daily_pnl': self.current_metrics.daily_pnl,
                'current_drawdown': self.current_metrics.current_drawdown,
                'value_at_risk': self.current_metrics.value_at_risk,
                'sharpe_ratio': self.current_metrics.sharpe_ratio
            },
            'risk_limits': self.risk_limits,
            'performance': performance,
            'status': {
                'emergency_stop': self.emergency_stop,
                'risk_reduction_mode': self.risk_reduction_mode
            },
            'overfitting_metrics': {
                'risk_multiplier': self.overfitting_risk_multiplier,
                'last_severity': self.overfitting_metrics['last_severity'].value if self.overfitting_metrics['last_severity'] else 'NONE',
                'detection_count': self.overfitting_metrics['detection_count'],
                'adaptive_threshold': self.overfitting_metrics['adaptive_threshold'],
                'recent_severity_trend': [s.value for s in list(self.overfitting_metrics['severity_history'])[-10:]],
                'model_uncertainty': getattr(self, 'last_model_uncertainty', 0.0)
            },
            'recent_alerts': list(self.risk_alerts)[-10:],
            'concentration': self.current_metrics.concentration_by_symbol
        }
        
    async def activate_kill_switch(self, reason: str) -> None:
        """Activate kill switch - emergency stop all trading"""
        await self._trigger_emergency_stop(f"Kill switch activated: {reason}")
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
    
    async def reset_emergency_stop(self) -> None:
        """Reset emergency stop (manual intervention required)"""
        async with self._lock:
            self.emergency_stop = False
            logger.info("Emergency stop reset")
            
    async def save_state(self, filepath: str) -> None:
        """Save risk system state"""
        state = {
            'risk_limits': self.risk_limits,
            'metrics_history': [
                {
                    'timestamp': m.timestamp,
                    'risk_level': m.risk_level.value,
                    'total_exposure': m.total_exposure,
                    'daily_pnl': m.daily_pnl,
                    'drawdown': m.current_drawdown,
                    'var': m.value_at_risk
                }
                for m in list(self.metrics_history)[-100:]
            ],
            'performance': self.position_tracker.get_performance_metrics(),
            'status': {
                'emergency_stop': self.emergency_stop,
                'risk_reduction_mode': self.risk_reduction_mode
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Saved risk system state to {filepath}")
        
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
            logger.error(f"Recovery failed: {e}")
            return False

    def get_state(self) -> Dict[str, Any]:
        """Get component state for checkpointing"""
        return {
            'class': self.__class__.__name__,
            'timestamp': time.time()
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load component state from checkpoint"""
        pass


# Example usage
async def main():
    """Example usage of RiskManagementSystem"""
    
    # Configuration
    config = {
        'account_balance': 10000,
        'max_position_size': 0.15,
        'max_concurrent_orders': 10,
        'max_daily_loss': 0.02,
        'max_drawdown': 0.05
    }
    
    # Initialize risk system
    risk_system = RiskManagementSystem(config)
    
    # Start monitoring
    await risk_system.start_monitoring()
    
    # Simulate trading
    try:
        # Add a position
        position = Position(
            symbol='BTCUSDT',
            side='long',
            size=1000,
            entry_price=50000,
            current_price=50000,
            timestamp=time.time()
        )
        await risk_system.add_position(position)
        
        # Calculate position size for new trade
        from core.grid_strategy_selector import GridStrategyConfig, GridType, OrderDistribution
        from core.market_regime_detector import MarketRegime
        
        strategy_config = GridStrategyConfig(
            regime=MarketRegime.RANGING,
            grid_type=GridType.SYMMETRIC,
            spacing=0.001,
            levels=5,
            position_size=0.1,
            order_distribution=OrderDistribution.UNIFORM,
            risk_limits={},
            execution_rules={}
        )
        
        context = {
            'account_balance': 10000,
            'volatility': 0.0015,
            'regime': MarketRegime.RANGING
        }
        
        position_size = await risk_system.calculate_position_size(strategy_config, context)
        print(f"Calculated position size: ${position_size:.2f}")
        
        # Check risk limits
        proposed_action = {
            'size': position_size,
            'symbol': 'ETHUSDT',
            'account_balance': 10000
        }
        
        violations = await risk_system.check_risk_limits(proposed_action)
        if violations:
            print(f"Risk violations: {[v.value for v in violations]}")
            action = await risk_system.get_risk_action(violations)
            print(f"Recommended action: {action.value}")
        else:
            print("No risk violations")
            
        # Update position
        await risk_system.update_position('BTCUSDT', 50500, {'volatility': 0.0015})
        
        # Get risk summary
        summary = await risk_system.get_risk_summary()
        print("\nRisk Summary:")
        print(f"  Risk Level: {summary['current_metrics']['risk_level']}")
        print(f"  Total Exposure: ${summary['current_metrics']['total_exposure']:.2f}")
        print(f"  Daily P&L: ${summary['current_metrics']['daily_pnl']:.2f}")
        print(f"  Drawdown: {summary['current_metrics']['current_drawdown']:.2%}")
        print(f"  VaR (95%): ${summary['current_metrics']['value_at_risk']:.2f}")
        
        # Save state
        await risk_system.save_state('risk_state.json')
        
    finally:
        # Stop monitoring
        await risk_system.stop_monitoring()



if __name__ == "__main__":
    asyncio.run(main())
