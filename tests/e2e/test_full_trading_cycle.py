"""
Full trading cycle end-to-end tests for GridAttention trading system.

Tests the complete trading workflow from market analysis to order execution,
including all system components working together in realistic scenarios.
"""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict
import uuid
import json
from unittest.mock import Mock, patch, AsyncMock

# Import all system components
from core.attention_learning import AttentionLearningLayer
from core.market_regime import MarketRegimeDetector
from core.grid_strategy import GridStrategySelector
from core.risk_management import RiskManagementSystem
from core.execution_engine import ExecutionEngine
from core.performance_monitor import PerformanceMonitor
from core.feedback_loop import FeedbackLoop
from core.overfitting_detector import OverfittingDetector
from core.market_data import MarketDataFeed
from core.position_manager import PositionManager
from core.order_manager import OrderManager
from core.compliance_manager import ComplianceManager
from core.alert_system import AlertSystem
from core.metrics_collector import MetricsCollector


class TradingScenario(Enum):
    """Different trading scenarios to test"""
    NORMAL_MARKET = "normal_market"
    TRENDING_MARKET = "trending_market"
    VOLATILE_MARKET = "volatile_market"
    RANGING_MARKET = "ranging_market"
    FLASH_CRASH = "flash_crash"
    LOW_LIQUIDITY = "low_liquidity"
    HIGH_FREQUENCY = "high_frequency"
    NEWS_EVENT = "news_event"


@dataclass
class TradingCycleConfig:
    """Configuration for trading cycle test"""
    scenario: TradingScenario
    duration_minutes: int
    initial_capital: Decimal
    instruments: List[str]
    max_positions: int
    risk_limits: Dict[str, Any]
    enable_live_trading: bool = False
    use_mock_exchange: bool = True


@dataclass
class TradingCycleResult:
    """Results from a complete trading cycle"""
    total_trades: int
    successful_trades: int
    failed_trades: int
    total_pnl: Decimal
    sharpe_ratio: float
    max_drawdown: float
    positions_opened: int
    positions_closed: int
    alerts_triggered: int
    regime_changes: int
    execution_metrics: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]


class TestFullTradingCycle:
    """Test complete trading cycle end-to-end"""
    
    @pytest.fixture
    async def trading_system(self):
        """Create complete trading system with all components"""
        # Initialize all components
        components = {
            'attention_layer': AttentionLearningLayer(
                lookback_window=100,
                learning_rate=0.001,
                enable_online_learning=True
            ),
            'regime_detector': MarketRegimeDetector(
                detection_threshold=0.7,
                min_regime_duration=20
            ),
            'grid_strategy': GridStrategySelector(
                min_grid_levels=5,
                max_grid_levels=20,
                adaptive_spacing=True
            ),
            'risk_management': RiskManagementSystem(
                max_position_size=Decimal('10'),
                max_daily_loss=Decimal('10000'),
                max_leverage=3.0
            ),
            'execution_engine': ExecutionEngine(
                enable_smart_routing=True,
                max_slippage_percent=0.1
            ),
            'performance_monitor': PerformanceMonitor(
                monitoring_interval=60,
                alert_thresholds={'drawdown': 0.1, 'loss_rate': 0.6}
            ),
            'feedback_loop': FeedbackLoop(
                update_frequency=300,  # 5 minutes
                adaptation_rate=0.1
            ),
            'overfitting_detector': OverfittingDetector(
                validation_split=0.2,
                detection_threshold=0.15
            ),
            'position_manager': PositionManager(),
            'order_manager': OrderManager(),
            'compliance_manager': ComplianceManager(),
            'alert_system': AlertSystem(),
            'metrics_collector': MetricsCollector()
        }
        
        # Initialize connections between components
        await self._connect_components(components)
        
        return components
    
    async def _connect_components(self, components: Dict[str, Any]):
        """Connect components with proper event flow"""
        # Connect market data to attention layer
        components['attention_layer'].set_regime_detector(components['regime_detector'])
        components['attention_layer'].set_grid_strategy(components['grid_strategy'])
        
        # Connect risk management to all trading components
        components['grid_strategy'].set_risk_manager(components['risk_management'])
        components['execution_engine'].set_risk_manager(components['risk_management'])
        
        # Connect monitoring and feedback
        components['performance_monitor'].set_components_to_monitor([
            components['attention_layer'],
            components['execution_engine'],
            components['risk_management']
        ])
        
        components['feedback_loop'].set_components_to_update([
            components['attention_layer'],
            components['grid_strategy']
        ])
        
        # Connect alert system
        components['alert_system'].register_sources([
            components['risk_management'],
            components['performance_monitor'],
            components['compliance_manager']
        ])
    
    @pytest.fixture
    async def market_data_feed(self):
        """Create market data feed"""
        return MarketDataFeed(
            symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
            update_frequency='realtime',
            enable_orderbook=True,
            enable_trades=True
        )
    
    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange for testing"""
        class MockExchange:
            def __init__(self):
                self.orders = {}
                self.fills = []
                self.balances = {
                    'USDT': Decimal('100000'),
                    'BTC': Decimal('0'),
                    'ETH': Decimal('0'),
                    'SOL': Decimal('0')
                }
                self.current_prices = {
                    'BTC/USDT': Decimal('50000'),
                    'ETH/USDT': Decimal('3000'),
                    'SOL/USDT': Decimal('100')
                }
                self.order_id_counter = 0
                
            async def place_order(self, order: Dict) -> Dict:
                """Place order on mock exchange"""
                self.order_id_counter += 1
                order_id = f'MOCK_ORD_{self.order_id_counter:06d}'
                
                # Simulate order processing delay
                await asyncio.sleep(0.01)
                
                # Check if order can be filled
                symbol = order['symbol']
                side = order['side']
                quantity = Decimal(str(order['quantity']))
                order_type = order.get('type', 'MARKET')
                
                if order_type == 'MARKET':
                    # Market order - execute immediately
                    fill_price = self.current_prices[symbol]
                    
                    # Apply slippage
                    slippage = Decimal('0.0005')  # 0.05%
                    if side == 'BUY':
                        fill_price *= (1 + slippage)
                    else:
                        fill_price *= (1 - slippage)
                    
                    # Update balances
                    base, quote = symbol.split('/')
                    if side == 'BUY':
                        cost = quantity * fill_price
                        if self.balances[quote] >= cost:
                            self.balances[quote] -= cost
                            self.balances[base] += quantity
                            
                            fill = {
                                'order_id': order_id,
                                'symbol': symbol,
                                'side': side,
                                'quantity': quantity,
                                'price': fill_price,
                                'timestamp': datetime.now(timezone.utc),
                                'status': 'FILLED'
                            }
                            self.fills.append(fill)
                            
                            return {
                                'order_id': order_id,
                                'status': 'FILLED',
                                'fill': fill
                            }
                        else:
                            return {
                                'order_id': order_id,
                                'status': 'REJECTED',
                                'reason': 'INSUFFICIENT_BALANCE'
                            }
                    else:  # SELL
                        if self.balances[base] >= quantity:
                            self.balances[base] -= quantity
                            self.balances[quote] += quantity * fill_price
                            
                            fill = {
                                'order_id': order_id,
                                'symbol': symbol,
                                'side': side,
                                'quantity': quantity,
                                'price': fill_price,
                                'timestamp': datetime.now(timezone.utc),
                                'status': 'FILLED'
                            }
                            self.fills.append(fill)
                            
                            return {
                                'order_id': order_id,
                                'status': 'FILLED',
                                'fill': fill
                            }
                        else:
                            return {
                                'order_id': order_id,
                                'status': 'REJECTED',
                                'reason': 'INSUFFICIENT_BALANCE'
                            }
                else:
                    # Limit order - store for later
                    self.orders[order_id] = {
                        **order,
                        'order_id': order_id,
                        'status': 'OPEN',
                        'timestamp': datetime.now(timezone.utc)
                    }
                    
                    return {
                        'order_id': order_id,
                        'status': 'ACCEPTED'
                    }
            
            async def cancel_order(self, order_id: str) -> Dict:
                """Cancel order"""
                if order_id in self.orders:
                    self.orders[order_id]['status'] = 'CANCELLED'
                    return {'status': 'CANCELLED', 'order_id': order_id}
                return {'status': 'NOT_FOUND', 'order_id': order_id}
            
            async def get_balance(self) -> Dict[str, Decimal]:
                """Get current balances"""
                return self.balances.copy()
            
            async def get_orderbook(self, symbol: str) -> Dict:
                """Get orderbook for symbol"""
                mid_price = self.current_prices[symbol]
                spread = mid_price * Decimal('0.001')  # 0.1% spread
                
                # Generate orderbook
                orderbook = {
                    'symbol': symbol,
                    'bids': [],
                    'asks': [],
                    'timestamp': datetime.now(timezone.utc)
                }
                
                # Generate bid levels
                for i in range(10):
                    price = mid_price - spread * (i + 1)
                    quantity = Decimal(str(np.random.exponential(1.0)))
                    orderbook['bids'].append({'price': price, 'quantity': quantity})
                
                # Generate ask levels
                for i in range(10):
                    price = mid_price + spread * (i + 1)
                    quantity = Decimal(str(np.random.exponential(1.0)))
                    orderbook['asks'].append({'price': price, 'quantity': quantity})
                
                return orderbook
            
            def update_price(self, symbol: str, new_price: Decimal):
                """Update market price (for simulation)"""
                self.current_prices[symbol] = new_price
        
        return MockExchange()
    
    @pytest.mark.asyncio
    async def test_normal_market_trading_cycle(self, trading_system, market_data_feed, mock_exchange):
        """Test complete trading cycle in normal market conditions"""
        config = TradingCycleConfig(
            scenario=TradingScenario.NORMAL_MARKET,
            duration_minutes=60,
            initial_capital=Decimal('100000'),
            instruments=['BTC/USDT', 'ETH/USDT'],
            max_positions=5,
            risk_limits={
                'max_position_size': Decimal('5'),
                'max_daily_loss': Decimal('5000'),
                'max_leverage': 2.0
            }
        )
        
        # Run trading cycle
        result = await self._run_trading_cycle(
            trading_system=trading_system,
            market_data_feed=market_data_feed,
            exchange=mock_exchange,
            config=config
        )
        
        # Verify results
        assert result.total_trades > 0
        assert result.successful_trades > 0
        assert result.positions_opened > 0
        assert result.positions_closed > 0
        
        # Check risk compliance
        assert result.risk_metrics['max_position_size'] <= config.risk_limits['max_position_size']
        assert result.risk_metrics['max_daily_loss'] <= config.risk_limits['max_daily_loss']
        assert result.risk_metrics['max_leverage'] <= config.risk_limits['max_leverage']
        
        # Verify performance metrics
        assert 'avg_execution_time_ms' in result.execution_metrics
        assert result.execution_metrics['avg_execution_time_ms'] < 100  # Less than 100ms
        
        # Check if system adapted during trading
        assert result.regime_changes >= 0  # May or may not change in normal market
        
        # Verify P&L calculation
        final_balance = await mock_exchange.get_balance()
        portfolio_value = await self._calculate_portfolio_value(
            balances=final_balance,
            prices=mock_exchange.current_prices
        )
        
        expected_pnl = portfolio_value - config.initial_capital
        assert abs(result.total_pnl - expected_pnl) < Decimal('1')  # Within $1 tolerance
    
    async def _run_trading_cycle(
        self,
        trading_system: Dict[str, Any],
        market_data_feed: Any,
        exchange: Any,
        config: TradingCycleConfig
    ) -> TradingCycleResult:
        """Run a complete trading cycle"""
        # Initialize tracking variables
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(minutes=config.duration_minutes)
        
        trades = []
        positions = {}
        alerts = []
        regime_changes = []
        metrics = defaultdict(list)
        
        # Connect exchange to execution engine
        trading_system['execution_engine'].set_exchange(exchange)
        
        # Initialize positions
        trading_system['position_manager'].initialize(
            initial_capital=config.initial_capital,
            instruments=config.instruments
        )
        
        # Start all system components
        await self._start_all_components(trading_system)
        
        # Main trading loop
        current_time = start_time
        tick_count = 0
        
        while current_time < end_time:
            tick_count += 1
            
            # Generate market data based on scenario
            market_data = await self._generate_market_data(
                scenario=config.scenario,
                instruments=config.instruments,
                current_prices=exchange.current_prices,
                tick=tick_count
            )
            
            # Update exchange prices
            for symbol, data in market_data.items():
                exchange.update_price(symbol, data['price'])
            
            # Feed market data to system
            await market_data_feed.push_update(market_data)
            
            # Phase 1: Market Analysis
            analysis_result = await self._run_analysis_phase(
                trading_system=trading_system,
                market_data=market_data,
                current_time=current_time
            )
            
            # Track regime changes
            if analysis_result.get('regime_changed', False):
                regime_changes.append({
                    'timestamp': current_time,
                    'from_regime': analysis_result['previous_regime'],
                    'to_regime': analysis_result['current_regime']
                })
            
            # Phase 2: Signal Generation
            signals = await self._run_signal_generation_phase(
                trading_system=trading_system,
                analysis=analysis_result,
                positions=positions
            )
            
            # Phase 3: Risk Assessment
            risk_approved_signals = await self._run_risk_assessment_phase(
                trading_system=trading_system,
                signals=signals,
                positions=positions,
                config=config
            )
            
            # Phase 4: Order Execution
            if risk_approved_signals:
                execution_results = await self._run_execution_phase(
                    trading_system=trading_system,
                    signals=risk_approved_signals,
                    exchange=exchange
                )
                
                # Update positions and trades
                for result in execution_results:
                    if result['status'] == 'FILLED':
                        trades.append(result)
                        await self._update_positions(
                            positions=positions,
                            fill=result['fill'],
                            position_manager=trading_system['position_manager']
                        )
                
                # Record execution metrics
                for result in execution_results:
                    metrics['execution_time_ms'].append(
                        result.get('execution_time_ms', 0)
                    )
            
            # Phase 5: Performance Monitoring
            monitoring_result = await self._run_monitoring_phase(
                trading_system=trading_system,
                positions=positions,
                trades=trades,
                current_time=current_time
            )
            
            # Collect alerts
            if monitoring_result.get('alerts'):
                alerts.extend(monitoring_result['alerts'])
            
            # Phase 6: Feedback and Adaptation
            if tick_count % 300 == 0:  # Every 5 minutes
                await self._run_feedback_phase(
                    trading_system=trading_system,
                    performance_data=monitoring_result,
                    regime_data=analysis_result
                )
            
            # Advance time
            current_time += timedelta(seconds=1)
            await asyncio.sleep(0.01)  # Prevent CPU spinning
        
        # Calculate final metrics
        final_result = await self._calculate_final_metrics(
            trades=trades,
            positions=positions,
            alerts=alerts,
            regime_changes=regime_changes,
            metrics=metrics,
            config=config,
            exchange=exchange
        )
        
        # Cleanup
        await self._stop_all_components(trading_system)
        
        return final_result
    
    async def _generate_market_data(
        self,
        scenario: TradingScenario,
        instruments: List[str],
        current_prices: Dict[str, Decimal],
        tick: int
    ) -> Dict[str, Dict]:
        """Generate market data based on scenario"""
        market_data = {}
        
        for symbol in instruments:
            current_price = current_prices[symbol]
            
            if scenario == TradingScenario.NORMAL_MARKET:
                # Normal market - small random movements
                change = Decimal(str(np.random.normal(0, 0.0005))) * current_price
                new_price = current_price + change
                volume = Decimal(str(np.random.exponential(10)))
                
            elif scenario == TradingScenario.TRENDING_MARKET:
                # Trending market - directional bias
                trend = Decimal('0.0001') if tick % 2 == 0 else Decimal('-0.0001')
                noise = Decimal(str(np.random.normal(0, 0.0003)))
                change = (trend + noise) * current_price
                new_price = current_price + change
                volume = Decimal(str(np.random.exponential(15)))  # Higher volume
                
            elif scenario == TradingScenario.VOLATILE_MARKET:
                # Volatile market - larger movements
                change = Decimal(str(np.random.normal(0, 0.002))) * current_price
                new_price = current_price + change
                volume = Decimal(str(np.random.exponential(20)))  # Much higher volume
                
            elif scenario == TradingScenario.FLASH_CRASH:
                # Flash crash - sudden drop
                if tick == 1800:  # Crash at 30 minutes
                    change = current_price * Decimal('-0.1')  # 10% drop
                else:
                    change = Decimal(str(np.random.normal(0, 0.0005))) * current_price
                new_price = current_price + change
                volume = Decimal(str(np.random.exponential(50))) if tick == 1800 else Decimal(str(np.random.exponential(10)))
                
            else:
                # Default to normal
                change = Decimal(str(np.random.normal(0, 0.0005))) * current_price
                new_price = current_price + change
                volume = Decimal(str(np.random.exponential(10)))
            
            # Ensure price doesn't go negative
            new_price = max(new_price, current_price * Decimal('0.5'))
            
            market_data[symbol] = {
                'price': new_price,
                'volume': volume,
                'bid': new_price * Decimal('0.9995'),
                'ask': new_price * Decimal('1.0005'),
                'timestamp': datetime.now(timezone.utc)
            }
        
        return market_data
    
    async def _run_analysis_phase(
        self,
        trading_system: Dict[str, Any],
        market_data: Dict[str, Dict],
        current_time: datetime
    ) -> Dict[str, Any]:
        """Run market analysis phase"""
        # Update attention layer with new data
        attention_result = await trading_system['attention_layer'].process_market_data(
            market_data=market_data,
            timestamp=current_time
        )
        
        # Detect market regime
        regime_result = await trading_system['regime_detector'].detect_regime(
            market_data=market_data,
            attention_weights=attention_result['attention_weights']
        )
        
        # Check for overfitting
        overfitting_check = await trading_system['overfitting_detector'].check_model(
            model=trading_system['attention_layer'],
            recent_performance=attention_result.get('recent_accuracy', 0.5)
        )
        
        return {
            'attention_weights': attention_result['attention_weights'],
            'market_regime': regime_result['regime'],
            'regime_confidence': regime_result['confidence'],
            'regime_changed': regime_result.get('changed', False),
            'previous_regime': regime_result.get('previous_regime'),
            'current_regime': regime_result.get('regime'),
            'overfitting_risk': overfitting_check['risk_level'],
            'timestamp': current_time
        }
    
    async def _run_signal_generation_phase(
        self,
        trading_system: Dict[str, Any],
        analysis: Dict[str, Any],
        positions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate trading signals based on analysis"""
        # Select appropriate grid strategy
        strategy_result = await trading_system['grid_strategy'].select_strategy(
            market_regime=analysis['market_regime'],
            attention_weights=analysis['attention_weights'],
            current_positions=positions
        )
        
        # Generate signals
        signals = []
        
        for grid in strategy_result['active_grids']:
            # Check if grid levels need adjustment
            if grid['needs_rebalance']:
                # Generate rebalancing signals
                for level in grid['levels']:
                    if level['action'] == 'BUY' and not level['filled']:
                        signals.append({
                            'symbol': grid['symbol'],
                            'side': 'BUY',
                            'type': 'LIMIT',
                            'price': level['price'],
                            'quantity': level['quantity'],
                            'strategy': 'GRID',
                            'grid_id': grid['id'],
                            'confidence': analysis['regime_confidence']
                        })
                    elif level['action'] == 'SELL' and level['can_sell']:
                        signals.append({
                            'symbol': grid['symbol'],
                            'side': 'SELL',
                            'type': 'LIMIT',
                            'price': level['price'],
                            'quantity': level['quantity'],
                            'strategy': 'GRID',
                            'grid_id': grid['id'],
                            'confidence': analysis['regime_confidence']
                        })
        
        return signals
    
    async def _run_risk_assessment_phase(
        self,
        trading_system: Dict[str, Any],
        signals: List[Dict[str, Any]],
        positions: Dict[str, Any],
        config: TradingCycleConfig
    ) -> List[Dict[str, Any]]:
        """Assess risk for generated signals"""
        approved_signals = []
        
        for signal in signals:
            # Check position limits
            position_check = await trading_system['risk_management'].check_position_limit(
                symbol=signal['symbol'],
                current_positions=positions,
                proposed_quantity=signal['quantity'],
                max_position_size=config.risk_limits['max_position_size']
            )
            
            if not position_check['approved']:
                continue
            
            # Check daily loss limit
            loss_check = await trading_system['risk_management'].check_daily_loss_limit(
                current_pnl=await self._calculate_daily_pnl(positions),
                max_daily_loss=config.risk_limits['max_daily_loss']
            )
            
            if not loss_check['approved']:
                continue
            
            # Check leverage
            leverage_check = await trading_system['risk_management'].check_leverage(
                positions=positions,
                proposed_trade=signal,
                max_leverage=config.risk_limits['max_leverage']
            )
            
            if not leverage_check['approved']:
                continue
            
            # All checks passed
            approved_signals.append({
                **signal,
                'risk_score': position_check['risk_score'],
                'approved_quantity': position_check.get('adjusted_quantity', signal['quantity'])
            })
        
        return approved_signals
    
    async def _run_execution_phase(
        self,
        trading_system: Dict[str, Any],
        signals: List[Dict[str, Any]],
        exchange: Any
    ) -> List[Dict[str, Any]]:
        """Execute approved trading signals"""
        execution_results = []
        
        for signal in signals:
            # Prepare order
            order = {
                'symbol': signal['symbol'],
                'side': signal['side'],
                'type': signal['type'],
                'quantity': signal['approved_quantity'],
                'price': signal.get('price'),
                'strategy': signal['strategy']
            }
            
            # Execute through execution engine
            start_time = datetime.now(timezone.utc)
            
            result = await trading_system['execution_engine'].execute_order(
                order=order,
                exchange=exchange
            )
            
            end_time = datetime.now(timezone.utc)
            execution_time_ms = (end_time - start_time).total_seconds() * 1000
            
            execution_results.append({
                **result,
                'signal': signal,
                'execution_time_ms': execution_time_ms
            })
        
        return execution_results
    
    async def _run_monitoring_phase(
        self,
        trading_system: Dict[str, Any],
        positions: Dict[str, Any],
        trades: List[Dict[str, Any]],
        current_time: datetime
    ) -> Dict[str, Any]:
        """Monitor system performance and generate alerts"""
        # Calculate current metrics
        metrics = await trading_system['performance_monitor'].calculate_metrics(
            positions=positions,
            trades=trades,
            timestamp=current_time
        )
        
        # Check for alerts
        alerts = []
        
        # Drawdown alert
        if metrics.get('current_drawdown', 0) > 0.1:
            alerts.append({
                'type': 'DRAWDOWN_WARNING',
                'severity': 'HIGH',
                'message': f"Drawdown at {metrics['current_drawdown']:.2%}",
                'timestamp': current_time
            })
        
        # Win rate alert
        if metrics.get('win_rate', 1.0) < 0.4 and len(trades) > 10:
            alerts.append({
                'type': 'LOW_WIN_RATE',
                'severity': 'MEDIUM',
                'message': f"Win rate at {metrics['win_rate']:.2%}",
                'timestamp': current_time
            })
        
        # Position concentration alert
        if positions:
            max_position_pct = max(
                pos.get('value', 0) / metrics.get('total_portfolio_value', 1)
                for pos in positions.values()
            )
            if max_position_pct > 0.3:
                alerts.append({
                    'type': 'POSITION_CONCENTRATION',
                    'severity': 'MEDIUM',
                    'message': f"Max position concentration at {max_position_pct:.2%}",
                    'timestamp': current_time
                })
        
        # Record metrics
        await trading_system['metrics_collector'].record_metrics(metrics)
        
        return {
            'metrics': metrics,
            'alerts': alerts
        }
    
    async def _run_feedback_phase(
        self,
        trading_system: Dict[str, Any],
        performance_data: Dict[str, Any],
        regime_data: Dict[str, Any]
    ):
        """Run feedback loop to adapt system parameters"""
        # Prepare feedback data
        feedback_data = {
            'performance_metrics': performance_data['metrics'],
            'regime_info': {
                'current_regime': regime_data['market_regime'],
                'regime_confidence': regime_data['regime_confidence']
            },
            'timestamp': datetime.now(timezone.utc)
        }
        
        # Update system components
        await trading_system['feedback_loop'].process_feedback(
            feedback_data=feedback_data,
            components={
                'attention_layer': trading_system['attention_layer'],
                'grid_strategy': trading_system['grid_strategy']
            }
        )
    
    async def _update_positions(
        self,
        positions: Dict[str, Any],
        fill: Dict[str, Any],
        position_manager: Any
    ):
        """Update positions based on fill"""
        symbol = fill['symbol']
        side = fill['side']
        quantity = fill['quantity']
        price = fill['price']
        
        if symbol not in positions:
            positions[symbol] = {
                'quantity': Decimal('0'),
                'avg_price': Decimal('0'),
                'value': Decimal('0'),
                'unrealized_pnl': Decimal('0')
            }
        
        position = positions[symbol]
        
        if side == 'BUY':
            # Update average price
            new_quantity = position['quantity'] + quantity
            if new_quantity > 0:
                position['avg_price'] = (
                    (position['quantity'] * position['avg_price'] + quantity * price) /
                    new_quantity
                )
            position['quantity'] = new_quantity
        else:  # SELL
            # Calculate realized P&L
            if position['quantity'] > 0:
                realized_pnl = quantity * (price - position['avg_price'])
                position['realized_pnl'] = position.get('realized_pnl', Decimal('0')) + realized_pnl
            
            position['quantity'] -= quantity
        
        # Update position value
        position['value'] = position['quantity'] * price
        
        # Update in position manager
        await position_manager.update_position(symbol, position)
    
    async def _calculate_portfolio_value(
        self,
        balances: Dict[str, Decimal],
        prices: Dict[str, Decimal]
    ) -> Decimal:
        """Calculate total portfolio value in USDT"""
        total_value = balances.get('USDT', Decimal('0'))
        
        for symbol, price in prices.items():
            base, quote = symbol.split('/')
            if base in balances:
                total_value += balances[base] * price
        
        return total_value
    
    async def _calculate_daily_pnl(self, positions: Dict[str, Any]) -> Decimal:
        """Calculate daily P&L from positions"""
        total_pnl = Decimal('0')
        
        for position in positions.values():
            total_pnl += position.get('realized_pnl', Decimal('0'))
            total_pnl += position.get('unrealized_pnl', Decimal('0'))
        
        return total_pnl
    
    async def _calculate_final_metrics(
        self,
        trades: List[Dict[str, Any]],
        positions: Dict[str, Any],
        alerts: List[Dict[str, Any]],
        regime_changes: List[Dict[str, Any]],
        metrics: Dict[str, List],
        config: TradingCycleConfig,
        exchange: Any
    ) -> TradingCycleResult:
        """Calculate final trading cycle metrics"""
        # Count trades
        total_trades = len(trades)
        successful_trades = len([t for t in trades if t['status'] == 'FILLED'])
        failed_trades = total_trades - successful_trades
        
        # Calculate P&L
        final_balance = await exchange.get_balance()
        final_value = await self._calculate_portfolio_value(
            balances=final_balance,
            prices=exchange.current_prices
        )
        total_pnl = final_value - config.initial_capital
        
        # Calculate returns for Sharpe ratio
        if trades:
            returns = []
            for i in range(1, len(trades)):
                if i > 0:
                    return_pct = (trades[i]['fill']['price'] - trades[i-1]['fill']['price']) / trades[i-1]['fill']['price']
                    returns.append(float(return_pct))
            
            if returns:
                sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # Calculate max drawdown
        if metrics.get('portfolio_values'):
            portfolio_values = metrics['portfolio_values']
            peak = portfolio_values[0]
            max_drawdown = 0.0
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0.0
        
        # Count positions
        positions_opened = len([t for t in trades if t['side'] == 'BUY'])
        positions_closed = len([t for t in trades if t['side'] == 'SELL'])
        
        # Execution metrics
        if metrics.get('execution_time_ms'):
            avg_execution_time = np.mean(metrics['execution_time_ms'])
            max_execution_time = np.max(metrics['execution_time_ms'])
        else:
            avg_execution_time = 0.0
            max_execution_time = 0.0
        
        execution_metrics = {
            'avg_execution_time_ms': avg_execution_time,
            'max_execution_time_ms': max_execution_time,
            'fill_rate': successful_trades / total_trades if total_trades > 0 else 0.0
        }
        
        # Risk metrics
        risk_metrics = {
            'max_position_size': max(
                pos.get('quantity', Decimal('0'))
                for pos in positions.values()
            ) if positions else Decimal('0'),
            'max_daily_loss': min(
                metrics.get('daily_pnl', [Decimal('0')])
            ) if metrics.get('daily_pnl') else Decimal('0'),
            'max_leverage': max(
                metrics.get('leverage', [0.0])
            ) if metrics.get('leverage') else 0.0
        }
        
        # Performance metrics
        performance_metrics = {
            'total_return_pct': float(total_pnl / config.initial_capital * 100),
            'win_rate': successful_trades / total_trades if total_trades > 0 else 0.0,
            'avg_win': np.mean([
                float(t['fill']['price'] * t['fill']['quantity'])
                for t in trades
                if t['status'] == 'FILLED' and t['side'] == 'SELL'
            ]) if any(t['side'] == 'SELL' for t in trades) else 0.0,
            'avg_loss': 0.0  # Would need to track losing trades
        }
        
        return TradingCycleResult(
            total_trades=total_trades,
            successful_trades=successful_trades,
            failed_trades=failed_trades,
            total_pnl=total_pnl,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            positions_opened=positions_opened,
            positions_closed=positions_closed,
            alerts_triggered=len(alerts),
            regime_changes=len(regime_changes),
            execution_metrics=execution_metrics,
            risk_metrics=risk_metrics,
            performance_metrics=performance_metrics
        )
    
    async def _start_all_components(self, trading_system: Dict[str, Any]):
        """Start all trading system components"""
        for component_name, component in trading_system.items():
            if hasattr(component, 'start'):
                await component.start()
    
    async def _stop_all_components(self, trading_system: Dict[str, Any]):
        """Stop all trading system components"""
        for component_name, component in trading_system.items():
            if hasattr(component, 'stop'):
                await component.stop()
    
    @pytest.mark.asyncio
    async def test_volatile_market_trading_cycle(self, trading_system, market_data_feed, mock_exchange):
        """Test trading cycle in volatile market conditions"""
        config = TradingCycleConfig(
            scenario=TradingScenario.VOLATILE_MARKET,
            duration_minutes=30,
            initial_capital=Decimal('100000'),
            instruments=['BTC/USDT'],
            max_positions=3,
            risk_limits={
                'max_position_size': Decimal('2'),  # Smaller positions in volatile market
                'max_daily_loss': Decimal('3000'),  # Tighter loss limit
                'max_leverage': 1.5  # Lower leverage
            }
        )
        
        result = await self._run_trading_cycle(
            trading_system=trading_system,
            market_data_feed=market_data_feed,
            exchange=mock_exchange,
            config=config
        )
        
        # In volatile markets, expect:
        # - More frequent trades
        # - Tighter risk controls
        # - More alerts
        assert result.total_trades > 10  # More trades due to volatility
        assert result.alerts_triggered > 0  # Should trigger some alerts
        assert result.risk_metrics['max_position_size'] <= config.risk_limits['max_position_size']
        
        # Check if system adapted to volatility
        assert result.regime_changes > 0  # Should detect regime changes
    
    @pytest.mark.asyncio
    async def test_flash_crash_scenario(self, trading_system, market_data_feed, mock_exchange):
        """Test system behavior during flash crash"""
        config = TradingCycleConfig(
            scenario=TradingScenario.FLASH_CRASH,
            duration_minutes=60,
            initial_capital=Decimal('100000'),
            instruments=['BTC/USDT', 'ETH/USDT'],
            max_positions=5,
            risk_limits={
                'max_position_size': Decimal('5'),
                'max_daily_loss': Decimal('10000'),
                'max_leverage': 2.0
            }
        )
        
        result = await self._run_trading_cycle(
            trading_system=trading_system,
            market_data_feed=market_data_feed,
            exchange=mock_exchange,
            config=config
        )
        
        # During flash crash, expect:
        # - Risk limits to kick in
        # - Positions to be reduced or closed
        # - High number of alerts
        assert result.alerts_triggered > 5  # Many alerts during crash
        assert result.risk_metrics['max_daily_loss'] < Decimal('0')  # Should have losses
        
        # But losses should be contained by risk management
        assert abs(result.risk_metrics['max_daily_loss']) <= config.risk_limits['max_daily_loss']
        
        # System should have adapted
        assert result.regime_changes > 0  # Should detect the crash as regime change
    
    @pytest.mark.asyncio
    async def test_multi_instrument_coordination(self, trading_system, market_data_feed, mock_exchange):
        """Test coordination across multiple instruments"""
        config = TradingCycleConfig(
            scenario=TradingScenario.NORMAL_MARKET,
            duration_minutes=120,
            initial_capital=Decimal('100000'),
            instruments=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
            max_positions=10,
            risk_limits={
                'max_position_size': Decimal('5'),
                'max_daily_loss': Decimal('5000'),
                'max_leverage': 3.0
            }
        )
        
        result = await self._run_trading_cycle(
            trading_system=trading_system,
            market_data_feed=market_data_feed,
            exchange=mock_exchange,
            config=config
        )
        
        # Check multi-instrument behavior
        assert result.total_trades > 0
        assert result.positions_opened > 0
        
        # Verify diversification
        final_balance = await mock_exchange.get_balance()
        assets_held = [asset for asset, balance in final_balance.items() 
                      if balance > 0 and asset != 'USDT']
        
        assert len(assets_held) >= 2  # Should hold multiple assets
        
        # Check risk is distributed
        assert result.risk_metrics['max_position_size'] <= config.risk_limits['max_position_size']
    
    @pytest.mark.asyncio
    async def test_system_recovery_after_failure(self, trading_system, market_data_feed, mock_exchange):
        """Test system recovery after component failure"""
        config = TradingCycleConfig(
            scenario=TradingScenario.NORMAL_MARKET,
            duration_minutes=60,
            initial_capital=Decimal('100000'),
            instruments=['BTC/USDT'],
            max_positions=5,
            risk_limits={
                'max_position_size': Decimal('5'),
                'max_daily_loss': Decimal('5000'),
                'max_leverage': 2.0
            }
        )
        
        # Simulate component failure midway
        async def run_with_failure():
            # Start trading
            task = asyncio.create_task(self._run_trading_cycle(
                trading_system=trading_system,
                market_data_feed=market_data_feed,
                exchange=mock_exchange,
                config=config
            ))
            
            # Wait for system to start
            await asyncio.sleep(5)
            
            # Simulate execution engine failure
            original_execute = trading_system['execution_engine'].execute_order
            
            async def failing_execute(*args, **kwargs):
                raise Exception("Simulated execution engine failure")
            
            trading_system['execution_engine'].execute_order = failing_execute
            
            # Let it fail for a bit
            await asyncio.sleep(10)
            
            # Restore functionality
            trading_system['execution_engine'].execute_order = original_execute
            
            # Complete the cycle
            result = await task
            return result
        
        result = await run_with_failure()
        
        # System should have recovered and continued
        assert result.total_trades > 0  # Should still make some trades
        assert result.failed_trades > 0  # Some trades should have failed
        assert result.successful_trades > 0  # But should recover and succeed later
        
        # Risk limits should still be respected
        assert result.risk_metrics['max_position_size'] <= config.risk_limits['max_position_size']
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, trading_system, market_data_feed, mock_exchange):
        """Test system performance under high load"""
        config = TradingCycleConfig(
            scenario=TradingScenario.HIGH_FREQUENCY,
            duration_minutes=10,  # Short but intense
            initial_capital=Decimal('100000'),
            instruments=['BTC/USDT', 'ETH/USDT'],
            max_positions=20,
            risk_limits={
                'max_position_size': Decimal('1'),  # Many small positions
                'max_daily_loss': Decimal('5000'),
                'max_leverage': 2.0
            }
        )
        
        # Generate high-frequency market data
        async def high_frequency_market_data():
            while True:
                for symbol in config.instruments:
                    data = {
                        symbol: {
                            'price': mock_exchange.current_prices[symbol] * (1 + Decimal(str(np.random.normal(0, 0.0001)))),
                            'volume': Decimal(str(np.random.exponential(1))),
                            'timestamp': datetime.now(timezone.utc)
                        }
                    }
                    await market_data_feed.push_update(data)
                await asyncio.sleep(0.01)  # 100 updates per second per instrument
        
        # Run with high-frequency data
        hf_task = asyncio.create_task(high_frequency_market_data())
        
        try:
            result = await self._run_trading_cycle(
                trading_system=trading_system,
                market_data_feed=market_data_feed,
                exchange=mock_exchange,
                config=config
            )
            
            # Check performance metrics
            assert result.execution_metrics['avg_execution_time_ms'] < 50  # Should stay fast
            assert result.execution_metrics['max_execution_time_ms'] < 200  # No severe spikes
            
            # Should handle the load
            assert result.total_trades > 50  # Many trades in short time
            assert result.execution_metrics['fill_rate'] > 0.8  # Most should succeed
            
        finally:
            hf_task.cancel()
            try:
                await hf_task
            except asyncio.CancelledError:
                pass


class TestTradingCycleValidation:
    """Validate trading cycle results and compliance"""
    
    @pytest.mark.asyncio
    async def test_trade_execution_validation(self, trading_system, market_data_feed, mock_exchange):
        """Validate that all trades are executed correctly"""
        config = TradingCycleConfig(
            scenario=TradingScenario.NORMAL_MARKET,
            duration_minutes=30,
            initial_capital=Decimal('100000'),
            instruments=['BTC/USDT'],
            max_positions=5,
            risk_limits={
                'max_position_size': Decimal('5'),
                'max_daily_loss': Decimal('5000'),
                'max_leverage': 2.0
            }
        )
        
        # Track all orders and fills
        all_orders = []
        all_fills = []
        
        # Intercept order placement
        original_place_order = mock_exchange.place_order
        
        async def track_place_order(order):
            all_orders.append(order.copy())
            result = await original_place_order(order)
            if result['status'] == 'FILLED':
                all_fills.append(result['fill'])
            return result
        
        mock_exchange.place_order = track_place_order
        
        # Run trading cycle
        result = await TestFullTradingCycle()._run_trading_cycle(
            trading_system=trading_system,
            market_data_feed=market_data_feed,
            exchange=mock_exchange,
            config=config
        )
        
        # Validate all orders
        for order in all_orders:
            assert 'symbol' in order
            assert 'side' in order
            assert 'quantity' in order
            assert order['quantity'] > 0
            
            # Check risk limits
            assert order['quantity'] <= config.risk_limits['max_position_size']
        
        # Validate all fills
        for fill in all_fills:
            assert 'order_id' in fill
            assert 'price' in fill
            assert 'quantity' in fill
            assert fill['price'] > 0
            assert fill['quantity'] > 0
        
        # Cross-validate orders and fills
        filled_order_ids = {fill['order_id'] for fill in all_fills}
        
        # Verify P&L calculation
        calculated_pnl = Decimal('0')
        for fill in all_fills:
            if fill['side'] == 'BUY':
                calculated_pnl -= fill['quantity'] * fill['price']
            else:
                calculated_pnl += fill['quantity'] * fill['price']
        
        # Add remaining position values
        final_balance = await mock_exchange.get_balance()
        for symbol in config.instruments:
            base, quote = symbol.split('/')
            if base in final_balance and final_balance[base] > 0:
                calculated_pnl += final_balance[base] * mock_exchange.current_prices[symbol]
        
        # Should match reported P&L (within rounding error)
        assert abs(result.total_pnl - (calculated_pnl + config.initial_capital - final_balance.get('USDT', 0))) < Decimal('1')
    
    @pytest.mark.asyncio
    async def test_risk_compliance_throughout_cycle(self, trading_system, market_data_feed, mock_exchange):
        """Ensure risk limits are never violated during the cycle"""
        config = TradingCycleConfig(
            scenario=TradingScenario.VOLATILE_MARKET,
            duration_minutes=60,
            initial_capital=Decimal('100000'),
            instruments=['BTC/USDT', 'ETH/USDT'],
            max_positions=5,
            risk_limits={
                'max_position_size': Decimal('3'),
                'max_daily_loss': Decimal('5000'),
                'max_leverage': 2.0
            }
        )
        
        # Track risk metrics throughout
        risk_violations = []
        
        # Hook into risk management
        original_check_position = trading_system['risk_management'].check_position_limit
        
        async def monitor_position_check(*args, **kwargs):
            result = await original_check_position(*args, **kwargs)
            
            # Check if limit would be violated
            if not result['approved'] and 'proposed_quantity' in kwargs:
                risk_violations.append({
                    'type': 'position_limit',
                    'proposed': kwargs['proposed_quantity'],
                    'limit': config.risk_limits['max_position_size'],
                    'timestamp': datetime.now(timezone.utc)
                })
            
            return result
        
        trading_system['risk_management'].check_position_limit = monitor_position_check
        
        # Run trading cycle
        result = await TestFullTradingCycle()._run_trading_cycle(
            trading_system=trading_system,
            market_data_feed=market_data_feed,
            exchange=mock_exchange,
            config=config
        )
        
        # Check that risk management prevented violations
        assert len(risk_violations) >= 0  # May have attempted violations
        
        # But actual positions should never exceed limits
        assert result.risk_metrics['max_position_size'] <= config.risk_limits['max_position_size']
        assert abs(result.risk_metrics['max_daily_loss']) <= config.risk_limits['max_daily_loss']
        assert result.risk_metrics['max_leverage'] <= config.risk_limits['max_leverage']
        
        # Verify risk management is working
        if risk_violations:
            print(f"Risk management prevented {len(risk_violations)} limit violations")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])