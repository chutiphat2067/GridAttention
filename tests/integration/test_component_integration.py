# tests/integration/test_component_integration.py

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from core.attention_learning_layer import AttentionLearningLayer
from core.market_regime_detector import MarketRegimeDetector
from core.grid_strategy_selector import GridStrategySelector
from core.risk_management_system import RiskManagementSystem
from core.execution_engine import ExecutionEngine
from core.performance_monitor import PerformanceMonitor
from core.feedback_loop import FeedbackLoop


class TestComponentIntegration:
    """Test integration between major system components"""
    
    @pytest.fixture
    async def integrated_system(self):
        """Create an integrated system with all components"""
        config = {
            'symbol': 'BTC/USDT',
            'timeframe': '5m',
            'max_position_size': 1.0,
            'risk_per_trade': 0.02,
            'attention_window': 50,
            'regime_lookback': 100,
            'grid_levels': 10,
            'grid_spacing': 0.001
        }
        
        # Initialize components
        attention_system = AttentionLearningSystem(config)
        regime_detector = MarketRegimeDetector(config)
        grid_manager = GridStrategyManager(config)
        risk_manager = RiskManager(config)
        execution_engine = ExecutionEngine(config)
        performance_monitor = PerformanceMonitor(config)
        feedback_loop = FeedbackLoop(config)
        
        # Connect components
        components = {
            'attention': attention_system,
            'regime': regime_detector,
            'grid': grid_manager,
            'risk': risk_manager,
            'execution': execution_engine,
            'performance': performance_monitor,
            'feedback': feedback_loop
        }
        
        return components, config
    
    @pytest.mark.asyncio
    async def test_attention_to_regime_integration(self, integrated_system):
        """Test attention system feeding into regime detector"""
        components, config = integrated_system
        
        # Generate market data
        market_data = self._generate_market_data(200)
        
        # Process through attention system
        attention_features = await components['attention'].process_market_data(market_data)
        
        # Feed to regime detector
        regime = await components['regime'].detect_regime(
            market_data, 
            attention_features=attention_features
        )
        
        assert regime is not None
        assert regime['type'] in ['trending', 'ranging', 'volatile']
        assert 'confidence' in regime
        assert 0 <= regime['confidence'] <= 1
    
    @pytest.mark.asyncio
    async def test_regime_to_grid_strategy_integration(self, integrated_system):
        """Test regime detector informing grid strategy"""
        components, config = integrated_system
        
        # Generate different market regimes
        trending_data = self._generate_trending_market(100)
        ranging_data = self._generate_ranging_market(100)
        
        # Test trending regime
        trending_regime = await components['regime'].detect_regime(trending_data)
        grid_params_trending = await components['grid'].adjust_for_regime(trending_regime)
        
        # Test ranging regime
        ranging_regime = await components['regime'].detect_regime(ranging_data)
        grid_params_ranging = await components['grid'].adjust_for_regime(ranging_regime)
        
        # Grid should adapt differently
        assert grid_params_trending['spacing'] != grid_params_ranging['spacing']
        assert grid_params_trending['levels'] != grid_params_ranging['levels']
    
    @pytest.mark.asyncio
    async def test_grid_to_risk_management_integration(self, integrated_system):
        """Test grid strategy with risk management constraints"""
        components, config = integrated_system
        
        # Setup grid
        grid_setup = {
            'levels': 10,
            'spacing': 0.001,
            'center_price': 50000,
            'size_per_level': 0.1
        }
        
        # Create grid orders
        grid_orders = await components['grid'].create_grid_orders(grid_setup)
        
        # Validate each order through risk management
        approved_orders = []
        for order in grid_orders:
            risk_check = await components['risk'].validate_order(order)
            if risk_check['approved']:
                approved_orders.append(order)
        
        # Some orders should be approved
        assert len(approved_orders) > 0
        assert len(approved_orders) <= len(grid_orders)
        
        # Total risk should be within limits
        total_risk = sum(o['size'] * o['price'] for o in approved_orders)
        assert total_risk <= config['max_position_size'] * grid_setup['center_price']
    
    @pytest.mark.asyncio
    async def test_risk_to_execution_integration(self, integrated_system):
        """Test risk-approved orders going to execution"""
        components, config = integrated_system
        
        # Create test order
        order = {
            'symbol': config['symbol'],
            'side': 'buy',
            'size': 0.01,
            'price': 50000,
            'type': 'limit'
        }
        
        # Risk validation
        risk_check = await components['risk'].validate_order(order)
        
        if risk_check['approved']:
            # Execute order
            with patch.object(components['execution'], '_send_order', new_callable=AsyncMock) as mock_send:
                mock_send.return_value = {'id': '12345', 'status': 'open'}
                
                execution_result = await components['execution'].execute_order(order)
                
                assert execution_result is not None
                assert 'id' in execution_result
                mock_send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execution_to_performance_integration(self, integrated_system):
        """Test execution results feeding performance monitor"""
        components, config = integrated_system
        
        # Simulate executed trades
        trades = [
            {
                'id': '1',
                'symbol': config['symbol'],
                'side': 'buy',
                'size': 0.01,
                'price': 50000,
                'timestamp': datetime.now() - timedelta(hours=2),
                'status': 'filled'
            },
            {
                'id': '2',
                'symbol': config['symbol'],
                'side': 'sell',
                'size': 0.01,
                'price': 50500,
                'timestamp': datetime.now() - timedelta(hours=1),
                'status': 'filled'
            }
        ]
        
        # Update performance monitor
        for trade in trades:
            await components['performance'].record_trade(trade)
        
        # Get performance metrics
        metrics = await components['performance'].calculate_metrics()
        
        assert 'total_trades' in metrics
        assert metrics['total_trades'] == 2
        assert 'pnl' in metrics
        assert metrics['pnl'] > 0  # Should be profitable
    
    @pytest.mark.asyncio
    async def test_performance_to_feedback_integration(self, integrated_system):
        """Test performance metrics feeding back to improve strategy"""
        components, config = integrated_system
        
        # Simulate performance history
        performance_data = {
            'win_rate': 0.45,  # Below target
            'sharpe_ratio': 0.8,
            'max_drawdown': 0.15,
            'total_trades': 100
        }
        
        # Feed to feedback loop
        adjustments = await components['feedback'].analyze_performance(performance_data)
        
        assert adjustments is not None
        assert 'attention_params' in adjustments
        assert 'risk_params' in adjustments
        assert 'grid_params' in adjustments
        
        # Should suggest improvements for low win rate
        assert adjustments['risk_params']['position_size_multiplier'] < 1.0
    
    @pytest.mark.asyncio
    async def test_full_cycle_integration(self, integrated_system):
        """Test complete cycle from market data to feedback"""
        components, config = integrated_system
        
        # Initialize system state
        system_state = {
            'positions': [],
            'orders': [],
            'balance': 10000,
            'performance': {'trades': [], 'pnl': 0}
        }
        
        # Simulate multiple market cycles
        for cycle in range(3):
            # Generate market data
            market_data = self._generate_market_data(100)
            
            # 1. Attention analysis
            attention_features = await components['attention'].process_market_data(market_data)
            
            # 2. Regime detection
            regime = await components['regime'].detect_regime(
                market_data, 
                attention_features=attention_features
            )
            
            # 3. Grid strategy adjustment
            grid_params = await components['grid'].adjust_for_regime(regime)
            
            # 4. Create orders with risk validation
            grid_orders = await components['grid'].create_grid_orders(grid_params)
            validated_orders = []
            
            for order in grid_orders:
                risk_check = await components['risk'].validate_order(
                    order, 
                    current_positions=system_state['positions']
                )
                if risk_check['approved']:
                    validated_orders.append(order)
            
            # 5. Execute orders (mock)
            executed_trades = []
            for order in validated_orders[:2]:  # Execute first 2 orders
                trade = {
                    'id': f'trade_{cycle}_{len(executed_trades)}',
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'size': order['size'],
                    'price': order['price'],
                    'timestamp': datetime.now(),
                    'status': 'filled'
                }
                executed_trades.append(trade)
                system_state['positions'].append(trade)
            
            # 6. Update performance
            for trade in executed_trades:
                await components['performance'].record_trade(trade)
            
            # 7. Get metrics and feedback
            metrics = await components['performance'].calculate_metrics()
            adjustments = await components['feedback'].analyze_performance(metrics)
            
            # 8. Apply adjustments (mock)
            if adjustments:
                # Would apply adjustments to components here
                pass
            
            system_state['performance']['trades'].extend(executed_trades)
        
        # Verify full cycle completed
        assert len(system_state['performance']['trades']) > 0
        final_metrics = await components['performance'].calculate_metrics()
        assert final_metrics['total_trades'] > 0
    
    @pytest.mark.asyncio
    async def test_error_propagation_integration(self, integrated_system):
        """Test error handling across components"""
        components, config = integrated_system
        
        # Test invalid data propagation
        invalid_data = pd.DataFrame()  # Empty dataframe
        
        # Attention should handle gracefully
        with pytest.raises(ValueError):
            await components['attention'].process_market_data(invalid_data)
        
        # Risk manager should reject invalid orders
        invalid_order = {'size': -1}  # Negative size
        risk_check = await components['risk'].validate_order(invalid_order)
        assert not risk_check['approved']
        assert 'error' in risk_check
    
    @pytest.mark.asyncio
    async def test_concurrent_component_access(self, integrated_system):
        """Test components handling concurrent requests"""
        components, config = integrated_system
        
        market_data = self._generate_market_data(100)
        
        # Concurrent tasks
        tasks = [
            components['attention'].process_market_data(market_data),
            components['regime'].detect_regime(market_data),
            components['risk'].get_current_exposure(),
            components['performance'].calculate_metrics()
        ]
        
        # All should complete without deadlock
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check no exceptions
        for result in results:
            assert not isinstance(result, Exception)
    
    # Helper methods
    def _generate_market_data(self, periods):
        """Generate synthetic market data"""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='5min')
        
        price = 50000
        prices = []
        for _ in range(periods):
            price *= (1 + np.random.normal(0, 0.001))
            prices.append(price)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.001 for p in prices],
            'low': [p * 0.999 for p in prices],
            'close': prices,
            'volume': np.random.uniform(100, 1000, periods)
        })
    
    def _generate_trending_market(self, periods):
        """Generate trending market data"""
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='5min')
        base_price = 50000
        trend = np.linspace(0, 1000, periods)
        noise = np.random.normal(0, 50, periods)
        prices = base_price + trend + noise
        
        return pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': np.random.uniform(100, 1000, periods)
        })
    
    def _generate_ranging_market(self, periods):
        """Generate ranging market data"""
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='5min')
        base_price = 50000
        prices = base_price + 100 * np.sin(np.linspace(0, 4*np.pi, periods))
        prices += np.random.normal(0, 20, periods)
        
        return pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': np.random.uniform(100, 1000, periods)
        })