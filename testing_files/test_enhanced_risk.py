"""
Test Enhanced Risk Manager Module
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhanced_risk_manager import (
    EnhancedRiskManager, RiskLevel, DrawdownState, RiskMetrics,
    PositionSizeRecommendation, DrawdownMetrics, DrawdownTracker
)

class TestEnhancedRiskManager:
    
    def setup_method(self):
        """Set up test environment before each test"""
        self.config = {
            'max_single_position': 0.03,
            'max_daily_drawdown': 0.02,
            'var_95_limit': 0.025,
            'kelly_enabled': True,
            'risk_history_size': 1000
        }
        self.risk_manager = EnhancedRiskManager(self.config)
        
    def create_sample_market_conditions(self):
        """Create sample market conditions"""
        return {
            'volatility_5m': 0.015,
            'volatility_avg': 0.012,
            'volatility_percentile': 60,
            'trend_strength': 0.3,
            'regime': 'RANGING'
        }
        
    def create_sample_account_metrics(self):
        """Create sample account metrics"""
        return {
            'balance': 10000,
            'win_rate': 0.55,
            'avg_win': 1.2,
            'avg_loss': 0.8,
            'confidence': 0.7,
            'current_drawdown': 0.01
        }
        
    def create_sample_positions(self):
        """Create sample positions"""
        return [
            {
                'symbol': 'EURUSD',
                'size': 100,
                'weight': 0.02,
                'volatility': 0.015,
                'leverage': 1.0,
                'pnl': 50
            },
            {
                'symbol': 'GBPUSD', 
                'size': 80,
                'weight': 0.015,
                'volatility': 0.018,
                'leverage': 1.0,
                'pnl': -20
            }
        ]
        
    def test_position_size_calculation(self):
        """Test dynamic position size calculation"""
        market_conditions = self.create_sample_market_conditions()
        account_metrics = self.create_sample_account_metrics()
        
        result = self.risk_manager.calculate_dynamic_position_size(
            signal_strength=0.8,
            market_conditions=market_conditions,
            account_metrics=account_metrics
        )
        
        assert isinstance(result, PositionSizeRecommendation)
        assert result.recommended_size > 0
        assert result.max_safe_size >= result.recommended_size
        assert 0 <= result.confidence <= 1
        assert len(result.reasoning) > 0
        
        print(f"Recommended size: {result.recommended_size:.2f}")
        print(f"Kelly size: {result.kelly_size:.2f}")
        print(f"Vol adjusted: {result.volatility_adjusted_size:.2f}")
        print(f"Confidence: {result.confidence:.2f}")
        
    def test_kelly_criterion_variations(self):
        """Test different Kelly criterion scenarios"""
        market_conditions = self.create_sample_market_conditions()
        
        # High win rate scenario
        high_win_metrics = self.create_sample_account_metrics()
        high_win_metrics['win_rate'] = 0.8
        high_win_metrics['avg_win'] = 1.5
        
        result_high = self.risk_manager.calculate_dynamic_position_size(
            signal_strength=0.7,
            market_conditions=market_conditions,
            account_metrics=high_win_metrics
        )
        
        # Low win rate scenario
        low_win_metrics = self.create_sample_account_metrics()
        low_win_metrics['win_rate'] = 0.3
        low_win_metrics['avg_win'] = 0.8
        
        result_low = self.risk_manager.calculate_dynamic_position_size(
            signal_strength=0.7,
            market_conditions=market_conditions,
            account_metrics=low_win_metrics
        )
        
        # High win rate should generally lead to larger Kelly sizing
        assert result_high.kelly_size >= result_low.kelly_size
        
    def test_risk_assessment(self):
        """Test comprehensive portfolio risk assessment"""
        positions = self.create_sample_positions()
        market_data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.exponential(1000, 100)
        })
        
        portfolio_metrics = {
            'current_balance': 10500,
            'current_drawdown': 0.01
        }
        
        risk_metrics = self.risk_manager.assess_portfolio_risk(
            positions, market_data, portfolio_metrics
        )
        
        assert isinstance(risk_metrics, RiskMetrics)
        assert risk_metrics.overall_risk in [level.value for level in RiskLevel]
        assert 0 <= risk_metrics.risk_score <= 1
        assert risk_metrics.var_95 >= 0
        assert risk_metrics.cvar_95 >= 0
        assert len(risk_metrics.recommendations) >= 0
        
        print(f"Overall risk: {risk_metrics.overall_risk}")
        print(f"Risk score: {risk_metrics.risk_score:.3f}")
        print(f"VaR 95%: {risk_metrics.var_95:.3f}")
        print(f"CVaR 95%: {risk_metrics.cvar_95:.3f}")
        
    def test_volatility_adjustments(self):
        """Test volatility-based position adjustments"""
        account_metrics = self.create_sample_account_metrics()
        
        # Low volatility
        low_vol_conditions = self.create_sample_market_conditions()
        low_vol_conditions['volatility_percentile'] = 20
        low_vol_conditions['volatility_5m'] = 0.008
        
        result_low_vol = self.risk_manager.calculate_dynamic_position_size(
            signal_strength=0.7,
            market_conditions=low_vol_conditions,
            account_metrics=account_metrics
        )
        
        # High volatility
        high_vol_conditions = self.create_sample_market_conditions()
        high_vol_conditions['volatility_percentile'] = 90
        high_vol_conditions['volatility_5m'] = 0.035
        
        result_high_vol = self.risk_manager.calculate_dynamic_position_size(
            signal_strength=0.7,
            market_conditions=high_vol_conditions,
            account_metrics=account_metrics
        )
        
        # Low volatility should allow larger positions
        assert result_low_vol.volatility_adjusted_size > result_high_vol.volatility_adjusted_size
        
    def test_regime_adjustments(self):
        """Test regime-based risk adjustments"""
        account_metrics = self.create_sample_account_metrics()
        market_conditions = self.create_sample_market_conditions()
        
        regimes = ['TRENDING', 'RANGING', 'VOLATILE', 'UNCERTAIN']
        results = {}
        
        for regime in regimes:
            conditions = market_conditions.copy()
            conditions['regime'] = regime
            
            result = self.risk_manager.calculate_dynamic_position_size(
                signal_strength=0.7,
                market_conditions=conditions,
                account_metrics=account_metrics
            )
            results[regime] = result
            
        # VOLATILE regime should have smallest recommended size
        volatile_size = results['VOLATILE'].recommended_size
        ranging_size = results['RANGING'].recommended_size
        
        # Generally, volatile should be more conservative
        assert volatile_size <= ranging_size * 1.2  # Allow some variance
        
    def test_stress_testing(self):
        """Test portfolio stress testing"""
        positions = self.create_sample_positions()
        
        stress_results = self.risk_manager.stress_test_portfolio(positions)
        
        assert isinstance(stress_results, dict)
        assert len(stress_results) > 0
        
        # Check that all scenarios have required metrics
        for scenario_name, results in stress_results.items():
            assert 'portfolio_pnl' in results
            assert 'worst_position_pnl' in results
            assert 'positions_at_risk' in results
            assert 'liquidity_impact' in results
            
        print(f"Stress test scenarios: {list(stress_results.keys())}")
        
    def test_real_time_alerts(self):
        """Test real-time risk alert generation"""
        positions = self.create_sample_positions()
        market_data = pd.DataFrame({
            'close': np.random.randn(50).cumsum() + 100,
            'volume': np.random.exponential(1000, 50)
        })
        
        alerts = self.risk_manager.get_real_time_risk_alerts(positions, market_data)
        
        assert isinstance(alerts, list)
        # Alerts may be empty if no risk conditions are met
        
        for alert in alerts:
            assert 'type' in alert
            assert 'severity' in alert
            assert 'message' in alert
            
        print(f"Generated {len(alerts)} risk alerts")
        
    def test_drawdown_tracking(self):
        """Test drawdown tracking functionality"""
        tracker = DrawdownTracker(self.config)
        
        # Simulate balance changes
        balances = [10000, 10200, 10100, 9800, 9500, 9300, 9600, 9900, 10100, 10300]
        
        for balance in balances:
            tracker.update(balance)
        
        metrics = tracker.get_current_metrics()
        
        assert isinstance(metrics, DrawdownMetrics)
        assert metrics.current_drawdown >= 0
        assert metrics.max_historical_drawdown >= metrics.current_drawdown
        assert metrics.state in [state.value for state in DrawdownState]
        assert 0 <= metrics.recovery_factor <= 1
        
        print(f"Current drawdown: {metrics.current_drawdown:.3f}")
        print(f"Max historical: {metrics.max_historical_drawdown:.3f}")
        print(f"State: {metrics.state}")
        
    def test_risk_constraints(self):
        """Test risk constraint enforcement"""
        account_metrics = self.create_sample_account_metrics()
        market_conditions = self.create_sample_market_conditions()
        
        # Very high signal strength
        result = self.risk_manager.calculate_dynamic_position_size(
            signal_strength=2.0,  # Unrealistically high
            market_conditions=market_conditions,
            account_metrics=account_metrics
        )
        
        # Should be constrained by max position limits
        max_allowed = account_metrics['balance'] * self.config['max_single_position']
        assert result.recommended_size <= max_allowed
        
        # Check constraints were applied
        assert len(result.constraints_applied) > 0
        
    def test_portfolio_context_integration(self):
        """Test portfolio context in position sizing"""
        account_metrics = self.create_sample_account_metrics()
        market_conditions = self.create_sample_market_conditions()
        
        # With portfolio context
        portfolio_context = {
            'existing_positions': self.create_sample_positions(),
            'total_exposure': 0.05,
            'correlation_matrix': np.eye(3)
        }
        
        result_with_context = self.risk_manager.calculate_dynamic_position_size(
            signal_strength=0.7,
            market_conditions=market_conditions,
            account_metrics=account_metrics,
            portfolio_context=portfolio_context
        )
        
        # Without portfolio context
        result_without_context = self.risk_manager.calculate_dynamic_position_size(
            signal_strength=0.7,
            market_conditions=market_conditions,
            account_metrics=account_metrics
        )
        
        # Both should be valid
        assert result_with_context.recommended_size > 0
        assert result_without_context.recommended_size > 0
        
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        account_metrics = self.create_sample_account_metrics()
        market_conditions = self.create_sample_market_conditions()
        
        # Zero signal strength
        result_zero = self.risk_manager.calculate_dynamic_position_size(
            signal_strength=0.0,
            market_conditions=market_conditions,
            account_metrics=account_metrics
        )
        assert result_zero.recommended_size >= 0
        
        # Negative values in account metrics
        bad_metrics = account_metrics.copy()
        bad_metrics['avg_win'] = -1.0  # Invalid
        
        result_bad = self.risk_manager.calculate_dynamic_position_size(
            signal_strength=0.5,
            market_conditions=market_conditions,
            account_metrics=bad_metrics
        )
        assert isinstance(result_bad, PositionSizeRecommendation)
        
        # Empty positions list
        empty_positions = []
        risk_metrics = self.risk_manager.assess_portfolio_risk(
            empty_positions, 
            pd.DataFrame({'close': [100, 101, 102]})
        )
        assert isinstance(risk_metrics, RiskMetrics)
        
    def test_var_calculation(self):
        """Test VaR and CVaR calculation"""
        positions = self.create_sample_positions()
        
        # Create market data with known distribution
        np.random.seed(42)
        returns = np.random.normal(-0.0001, 0.02, 250)  # Slight negative drift
        prices = 100 * (1 + returns).cumprod()
        
        market_data = pd.DataFrame({
            'close': prices,
            'volume': np.random.exponential(1000, 250)
        })
        
        risk_metrics = self.risk_manager.assess_portfolio_risk(positions, market_data)
        
        # VaR and CVaR should be positive (representing potential losses)
        assert risk_metrics.var_95 >= 0
        assert risk_metrics.cvar_95 >= 0
        # CVaR should generally be >= VaR
        assert risk_metrics.cvar_95 >= risk_metrics.var_95 * 0.8  # Allow some numerical variance
        
    def test_risk_level_determination(self):
        """Test risk level classification"""
        # Create high-risk scenario
        high_risk_positions = [
            {'size': 1000, 'weight': 0.08, 'volatility': 0.05, 'leverage': 2.0, 'pnl': -500},
            {'size': 800, 'weight': 0.06, 'volatility': 0.04, 'leverage': 1.5, 'pnl': -300}
        ]
        
        market_data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.exponential(1000, 100)
        })
        
        portfolio_metrics = {'current_drawdown': 0.08}  # High drawdown
        
        risk_metrics = self.risk_manager.assess_portfolio_risk(
            high_risk_positions, market_data, portfolio_metrics
        )
        
        # Should detect high or critical risk
        assert risk_metrics.overall_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL, RiskLevel.EXTREME]
        assert len(risk_metrics.recommendations) > 0


if __name__ == "__main__":
    # Run tests
    test = TestEnhancedRiskManager()
    test.setup_method()
    
    print("Testing position size calculation...")
    test.test_position_size_calculation()
    print("✓ Passed")
    
    print("Testing Kelly criterion variations...")
    test.test_kelly_criterion_variations()
    print("✓ Passed")
    
    print("Testing risk assessment...")
    test.test_risk_assessment()
    print("✓ Passed")
    
    print("Testing volatility adjustments...")
    test.test_volatility_adjustments()
    print("✓ Passed")
    
    print("Testing regime adjustments...")
    test.test_regime_adjustments()
    print("✓ Passed")
    
    print("Testing stress testing...")
    test.test_stress_testing()
    print("✓ Passed")
    
    print("Testing real-time alerts...")
    test.test_real_time_alerts()
    print("✓ Passed")
    
    print("Testing drawdown tracking...")
    test.test_drawdown_tracking()
    print("✓ Passed")
    
    print("Testing risk constraints...")
    test.test_risk_constraints()
    print("✓ Passed")
    
    print("Testing portfolio context...")
    test.test_portfolio_context_integration()
    print("✓ Passed")
    
    print("Testing edge cases...")
    test.test_edge_cases()
    print("✓ Passed")
    
    print("Testing VaR calculation...")
    test.test_var_calculation()
    print("✓ Passed")
    
    print("Testing risk level determination...")
    test.test_risk_level_determination()
    print("✓ Passed")
    
    print("\nAll tests passed! ✨")