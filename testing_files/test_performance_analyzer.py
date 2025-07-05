"""
Test Performance Analyzer Module
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.performance_analyzer import (
    PerformanceAnalyzer, PerformancePattern, RecommendationType,
    HourlyPerformanceMetrics, PatternAnalysis, AutoRecommendation,
    PerformanceReport
)

class TestPerformanceAnalyzer:
    
    def setup_method(self):
        """Set up test environment before each test"""
        self.config = {
            'trade_history_size': 1000,
            'pattern_detection': True,
            'auto_recommendations': True
        }
        self.analyzer = PerformanceAnalyzer(self.config)
        
    def create_sample_trade_data(self, days: int = 30):
        """Create sample trade data"""
        
        # Generate realistic trading data
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                             end=datetime.now(), freq='1H')
        
        # Random walk with some trend and volatility
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Small positive drift
        
        # Add some hour-based patterns
        for i, date in enumerate(dates):
            hour = date.hour
            if 8 <= hour <= 10:  # Morning session boost
                returns[i] += 0.001
            elif 22 <= hour <= 23:  # Late night decline
                returns[i] -= 0.001
        
        cumulative_pnl = np.cumsum(returns) * 100  # Scale to reasonable PnL
        
        trade_data = pd.DataFrame({
            'pnl': returns * 100,
            'cumulative_pnl': cumulative_pnl,
            'volume': np.random.exponential(1000, len(dates)),
            'regime': np.random.choice(['TRENDING', 'RANGING', 'VOLATILE'], len(dates))
        }, index=dates)
        
        return trade_data
        
    def create_sample_market_data(self, days: int = 30):
        """Create sample market data"""
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                             end=datetime.now(), freq='1H')
        
        # Generate price data
        returns = np.random.normal(0, 0.01, len(dates))
        prices = 100 * (1 + returns).cumprod()
        
        market_data = pd.DataFrame({
            'close': prices,
            'volume': np.random.exponential(1000, len(dates)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates))))
        }, index=dates)
        
        return market_data
        
    def test_basic_performance_analysis(self):
        """Test basic performance analysis"""
        
        trade_data = self.create_sample_trade_data(30)
        market_data = self.create_sample_market_data(30)
        
        report = self.analyzer.analyze_performance(trade_data, market_data)
        
        assert isinstance(report, PerformanceReport)
        assert isinstance(report.total_return, float)
        assert isinstance(report.sharpe_ratio, float)
        assert isinstance(report.max_drawdown, float)
        assert 0 <= report.win_rate <= 1
        assert report.profit_factor >= 0
        assert isinstance(report.hourly_analysis, list)
        assert isinstance(report.detected_patterns, list)
        assert isinstance(report.recommendations, list)
        
        print(f"Total return: {report.total_return:.3f}")
        print(f"Sharpe ratio: {report.sharpe_ratio:.3f}")
        print(f"Max drawdown: {report.max_drawdown:.3f}")
        print(f"Win rate: {report.win_rate:.3f}")
        print(f"Profit factor: {report.profit_factor:.3f}")
        
    def test_hourly_analysis(self):
        """Test hourly performance analysis"""
        
        trade_data = self.create_sample_trade_data(15)
        market_data = self.create_sample_market_data(15)
        
        report = self.analyzer.analyze_performance(trade_data, market_data)
        
        assert len(report.hourly_analysis) > 0
        
        for hour_metrics in report.hourly_analysis:
            assert isinstance(hour_metrics, HourlyPerformanceMetrics)
            assert 0 <= hour_metrics.hour <= 23
            assert isinstance(hour_metrics.avg_return, float)
            assert hour_metrics.volatility >= 0
            assert 0 <= hour_metrics.win_rate <= 1
            assert 0 <= hour_metrics.recommendation_strength <= 1
            
        # Check if we have good coverage of hours
        hours_covered = [h.hour for h in report.hourly_analysis]
        print(f"Hours analyzed: {sorted(hours_covered)}")
        
        # Find best and worst performing hours
        best_hour = max(report.hourly_analysis, key=lambda x: x.sharpe_ratio)
        worst_hour = min(report.hourly_analysis, key=lambda x: x.sharpe_ratio)
        
        print(f"Best hour: {best_hour.hour} (Sharpe: {best_hour.sharpe_ratio:.3f})")
        print(f"Worst hour: {worst_hour.hour} (Sharpe: {worst_hour.sharpe_ratio:.3f})")
        
    def test_pattern_detection(self):
        """Test pattern detection functionality"""
        
        # Create trending data
        dates = pd.date_range(start=datetime.now() - timedelta(days=10), 
                             end=datetime.now(), freq='1H')
        
        # Strong uptrend
        trend_returns = np.random.normal(0.002, 0.01, len(dates))  # Positive drift
        trending_data = pd.DataFrame({
            'pnl': trend_returns * 100,
            'cumulative_pnl': np.cumsum(trend_returns) * 100
        }, index=dates)
        
        market_data = self.create_sample_market_data(10)
        
        report = self.analyzer.analyze_performance(trending_data, market_data)
        
        assert len(report.detected_patterns) > 0
        
        for pattern in report.detected_patterns:
            assert isinstance(pattern, PatternAnalysis)
            assert pattern.pattern_type in PerformancePattern
            assert 0 <= pattern.confidence <= 1
            assert pattern.duration_hours > 0
            assert pattern.strength >= 0
            assert pattern.risk_level >= 0
            assert isinstance(pattern.supporting_indicators, list)
            
        print(f"Detected patterns: {[p.pattern_type.value for p in report.detected_patterns]}")
        
        # Test with volatile data
        volatile_returns = np.random.normal(0, 0.05, len(dates))  # High volatility
        volatile_data = pd.DataFrame({
            'pnl': volatile_returns * 100,
            'cumulative_pnl': np.cumsum(volatile_returns) * 100
        }, index=dates)
        
        volatile_report = self.analyzer.analyze_performance(volatile_data, market_data)
        print(f"Volatile patterns: {[p.pattern_type.value for p in volatile_report.detected_patterns]}")
        
    def test_auto_recommendations(self):
        """Test automatic recommendation generation"""
        
        # Create data with clear patterns
        trade_data = self.create_sample_trade_data(20)
        market_data = self.create_sample_market_data(20)
        
        report = self.analyzer.analyze_performance(trade_data, market_data)
        
        assert isinstance(report.recommendations, list)
        
        for recommendation in report.recommendations:
            assert isinstance(recommendation, AutoRecommendation)
            assert recommendation.type in RecommendationType
            assert recommendation.priority in ['low', 'medium', 'high']
            assert len(recommendation.description) > 0
            assert recommendation.expected_improvement >= 0
            assert recommendation.implementation_difficulty in ['easy', 'medium', 'hard']
            assert 0 <= recommendation.confidence <= 1
            assert isinstance(recommendation.parameters, dict)
            assert isinstance(recommendation.reasoning, list)
            
        print(f"Generated {len(report.recommendations)} recommendations:")
        for rec in report.recommendations:
            print(f"- {rec.type.value}: {rec.description} (Priority: {rec.priority})")
            
    def test_regime_performance_analysis(self):
        """Test regime-based performance analysis"""
        
        trade_data = self.create_sample_trade_data(25)
        market_data = self.create_sample_market_data(25)
        
        report = self.analyzer.analyze_performance(trade_data, market_data)
        
        assert isinstance(report.regime_performance, dict)
        assert len(report.regime_performance) > 0
        
        for regime, metrics in report.regime_performance.items():
            assert isinstance(metrics, dict)
            assert 'avg_return' in metrics
            assert 'sharpe_ratio' in metrics
            assert 'win_rate' in metrics
            assert 'max_drawdown' in metrics
            assert 'trade_count' in metrics
            
        print("Regime performance:")
        for regime, metrics in report.regime_performance.items():
            print(f"{regime}: Return={metrics['avg_return']:.4f}, "
                  f"Sharpe={metrics['sharpe_ratio']:.3f}, "
                  f"WinRate={metrics['win_rate']:.3f}")
            
    def test_optimization_opportunities(self):
        """Test optimization opportunity identification"""
        
        trade_data = self.create_sample_trade_data(30)
        market_data = self.create_sample_market_data(30)
        
        report = self.analyzer.analyze_performance(trade_data, market_data)
        
        assert isinstance(report.optimization_opportunities, dict)
        assert len(report.optimization_opportunities) > 0
        
        for opportunity, improvement in report.optimization_opportunities.items():
            assert isinstance(opportunity, str)
            assert isinstance(improvement, float)
            assert improvement >= 0
            
        print("Optimization opportunities:")
        for opp, improvement in report.optimization_opportunities.items():
            print(f"- {opp}: {improvement:.1%} potential improvement")
            
    def test_performance_summary(self):
        """Test performance summary generation"""
        
        summary = self.analyzer.get_performance_summary(days=30)
        
        assert isinstance(summary, dict)
        assert 'total_trades' in summary
        assert 'avg_daily_return' in summary
        assert 'best_hour' in summary
        assert 'worst_hour' in summary
        assert 'dominant_pattern' in summary
        assert 'optimization_score' in summary
        assert 'recommendation_count' in summary
        
        print("Performance summary:")
        for key, value in summary.items():
            print(f"- {key}: {value}")
            
    def test_report_export(self):
        """Test report export functionality"""
        
        trade_data = self.create_sample_trade_data(10)
        market_data = self.create_sample_market_data(10)
        
        report = self.analyzer.analyze_performance(trade_data, market_data)
        
        # Test export
        export_path = "testing_files/test_performance_report.json"
        success = self.analyzer.export_analysis_report(report, export_path)
        
        assert success is True
        
        # Check if file exists
        assert os.path.exists(export_path)
        
        # Read and verify content
        import json
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
            
        assert 'total_return' in exported_data
        assert 'sharpe_ratio' in exported_data
        assert 'hourly_analysis' in exported_data
        assert 'patterns' in exported_data
        assert 'recommendations' in exported_data
        
        # Cleanup
        os.remove(export_path)
        
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        
        # Empty data
        empty_data = pd.DataFrame()
        market_data = self.create_sample_market_data(5)
        
        report_empty = self.analyzer.analyze_performance(empty_data, market_data)
        assert isinstance(report_empty, PerformanceReport)
        
        # Very short data
        short_data = self.create_sample_trade_data(1)
        report_short = self.analyzer.analyze_performance(short_data, market_data)
        assert isinstance(report_short, PerformanceReport)
        
        # Data with NaN values
        nan_data = self.create_sample_trade_data(5)
        nan_data.iloc[2:4] = np.nan
        report_nan = self.analyzer.analyze_performance(nan_data, market_data)
        assert isinstance(report_nan, PerformanceReport)
        
        print("Edge cases handled successfully")
        
    def test_calculation_methods(self):
        """Test individual calculation methods"""
        
        # Test total return calculation
        simple_data = pd.DataFrame({
            'pnl': [10, -5, 15, -3, 8]
        })
        
        total_return = self.analyzer._calculate_total_return(simple_data)
        expected_return = 25 / 10000  # Sum of PnL / initial balance
        assert abs(total_return - expected_return) < 0.001
        
        # Test win rate calculation
        win_rate = self.analyzer._calculate_win_rate(simple_data)
        expected_win_rate = 3/5  # 3 profitable trades out of 5
        assert abs(win_rate - expected_win_rate) < 0.001
        
        # Test profit factor calculation
        profit_factor = self.analyzer._calculate_profit_factor(simple_data)
        gross_profit = 10 + 15 + 8  # 33
        gross_loss = 5 + 3  # 8
        expected_pf = gross_profit / gross_loss
        assert abs(profit_factor - expected_pf) < 0.001
        
        print(f"Total return: {total_return:.4f}")
        print(f"Win rate: {win_rate:.3f}")
        print(f"Profit factor: {profit_factor:.3f}")


if __name__ == "__main__":
    # Run tests
    test = TestPerformanceAnalyzer()
    test.setup_method()
    
    print("Testing basic performance analysis...")
    test.test_basic_performance_analysis()
    print("✓ Passed")
    
    print("Testing hourly analysis...")
    test.test_hourly_analysis()
    print("✓ Passed")
    
    print("Testing pattern detection...")
    test.test_pattern_detection()
    print("✓ Passed")
    
    print("Testing auto recommendations...")
    test.test_auto_recommendations()
    print("✓ Passed")
    
    print("Testing regime performance...")
    test.test_regime_performance_analysis()
    print("✓ Passed")
    
    print("Testing optimization opportunities...")
    test.test_optimization_opportunities()
    print("✓ Passed")
    
    print("Testing performance summary...")
    test.test_performance_summary()
    print("✓ Passed")
    
    print("Testing report export...")
    test.test_report_export()
    print("✓ Passed")
    
    print("Testing edge cases...")
    test.test_edge_cases()
    print("✓ Passed")
    
    print("Testing calculation methods...")
    test.test_calculation_methods()
    print("✓ Passed")
    
    print("\nAll tests passed! ✨")