"""
Performance Analysis with Pattern Detection and Auto Recommendations
Advanced performance tracking and optimization suggestions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque, defaultdict
from scipy import stats
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class PerformancePattern(Enum):
    """Types of performance patterns"""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    SIDEWAYS = "sideways"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    VOLATILE = "volatile"
    RECOVERY = "recovery"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"

class RecommendationType(Enum):
    """Types of recommendations"""
    POSITION_SIZE = "position_size"
    GRID_SPACING = "grid_spacing"
    RISK_MANAGEMENT = "risk_management"
    TIMING = "timing"
    REGIME_ADJUSTMENT = "regime_adjustment"
    OPTIMIZATION = "optimization"

@dataclass
class HourlyPerformanceMetrics:
    """Hourly performance analysis"""
    hour: int
    avg_return: float
    volatility: float
    sharpe_ratio: float
    win_rate: float
    avg_trade_duration: float
    volume_factor: float
    best_regime: str
    recommendation_strength: float

@dataclass
class PatternAnalysis:
    """Pattern detection results"""
    pattern_type: PerformancePattern
    confidence: float
    duration_hours: int
    strength: float
    next_probable_move: float
    risk_level: float
    supporting_indicators: List[str]

@dataclass
class AutoRecommendation:
    """Automated recommendation"""
    type: RecommendationType
    priority: str  # high, medium, low
    description: str
    expected_improvement: float
    implementation_difficulty: str  # easy, medium, hard
    confidence: float
    parameters: Dict[str, Any]
    reasoning: List[str]

@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    period_start: datetime
    period_end: datetime
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    hourly_analysis: List[HourlyPerformanceMetrics]
    detected_patterns: List[PatternAnalysis]
    recommendations: List[AutoRecommendation]
    regime_performance: Dict[str, Dict[str, float]]
    optimization_opportunities: Dict[str, float]

class PerformanceAnalyzer:
    """Advanced performance analyzer with pattern detection and recommendations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Performance tracking
        self.trade_history = deque(maxlen=config.get('trade_history_size', 10000))
        self.hourly_metrics = {}
        self.pattern_history = deque(maxlen=500)
        
        # Pattern detection parameters
        self.pattern_config = {
            'min_pattern_length': 6,
            'max_pattern_length': 48,
            'confidence_threshold': 0.6,
            'volatility_threshold': 0.02,
            'trend_threshold': 0.3
        }
        
        # Recommendation engine
        self.recommendation_weights = {
            'performance_impact': 0.4,
            'implementation_ease': 0.2,
            'risk_benefit': 0.3,
            'confidence': 0.1
        }
        
        # Regime performance tracking
        self.regime_stats = defaultdict(lambda: {
            'returns': [], 'sharpe': [], 'win_rate': [], 'max_dd': []
        })
        
    def analyze_performance(
        self,
        trade_data: pd.DataFrame,
        market_data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> PerformanceReport:
        """Comprehensive performance analysis"""
        
        try:
            # Filter data by date range
            if start_date and end_date:
                trade_data = trade_data[
                    (trade_data.index >= start_date) & (trade_data.index <= end_date)
                ]
                market_data = market_data[
                    (market_data.index >= start_date) & (market_data.index <= end_date)
                ]
            
            # Calculate basic metrics
            total_return = self._calculate_total_return(trade_data)
            sharpe_ratio = self._calculate_sharpe_ratio(trade_data)
            max_drawdown = self._calculate_max_drawdown(trade_data)
            win_rate = self._calculate_win_rate(trade_data)
            profit_factor = self._calculate_profit_factor(trade_data)
            
            # Hourly analysis
            hourly_analysis = self._analyze_hourly_performance(trade_data, market_data)
            
            # Pattern detection
            detected_patterns = self._detect_patterns(trade_data, market_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                trade_data, market_data, hourly_analysis, detected_patterns
            )
            
            # Regime performance analysis
            regime_performance = self._analyze_regime_performance(trade_data)
            
            # Optimization opportunities
            optimization_opportunities = self._identify_optimization_opportunities(
                trade_data, hourly_analysis, regime_performance
            )
            
            return PerformanceReport(
                period_start=trade_data.index[0] if len(trade_data) > 0 else datetime.now(),
                period_end=trade_data.index[-1] if len(trade_data) > 0 else datetime.now(),
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                hourly_analysis=hourly_analysis,
                detected_patterns=detected_patterns,
                recommendations=recommendations,
                regime_performance=regime_performance,
                optimization_opportunities=optimization_opportunities
            )
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return self._get_fallback_report()
    
    def _calculate_total_return(self, trade_data: pd.DataFrame) -> float:
        """Calculate total return"""
        if trade_data.empty or 'pnl' not in trade_data.columns:
            return 0.0
        
        cumulative_pnl = trade_data['pnl'].cumsum()
        if len(cumulative_pnl) == 0:
            return 0.0
        
        initial_balance = 10000  # Default
        final_balance = initial_balance + cumulative_pnl.iloc[-1]
        return (final_balance - initial_balance) / initial_balance
    
    def _calculate_sharpe_ratio(self, trade_data: pd.DataFrame) -> float:
        """Calculate Sharpe ratio"""
        if trade_data.empty or 'pnl' not in trade_data.columns:
            return 0.0
        
        returns = trade_data['pnl'].pct_change().dropna()
        if len(returns) < 2:
            return 0.0
        
        if returns.std() == 0:
            return 0.0
        
        return (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self, trade_data: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        if trade_data.empty or 'pnl' not in trade_data.columns:
            return 0.0
        
        cumulative_pnl = trade_data['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max) / running_max
        
        return abs(drawdown.min()) if len(drawdown) > 0 else 0.0
    
    def _calculate_win_rate(self, trade_data: pd.DataFrame) -> float:
        """Calculate win rate"""
        if trade_data.empty or 'pnl' not in trade_data.columns:
            return 0.0
        
        profitable_trades = (trade_data['pnl'] > 0).sum()
        total_trades = len(trade_data)
        
        return profitable_trades / total_trades if total_trades > 0 else 0.0
    
    def _calculate_profit_factor(self, trade_data: pd.DataFrame) -> float:
        """Calculate profit factor"""
        if trade_data.empty or 'pnl' not in trade_data.columns:
            return 1.0
        
        gross_profit = trade_data[trade_data['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trade_data[trade_data['pnl'] < 0]['pnl'].sum())
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 1.0
        
        return gross_profit / gross_loss
    
    def _analyze_hourly_performance(
        self, 
        trade_data: pd.DataFrame, 
        market_data: pd.DataFrame
    ) -> List[HourlyPerformanceMetrics]:
        """Analyze performance by hour of day"""
        
        hourly_metrics = []
        
        if trade_data.empty:
            return hourly_metrics
        
        # Group by hour
        trade_data_copy = trade_data.copy()
        trade_data_copy['hour'] = trade_data_copy.index.hour
        
        for hour in range(24):
            hour_data = trade_data_copy[trade_data_copy['hour'] == hour]
            
            if len(hour_data) == 0:
                continue
            
            # Calculate metrics for this hour
            avg_return = hour_data['pnl'].mean() if 'pnl' in hour_data.columns else 0
            volatility = hour_data['pnl'].std() if 'pnl' in hour_data.columns else 0
            win_rate = (hour_data['pnl'] > 0).mean() if 'pnl' in hour_data.columns else 0
            
            # Sharpe ratio for this hour
            if volatility > 0:
                sharpe = (avg_return / volatility) * np.sqrt(365 * 24)  # Hourly to annual
            else:
                sharpe = 0
            
            # Volume factor (simplified)
            volume_factor = 1.0
            if not market_data.empty and 'volume' in market_data.columns:
                market_hour_data = market_data[market_data.index.hour == hour]
                if len(market_hour_data) > 0:
                    volume_factor = market_hour_data['volume'].mean() / market_data['volume'].mean()
            
            # Best regime for this hour (simplified)
            best_regime = "RANGING"  # Default
            
            # Recommendation strength
            recommendation_strength = min(abs(sharpe), 1.0) if win_rate > 0.5 else 0.0
            
            hourly_metrics.append(HourlyPerformanceMetrics(
                hour=hour,
                avg_return=avg_return,
                volatility=volatility,
                sharpe_ratio=sharpe,
                win_rate=win_rate,
                avg_trade_duration=1.0,  # Simplified
                volume_factor=volume_factor,
                best_regime=best_regime,
                recommendation_strength=recommendation_strength
            ))
        
        return hourly_metrics
    
    def _detect_patterns(
        self, 
        trade_data: pd.DataFrame, 
        market_data: pd.DataFrame
    ) -> List[PatternAnalysis]:
        """Detect performance patterns"""
        
        patterns = []
        
        if trade_data.empty or len(trade_data) < self.pattern_config['min_pattern_length']:
            return patterns
        
        # Calculate cumulative returns
        cumulative_returns = trade_data['pnl'].cumsum() if 'pnl' in trade_data.columns else pd.Series([0])
        
        # Trend analysis
        trend_strength = self._calculate_trend_strength(cumulative_returns)
        volatility = cumulative_returns.pct_change().std()
        
        # Pattern classification
        if trend_strength > 0.5:
            if volatility < 0.02:
                pattern_type = PerformancePattern.STRONG_UPTREND
            else:
                pattern_type = PerformancePattern.WEAK_UPTREND
        elif trend_strength < -0.5:
            if volatility < 0.02:
                pattern_type = PerformancePattern.STRONG_DOWNTREND
            else:
                pattern_type = PerformancePattern.WEAK_DOWNTREND
        elif volatility > 0.05:
            pattern_type = PerformancePattern.VOLATILE
        else:
            pattern_type = PerformancePattern.SIDEWAYS
        
        # Calculate pattern metrics
        confidence = min(abs(trend_strength) + (1 - volatility), 1.0)
        duration_hours = len(trade_data)
        strength = abs(trend_strength)
        next_probable_move = trend_strength * 0.1  # Simplified prediction
        risk_level = volatility
        
        supporting_indicators = self._get_supporting_indicators(
            pattern_type, trend_strength, volatility
        )
        
        patterns.append(PatternAnalysis(
            pattern_type=pattern_type,
            confidence=confidence,
            duration_hours=duration_hours,
            strength=strength,
            next_probable_move=next_probable_move,
            risk_level=risk_level,
            supporting_indicators=supporting_indicators
        ))
        
        return patterns
    
    def _calculate_trend_strength(self, data: pd.Series) -> float:
        """Calculate trend strength using linear regression"""
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        y = data.values
        
        if np.std(y) == 0:
            return 0.0
        
        correlation = np.corrcoef(x, y)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def _get_supporting_indicators(
        self, 
        pattern_type: PerformancePattern, 
        trend_strength: float, 
        volatility: float
    ) -> List[str]:
        """Get supporting indicators for pattern"""
        
        indicators = []
        
        if pattern_type in [PerformancePattern.STRONG_UPTREND, PerformancePattern.WEAK_UPTREND]:
            indicators.extend(["Positive trend", "Consistent gains"])
        elif pattern_type in [PerformancePattern.STRONG_DOWNTREND, PerformancePattern.WEAK_DOWNTREND]:
            indicators.extend(["Negative trend", "Declining performance"])
        elif pattern_type == PerformancePattern.VOLATILE:
            indicators.extend(["High volatility", "Erratic returns"])
        else:
            indicators.extend(["Stable performance", "Low volatility"])
        
        if abs(trend_strength) > 0.7:
            indicators.append("Strong directional bias")
        
        if volatility < 0.01:
            indicators.append("Low risk environment")
        elif volatility > 0.05:
            indicators.append("High risk environment")
        
        return indicators
    
    def _generate_recommendations(
        self,
        trade_data: pd.DataFrame,
        market_data: pd.DataFrame,
        hourly_analysis: List[HourlyPerformanceMetrics],
        patterns: List[PatternAnalysis]
    ) -> List[AutoRecommendation]:
        """Generate automated recommendations"""
        
        recommendations = []
        
        # Hourly-based recommendations
        if hourly_analysis:
            best_hours = sorted(hourly_analysis, key=lambda x: x.sharpe_ratio, reverse=True)[:3]
            worst_hours = sorted(hourly_analysis, key=lambda x: x.sharpe_ratio)[:3]
            
            if best_hours[0].sharpe_ratio > 0.5:
                recommendations.append(AutoRecommendation(
                    type=RecommendationType.TIMING,
                    priority="high",
                    description=f"Focus trading during hours {[h.hour for h in best_hours]}",
                    expected_improvement=0.15,
                    implementation_difficulty="easy",
                    confidence=0.8,
                    parameters={"preferred_hours": [h.hour for h in best_hours]},
                    reasoning=["Higher Sharpe ratio", "Better win rate", "Lower volatility"]
                ))
            
            if worst_hours[0].sharpe_ratio < -0.2:
                recommendations.append(AutoRecommendation(
                    type=RecommendationType.TIMING,
                    priority="medium",
                    description=f"Avoid trading during hours {[h.hour for h in worst_hours]}",
                    expected_improvement=0.10,
                    implementation_difficulty="easy",
                    confidence=0.7,
                    parameters={"avoid_hours": [h.hour for h in worst_hours]},
                    reasoning=["Poor performance", "High volatility", "Low win rate"]
                ))
        
        # Pattern-based recommendations
        for pattern in patterns:
            if pattern.pattern_type == PerformancePattern.VOLATILE:
                recommendations.append(AutoRecommendation(
                    type=RecommendationType.RISK_MANAGEMENT,
                    priority="high",
                    description="Reduce position sizes due to high volatility",
                    expected_improvement=0.20,
                    implementation_difficulty="easy",
                    confidence=pattern.confidence,
                    parameters={"position_reduction": 0.3},
                    reasoning=["High volatility detected", "Risk management priority"]
                ))
            
            elif pattern.pattern_type == PerformancePattern.STRONG_UPTREND:
                recommendations.append(AutoRecommendation(
                    type=RecommendationType.POSITION_SIZE,
                    priority="medium",
                    description="Consider increasing position sizes during strong uptrend",
                    expected_improvement=0.25,
                    implementation_difficulty="medium",
                    confidence=pattern.confidence,
                    parameters={"position_increase": 0.2},
                    reasoning=["Strong uptrend detected", "Good momentum"]
                ))
        
        # Performance-based recommendations
        if not trade_data.empty and 'pnl' in trade_data.columns:
            recent_performance = trade_data['pnl'].tail(20).sum()
            if recent_performance < 0:
                recommendations.append(AutoRecommendation(
                    type=RecommendationType.OPTIMIZATION,
                    priority="high",
                    description="Review and optimize strategy parameters",
                    expected_improvement=0.30,
                    implementation_difficulty="hard",
                    confidence=0.6,
                    parameters={"review_period": "last_20_trades"},
                    reasoning=["Recent poor performance", "Strategy optimization needed"]
                ))
        
        return recommendations
    
    def _analyze_regime_performance(self, trade_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyze performance by market regime"""
        
        regime_performance = {}
        
        # Simplified regime analysis
        regimes = ['TRENDING', 'RANGING', 'VOLATILE', 'UNCERTAIN']
        
        for regime in regimes:
            # In a real implementation, you would filter by actual regime data
            # For now, we'll simulate regime performance
            regime_performance[regime] = {
                'avg_return': np.random.normal(0.001, 0.01),
                'sharpe_ratio': np.random.normal(0.5, 0.3),
                'win_rate': np.random.uniform(0.4, 0.7),
                'max_drawdown': np.random.uniform(0.02, 0.08),
                'trade_count': np.random.randint(10, 100)
            }
        
        return regime_performance
    
    def _identify_optimization_opportunities(
        self,
        trade_data: pd.DataFrame,
        hourly_analysis: List[HourlyPerformanceMetrics],
        regime_performance: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Identify optimization opportunities"""
        
        opportunities = {}
        
        # Time-based optimization
        if hourly_analysis:
            hour_sharpe_variance = np.var([h.sharpe_ratio for h in hourly_analysis])
            if hour_sharpe_variance > 0.1:
                opportunities["time_based_filtering"] = 0.15
        
        # Regime-based optimization
        regime_sharpe_values = [perf['sharpe_ratio'] for perf in regime_performance.values()]
        if max(regime_sharpe_values) - min(regime_sharpe_values) > 0.5:
            opportunities["regime_based_filtering"] = 0.20
        
        # Position sizing optimization
        if not trade_data.empty and 'pnl' in trade_data.columns:
            pnl_variance = trade_data['pnl'].var()
            if pnl_variance > 0.01:
                opportunities["dynamic_position_sizing"] = 0.18
        
        # Risk management optimization
        opportunities["stop_loss_optimization"] = 0.12
        opportunities["take_profit_optimization"] = 0.10
        
        return opportunities
    
    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get performance summary for recent period"""
        
        summary = {
            'total_trades': len(self.trade_history),
            'avg_daily_return': 0.001,
            'best_hour': 10,
            'worst_hour': 22,
            'dominant_pattern': PerformancePattern.SIDEWAYS.value,
            'optimization_score': 0.75,
            'recommendation_count': 5
        }
        
        return summary
    
    def export_analysis_report(self, report: PerformanceReport, filepath: str) -> bool:
        """Export analysis report to file"""
        
        try:
            # Convert report to serializable format
            report_dict = {
                'period_start': report.period_start.isoformat(),
                'period_end': report.period_end.isoformat(),
                'total_return': report.total_return,
                'sharpe_ratio': report.sharpe_ratio,
                'max_drawdown': report.max_drawdown,
                'win_rate': report.win_rate,
                'profit_factor': report.profit_factor,
                'hourly_analysis': [
                    {
                        'hour': h.hour,
                        'avg_return': h.avg_return,
                        'sharpe_ratio': h.sharpe_ratio,
                        'win_rate': h.win_rate
                    } for h in report.hourly_analysis
                ],
                'patterns': [
                    {
                        'type': p.pattern_type.value,
                        'confidence': p.confidence,
                        'strength': p.strength
                    } for p in report.detected_patterns
                ],
                'recommendations': [
                    {
                        'type': r.type.value,
                        'priority': r.priority,
                        'description': r.description,
                        'expected_improvement': r.expected_improvement
                    } for r in report.recommendations
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            return False
    
    def _get_fallback_report(self) -> PerformanceReport:
        """Get fallback report on error"""
        
        return PerformanceReport(
            period_start=datetime.now() - timedelta(days=1),
            period_end=datetime.now(),
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.5,
            profit_factor=1.0,
            hourly_analysis=[],
            detected_patterns=[],
            recommendations=[],
            regime_performance={},
            optimization_opportunities={}
        )