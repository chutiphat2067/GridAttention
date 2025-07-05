"""
Enhanced Risk Management with Dynamic Position Sizing
Simplified but Complete Implementation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
from scipy import stats

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EXTREME = "extreme"

class DrawdownState(Enum):
    """Drawdown state classification"""
    NORMAL = "normal"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"
    RECOVERY = "recovery"

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics container"""
    position_risk: float
    portfolio_risk: float
    correlation_risk: float
    volatility_risk: float
    drawdown_risk: float
    liquidity_risk: float
    concentration_risk: float
    leverage_risk: float
    overall_risk: RiskLevel
    risk_score: float
    recommendations: List[str]
    risk_budget_usage: float
    var_95: float
    cvar_95: float
    maximum_entropy: float

@dataclass
class PositionSizeRecommendation:
    """Position sizing recommendation"""
    recommended_size: float
    max_safe_size: float
    kelly_size: float
    volatility_adjusted_size: float
    risk_parity_size: float
    confidence: float
    reasoning: List[str]
    constraints_applied: List[str]

@dataclass
class DrawdownMetrics:
    """Drawdown tracking metrics"""
    current_drawdown: float
    max_historical_drawdown: float
    drawdown_duration_days: int
    recovery_factor: float
    drawdown_frequency: float
    underwater_curve: List[float]
    peak_to_valley_ratio: float
    state: DrawdownState

class DrawdownTracker:
    """Track drawdown metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.balance_history = deque(maxlen=1000)
        self.peak_balance = 0
        
    def update(self, current_balance: float):
        """Update with new balance"""
        self.balance_history.append(current_balance)
        self.peak_balance = max(self.peak_balance, current_balance)
        
    def get_current_metrics(self) -> DrawdownMetrics:
        """Get current drawdown metrics"""
        if not self.balance_history:
            return DrawdownMetrics(
                current_drawdown=0,
                max_historical_drawdown=0,
                drawdown_duration_days=0,
                recovery_factor=1.0,
                drawdown_frequency=0,
                underwater_curve=[],
                peak_to_valley_ratio=1.0,
                state=DrawdownState.NORMAL
            )
        
        current_balance = self.balance_history[-1]
        current_dd = (self.peak_balance - current_balance) / self.peak_balance if self.peak_balance > 0 else 0
        
        # Determine state
        if current_dd > 0.1:
            state = DrawdownState.CRITICAL
        elif current_dd > 0.05:
            state = DrawdownState.DANGER
        elif current_dd > 0.02:
            state = DrawdownState.WARNING
        else:
            state = DrawdownState.NORMAL
            
        return DrawdownMetrics(
            current_drawdown=current_dd,
            max_historical_drawdown=current_dd,
            drawdown_duration_days=1,
            recovery_factor=0.5,
            drawdown_frequency=0.1,
            underwater_curve=list(self.balance_history),
            peak_to_valley_ratio=0.9,
            state=state
        )

class EnhancedRiskManager:
    """Simplified Enhanced Risk Manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_limits = {
            'max_single_position': config.get('max_single_position', 0.03),
            'max_daily_drawdown': config.get('max_daily_drawdown', 0.02),
            'var_95_limit': config.get('var_95_limit', 0.025),
        }
        self.drawdown_tracker = DrawdownTracker(config)
    
    def calculate_dynamic_position_size(
        self,
        signal_strength: float,
        market_conditions: Dict[str, Any],
        account_metrics: Dict[str, Any],
        portfolio_context: Optional[Dict[str, Any]] = None
    ) -> PositionSizeRecommendation:
        """Calculate dynamic position size"""
        
        try:
            # Kelly calculation
            win_rate = account_metrics.get('win_rate', 0.52)
            avg_win = account_metrics.get('avg_win', 1.0)
            avg_loss = abs(account_metrics.get('avg_loss', 0.8))
            
            if avg_win > 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = self._adjust_kelly_for_conditions(kelly_fraction, market_conditions, account_metrics)
            else:
                kelly_fraction = 0.001
            
            # Volatility adjustment
            vol_5m = market_conditions.get('volatility_5m', 0.015)
            target_vol = 0.02
            vol_adjusted_size = (target_vol / vol_5m) * 0.01 if vol_5m > 0 else 0.01
            
            # Risk parity
            risk_parity_size = 0.02  # Simple default
            
            # Ensemble sizing
            sizes = [kelly_fraction, vol_adjusted_size, risk_parity_size]
            sizes = [max(0.001, min(s, 0.05)) for s in sizes]  # Bounds
            recommended_size = np.mean(sizes) * signal_strength
            
            # Apply account balance
            balance = account_metrics.get('balance', 10000)
            recommended_size = min(recommended_size * balance, balance * 0.03)
            
            # Confidence calculation
            confidence = min(account_metrics.get('confidence', 0.7), 1.0)
            
            return PositionSizeRecommendation(
                recommended_size=recommended_size,
                max_safe_size=balance * 0.05,
                kelly_size=kelly_fraction * balance,
                volatility_adjusted_size=vol_adjusted_size * balance,
                risk_parity_size=risk_parity_size * balance,
                confidence=confidence,
                reasoning=["Kelly criterion", "Volatility adjustment", "Risk parity"],
                constraints_applied=["Max position limit"]
            )
            
        except Exception as e:
            logger.error(f"Position sizing calculation failed: {e}")
            balance = account_metrics.get('balance', 10000)
            return PositionSizeRecommendation(
                recommended_size=balance * 0.01,
                max_safe_size=balance * 0.05,
                kelly_size=0,
                volatility_adjusted_size=0,
                risk_parity_size=0,
                confidence=0.3,
                reasoning=["Fallback calculation"],
                constraints_applied=["Error recovery"]
            )
    
    def assess_portfolio_risk(
        self,
        positions: List[Dict[str, Any]],
        market_data: pd.DataFrame,
        portfolio_metrics: Optional[Dict[str, Any]] = None
    ) -> RiskMetrics:
        """Assess comprehensive portfolio risk"""
        
        try:
            # Basic risk calculations
            position_risk = self._calculate_position_risk(positions)
            portfolio_risk = self._calculate_portfolio_risk(positions, market_data)
            correlation_risk = self._calculate_correlation_risk(positions)
            volatility_risk = self._calculate_volatility_risk(positions, market_data)
            drawdown_risk = portfolio_metrics.get('current_drawdown', 0) if portfolio_metrics else 0
            liquidity_risk = self._calculate_liquidity_risk(positions, market_data)
            concentration_risk = self._calculate_concentration_risk(positions)
            leverage_risk = self._calculate_leverage_risk(positions)
            
            # Overall risk score
            risk_components = [
                position_risk, portfolio_risk, correlation_risk, volatility_risk,
                drawdown_risk, liquidity_risk, concentration_risk, leverage_risk
            ]
            risk_score = np.mean(risk_components)
            
            # Risk level determination
            if risk_score > 0.8:
                overall_risk = RiskLevel.EXTREME
            elif risk_score > 0.6:
                overall_risk = RiskLevel.CRITICAL
            elif risk_score > 0.4:
                overall_risk = RiskLevel.HIGH
            elif risk_score > 0.2:
                overall_risk = RiskLevel.MEDIUM
            else:
                overall_risk = RiskLevel.LOW
            
            # VaR calculation
            var_95, cvar_95 = self._calculate_var_cvar(positions, market_data)
            
            # Recommendations
            recommendations = self._generate_recommendations(risk_score, positions)
            
            return RiskMetrics(
                position_risk=position_risk,
                portfolio_risk=portfolio_risk,
                correlation_risk=correlation_risk,
                volatility_risk=volatility_risk,
                drawdown_risk=drawdown_risk,
                liquidity_risk=liquidity_risk,
                concentration_risk=concentration_risk,
                leverage_risk=leverage_risk,
                overall_risk=overall_risk,
                risk_score=risk_score,
                recommendations=recommendations,
                risk_budget_usage=risk_score,
                var_95=var_95,
                cvar_95=cvar_95,
                maximum_entropy=0.5
            )
            
        except Exception as e:
            logger.error(f"Portfolio risk assessment failed: {e}")
            return self._get_fallback_risk_metrics()
    
    def stress_test_portfolio(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stress test portfolio"""
        
        scenarios = {}
        
        # Market crash scenario
        scenarios['market_crash'] = {
            'portfolio_pnl': -sum(pos.get('size', 0) * 0.1 for pos in positions),
            'worst_position_pnl': -max(pos.get('size', 0) * 0.15 for pos in positions) if positions else 0,
            'positions_at_risk': len([pos for pos in positions if pos.get('pnl', 0) < 0]),
            'liquidity_impact': 0.3
        }
        
        # Volatility spike
        scenarios['volatility_spike'] = {
            'portfolio_pnl': -sum(pos.get('size', 0) * 0.05 for pos in positions),
            'worst_position_pnl': -max(pos.get('size', 0) * 0.08 for pos in positions) if positions else 0,
            'positions_at_risk': len(positions),
            'liquidity_impact': 0.2
        }
        
        return scenarios
    
    def get_real_time_risk_alerts(
        self,
        positions: List[Dict[str, Any]],
        market_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Generate real-time risk alerts"""
        
        alerts = []
        
        # Check drawdown
        if self.drawdown_tracker.balance_history:
            current_dd = self.drawdown_tracker.get_current_metrics().current_drawdown
            if current_dd > 0.05:
                alerts.append({
                    'type': 'drawdown_critical',
                    'severity': 'critical',
                    'message': f'Critical drawdown: {current_dd:.1%}'
                })
        
        # Check concentration
        if positions:
            max_weight = max(pos.get('weight', 0) for pos in positions)
            if max_weight > 0.05:
                alerts.append({
                    'type': 'concentration_risk',
                    'severity': 'medium',
                    'message': f'High concentration: {max_weight:.1%}'
                })
        
        return alerts
    
    # Helper methods
    def _adjust_kelly_for_conditions(self, kelly_fraction: float, market_conditions: Dict[str, Any], account_metrics: Dict[str, Any]) -> float:
        """Adjust Kelly fraction"""
        adjusted = kelly_fraction
        
        # Volatility adjustment
        vol_ratio = market_conditions.get('volatility_5m', 0.015) / 0.015
        if vol_ratio > 1.5:
            adjusted *= 0.7
        
        # Confidence adjustment
        confidence = account_metrics.get('confidence', 0.7)
        adjusted *= confidence
        
        return max(0.001, min(adjusted, 0.25))
    
    def _calculate_position_risk(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate position risk"""
        if not positions:
            return 0.0
        
        total_weight = sum(pos.get('weight', 0) for pos in positions)
        return min(total_weight / 0.15, 1.0)  # Normalize to 15% max
    
    def _calculate_portfolio_risk(self, positions: List[Dict[str, Any]], market_data: pd.DataFrame) -> float:
        """Calculate portfolio risk"""
        if market_data.empty:
            return 0.3
        
        returns = market_data['close'].pct_change().dropna()
        if len(returns) < 5:
            return 0.3
        
        portfolio_vol = returns.std()
        return min(portfolio_vol / 0.03, 1.0)
    
    def _calculate_correlation_risk(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate correlation risk"""
        if len(positions) < 2:
            return 0.0
        
        # Simple correlation estimation
        correlation_risk = 0
        symbols = [pos.get('symbol', '') for pos in positions]
        
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols[i+1:], i+1):
                if len(sym1) >= 6 and len(sym2) >= 6:
                    if sym1[:3] == sym2[:3]:  # Same base currency
                        weight1 = positions[i].get('weight', 0)
                        weight2 = positions[j].get('weight', 0)
                        correlation_risk += weight1 * weight2 * 0.7
        
        return min(correlation_risk, 1.0)
    
    def _calculate_volatility_risk(self, positions: List[Dict[str, Any]], market_data: pd.DataFrame) -> float:
        """Calculate volatility risk"""
        if market_data.empty or not positions:
            return 0.0
        
        returns = market_data['close'].pct_change().dropna()
        if len(returns) < 10:
            return 0.1
        
        daily_vol = returns.std()
        return min(daily_vol / 0.05, 1.0)
    
    def _calculate_liquidity_risk(self, positions: List[Dict[str, Any]], market_data: pd.DataFrame) -> float:
        """Calculate liquidity risk"""
        if not positions:
            return 0.0
        
        # Simple volume-based liquidity risk
        volume = market_data.get('volume', pd.Series([1000] * len(market_data)))
        if volume.empty:
            return 0.2
        
        avg_volume = volume.mean()
        if avg_volume < 1000:
            return 0.5
        
        return 0.1
    
    def _calculate_concentration_risk(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate concentration risk"""
        if not positions:
            return 0.0
        
        max_weight = max(pos.get('weight', 0) for pos in positions)
        return min(max_weight / 0.05, 1.0)
    
    def _calculate_leverage_risk(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate leverage risk"""
        if not positions:
            return 0.0
        
        total_leverage = sum(pos.get('leverage', 1.0) * pos.get('weight', 0) for pos in positions)
        return min(total_leverage / 3.0, 1.0)
    
    def _calculate_var_cvar(self, positions: List[Dict[str, Any]], market_data: pd.DataFrame) -> Tuple[float, float]:
        """Calculate VaR and CVaR"""
        if market_data.empty:
            return 0.02, 0.025
        
        returns = market_data['close'].pct_change().dropna()
        if len(returns) < 20:
            return 0.02, 0.025
        
        var_95 = np.percentile(returns, 5) * -1  # 5th percentile loss
        cvar_95 = returns[returns <= -var_95].mean() * -1 if (returns <= -var_95).any() else var_95
        
        return max(0, var_95), max(0, cvar_95)
    
    def _generate_recommendations(self, risk_score: float, positions: List[Dict[str, Any]]) -> List[str]:
        """Generate risk recommendations"""
        recommendations = []
        
        if risk_score > 0.6:
            recommendations.append("Reduce position sizes immediately")
            recommendations.append("Review correlation exposure")
        elif risk_score > 0.4:
            recommendations.append("Monitor positions closely")
            recommendations.append("Consider reducing leverage")
        else:
            recommendations.append("Risk levels acceptable")
        
        return recommendations
    
    def _get_fallback_risk_metrics(self) -> RiskMetrics:
        """Get fallback risk metrics on error"""
        return RiskMetrics(
            position_risk=0.3,
            portfolio_risk=0.3,
            correlation_risk=0.2,
            volatility_risk=0.3,
            drawdown_risk=0.1,
            liquidity_risk=0.2,
            concentration_risk=0.2,
            leverage_risk=0.1,
            overall_risk=RiskLevel.MEDIUM,
            risk_score=0.3,
            recommendations=["Error in risk calculation - review manually"],
            risk_budget_usage=0.3,
            var_95=0.02,
            cvar_95=0.025,
            maximum_entropy=0.5
        )