#!/usr/bin/env python3
"""
Configuration Data Test Fixtures for GridAttention Trading System
Provides comprehensive configuration templates for testing all components
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import copy


class TradingMode(Enum):
    """Trading modes for the system"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"
    TEST = "test"


class ConfigProfile(Enum):
    """Pre-defined configuration profiles"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    HIGH_FREQUENCY = "high_frequency"
    LEARNING = "learning"
    TEST = "test"


@dataclass
class AttentionConfig:
    """Configuration for Attention Learning Layer"""
    # Phase settings
    initial_phase: str = "learning"
    min_trades_for_learning: int = 2000
    min_trades_for_shadow: int = 500
    min_trades_for_active: int = 200
    
    # Learning parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    
    # Attention mechanism
    feature_attention_dim: int = 64
    temporal_attention_dim: int = 32
    regime_attention_dim: int = 16
    attention_dropout: float = 0.1
    
    # Regularization
    l2_regularization: float = 0.01
    gradient_clipping: float = 1.0
    max_grad_norm: float = 1.0
    
    # Feature settings
    feature_window_size: int = 100
    max_features: int = 50
    feature_selection_method: str = "mutual_information"
    
    # Memory and performance
    max_memory_size: int = 10000
    checkpoint_interval: int = 100
    enable_gpu: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MarketRegimeConfig:
    """Configuration for Market Regime Detector"""
    # Detection parameters
    lookback_period: int = 200
    update_frequency: str = "5m"
    min_confidence: float = 0.7
    
    # Regime definitions
    trend_threshold: float = 0.02
    volatility_threshold: float = 0.015
    volume_threshold: float = 1.5
    
    # Statistical parameters
    gmm_components: int = 4
    gmm_covariance_type: str = "full"
    gmm_max_iter: int = 100
    gmm_convergence_tol: float = 0.001
    
    # Smoothing
    regime_smoothing_window: int = 5
    confidence_smoothing_alpha: float = 0.3
    
    # Features for regime detection
    regime_features: List[str] = field(default_factory=lambda: [
        "returns_mean", "returns_std", "volume_ratio",
        "high_low_ratio", "close_open_ratio", "trend_strength"
    ])
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class GridStrategyConfig:
    """Configuration for Grid Strategy Selector"""
    # Grid parameters
    grid_type: str = "symmetric"  # symmetric, asymmetric, dynamic
    num_levels: int = 10
    grid_spacing: float = 0.005  # 0.5% between levels
    
    # Position sizing
    position_size_per_level: float = 0.1
    max_position_size: float = 1.0
    use_martingale: bool = False
    martingale_multiplier: float = 1.5
    
    # Order management
    order_type: str = "limit"
    time_in_force: str = "GTC"
    post_only: bool = True
    
    # Grid adjustment
    enable_dynamic_adjustment: bool = True
    adjustment_frequency: str = "1h"
    min_price_movement: float = 0.002  # 0.2% minimum move
    
    # Strategy selection criteria
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        "profit_factor": 0.3,
        "sharpe_ratio": 0.3,
        "max_drawdown": 0.2,
        "win_rate": 0.2
    })
    
    # Regime-specific settings
    regime_configs: Dict[str, Dict] = field(default_factory=lambda: {
        "trending": {"grid_spacing": 0.01, "num_levels": 5},
        "ranging": {"grid_spacing": 0.005, "num_levels": 10},
        "volatile": {"grid_spacing": 0.015, "num_levels": 7}
    })
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RiskManagementConfig:
    """Configuration for Risk Management System"""
    # Position limits
    max_position_size_usd: float = 10000
    max_positions: int = 5
    max_correlation: float = 0.7
    
    # Risk metrics
    max_var_pct: float = 2.0  # Value at Risk
    max_drawdown_pct: float = 10.0
    max_daily_loss_pct: float = 5.0
    
    # Stop loss and take profit
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 5.0
    trailing_stop_pct: float = 1.0
    use_dynamic_stops: bool = True
    
    # Leverage and margin
    max_leverage: float = 3.0
    initial_margin_pct: float = 20.0
    maintenance_margin_pct: float = 10.0
    margin_call_pct: float = 15.0
    
    # Risk per trade
    risk_per_trade_pct: float = 1.0
    kelly_fraction: float = 0.25
    use_kelly_criterion: bool = False
    
    # Emergency settings
    emergency_stop_loss_pct: float = 5.0
    max_consecutive_losses: int = 5
    pause_after_max_losses: int = 3600  # seconds
    
    # Correlation and exposure
    max_sector_exposure_pct: float = 40.0
    correlation_window: int = 100
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExecutionConfig:
    """Configuration for Execution Engine"""
    # Order execution
    order_timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: int = 1
    
    # Slippage and fees
    expected_slippage_bps: float = 10  # basis points
    maker_fee_bps: float = 10
    taker_fee_bps: float = 15
    
    # Rate limiting
    max_orders_per_minute: int = 60
    max_orders_per_hour: int = 1000
    rate_limit_buffer: float = 0.8
    
    # Order routing
    enable_smart_routing: bool = True
    preferred_exchanges: List[str] = field(default_factory=lambda: ["binance", "bybit"])
    
    # Execution algorithms
    use_twap: bool = False
    twap_duration_minutes: int = 5
    use_iceberg: bool = False
    iceberg_visible_pct: float = 20.0
    
    # Latency settings
    max_latency_ms: int = 100
    latency_monitoring: bool = True
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PerformanceMonitorConfig:
    """Configuration for Performance Monitor"""
    # Metrics calculation
    metrics_update_interval: str = "1m"
    rolling_window_size: int = 1000
    
    # Performance metrics
    calculate_sharpe: bool = True
    calculate_sortino: bool = True
    calculate_calmar: bool = True
    risk_free_rate: float = 0.02
    
    # Benchmarks
    benchmark_symbol: str = "BTC/USDT"
    benchmark_return_period: str = "1d"
    
    # Alerts
    enable_alerts: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["log", "email"])
    
    # Alert thresholds
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "drawdown_pct": 5.0,
        "daily_loss_pct": 3.0,
        "win_rate_min": 0.4,
        "sharpe_ratio_min": 0.5
    })
    
    # Storage
    store_metrics_history: bool = True
    metrics_retention_days: int = 90
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OverfittingDetectorConfig:
    """Configuration for Overfitting Detector"""
    # Detection parameters
    lookback_window: int = 500
    validation_ratio: float = 0.3
    detection_threshold: float = 0.15
    
    # Statistical tests
    enable_distribution_test: bool = True
    enable_autocorrelation_test: bool = True
    enable_stability_test: bool = True
    
    # Regularization
    auto_regularization: bool = True
    regularization_strength: float = 0.1
    dropout_increase_step: float = 0.05
    max_dropout: float = 0.5
    
    # Model complexity
    complexity_penalty: float = 0.01
    max_feature_correlation: float = 0.9
    
    # Alerts
    alert_on_detection: bool = True
    auto_pause_on_severe: bool = True
    severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.1,
        "medium": 0.2,
        "high": 0.3,
        "critical": 0.5
    })
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FeedbackLoopConfig:
    """Configuration for Feedback Loop"""
    # Update frequencies
    performance_update_freq: str = "5m"
    parameter_update_freq: str = "1h"
    model_retrain_freq: str = "1d"
    
    # Adaptation rates
    learning_rate_adaptation: float = 0.001
    parameter_adaptation_rate: float = 0.01
    
    # Feedback weights
    feedback_weights: Dict[str, float] = field(default_factory=lambda: {
        "profit_loss": 0.4,
        "risk_adjusted_return": 0.3,
        "prediction_accuracy": 0.2,
        "execution_quality": 0.1
    })
    
    # Constraints
    max_parameter_change_pct: float = 10.0
    min_sample_size: int = 100
    confidence_threshold: float = 0.8
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DataConfig:
    """Configuration for Data Processing"""
    # Market data
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h"])
    
    # Data collection
    history_days: int = 30
    real_time_buffer_size: int = 1000
    
    # Feature engineering
    technical_indicators: List[str] = field(default_factory=lambda: [
        "sma", "ema", "rsi", "macd", "bbands", "atr", "adx"
    ])
    
    # Data quality
    max_missing_pct: float = 5.0
    outlier_std_threshold: float = 4.0
    
    # Storage
    use_database: bool = True
    database_type: str = "postgresql"
    cache_size_mb: int = 1000
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SystemConfig:
    """Main system configuration"""
    # Component configs
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    market_regime: MarketRegimeConfig = field(default_factory=MarketRegimeConfig)
    grid_strategy: GridStrategyConfig = field(default_factory=GridStrategyConfig)
    risk_management: RiskManagementConfig = field(default_factory=RiskManagementConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    performance_monitor: PerformanceMonitorConfig = field(default_factory=PerformanceMonitorConfig)
    overfitting_detector: OverfittingDetectorConfig = field(default_factory=OverfittingDetectorConfig)
    feedback_loop: FeedbackLoopConfig = field(default_factory=FeedbackLoopConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # System settings
    mode: str = "test"
    debug: bool = True
    log_level: str = "INFO"
    
    # API settings
    exchange: str = "binance"
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    
    # Performance settings
    enable_profiling: bool = False
    max_cpu_cores: int = 4
    max_memory_gb: int = 8
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "attention": self.attention.to_dict(),
            "market_regime": self.market_regime.to_dict(),
            "grid_strategy": self.grid_strategy.to_dict(),
            "risk_management": self.risk_management.to_dict(),
            "execution": self.execution.to_dict(),
            "performance_monitor": self.performance_monitor.to_dict(),
            "overfitting_detector": self.overfitting_detector.to_dict(),
            "feedback_loop": self.feedback_loop.to_dict(),
            "data": self.data.to_dict(),
            "mode": self.mode,
            "debug": self.debug,
            "log_level": self.log_level,
            "exchange": self.exchange,
            "api_key": self.api_key,
            "api_secret": self.api_secret,
            "testnet": self.testnet,
            "enable_profiling": self.enable_profiling,
            "max_cpu_cores": self.max_cpu_cores,
            "max_memory_gb": self.max_memory_gb
        }
    
    def to_yaml(self, file_path: Optional[Path] = None) -> str:
        """Convert to YAML format"""
        yaml_str = yaml.dump(self.to_dict(), default_flow_style=False)
        if file_path:
            with open(file_path, 'w') as f:
                f.write(yaml_str)
        return yaml_str
    
    def to_json(self, file_path: Optional[Path] = None) -> str:
        """Convert to JSON format"""
        json_str = json.dumps(self.to_dict(), indent=2)
        if file_path:
            with open(file_path, 'w') as f:
                f.write(json_str)
        return json_str


class ConfigFactory:
    """Factory for creating pre-configured test configurations"""
    
    @staticmethod
    def create_config(profile: ConfigProfile = ConfigProfile.TEST) -> SystemConfig:
        """Create configuration based on profile"""
        
        if profile == ConfigProfile.CONSERVATIVE:
            return ConfigFactory._create_conservative_config()
        elif profile == ConfigProfile.AGGRESSIVE:
            return ConfigFactory._create_aggressive_config()
        elif profile == ConfigProfile.HIGH_FREQUENCY:
            return ConfigFactory._create_high_frequency_config()
        elif profile == ConfigProfile.LEARNING:
            return ConfigFactory._create_learning_config()
        else:
            return ConfigFactory._create_test_config()
    
    @staticmethod
    def _create_test_config() -> SystemConfig:
        """Create test configuration"""
        config = SystemConfig()
        
        # Optimize for testing
        config.attention.min_trades_for_learning = 100
        config.attention.min_trades_for_shadow = 50
        config.attention.min_trades_for_active = 20
        config.attention.epochs = 10
        
        config.risk_management.max_position_size_usd = 1000
        config.risk_management.max_drawdown_pct = 20.0
        
        config.mode = "test"
        config.debug = True
        
        return config
    
    @staticmethod
    def _create_conservative_config() -> SystemConfig:
        """Create conservative configuration"""
        config = SystemConfig()
        
        # Conservative risk settings
        config.risk_management.max_position_size_usd = 5000
        config.risk_management.max_positions = 3
        config.risk_management.max_leverage = 1.0
        config.risk_management.stop_loss_pct = 1.0
        config.risk_management.max_drawdown_pct = 5.0
        config.risk_management.risk_per_trade_pct = 0.5
        
        # Conservative grid
        config.grid_strategy.num_levels = 5
        config.grid_strategy.grid_spacing = 0.01
        
        # Slower adaptation
        config.feedback_loop.parameter_adaptation_rate = 0.005
        
        return config
    
    @staticmethod
    def _create_aggressive_config() -> SystemConfig:
        """Create aggressive configuration"""
        config = SystemConfig()
        
        # Aggressive settings
        config.risk_management.max_position_size_usd = 50000
        config.risk_management.max_positions = 10
        config.risk_management.max_leverage = 5.0
        config.risk_management.stop_loss_pct = 5.0
        config.risk_management.max_drawdown_pct = 20.0
        config.risk_management.risk_per_trade_pct = 2.0
        
        # Dense grid
        config.grid_strategy.num_levels = 20
        config.grid_strategy.grid_spacing = 0.003
        
        # Faster adaptation
        config.feedback_loop.parameter_adaptation_rate = 0.02
        config.attention.learning_rate = 0.01
        
        return config
    
    @staticmethod
    def _create_high_frequency_config() -> SystemConfig:
        """Create high-frequency trading configuration"""
        config = SystemConfig()
        
        # HFT settings
        config.execution.max_orders_per_minute = 300
        config.execution.max_latency_ms = 10
        config.execution.expected_slippage_bps = 5
        
        # Fast updates
        config.market_regime.update_frequency = "1m"
        config.performance_monitor.metrics_update_interval = "10s"
        config.feedback_loop.performance_update_freq = "1m"
        
        # Tight risk
        config.risk_management.stop_loss_pct = 0.5
        config.risk_management.max_drawdown_pct = 3.0
        
        # Small positions
        config.grid_strategy.position_size_per_level = 0.01
        
        return config
    
    @staticmethod
    def _create_learning_config() -> SystemConfig:
        """Create configuration optimized for learning phase"""
        config = SystemConfig()
        
        # Extended learning
        config.attention.min_trades_for_learning = 5000
        config.attention.min_trades_for_shadow = 2000
        config.attention.min_trades_for_active = 1000
        config.attention.epochs = 200
        config.attention.validation_split = 0.3
        
        # More conservative during learning
        config.risk_management.max_position_size_usd = 1000
        config.risk_management.risk_per_trade_pct = 0.1
        
        # Enhanced monitoring
        config.overfitting_detector.detection_threshold = 0.1
        config.overfitting_detector.auto_regularization = True
        
        return config


def create_test_config(
    profile: ConfigProfile = ConfigProfile.TEST,
    overrides: Optional[Dict] = None
) -> SystemConfig:
    """Convenience function to create test configuration"""
    config = ConfigFactory.create_config(profile)
    
    if overrides:
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return config


def load_config_from_file(file_path: Path) -> SystemConfig:
    """Load configuration from file"""
    with open(file_path, 'r') as f:
        if file_path.suffix == '.yaml':
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    
    # Reconstruct config object
    config = SystemConfig()
    
    # Map data to config attributes
    for component, settings in data.items():
        if hasattr(config, component) and isinstance(settings, dict):
            component_config = getattr(config, component)
            for key, value in settings.items():
                if hasattr(component_config, key):
                    setattr(component_config, key, value)
        elif hasattr(config, component):
            setattr(config, component, settings)
    
    return config


# Pre-defined test configurations
TEST_CONFIGS = {
    "minimal": {
        "attention": {"min_trades_for_learning": 10},
        "risk_management": {"max_position_size_usd": 100},
        "mode": "test"
    },
    "integration": {
        "attention": {"min_trades_for_learning": 100},
        "market_regime": {"lookback_period": 50},
        "grid_strategy": {"num_levels": 5},
        "mode": "test"
    },
    "stress": {
        "risk_management": {"max_drawdown_pct": 50.0},
        "execution": {"max_orders_per_minute": 1000},
        "mode": "test"
    },
    "production_like": {
        "attention": {"min_trades_for_learning": 2000},
        "risk_management": {"max_drawdown_pct": 10.0},
        "execution": {"max_latency_ms": 100},
        "mode": "paper"
    }
}


# Configuration validation
def validate_config(config: SystemConfig) -> List[str]:
    """Validate configuration and return list of issues"""
    issues = []
    
    # Risk management validation
    if config.risk_management.stop_loss_pct >= config.risk_management.max_drawdown_pct:
        issues.append("Stop loss should be less than max drawdown")
    
    if config.risk_management.max_leverage > 10:
        issues.append("Leverage seems dangerously high")
    
    # Attention validation
    if config.attention.min_trades_for_active > config.attention.min_trades_for_shadow:
        issues.append("Active phase requires fewer trades than shadow phase")
    
    # Grid validation
    if config.grid_strategy.num_levels < 3:
        issues.append("Grid should have at least 3 levels")
    
    return issues


if __name__ == "__main__":
    # Example usage
    
    # Create test config
    test_config = create_test_config()
    print("Test Config:")
    print(json.dumps(test_config.to_dict(), indent=2))
    
    # Create conservative config
    conservative = create_test_config(ConfigProfile.CONSERVATIVE)
    print("\nConservative Risk Settings:")
    print(f"Max Position: ${conservative.risk_management.max_position_size_usd}")
    print(f"Max Leverage: {conservative.risk_management.max_leverage}x")
    print(f"Stop Loss: {conservative.risk_management.stop_loss_pct}%")
    
    # Save to file
    test_config.to_yaml(Path("test_config.yaml"))
    
    # Validate config
    issues = validate_config(test_config)
    if issues:
        print("\nConfiguration Issues:")
        for issue in issues:
            print(f"- {issue}")