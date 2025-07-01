# GridAttention Trading System - Project Knowledge

## Core Architecture
```
GridAttention/
├── Core System
│   ├── attention_learning_layer.py      # AI attention mechanism + overfitting prevention
│   ├── market_regime_detector.py        # Market regime classification
│   ├── grid_strategy_selector.py        # Strategy selection with validation
│   ├── execution_engine.py              # Trade execution
│   ├── risk_management_system.py        # Risk controls
│   ├── performance_monitor.py           # Performance tracking
│   ├── feedback_loop.py                 # Learning & adaptation
│   └── overfitting_detector.py          # Overfitting prevention
├── Data Processing
│   ├── market_data_input.py             # Market data ingestion
│   ├── feature_engineering_pipeline.py  # Feature creation
│   └── data_augmentation.py             # Data enhancement
├── Scaling & Monitoring
│   ├── scaling_api.py                   # API endpoints
│   ├── scaling_monitor.py               # System monitoring
│   └── scaling_dashboard.html           # Web dashboard
├── Configuration
│   ├── config.yaml                      # System configuration
│   └── requirements.txt                 # Dependencies
├── Warmup System
│   ├── warm_up_system.py                # System warmup
│   ├── warm_up_config.py                # Warmup configuration
│   ├── warmup_main.py                   # Warmup execution
│   └── warmup_main.ipynb                # Jupyter notebook
└── Testing
    └── testing_files/
        └── test_warmup_integration.py   # Integration tests
```

## Key Features Implemented
- **Overfitting Prevention**: Integrated across all components
- **Cross-validation**: Strategy selection & parameter tuning
- **Conservative Learning**: Gradual parameter adjustment
- **Regime Detection**: Market condition classification
- **Risk Management**: Multi-layer risk controls
- **Performance Monitoring**: Real-time metrics
- **Scaling Infrastructure**: API & monitoring

## Latest Updates (Commit: 0f5b280)
- Added OverfittingDetector integration
- Enhanced attention layer with batch normalization
- Implemented conservative parameter adjustment
- Added cross-validation for strategy selection
- Enhanced feedback loop with cooldown mechanisms

## Dependencies
```python
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
aiohttp>=3.8.0
PyYAML>=6.0
asyncio
logging
```

## Usage Pattern
```python
# Main execution flow
python main.py

# Warmup system
python warmup_main.py

# API server
python scaling_api.py
```

## Architecture Notes
- **Async-first design**: All components use asyncio
- **Event-driven**: Components communicate via feedback loop
- **Modular**: Each component can be tested independently
- **Validation-heavy**: Multiple validation layers prevent overfitting
- **Conservative**: Parameters adjust gradually with safeguards