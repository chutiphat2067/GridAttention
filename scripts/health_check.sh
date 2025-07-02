#!/bin/bash
echo "🏥 GridAttention System Health Check..."

# Check Python version
echo "🐍 Python version:"
python --version

# Check required directories
echo ""
echo "📁 Directory structure:"
for dir in core infrastructure data monitoring config tests utils; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir/"
    else
        echo "  ❌ $dir/ missing"
    fi
done

# Check main dependencies
echo ""
echo "📦 Dependencies:"
python -c "
try:
    import yaml
    print('  ✅ PyYAML')
except ImportError:
    print('  ❌ PyYAML missing')

try:
    import numpy
    print('  ✅ NumPy')
except ImportError:
    print('  ❌ NumPy missing')

try:
    import pandas
    print('  ✅ Pandas')
except ImportError:
    print('  ❌ Pandas missing')

try:
    import sklearn
    print('  ✅ Scikit-learn')
except ImportError:
    print('  ❌ Scikit-learn missing')
"

# Test imports
echo ""
echo "🔧 Testing imports:"
python -c "
try:
    from core.attention_learning_layer import AttentionLearningLayer
    print('  ✅ Core modules')
except Exception as e:
    print(f'  ❌ Core modules: {e}')

try:
    from infrastructure.system_coordinator import SystemCoordinator
    print('  ✅ Infrastructure modules')
except Exception as e:
    print(f'  ❌ Infrastructure modules: {e}')

try:
    from data.market_data_input import MarketDataInput
    print('  ✅ Data modules')
except Exception as e:
    print(f'  ❌ Data modules: {e}')

try:
    from monitoring.dashboard_integration import integrate_dashboard_optimized
    print('  ✅ Monitoring modules')
except Exception as e:
    print(f'  ❌ Monitoring modules: {e}')
"

# Test main.py
echo ""
echo "🚀 Testing main.py:"
python main.py --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "  ✅ main.py executable"
else
    echo "  ❌ main.py has issues"
fi

echo ""
echo "✅ Health check completed!"