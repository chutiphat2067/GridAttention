#!/bin/bash
echo "🧪 Running GridAttention tests..."

# Check if pytest is available
if command -v pytest &> /dev/null; then
    echo "Using pytest for testing..."
    python -m pytest tests/ -v --tb=short
else
    echo "pytest not found, running tests individually..."
    
    # Run individual test files
    for test_file in tests/test_*.py; do
        if [ -f "$test_file" ]; then
            echo "Running $test_file..."
            python "$test_file"
        fi
    done
    
    # Run final test
    if [ -f "tests/final_test.py" ]; then
        echo "Running final integration test..."
        python tests/final_test.py
    fi
fi

echo "✅ Tests completed!"