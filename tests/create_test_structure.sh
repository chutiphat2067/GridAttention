#!/bin/bash
# create_test_structure.sh - สร้างโครงสร้าง test directories

echo "🏗️ Creating GridAttention Test Structure..."
cd tests/

# Create main directories
echo "📁 Creating main directories..."
mkdir -p unit/{core,utils,models}
mkdir -p integration
mkdir -p functional
mkdir -p performance
mkdir -p security
mkdir -p edge_cases
mkdir -p compliance
mkdir -p monitoring
mkdir -p e2e
mkdir -p fixtures
mkdir -p mocks
mkdir -p utils
mkdir -p reports/{coverage,performance,junit}
mkdir -p scripts

# Create __init__.py files
echo "📝 Creating __init__.py files..."
find . -type d -name "reports" -prune -o -type d -exec touch {}/__init__.py \;

# Create .gitignore for reports
echo "🚫 Creating .gitignore for reports..."
echo "*" > reports/.gitignore
echo "!.gitignore" >> reports/.gitignore

# Create placeholder files for empty directories
echo "📋 Creating placeholder files..."
touch unit/core/.gitkeep
touch unit/utils/.gitkeep
touch unit/models/.gitkeep
touch integration/.gitkeep
touch functional/.gitkeep
touch performance/.gitkeep
touch security/.gitkeep
touch edge_cases/.gitkeep
touch compliance/.gitkeep
touch monitoring/.gitkeep
touch e2e/.gitkeep
touch fixtures/.gitkeep
touch mocks/.gitkeep
touch utils/.gitkeep
touch scripts/.gitkeep

echo "✅ Test structure created successfully!"
echo ""
echo "📊 Directory structure:"
tree -a -I '__pycache__'