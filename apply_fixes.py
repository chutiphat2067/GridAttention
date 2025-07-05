#!/usr/bin/env python3
"""
Apply all priority 1-2 fixes to GridAttention system
"""
import os
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_patches():
    """Apply all patches to the system"""
    
    # 1. Backup original files
    logger.info("Creating backups...")
    files_to_patch = [
        'core/attention_learning_layer.py',
        'data/market_data_input.py',
        'core/execution_engine.py',
        'data/feature_engineering_pipeline.py',
        'core/market_regime_detector.py'
    ]
    
    for file in files_to_patch:
        if Path(file).exists():
            shutil.copy(file, f"{file}.backup")
            logger.info(f"Backed up {file}")
            
    # 2. Copy new files
    logger.info("Copying new components...")
    new_files = [
        ('fixes/memory/memory_manager.py', 'utils/memory_manager.py'),
        ('fixes/error_recovery/resilient_components.py', 'utils/resilient_components.py'),
        ('fixes/validation/validators.py', 'utils/validators.py'),
        ('fixes/performance/optimizations.py', 'utils/optimizations.py')
    ]
    
    for src, dst in new_files:
        if Path(src).exists():
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
            logger.info(f"Copied {src} -> {dst}")
            
    # 3. Update imports
    logger.info("Updating imports...")
    
    # Add imports to utils/__init__.py
    utils_init_path = 'utils/__init__.py'
    if Path(utils_init_path).exists():
        with open(utils_init_path, 'a') as f:
            f.write('\n# Priority 1-2 fixes\n')
            f.write('from .memory_manager import MemoryManager, DataRetentionMixin\n')
            f.write('from .resilient_components import retry_with_backoff, ResilientConnection, CircuitBreaker\n')
            f.write('from .validators import FeatureValidator, OrderValidator, DataIntegrityChecker\n')
            f.write('from .optimizations import PerformanceCache, cached_async, BatchProcessor\n')
    else:
        # Create utils/__init__.py if it doesn't exist
        os.makedirs('utils', exist_ok=True)
        with open(utils_init_path, 'w') as f:
            f.write('# GridAttention utilities\n')
            f.write('from .memory_manager import MemoryManager, DataRetentionMixin\n')
            f.write('from .resilient_components import retry_with_backoff, ResilientConnection, CircuitBreaker\n')
            f.write('from .validators import FeatureValidator, OrderValidator, DataIntegrityChecker\n')
            f.write('from .optimizations import PerformanceCache, cached_async, BatchProcessor\n')
                
    logger.info("Patches applied successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Run: python test_fixes.py")
    logger.info("2. Run: python main.py --config config/config.yaml")
    logger.info("3. Monitor memory usage and performance")
    
if __name__ == '__main__':
    apply_patches()