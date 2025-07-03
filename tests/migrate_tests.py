#!/usr/bin/env python3
"""
Migrate existing test files to new organized structure
"""

import os
import shutil
from pathlib import Path

def migrate_test_files():
    """ย้าย test files ไปยัง structure ใหม่"""
    
    print("🚚 Migrating test files to new structure...")
    
    # Current directory
    tests_dir = Path("tests")
    
    # Migration mapping
    migrations = {
        # Integration Tests
        'test_integration.py': 'integration/test_component_integration.py',
        'test_warmup_integration.py': 'integration/test_warmup_integration.py',
        'test_phase_augmentation.py': 'integration/test_phase_transitions.py',
        
        # Functional Tests
        'test_gridattention_complete.py': 'functional/test_trading_scenarios.py',
        'test_market_scenarios.py': 'functional/test_market_scenarios.py',
        
        # Performance Tests
        'test_stress_load.py': 'performance/test_stress_load.py',
        
        # Security Tests
        'test_security.py': 'security/test_security.py',
        
        # Edge Cases
        'test_edge_recovery.py': 'edge_cases/test_edge_recovery.py',
        
        # Compliance Tests
        'test_compliance.py': 'compliance/test_compliance.py',
        
        # Monitoring Tests
        'test_augmentation_monitoring.py': 'monitoring/test_augmentation_monitoring.py',
        'test_monitoring_alerts.py': 'monitoring/test_monitoring_alerts.py',
        
        # E2E Tests
        'final_test.py': 'e2e/test_full_trading_cycle.py',
        'test_system_validation.py': 'e2e/test_system_validation.py',
        
        # Unit Tests (core)
        'test_overfitting_detection.py': 'unit/core/test_overfitting_detection.py',
        
        # Scripts
        'run_all_tests.py': 'scripts/run_all_tests.py'
    }
    
    migrated_count = 0
    
    for old_path, new_path in migrations.items():
        old_file = tests_dir / old_path
        new_file = tests_dir / new_path
        
        if old_file.exists():
            try:
                # Create directory if it doesn't exist
                new_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file to new location
                shutil.copy2(old_file, new_file)
                
                print(f"   ✅ {old_path} → {new_path}")
                migrated_count += 1
                
            except Exception as e:
                print(f"   ❌ Failed to migrate {old_path}: {e}")
        else:
            print(f"   ⚠️  {old_path} - file not found")
    
    print(f"\n📊 Migration Summary:")
    print(f"   Migrated: {migrated_count}/{len(migrations)} files")
    
    return migrated_count

def cleanup_old_files():
    """ลบ test files เก่าหลังจากย้ายแล้ว"""
    
    print("\n🧹 Cleaning up old test files...")
    
    tests_dir = Path("tests")
    
    # Files to remove (already migrated)
    files_to_remove = [
        'test_integration.py',
        'test_warmup_integration.py', 
        'test_phase_augmentation.py',
        'test_gridattention_complete.py',
        'test_market_scenarios.py',
        'test_stress_load.py',
        'test_security.py',
        'test_edge_recovery.py',
        'test_compliance.py',
        'test_augmentation_monitoring.py',
        'test_monitoring_alerts.py',
        'final_test.py',
        'test_system_validation.py',
        'test_overfitting_detection.py',
        'run_all_tests.py'
    ]
    
    removed_count = 0
    
    for file_name in files_to_remove:
        file_path = tests_dir / file_name
        if file_path.exists():
            try:
                os.remove(file_path)
                print(f"   ✅ Removed {file_name}")
                removed_count += 1
            except Exception as e:
                print(f"   ❌ Failed to remove {file_name}: {e}")
    
    print(f"\n   Removed: {removed_count}/{len(files_to_remove)} files")
    
    return removed_count

def verify_migration():
    """ตรวจสอบว่าการย้ายไฟล์เสร็จสมบูรณ์"""
    
    print("\n🔍 Verifying migration...")
    
    tests_dir = Path("tests")
    
    # Check new structure
    expected_files = [
        'integration/test_component_integration.py',
        'integration/test_warmup_integration.py',
        'integration/test_phase_transitions.py',
        'functional/test_trading_scenarios.py',
        'functional/test_market_scenarios.py',
        'performance/test_stress_load.py',
        'security/test_security.py',
        'edge_cases/test_edge_recovery.py',
        'compliance/test_compliance.py',
        'monitoring/test_augmentation_monitoring.py',
        'monitoring/test_monitoring_alerts.py',
        'e2e/test_full_trading_cycle.py',
        'e2e/test_system_validation.py',
        'unit/core/test_overfitting_detection.py',
        'scripts/run_all_tests.py'
    ]
    
    found_files = 0
    
    for file_path in expected_files:
        full_path = tests_dir / file_path
        if full_path.exists():
            print(f"   ✅ {file_path}")
            found_files += 1
        else:
            print(f"   ❌ {file_path} - missing")
    
    print(f"\n📊 Verification Summary:")
    print(f"   Found: {found_files}/{len(expected_files)} files")
    
    success_rate = (found_files / len(expected_files)) * 100
    print(f"   Success Rate: {success_rate:.1f}%")
    
    return success_rate >= 80

if __name__ == "__main__":
    print("🏗️ GridAttention Test Migration")
    print("=" * 50)
    
    try:
        # Step 1: Migrate files
        migrated = migrate_test_files()
        
        # Step 2: Cleanup old files (only if migration was successful)
        if migrated > 0:
            print("\n❓ Clean up old files? (y/n): ", end="")
            response = input().lower().strip()
            
            if response in ['y', 'yes']:
                cleanup_old_files()
            else:
                print("   Keeping old files for safety")
        
        # Step 3: Verify migration
        success = verify_migration()
        
        if success:
            print("\n🎉 Migration completed successfully!")
            print("\nNext steps:")
            print("1. Run: python tests/test_health_check.py")
            print("2. Test new structure: python tests/scripts/run_all_tests.py")
        else:
            print("\n⚠️  Migration completed with issues")
            print("   Please review the missing files")
            
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        print("   Please check file permissions and paths")