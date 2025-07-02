#!/usr/bin/env python3
"""
Test Suite Health Check - ตรวจสอบความพร้อมของ test files
"""

import os
import sys
import importlib
import subprocess
from pathlib import Path


def check_test_environment():
    """ตรวจสอบสภาพแวดล้อมสำหรับการทดสอบ"""
    
    print("🔍 GridAttention Test Suite Health Check")
    print("=" * 60)
    
    # 1. ตรวจสอบ Python version
    print("\n1️⃣ Python Version:")
    python_version = sys.version.split()[0]
    if sys.version_info >= (3, 8):
        print(f"   ✅ Python {python_version} (OK)")
    else:
        print(f"   ❌ Python {python_version} (ต้องการ 3.8+)")
    
    # 2. ตรวจสอบ test dependencies
    print("\n2️⃣ Test Dependencies:")
    test_deps = {
        'pytest': 'pytest',
        'pytest-asyncio': 'pytest-asyncio',
        'pytest-cov': 'pytest-cov'
    }
    
    for name, package in test_deps.items():
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - ติดตั้งด้วย: pip install {package}")
    
    # 3. ตรวจสอบ test files
    print("\n3️⃣ Test Files:")
    test_dir = Path("tests")
    
    if not test_dir.exists():
        print(f"   ❌ Directory 'tests/' not found!")
        return False
    
    test_files = [
        'test_gridattention_complete.py',
        'test_warmup_integration.py', 
        'test_system_validation.py',
        'test_overfitting_detection.py',
        'test_phase_augmentation.py',
        'test_integration.py',
        'test_augmentation_monitoring.py',
        'final_test.py',
        'run_all_tests.py'
    ]
    
    found_files = 0
    for test_file in test_files:
        file_path = test_dir / test_file
        if file_path.exists():
            print(f"   ✅ {test_file}")
            found_files += 1
        else:
            print(f"   ❌ {test_file} - ไม่พบไฟล์")
    
    print(f"\n   พบ {found_files}/{len(test_files)} test files")
    
    # 4. ตรวจสอบ main components ที่ tests ต้องใช้
    print("\n4️⃣ Required Components:")
    components_check = [
        ('core/attention_learning_layer.py', 'attention_learning_layer.py'),
        ('core/market_regime_detector.py', 'market_regime_detector.py'),
        ('core/grid_strategy_selector.py', 'grid_strategy_selector.py'),
        ('core/risk_management_system.py', 'risk_management_system.py'),
        ('core/execution_engine.py', 'execution_engine.py'),
        ('core/performance_monitor.py', 'performance_monitor.py'),
        ('core/feedback_loop.py', 'feedback_loop.py'),
        ('core/overfitting_detector.py', 'overfitting_detector.py')
    ]
    
    found_components = 0
    for new_path, old_path in components_check:
        if Path(new_path).exists():
            print(f"   ✅ {new_path}")
            found_components += 1
        elif Path(old_path).exists():
            print(f"   ⚠️  {old_path} - ใช้ old structure")
            found_components += 1
        else:
            print(f"   ❌ {new_path} - ไม่พบไฟล์")
    
    print(f"\n   พบ {found_components}/{len(components_check)} components")
    
    # 5. ตรวจสอบ config files
    print("\n5️⃣ Configuration Files:")
    config_files = [
        'config/config.yaml',
        'config/overfitting_config.yaml', 
        'config/config_production.yaml',
        'config/config_minimal.yaml'
    ]
    
    for config in config_files:
        if Path(config).exists():
            print(f"   ✅ {config}")
        else:
            print(f"   ⚠️  {config} - ไม่พบ (อาจสร้างอัตโนมัติ)")
    
    # 6. ตรวจสอบ main.py
    print("\n6️⃣ Main Entry Point:")
    if Path("main.py").exists():
        print("   ✅ main.py - พบไฟล์")
        
        # ตรวจสอบ import paths ใน main.py
        try:
            with open("main.py", 'r') as f:
                content = f.read()
                
            required_imports = [
                'OptimizedDashboardCollector',
                'patch_system_buffers',
                'unified_monitor'
            ]
            
            found_imports = 0
            for imp in required_imports:
                if imp in content:
                    found_imports += 1
            
            print(f"   ✅ พบ {found_imports}/{len(required_imports)} required imports")
            
        except Exception as e:
            print(f"   ⚠️  ไม่สามารถตรวจสอบ imports: {e}")
    else:
        print("   ❌ main.py - ไม่พบไฟล์")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 สรุปผลการตรวจสอบ:")
    
    issues = []
    
    if sys.version_info < (3, 8):
        issues.append("- อัพเกรด Python เป็น 3.8+")
    
    if found_files < len(test_files):
        issues.append("- ตรวจสอบว่า test files ครบถ้วน")
    
    if found_components < len(components_check) * 0.8:
        issues.append("- ตรวจสอบว่า component files ครบถ้วน")
    
    if not issues:
        print("✅ ระบบพร้อมสำหรับการทดสอบ!")
        print("\nรัน tests ด้วย:")
        print("   python tests/run_all_tests.py")
        print("   python tests/run_all_tests.py --quick")
    else:
        print("⚠️  พบปัญหาที่ต้องแก้ไข:")
        for issue in issues:
            print(issue)
    
    return len(issues) == 0


def suggest_fixes():
    """แนะนำวิธีแก้ไขปัญหาที่พบ"""
    
    print("\n💡 คำแนะนำการแก้ไข:")
    print("-" * 60)
    
    print("\n1. ติดตั้ง test dependencies:")
    print("   pip install pytest pytest-asyncio pytest-cov")
    
    print("\n2. ตรวจสอบ import paths:")
    print("   - ตรวจสอบว่ารันจาก root directory ของ project")
    print("   - ตรวจสอบ __init__.py ใน tests/")
    
    print("\n3. รัน test แบบ verbose เพื่อดู error:")
    print("   python tests/run_all_tests.py --stop-on-failure")
    
    print("\n4. รัน health check อื่นๆ:")
    print("   ./scripts/health_check.sh")
    print("   python scripts/quick_verify.py")


if __name__ == "__main__":
    print("🚀 เริ่มตรวจสอบ Test Suite...\n")
    
    success = check_test_environment()
    
    if not success:
        suggest_fixes()
    
    print("\n✅ การตรวจสอบเสร็จสิ้น")
    sys.exit(0 if success else 1)