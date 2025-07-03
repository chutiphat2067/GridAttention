#!/usr/bin/env python3
"""
Master Test Runner for GridAttention Trading System
Run all tests and generate comprehensive report
"""

import asyncio
import sys
import time
import json
from datetime import datetime
from pathlib import Path
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestSuite:
    """Test suite information"""
    def __init__(self, name: str, module: str, description: str):
        self.name = name
        self.module = module
        self.description = description
        self.result = None
        self.execution_time = 0
        self.error = None


# Define all test suites
TEST_SUITES = [
    TestSuite(
        name="System Validation",
        module="test_system_validation",
        description="Validates system configuration, dependencies, and component integration"
    ),
    TestSuite(
        name="Component Tests",
        module="test_gridattention_complete",
        description="Tests all individual components (Attention, Regime Detector, Strategy Selector, etc.)"
    ),
    TestSuite(
        name="Warmup Integration",
        module="test_warmup_integration",
        description="Tests warmup system and accelerated learning functionality"
    )
]


async def run_test_suite(suite: TestSuite) -> bool:
    """Run a single test suite"""
    print(f"\n{'='*80}")
    print(f"Running: {suite.name}")
    print(f"Description: {suite.description}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Import and run the test module
        if suite.module == "test_system_validation":
            from test_system_validation import main as test_main
        elif suite.module == "test_gridattention_complete":
            from test_gridattention_complete import main as test_main
        elif suite.module == "test_warmup_integration":
            from test_warmup_integration import run_warmup_integration_tests as test_main
        else:
            raise ValueError(f"Unknown test module: {suite.module}")
            
        # Run the test
        result = await test_main()
        
        suite.result = "PASSED" if result else "FAILED"
        suite.execution_time = time.time() - start_time
        
        print(f"\n✅ {suite.name} completed in {suite.execution_time:.2f}s")
        return result
        
    except Exception as e:
        suite.result = "ERROR"
        suite.error = str(e)
        suite.execution_time = time.time() - start_time
        
        print(f"\n❌ {suite.name} failed with error: {e}")
        logger.error(f"Test suite {suite.name} error: {e}", exc_info=True)
        return False


def print_summary(suites: list, total_time: float):
    """Print test summary"""
    print("\n" + "="*80)
    print("TEST EXECUTION SUMMARY")
    print("="*80)
    
    print(f"\nTotal Execution Time: {total_time:.2f} seconds")
    
    print(f"\n{'Test Suite':<30} {'Result':<10} {'Time':<10} {'Details'}")
    print("-"*80)
    
    passed = 0
    failed = 0
    errors = 0
    
    for suite in suites:
        status_icon = {
            "PASSED": "✅",
            "FAILED": "❌",
            "ERROR": "⚠️",
            None: "⏭️"
        }.get(suite.result, "?")
        
        details = ""
        if suite.error:
            details = f"Error: {suite.error[:50]}..."
            
        print(f"{suite.name:<30} {status_icon} {suite.result or 'SKIPPED':<8} "
              f"{suite.execution_time:>6.2f}s  {details}")
              
        if suite.result == "PASSED":
            passed += 1
        elif suite.result == "FAILED":
            failed += 1
        elif suite.result == "ERROR":
            errors += 1
            
    print("-"*80)
    print(f"{'TOTAL':<30} Passed: {passed}, Failed: {failed}, Errors: {errors}")
    
    # Overall result
    all_passed = passed == len(suites) and failed == 0 and errors == 0
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED! System is ready for use.")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
    print("="*80)
    
    return all_passed


def save_test_report(suites: list, total_time: float, output_file: str = "test_report.json"):
    """Save detailed test report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_execution_time": total_time,
        "test_suites": [
            {
                "name": suite.name,
                "module": suite.module,
                "description": suite.description,
                "result": suite.result,
                "execution_time": suite.execution_time,
                "error": suite.error
            }
            for suite in suites
        ],
        "summary": {
            "total_tests": len(suites),
            "passed": sum(1 for s in suites if s.result == "PASSED"),
            "failed": sum(1 for s in suites if s.result == "FAILED"),
            "errors": sum(1 for s in suites if s.result == "ERROR"),
            "skipped": sum(1 for s in suites if s.result is None)
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"\nDetailed test report saved to: {output_file}")


async def main(args):
    """Main test execution"""
    print("="*80)
    print("GRIDATTENTION TRADING SYSTEM - MASTER TEST RUNNER")
    print("="*80)
    print(f"\nRunning tests at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Select test suites to run
    if args.suite:
        # Run specific suite
        suites_to_run = [s for s in TEST_SUITES if s.name.lower() == args.suite.lower()]
        if not suites_to_run:
            print(f"Error: Unknown test suite '{args.suite}'")
            print(f"Available suites: {', '.join(s.name for s in TEST_SUITES)}")
            return False
    else:
        # Run all suites
        suites_to_run = TEST_SUITES
        
    # Run selected test suites
    for suite in suites_to_run:
        if args.quick and suite.name == "Component Tests":
            print(f"\nSkipping {suite.name} in quick mode...")
            suite.result = None
            continue
            
        await run_test_suite(suite)
        
        if args.stop_on_failure and suite.result != "PASSED":
            print(f"\nStopping due to test failure (--stop-on-failure)")
            break
            
    total_time = time.time() - start_time
    
    # Print summary
    all_passed = print_summary(TEST_SUITES, total_time)
    
    # Save report
    if args.output:
        save_test_report(TEST_SUITES, total_time, args.output)
    else:
        save_test_report(TEST_SUITES, total_time)
        
    return all_passed


def create_test_files():
    """Create placeholder test files if they don't exist"""
    test_files = [
        ("test_system_validation.py", "from test_system_validation import *"),
        ("test_gridattention_complete.py", "from test_gridattention_complete import *"),
        ("test_warmup_integration.py", "from test_warmup_integration import *")
    ]
    
    for filename, content in test_files:
        if not Path(filename).exists():
            print(f"Creating placeholder for {filename}...")
            with open(filename, 'w') as f:
                f.write(f"# Placeholder for {filename}\n")
                f.write("async def main():\n")
                f.write("    print(f'Running {filename}...')\n")
                f.write("    return True\n")
                f.write("\n")
                f.write("async def run_warmup_integration_tests():\n")
                f.write("    return await main()\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GridAttention system tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_tests.py                    # Run all tests
  python run_all_tests.py --quick            # Quick validation only
  python run_all_tests.py --suite "Warmup"   # Run specific suite
  python run_all_tests.py --stop-on-failure  # Stop on first failure
        """
    )
    
    parser.add_argument(
        "--suite",
        type=str,
        help="Run specific test suite by name"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only (skip comprehensive component tests)"
    )
    
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop execution on first test failure"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for test report (default: test_report.json)"
    )
    
    parser.add_argument(
        "--create-placeholders",
        action="store_true",
        help="Create placeholder test files if missing"
    )
    
    args = parser.parse_args()
    
    # Create placeholder files if requested
    if args.create_placeholders:
        create_test_files()
        print("Placeholder files created. Please add actual test implementations.")
        sys.exit(0)
        
    # Run tests
    try:
        success = asyncio.run(main(args))
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test runner failed: {e}", exc_info=True)
        sys.exit(1)