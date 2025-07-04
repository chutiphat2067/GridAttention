#!/usr/bin/env python3
"""
Test Helpers for GridAttention Trading System
Provides comprehensive testing utilities and helper functions
"""

import asyncio
import time
import functools
import contextlib
import tempfile
import shutil
import json
import pickle
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Type, TypeVar
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from decimal import Decimal
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import aiofiles
import psutil
import gc
import tracemalloc
import warnings
from collections import defaultdict, deque
import yaml
import random
import string
import inspect
import threading
import signal


logger = logging.getLogger(__name__)
T = TypeVar('T')


# Test Decorators

def async_test(timeout: float = 30.0):
    """Decorator for async test functions with timeout"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            async def run():
                return await func(*args, **kwargs)
            
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                asyncio.wait_for(run(), timeout=timeout)
            )
        return wrapper
    return decorator


def retry_test(retries: int = 3, delay: float = 1.0, exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """Decorator to retry flaky tests"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            for i in range(retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if i < retries - 1:
                        await asyncio.sleep(delay)
                    else:
                        raise
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if i < retries - 1:
                        time.sleep(delay)
                    else:
                        raise
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator


def performance_test(threshold_ms: float = 1000):
    """Decorator to ensure test runs within performance threshold"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            elapsed_ms = (time.time() - start_time) * 1000
            
            if elapsed_ms > threshold_ms:
                warnings.warn(
                    f"Performance test {func.__name__} took {elapsed_ms:.2f}ms "
                    f"(threshold: {threshold_ms}ms)"
                )
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_ms = (time.time() - start_time) * 1000
            
            if elapsed_ms > threshold_ms:
                warnings.warn(
                    f"Performance test {func.__name__} took {elapsed_ms:.2f}ms "
                    f"(threshold: {threshold_ms}ms)"
                )
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator


def skip_if_slow(condition: bool = True, reason: str = "Test skipped in slow mode"):
    """Skip test if running in slow mode"""
    return pytest.mark.skipif(
        os.environ.get("SKIP_SLOW_TESTS", "").lower() == "true" and condition,
        reason=reason
    )


def requires_env(*env_vars: str):
    """Decorator to skip test if required environment variables are missing"""
    def decorator(func):
        missing_vars = [var for var in env_vars if not os.environ.get(var)]
        
        if missing_vars:
            return pytest.mark.skip(
                reason=f"Missing environment variables: {', '.join(missing_vars)}"
            )(func)
        return func
    return decorator


# Context Managers

@contextlib.contextmanager
def temporary_directory():
    """Create a temporary directory that's cleaned up automatically"""
    temp_dir = tempfile.mkdtemp()
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@contextlib.contextmanager
def temporary_file(suffix: str = "", content: Optional[str] = None):
    """Create a temporary file that's cleaned up automatically"""
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        if content:
            with open(path, 'w') as f:
                f.write(content)
        yield Path(path)
    finally:
        os.close(fd)
        os.unlink(path)


@contextlib.contextmanager
def mock_time(target_time: Union[datetime, float]):
    """Mock time for testing time-dependent code"""
    with patch('time.time') as mock_time_func:
        if isinstance(target_time, datetime):
            mock_time_func.return_value = target_time.timestamp()
        else:
            mock_time_func.return_value = target_time
        
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = (
                target_time if isinstance(target_time, datetime)
                else datetime.fromtimestamp(target_time)
            )
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            
            yield mock_time_func


@contextlib.contextmanager
def capture_logs(logger_name: Optional[str] = None, level: int = logging.INFO):
    """Capture log messages for testing"""
    logs = []
    
    class TestHandler(logging.Handler):
        def emit(self, record):
            logs.append({
                'level': record.levelname,
                'message': record.getMessage(),
                'time': datetime.fromtimestamp(record.created),
                'name': record.name
            })
    
    handler = TestHandler()
    handler.setLevel(level)
    
    logger_obj = logging.getLogger(logger_name)
    logger_obj.addHandler(handler)
    old_level = logger_obj.level
    logger_obj.setLevel(level)
    
    try:
        yield logs
    finally:
        logger_obj.removeHandler(handler)
        logger_obj.setLevel(old_level)


@contextlib.contextmanager
def suppress_warnings(*warning_types):
    """Suppress specific warnings during test"""
    with warnings.catch_warnings():
        for warning_type in warning_types:
            warnings.filterwarnings("ignore", category=warning_type)
        yield


# Memory and Performance Utilities

@contextlib.contextmanager
def measure_memory_usage():
    """Measure memory usage during test"""
    gc.collect()
    process = psutil.Process()
    
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    tracemalloc.start()
    
    try:
        yield
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_stats = {
            'start_mb': start_memory,
            'end_mb': end_memory,
            'delta_mb': end_memory - start_memory,
            'peak_mb': peak / 1024 / 1024,
            'current_mb': current / 1024 / 1024
        }
        
        logger.info(f"Memory usage: {memory_stats}")


def measure_execution_time(func: Callable) -> Tuple[Any, float]:
    """Measure function execution time"""
    start_time = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start_time
    return result, elapsed


async def measure_async_execution_time(func: Callable) -> Tuple[Any, float]:
    """Measure async function execution time"""
    start_time = time.perf_counter()
    result = await func()
    elapsed = time.perf_counter() - start_time
    return result, elapsed


# Data Generation Utilities

def generate_random_string(length: int = 10) -> str:
    """Generate random string"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_random_symbol() -> str:
    """Generate random trading symbol"""
    bases = ["BTC", "ETH", "BNB", "SOL", "ADA", "DOT", "MATIC"]
    quotes = ["USDT", "BUSD", "USDC"]
    return f"{random.choice(bases)}{random.choice(quotes)}"


def generate_random_price(base_price: float = 100.0, volatility: float = 0.1) -> float:
    """Generate random price with volatility"""
    return base_price * (1 + random.uniform(-volatility, volatility))


def generate_random_order(symbol: Optional[str] = None) -> Dict[str, Any]:
    """Generate random order data"""
    return {
        "order_id": generate_random_string(16),
        "symbol": symbol or generate_random_symbol(),
        "side": random.choice(["buy", "sell"]),
        "type": random.choice(["limit", "market"]),
        "price": round(generate_random_price(45000 if "BTC" in (symbol or "") else 3000), 2),
        "quantity": round(random.uniform(0.01, 1.0), 8),
        "status": "new",
        "timestamp": datetime.now()
    }


def generate_ohlcv_data(
    symbol: str = "BTCUSDT",
    periods: int = 100,
    interval: str = "5m",
    start_time: Optional[datetime] = None
) -> pd.DataFrame:
    """Generate OHLCV data for testing"""
    if not start_time:
        start_time = datetime.now() - timedelta(minutes=periods * 5)
    
    # Generate timestamps
    interval_map = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1)
    }
    
    delta = interval_map.get(interval, timedelta(minutes=5))
    timestamps = [start_time + delta * i for i in range(periods)]
    
    # Generate price data
    base_price = 45000 if "BTC" in symbol else 3000
    prices = [base_price]
    
    for _ in range(periods - 1):
        change = random.uniform(-0.002, 0.002)
        prices.append(prices[-1] * (1 + change))
    
    # Generate OHLCV
    data = []
    for i, (timestamp, close_price) in enumerate(zip(timestamps, prices)):
        open_price = prices[i-1] if i > 0 else close_price
        high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.001))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.001))
        volume = random.uniform(100, 1000)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    return pd.DataFrame(data).set_index('timestamp')


# Configuration Utilities

def create_test_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create test configuration with optional overrides"""
    base_config = {
        "mode": "test",
        "debug": True,
        "log_level": "DEBUG",
        
        "attention": {
            "min_trades_for_learning": 100,
            "min_trades_for_shadow": 50,
            "min_trades_for_active": 20,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        },
        
        "market_regime": {
            "lookback_period": 100,
            "update_frequency": "5m",
            "min_confidence": 0.6
        },
        
        "grid_strategy": {
            "num_levels": 10,
            "grid_spacing": 0.005,
            "position_size_per_level": 0.1
        },
        
        "risk_management": {
            "max_position_size_usd": 1000,
            "max_positions": 5,
            "stop_loss_pct": 2.0,
            "max_drawdown_pct": 10.0
        },
        
        "execution": {
            "order_timeout_seconds": 30,
            "max_retries": 3
        }
    }
    
    if overrides:
        deep_update(base_config, overrides)
    
    return base_config


def deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """Deep update dictionary"""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def load_test_data(filename: str) -> Any:
    """Load test data from file"""
    test_data_dir = Path(__file__).parent / "test_data"
    file_path = test_data_dir / filename
    
    if file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            return json.load(f)
    elif file_path.suffix == '.yaml':
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    elif file_path.suffix == '.pkl':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif file_path.suffix == '.csv':
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


# Mock Factory Utilities

class MockFactory:
    """Factory for creating mock objects"""
    
    @staticmethod
    def create_mock_exchange(name: str = "test_exchange") -> Mock:
        """Create mock exchange"""
        exchange = Mock()
        exchange.name = name
        exchange.has = Mock(return_value=True)
        exchange.fetch_ticker = AsyncMock(return_value={
            'symbol': 'BTC/USDT',
            'last': 45000,
            'bid': 44999,
            'ask': 45001,
            'volume': 1000
        })
        exchange.fetch_order_book = AsyncMock(return_value={
            'bids': [[44999, 1], [44998, 2]],
            'asks': [[45001, 1], [45002, 2]],
            'timestamp': datetime.now().timestamp() * 1000
        })
        exchange.create_order = AsyncMock(return_value={
            'id': generate_random_string(),
            'status': 'open',
            'symbol': 'BTC/USDT',
            'type': 'limit',
            'side': 'buy',
            'price': 45000,
            'amount': 0.1
        })
        exchange.cancel_order = AsyncMock(return_value={'status': 'canceled'})
        
        return exchange
    
    @staticmethod
    def create_mock_database() -> Mock:
        """Create mock database"""
        db = Mock()
        db.insert_one = AsyncMock(return_value="mock_id")
        db.find_one = AsyncMock(return_value={"_id": "mock_id", "data": "test"})
        db.find = AsyncMock(return_value=Mock(to_list=AsyncMock(return_value=[])))
        db.update_one = AsyncMock(return_value=Mock(modified_count=1))
        db.delete_one = AsyncMock(return_value=Mock(deleted_count=1))
        
        return db
    
    @staticmethod
    def create_mock_websocket() -> Mock:
        """Create mock websocket"""
        ws = Mock()
        ws.connected = True
        ws.send = AsyncMock()
        ws.recv = AsyncMock(return_value=json.dumps({
            'e': 'trade',
            's': 'BTCUSDT',
            'p': '45000',
            'q': '0.1',
            'T': datetime.now().timestamp() * 1000
        }))
        ws.close = AsyncMock()
        
        return ws


# Assertion Helpers

class AssertionHelpers:
    """Custom assertion helpers"""
    
    @staticmethod
    def assert_almost_equal(actual: float, expected: float, tolerance: float = 0.0001):
        """Assert floating point values are almost equal"""
        assert abs(actual - expected) <= tolerance, \
            f"Expected {expected} ± {tolerance}, got {actual}"
    
    @staticmethod
    def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, check_exact: bool = False):
        """Assert dataframes are equal"""
        if check_exact:
            pd.testing.assert_frame_equal(df1, df2)
        else:
            pd.testing.assert_frame_equal(df1, df2, check_dtype=False, check_exact=False)
    
    @staticmethod
    def assert_dict_contains(actual: Dict, expected: Dict):
        """Assert dictionary contains expected key-value pairs"""
        for key, value in expected.items():
            assert key in actual, f"Key '{key}' not found in dictionary"
            if isinstance(value, dict) and isinstance(actual[key], dict):
                AssertionHelpers.assert_dict_contains(actual[key], value)
            else:
                assert actual[key] == value, f"Expected {key}={value}, got {key}={actual[key]}"
    
    @staticmethod
    def assert_event_emitted(events: List[Dict], event_type: str, data: Optional[Dict] = None):
        """Assert specific event was emitted"""
        matching_events = [e for e in events if e.get('type') == event_type]
        assert matching_events, f"No events of type '{event_type}' found"
        
        if data:
            for event in matching_events:
                try:
                    AssertionHelpers.assert_dict_contains(event.get('data', {}), data)
                    return  # Found matching event
                except AssertionError:
                    continue
            
            assert False, f"No events of type '{event_type}' with matching data found"


# Async Utilities

async def wait_for_condition(
    condition: Callable[[], bool],
    timeout: float = 5.0,
    interval: float = 0.1,
    message: str = "Condition not met"
) -> bool:
    """Wait for condition to become true"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if await asyncio.coroutine(condition)() if asyncio.iscoroutinefunction(condition) else condition():
            return True
        await asyncio.sleep(interval)
    
    raise TimeoutError(f"{message} after {timeout}s")


async def run_concurrent_tests(test_funcs: List[Callable], max_concurrent: int = 10):
    """Run multiple test functions concurrently"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_with_semaphore(func):
        async with semaphore:
            if asyncio.iscoroutinefunction(func):
                return await func()
            else:
                return await asyncio.get_event_loop().run_in_executor(None, func)
    
    tasks = [run_with_semaphore(func) for func in test_funcs]
    return await asyncio.gather(*tasks, return_exceptions=True)


# Test Data Fixtures

class TestDataFixtures:
    """Common test data fixtures"""
    
    @staticmethod
    def sample_market_data() -> Dict[str, Any]:
        """Sample market data"""
        return {
            "symbol": "BTC/USDT",
            "timestamp": datetime.now(),
            "bid": 44999,
            "ask": 45001,
            "last": 45000,
            "volume": 12345.67,
            "high": 45500,
            "low": 44500
        }
    
    @staticmethod
    def sample_order() -> Dict[str, Any]:
        """Sample order data"""
        return {
            "id": generate_random_string(),
            "symbol": "BTC/USDT",
            "type": "limit",
            "side": "buy",
            "price": 45000,
            "amount": 0.1,
            "status": "open",
            "timestamp": datetime.now()
        }
    
    @staticmethod
    def sample_position() -> Dict[str, Any]:
        """Sample position data"""
        return {
            "symbol": "BTC/USDT",
            "side": "long",
            "amount": 0.5,
            "entry_price": 44500,
            "current_price": 45000,
            "unrealized_pnl": 250,
            "margin_used": 2225
        }
    
    @staticmethod
    def sample_grid_config() -> Dict[str, Any]:
        """Sample grid configuration"""
        return {
            "symbol": "BTC/USDT",
            "grid_type": "symmetric",
            "num_levels": 10,
            "upper_price": 46000,
            "lower_price": 44000,
            "amount_per_level": 0.01
        }


# Process Management

class ProcessManager:
    """Manage test processes"""
    
    def __init__(self):
        self.processes = []
    
    def start_background_process(self, target: Callable, args: Tuple = ()):
        """Start background process"""
        process = threading.Thread(target=target, args=args, daemon=True)
        process.start()
        self.processes.append(process)
        return process
    
    def stop_all_processes(self, timeout: float = 5.0):
        """Stop all background processes"""
        for process in self.processes:
            if process.is_alive():
                process.join(timeout=timeout)
        self.processes.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_all_processes()


# Signal Handling

@contextlib.contextmanager
def timeout_context(seconds: float):
    """Context manager for timeout using signals (Unix only)"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    if sys.platform != 'win32':
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(seconds))
        
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Fallback for Windows
        yield


# Test Report Generation

class TestReporter:
    """Generate test reports"""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
    
    def add_result(self, test_name: str, passed: bool, duration: float, details: Optional[Dict] = None):
        """Add test result"""
        self.results.append({
            'test_name': test_name,
            'passed': passed,
            'duration': duration,
            'details': details or {},
            'timestamp': datetime.now()
        })
    
    def generate_report(self, format: str = 'json') -> Union[str, Dict]:
        """Generate test report"""
        total_duration = time.time() - self.start_time
        passed_count = sum(1 for r in self.results if r['passed'])
        failed_count = len(self.results) - passed_count
        
        report = {
            'summary': {
                'total_tests': len(self.results),
                'passed': passed_count,
                'failed': failed_count,
                'duration': total_duration,
                'pass_rate': passed_count / len(self.results) if self.results else 0
            },
            'results': self.results
        }
        
        if format == 'json':
            return json.dumps(report, indent=2, default=str)
        elif format == 'dict':
            return report
        else:
            raise ValueError(f"Unsupported format: {format}")


# Environment Setup

def setup_test_environment():
    """Setup test environment"""
    # Set test mode
    os.environ['TRADING_MODE'] = 'test'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Disable warnings in test mode
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)


def cleanup_test_environment():
    """Cleanup test environment"""
    # Clear environment variables
    os.environ.pop('TRADING_MODE', None)
    os.environ.pop('LOG_LEVEL', None)
    
    # Collect garbage
    gc.collect()
    
    # Close any open file handles
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)


# Main test runner helper

def run_test_suite(test_dir: str = "tests", pattern: str = "test_*.py"):
    """Run test suite with custom configuration"""
    import pytest
    
    args = [
        test_dir,
        f"--pattern={pattern}",
        "-v",  # Verbose
        "--tb=short",  # Short traceback
        "--strict-markers",  # Strict marker checking
        "-p", "no:warnings",  # Disable warnings plugin
    ]
    
    # Add coverage if requested
    if os.environ.get('COVERAGE'):
        args.extend([
            "--cov=.",
            "--cov-report=html",
            "--cov-report=term"
        ])
    
    # Add parallel execution if requested
    if os.environ.get('PARALLEL'):
        args.extend(["-n", "auto"])
    
    return pytest.main(args)


if __name__ == "__main__":
    # Example usage
    setup_test_environment()
    
    # Run some example tests
    @async_test()
    async def test_example():
        with temporary_directory() as temp_dir:
            test_file = temp_dir / "test.json"
            test_file.write_text(json.dumps({"test": "data"}))
            
            data = json.loads(test_file.read_text())
            assert data["test"] == "data"
    
    # Generate sample data
    ohlcv_data = generate_ohlcv_data(periods=50)
    print(f"Generated {len(ohlcv_data)} OHLCV records")
    
    # Test configuration
    config = create_test_config({"attention": {"learning_rate": 0.01}})
    print(f"Test config: {json.dumps(config, indent=2)}")
    
    cleanup_test_environment()