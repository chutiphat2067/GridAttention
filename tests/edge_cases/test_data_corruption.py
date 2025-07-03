"""
Data Corruption Testing Suite for GridAttention Trading System
Tests data integrity, corruption detection, recovery mechanisms, and validation
"""

import pytest
import asyncio
import json
import hashlib
import random
import struct
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Tuple, Any, Union
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import logging
from dataclasses import dataclass, field
from enum import Enum
import zlib
import base64
import msgpack
import sqlite3

# GridAttention imports - aligned with system structure
from src.grid_attention_layer import GridAttentionLayer
from src.data.integrity_checker import DataIntegrityChecker
from src.data.validation_engine import ValidationEngine
from src.data.corruption_detector import CorruptionDetector
from src.data.recovery_manager import DataRecoveryManager
from src.data.checksum_validator import ChecksumValidator
from src.data.data_sanitizer import DataSanitizer
from src.database.db_integrity import DatabaseIntegrityChecker
from src.storage.data_store import SecureDataStore

# Test utilities
from tests.utils.test_helpers import async_test, create_test_config
from tests.utils.corruption_simulator import (
    corrupt_bytes,
    corrupt_json,
    corrupt_database,
    simulate_bit_flip
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorruptionType(Enum):
    """Types of data corruption"""
    BIT_FLIP = "bit_flip"
    TRUNCATION = "truncation"
    ENCODING_ERROR = "encoding_error"
    TYPE_MISMATCH = "type_mismatch"
    NULL_INJECTION = "null_injection"
    OVERFLOW = "overflow"
    PRECISION_LOSS = "precision_loss"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    SERIALIZATION_ERROR = "serialization_error"
    MEMORY_CORRUPTION = "memory_corruption"


@dataclass
class DataIntegrityReport:
    """Data integrity check report"""
    timestamp: datetime
    data_type: str
    corruption_detected: bool
    corruption_type: Optional[CorruptionType]
    affected_fields: List[str]
    severity: str  # low, medium, high, critical
    recoverable: bool
    recovery_method: Optional[str]
    data_loss: bool
    checksum_before: Optional[str]
    checksum_after: Optional[str]


class TestMarketDataCorruption:
    """Test market data corruption scenarios"""
    
    @pytest.fixture
    def integrity_checker(self):
        """Create data integrity checker"""
        config = create_test_config()
        config['integrity'] = {
            'enable_checksums': True,
            'checksum_algorithm': 'sha256',
            'validation_level': 'strict',
            'auto_recovery': True,
            'backup_retention': 3
        }
        return DataIntegrityChecker(config)
    
    @pytest.fixture
    def corruption_detector(self):
        """Create corruption detector"""
        config = create_test_config()
        config['corruption'] = {
            'detection_methods': ['checksum', 'range', 'type', 'pattern'],
            'sensitivity': 'high',
            'quarantine_corrupt_data': True
        }
        return CorruptionDetector(config)
    
    @async_test
    async def test_price_data_corruption(self, grid_attention, integrity_checker):
        """Test detection of corrupted price data"""
        # Valid market data
        valid_data = {
            'symbol': 'BTC/USDT',
            'price': 50000.0,
            'volume': 123.45,
            'timestamp': datetime.now(),
            'bid': 49995.0,
            'ask': 50005.0
        }
        
        # Create corrupted versions
        corruption_tests = [
            # Negative price
            {**valid_data, 'price': -50000.0, 'expected_error': 'negative_price'},
            
            # Extreme price (overflow)
            {**valid_data, 'price': 1e20, 'expected_error': 'price_out_of_range'},
            
            # NaN price
            {**valid_data, 'price': float('nan'), 'expected_error': 'invalid_numeric'},
            
            # Infinity price
            {**valid_data, 'price': float('inf'), 'expected_error': 'infinite_value'},
            
            # String price
            {**valid_data, 'price': '50000', 'expected_error': 'type_mismatch'},
            
            # Bid > Ask
            {**valid_data, 'bid': 50010.0, 'ask': 49990.0, 'expected_error': 'invalid_spread'},
            
            # Future timestamp
            {**valid_data, 'timestamp': datetime.now() + timedelta(days=1), 'expected_error': 'future_timestamp'}
        ]
        
        for corrupt_data in corruption_tests:
            expected_error = corrupt_data.pop('expected_error')
            
            # Check if corruption is detected
            validation_result = await integrity_checker.validate_market_data(corrupt_data)
            
            assert validation_result['valid'] is False
            assert validation_result['error_type'] == expected_error
            assert len(validation_result['errors']) > 0
            
            # Verify data is not processed
            process_result = await grid_attention.process_market_update(corrupt_data)
            assert process_result['processed'] is False
            assert process_result['reason'] == 'data_validation_failed'
    
    @async_test
    async def test_volume_data_corruption(self, corruption_detector):
        """Test volume data corruption detection"""
        # Generate volume series
        volume_data = [
            {'timestamp': datetime.now() - timedelta(minutes=i), 'volume': 100 + i * 10}
            for i in range(10)
        ]
        
        # Corrupt some entries
        volume_data[5]['volume'] = -100  # Negative volume
        volume_data[7]['volume'] = 1e15  # Unrealistic volume
        volume_data[8]['volume'] = None  # Null volume
        
        # Detect corruption
        detection_results = []
        for data in volume_data:
            result = await corruption_detector.check_volume_integrity(data)
            detection_results.append(result)
        
        # Verify corruption detected
        corrupt_indices = [5, 7, 8]
        for i, result in enumerate(detection_results):
            if i in corrupt_indices:
                assert result['corruption_detected'] is True
                assert result['corruption_type'] is not None
            else:
                assert result['corruption_detected'] is False
    
    @async_test
    async def test_order_book_corruption(self, grid_attention):
        """Test order book data corruption"""
        # Valid order book
        valid_order_book = {
            'symbol': 'BTC/USDT',
            'bids': [
                {'price': 49990, 'size': 1.0},
                {'price': 49980, 'size': 2.0},
                {'price': 49970, 'size': 3.0}
            ],
            'asks': [
                {'price': 50010, 'size': 1.0},
                {'price': 50020, 'size': 2.0},
                {'price': 50030, 'size': 3.0}
            ],
            'timestamp': datetime.now()
        }
        
        # Test various corruptions
        corruptions = [
            # Crossed book (bid > ask)
            lambda ob: {**ob, 'bids': [{'price': 50020, 'size': 1.0}] + ob['bids'][1:]},
            
            # Non-sorted bids
            lambda ob: {**ob, 'bids': [ob['bids'][2], ob['bids'][0], ob['bids'][1]]},
            
            # Duplicate price levels
            lambda ob: {**ob, 'asks': [ob['asks'][0], ob['asks'][0], ob['asks'][2]]},
            
            # Negative sizes
            lambda ob: {**ob, 'bids': [{'price': 49990, 'size': -1.0}] + ob['bids'][1:]},
            
            # Missing price
            lambda ob: {**ob, 'asks': [{'size': 1.0}] + ob['asks'][1:]}
        ]
        
        for corrupt_func in corruptions:
            corrupt_book = corrupt_func(valid_order_book.copy())
            
            result = await grid_attention.validate_order_book(corrupt_book)
            assert result['valid'] is False
            assert result['corruption_detected'] is True
            assert len(result['issues']) > 0


class TestTradingDataCorruption:
    """Test trading data corruption scenarios"""
    
    @async_test
    async def test_order_data_corruption(self, execution_engine, integrity_checker):
        """Test order data corruption detection"""
        # Valid order
        valid_order = {
            'order_id': 'ORD123456',
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'price': Decimal('50000.00'),
            'quantity': Decimal('0.1'),
            'type': 'limit',
            'timestamp': datetime.now()
        }
        
        # Test decimal precision corruption
        corrupt_orders = [
            # Precision loss
            {**valid_order, 'price': 50000.000000000001},  # Float instead of Decimal
            
            # Overflow
            {**valid_order, 'quantity': Decimal('1e50')},
            
            # Invalid decimal
            {**valid_order, 'price': Decimal('NaN')},
            
            # Wrong type
            {**valid_order, 'side': 1},  # Should be string
            
            # Invalid enum value
            {**valid_order, 'type': 'invalid_type'}
        ]
        
        for corrupt_order in corrupt_orders:
            # Validate order
            validation = await integrity_checker.validate_order_data(corrupt_order)
            assert validation['valid'] is False
            
            # Try to submit (should fail)
            result = await execution_engine.submit_order(corrupt_order)
            assert result['success'] is False
            assert 'corruption' in result['error'].lower() or 'invalid' in result['error'].lower()
    
    @async_test
    async def test_position_data_corruption(self, grid_attention):
        """Test position data corruption and recovery"""
        # Create valid position
        valid_position = {
            'symbol': 'BTC/USDT',
            'side': 'long',
            'size': Decimal('1.5'),
            'entry_price': Decimal('48000'),
            'unrealized_pnl': Decimal('3000'),
            'margin_used': Decimal('7200')
        }
        
        # Store position
        await grid_attention.update_position(valid_position)
        
        # Simulate corruption in storage
        with patch.object(grid_attention.storage, 'get_position') as mock_get:
            # Return corrupted data
            mock_get.return_value = {
                'symbol': 'BTC/USDT',
                'side': 'long',
                'size': -1.5,  # Negative size for long position
                'entry_price': 0,  # Zero entry price
                'unrealized_pnl': 'invalid',  # String instead of Decimal
                'margin_used': float('inf')  # Infinity
            }
            
            # Attempt to load position
            loaded_position = await grid_attention.get_position('BTC/USDT')
            
            # Should detect corruption and attempt recovery
            assert loaded_position['corruption_detected'] is True
            assert loaded_position['recovery_attempted'] is True
            
            # Check if recovery was successful
            if loaded_position['recovery_successful']:
                # Recovered data should be valid
                assert loaded_position['data']['size'] > 0
                assert loaded_position['data']['entry_price'] > 0
                assert isinstance(loaded_position['data']['unrealized_pnl'], Decimal)
    
    @async_test
    async def test_trade_history_corruption(self, grid_attention, corruption_detector):
        """Test trade history data corruption"""
        # Generate trade history
        trades = []
        for i in range(100):
            trade = {
                'trade_id': f'TRD{i:06d}',
                'symbol': 'BTC/USDT',
                'side': 'buy' if i % 2 == 0 else 'sell',
                'price': Decimal('50000') + Decimal(str(i * 10)),
                'quantity': Decimal('0.1'),
                'fee': Decimal('0.0001'),
                'timestamp': datetime.now() - timedelta(hours=i)
            }
            trades.append(trade)
        
        # Corrupt random trades
        corruption_indices = random.sample(range(100), 10)
        for idx in corruption_indices:
            corruption_type = random.choice([
                'duplicate_id', 'missing_field', 'type_error', 'range_error'
            ])
            
            if corruption_type == 'duplicate_id':
                trades[idx]['trade_id'] = trades[0]['trade_id']
            elif corruption_type == 'missing_field':
                del trades[idx]['price']
            elif corruption_type == 'type_error':
                trades[idx]['quantity'] = '0.1'  # String instead of Decimal
            else:  # range_error
                trades[idx]['fee'] = Decimal('-0.01')  # Negative fee
        
        # Validate trade history
        validation_results = await corruption_detector.validate_trade_history(trades)
        
        assert validation_results['total_trades'] == 100
        assert validation_results['corrupt_trades'] == len(corruption_indices)
        assert len(validation_results['corruption_details']) == len(corruption_indices)
        
        # Verify each corruption was detected
        for detail in validation_results['corruption_details']:
            assert detail['trade_index'] in corruption_indices
            assert detail['corruption_type'] is not None
            assert detail['recoverable'] in [True, False]


class TestDatabaseCorruption:
    """Test database corruption scenarios"""
    
    @pytest.fixture
    def db_integrity_checker(self):
        """Create database integrity checker"""
        config = create_test_config()
        config['database'] = {
            'type': 'sqlite',
            'path': ':memory:',
            'enable_wal': True,
            'enable_integrity_checks': True,
            'backup_interval': 3600
        }
        return DatabaseIntegrityChecker(config)
    
    @async_test
    async def test_database_checksum_corruption(self, db_integrity_checker):
        """Test database checksum corruption detection"""
        # Create test database
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE orders (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL,
                checksum TEXT NOT NULL
            )
        ''')
        
        # Insert valid data with checksum
        order_data = {
            'id': 'ORD001',
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'price': 50000.0,
            'quantity': 0.1
        }
        
        # Calculate checksum
        checksum = hashlib.sha256(
            json.dumps(order_data, sort_keys=True).encode()
        ).hexdigest()
        
        cursor.execute(
            'INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)',
            (order_data['id'], order_data['symbol'], order_data['side'],
             order_data['price'], order_data['quantity'], checksum)
        )
        conn.commit()
        
        # Corrupt data (change price without updating checksum)
        cursor.execute(
            'UPDATE orders SET price = ? WHERE id = ?',
            (45000.0, 'ORD001')
        )
        conn.commit()
        
        # Check integrity
        integrity_result = await db_integrity_checker.check_table_integrity(
            conn, 'orders'
        )
        
        assert integrity_result['corruption_detected'] is True
        assert integrity_result['corrupt_rows'] == 1
        assert 'checksum_mismatch' in integrity_result['issues']
    
    @async_test
    async def test_foreign_key_corruption(self, db_integrity_checker):
        """Test foreign key constraint corruption"""
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        
        # Enable foreign keys
        cursor.execute('PRAGMA foreign_keys = ON')
        
        # Create related tables
        cursor.execute('''
            CREATE TABLE users (
                user_id INTEGER PRIMARY KEY,
                username TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE positions (
                position_id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                size REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        # Insert valid user
        cursor.execute('INSERT INTO users VALUES (1, "trader1")')
        
        # Corrupt foreign key by disabling constraints temporarily
        cursor.execute('PRAGMA foreign_keys = OFF')
        cursor.execute(
            'INSERT INTO positions VALUES (1, 999, "BTC/USDT", 1.0)'
        )  # user_id 999 doesn't exist
        cursor.execute('PRAGMA foreign_keys = ON')
        
        # Check integrity
        integrity_result = await db_integrity_checker.check_foreign_keys(conn)
        
        assert integrity_result['violations'] > 0
        assert integrity_result['tables_affected'] == ['positions']
        assert integrity_result['orphaned_records'] == 1
    
    @async_test
    async def test_data_type_corruption(self, db_integrity_checker):
        """Test data type corruption in database"""
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        
        # Create table with specific types
        cursor.execute('''
            CREATE TABLE trades (
                trade_id INTEGER PRIMARY KEY,
                price REAL NOT NULL CHECK(price > 0),
                quantity REAL NOT NULL CHECK(quantity > 0),
                timestamp INTEGER NOT NULL
            )
        ''')
        
        # Attempt to insert corrupted data
        corrupt_inserts = [
            (1, 'not_a_number', 0.1, int(datetime.now().timestamp())),  # String price
            (2, 50000, -0.1, int(datetime.now().timestamp())),  # Negative quantity
            (3, 50000, 0.1, 'not_a_timestamp'),  # String timestamp
        ]
        
        violations = []
        for data in corrupt_inserts:
            try:
                cursor.execute(
                    'INSERT INTO trades VALUES (?, ?, ?, ?)',
                    data
                )
            except sqlite3.IntegrityError as e:
                violations.append(str(e))
        
        assert len(violations) >= 2  # At least constraint violations


class TestSerializationCorruption:
    """Test data serialization corruption"""
    
    @async_test
    async def test_json_serialization_corruption(self, data_sanitizer):
        """Test JSON serialization corruption"""
        # Valid data structure
        valid_data = {
            'orders': [
                {'id': 1, 'price': 50000.0},
                {'id': 2, 'price': 50100.0}
            ],
            'timestamp': datetime.now().isoformat(),
            'metadata': {'version': '1.0'}
        }
        
        # Serialize
        json_str = json.dumps(valid_data)
        
        # Corrupt JSON in various ways
        corruptions = [
            # Missing closing brace
            json_str[:-1],
            
            # Invalid escape sequence
            json_str.replace('"orders"', '"ord\\ers"'),
            
            # Truncated
            json_str[:len(json_str)//2],
            
            # Invalid UTF-8
            json_str.encode('utf-8')[:-1].decode('utf-8', errors='ignore'),
            
            # Circular reference (can't actually create in JSON)
            '{"a": {"b": {"c": "{{CIRCULAR}}"}}}',
        ]
        
        for corrupt_json in corruptions:
            result = await data_sanitizer.safe_deserialize_json(corrupt_json)
            
            assert result['success'] is False
            assert result['error_type'] in ['json_decode_error', 'validation_error']
            assert result['fallback_used'] in [True, False]
    
    @async_test
    async def test_msgpack_corruption(self, data_sanitizer):
        """Test MessagePack serialization corruption"""
        # Valid data
        data = {
            'symbol': 'BTC/USDT',
            'prices': [50000, 50100, 50200],
            'volumes': [1.5, 2.0, 1.8],
            'timestamp': int(datetime.now().timestamp())
        }
        
        # Serialize with msgpack
        packed = msgpack.packb(data)
        
        # Corrupt packed data
        corruptions = [
            # Bit flip
            corrupt_bytes(packed, num_flips=5),
            
            # Truncation
            packed[:len(packed)//2],
            
            # Header corruption
            b'\x00' + packed[1:],
            
            # Random bytes
            b''.join([bytes([random.randint(0, 255)]) for _ in range(len(packed))])
        ]
        
        for corrupt_data in corruptions:
            result = await data_sanitizer.safe_deserialize_msgpack(corrupt_data)
            
            assert result['success'] is False or result['data'] != data
            if result['success'] is False:
                assert result['error_type'] == 'msgpack_decode_error'
    
    @async_test
    async def test_pickle_corruption_security(self, data_sanitizer):
        """Test pickle deserialization security (should never use pickle for untrusted data)"""
        # Dangerous pickle payload (should be rejected)
        dangerous_pickle = pickle.dumps(
            {'__reduce__': (eval, ('print("Code execution!")',))}
        )
        
        # Should reject pickle entirely for security
        result = await data_sanitizer.safe_deserialize(
            dangerous_pickle,
            format='pickle'
        )
        
        assert result['success'] is False
        assert result['error_type'] == 'format_not_allowed'
        assert 'security' in result['reason'].lower()


class TestChecksumValidation:
    """Test checksum validation mechanisms"""
    
    @pytest.fixture
    def checksum_validator(self):
        """Create checksum validator"""
        config = create_test_config()
        config['checksum'] = {
            'algorithms': ['sha256', 'crc32', 'md5'],
            'default_algorithm': 'sha256',
            'verify_on_read': True,
            'verify_on_write': True
        }
        return ChecksumValidator(config)
    
    @async_test
    async def test_data_checksum_validation(self, checksum_validator):
        """Test basic checksum validation"""
        # Original data
        data = {
            'order_id': 'ORD123',
            'symbol': 'BTC/USDT',
            'price': 50000.0,
            'quantity': 0.1
        }
        
        # Calculate checksum
        checksum = await checksum_validator.calculate_checksum(data)
        
        # Verify valid data
        is_valid = await checksum_validator.verify_checksum(data, checksum)
        assert is_valid is True
        
        # Corrupt data
        data['price'] = 45000.0
        
        # Verify corrupted data
        is_valid = await checksum_validator.verify_checksum(data, checksum)
        assert is_valid is False
    
    @async_test
    async def test_streaming_checksum(self, checksum_validator):
        """Test checksum calculation for streaming data"""
        # Simulate streaming market data
        stream_checksum = checksum_validator.create_stream_checksum()
        
        data_points = []
        for i in range(1000):
            data = {
                'timestamp': datetime.now().timestamp() + i,
                'price': 50000 + random.uniform(-100, 100),
                'volume': random.uniform(0.1, 10)
            }
            data_points.append(data)
            
            # Update streaming checksum
            await stream_checksum.update(data)
        
        # Get final checksum
        final_checksum = await stream_checksum.finalize()
        
        # Verify by recalculating
        verification_checksum = checksum_validator.create_stream_checksum()
        for data in data_points:
            await verification_checksum.update(data)
        
        verify_final = await verification_checksum.finalize()
        
        assert final_checksum == verify_final
        
        # Corrupt one data point
        data_points[500]['price'] = -1000
        
        # Recalculate with corruption
        corrupt_checksum = checksum_validator.create_stream_checksum()
        for data in data_points:
            await corrupt_checksum.update(data)
        
        corrupt_final = await corrupt_checksum.finalize()
        
        assert corrupt_final != final_checksum


class TestDataRecovery:
    """Test data recovery mechanisms"""
    
    @pytest.fixture
    def recovery_manager(self):
        """Create data recovery manager"""
        config = create_test_config()
        config['recovery'] = {
            'enable_auto_recovery': True,
            'backup_locations': ['primary', 'secondary', 'archive'],
            'recovery_strategies': ['backup', 'interpolation', 'last_known_good'],
            'max_recovery_attempts': 3
        }
        return DataRecoveryManager(config)
    
    @async_test
    async def test_backup_recovery(self, recovery_manager):
        """Test recovery from backup"""
        # Original data
        original_data = {
            'position_id': 'POS123',
            'symbol': 'BTC/USDT',
            'size': 1.5,
            'entry_price': 48000.0,
            'current_price': 50000.0,
            'pnl': 3000.0
        }
        
        # Create backup
        backup_id = await recovery_manager.create_backup(
            data=original_data,
            data_type='position'
        )
        
        # Corrupt current data
        corrupt_data = {
            'position_id': 'POS123',
            'symbol': 'BTC/USDT',
            'size': -999,  # Corrupted
            'entry_price': 0,  # Corrupted
            'current_price': float('inf'),  # Corrupted
            'pnl': 'error'  # Corrupted
        }
        
        # Attempt recovery
        recovery_result = await recovery_manager.recover_data(
            corrupt_data=corrupt_data,
            data_type='position',
            recovery_strategy='backup'
        )
        
        assert recovery_result['success'] is True
        assert recovery_result['strategy_used'] == 'backup'
        assert recovery_result['data'] == original_data
        assert recovery_result['data_loss'] is False
    
    @async_test
    async def test_interpolation_recovery(self, recovery_manager):
        """Test data recovery using interpolation"""
        # Time series data with corruption
        price_series = []
        for i in range(100):
            price = 50000 + i * 10 + random.uniform(-5, 5)
            price_series.append({
                'timestamp': datetime.now() + timedelta(minutes=i),
                'price': price
            })
        
        # Corrupt some entries
        corrupt_indices = [25, 26, 27, 50, 75]
        for idx in corrupt_indices:
            price_series[idx]['price'] = None  # Missing data
        
        # Attempt interpolation recovery
        recovery_result = await recovery_manager.recover_time_series(
            data=price_series,
            corrupt_indices=corrupt_indices,
            strategy='interpolation'
        )
        
        assert recovery_result['success'] is True
        assert recovery_result['recovered_count'] == len(corrupt_indices)
        
        # Verify interpolated values are reasonable
        for idx in corrupt_indices:
            recovered_price = recovery_result['data'][idx]['price']
            assert recovered_price is not None
            
            # Check if interpolated value is between neighbors
            if 0 < idx < len(price_series) - 1:
                prev_price = price_series[idx-1]['price']
                next_price = price_series[idx+1]['price']
                if prev_price and next_price:
                    assert min(prev_price, next_price) <= recovered_price <= max(prev_price, next_price)
    
    @async_test
    async def test_partial_recovery(self, recovery_manager):
        """Test partial data recovery when full recovery isn't possible"""
        # Complex data structure with partial corruption
        account_data = {
            'account_id': 'ACC123',
            'balances': {
                'BTC': 1.5,
                'ETH': None,  # Corrupted
                'USDT': 'error'  # Corrupted
            },
            'positions': [
                {'symbol': 'BTC/USDT', 'size': 1.0},
                {'symbol': 'ETH/USDT', 'size': -999},  # Corrupted
            ],
            'orders': [
                {'id': 'ORD1', 'status': 'open'},
                {'id': 'ORD2', 'status': None}  # Corrupted
            ],
            'settings': {
                'risk_limit': 0.02,
                'max_positions': 'invalid'  # Corrupted
            }
        }
        
        # Attempt recovery
        recovery_result = await recovery_manager.recover_complex_data(
            data=account_data,
            schema='account',
            partial_recovery_allowed=True
        )
        
        assert recovery_result['success'] is True
        assert recovery_result['partial_recovery'] is True
        assert recovery_result['fields_recovered'] > 0
        assert recovery_result['fields_lost'] > 0
        
        # Check recovered data
        recovered_data = recovery_result['data']
        assert recovered_data['balances']['BTC'] == 1.5  # Preserved
        assert 'ETH' in recovery_result['unrecoverable_fields']


class TestMemoryCorruption:
    """Test in-memory data corruption scenarios"""
    
    @async_test
    async def test_race_condition_corruption(self, grid_attention):
        """Test data corruption due to race conditions"""
        # Shared position data
        position = {
            'symbol': 'BTC/USDT',
            'size': 0.0,
            'total_cost': 0.0
        }
        
        # Simulate concurrent updates without proper locking
        async def update_position(order_size: float, order_price: float):
            # Read (with simulated delay)
            current_size = position['size']
            current_cost = position['total_cost']
            await asyncio.sleep(random.uniform(0.001, 0.01))
            
            # Modify
            new_size = current_size + order_size
            new_cost = current_cost + (order_size * order_price)
            
            # Write (with simulated delay)
            await asyncio.sleep(random.uniform(0.001, 0.01))
            position['size'] = new_size
            position['total_cost'] = new_cost
        
        # Run concurrent updates
        tasks = []
        expected_size = 0.0
        expected_cost = 0.0
        
        for i in range(10):
            size = 0.1
            price = 50000 + i * 100
            expected_size += size
            expected_cost += size * price
            
            task = update_position(size, price)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Check for corruption due to race condition
        size_corrupted = abs(position['size'] - expected_size) > 0.001
        cost_corrupted = abs(position['total_cost'] - expected_cost) > 1.0
        
        # In a real race condition, data will likely be corrupted
        # This test demonstrates the issue
        if size_corrupted or cost_corrupted:
            logger.warning(f"Race condition detected: Expected size={expected_size}, got={position['size']}")
            logger.warning(f"Expected cost={expected_cost}, got={position['total_cost']}")
    
    @async_test
    async def test_buffer_overflow_protection(self, data_sanitizer):
        """Test protection against buffer overflow attempts"""
        # Attempt to create oversized data
        oversized_inputs = [
            'A' * (10 ** 6),  # 1MB string
            [0] * (10 ** 6),  # 1M element list
            {'key': 'value' * (10 ** 5)},  # Large value
        ]
        
        for oversized_input in oversized_inputs:
            result = await data_sanitizer.sanitize_input(
                data=oversized_input,
                max_size_bytes=1024 * 100  # 100KB limit
            )
            
            assert result['sanitized'] is True
            assert result['size_reduced'] is True or result['rejected'] is True


# Helper Functions

def corrupt_bytes(data: bytes, num_flips: int = 1) -> bytes:
    """Corrupt bytes by flipping random bits"""
    corrupted = bytearray(data)
    
    for _ in range(num_flips):
        if len(corrupted) > 0:
            byte_idx = random.randint(0, len(corrupted) - 1)
            bit_idx = random.randint(0, 7)
            corrupted[byte_idx] ^= (1 << bit_idx)
    
    return bytes(corrupted)


class DataSanitizer:
    """Sanitize and validate data"""
    
    def __init__(self, config):
        self.config = config
        self.max_string_length = config.get('max_string_length', 10000)
        self.max_array_length = config.get('max_array_length', 1000)
        
    async def safe_deserialize_json(self, json_str: str) -> Dict:
        """Safely deserialize JSON with error handling"""
        try:
            data = json.loads(json_str)
            return {'success': True, 'data': data}
        except json.JSONDecodeError as e:
            return {
                'success': False,
                'error_type': 'json_decode_error',
                'error': str(e),
                'fallback_used': False
            }
        except Exception as e:
            return {
                'success': False,
                'error_type': 'unknown_error',
                'error': str(e)
            }
    
    async def safe_deserialize_msgpack(self, packed_data: bytes) -> Dict:
        """Safely deserialize MessagePack data"""
        try:
            data = msgpack.unpackb(packed_data, raw=False)
            return {'success': True, 'data': data}
        except Exception as e:
            return {
                'success': False,
                'error_type': 'msgpack_decode_error',
                'error': str(e)
            }
    
    async def safe_deserialize(self, data: bytes, format: str) -> Dict:
        """Safely deserialize data in specified format"""
        if format == 'pickle':
            # Never deserialize pickle from untrusted sources
            return {
                'success': False,
                'error_type': 'format_not_allowed',
                'reason': 'Pickle deserialization is disabled for security'
            }
        elif format == 'json':
            return await self.safe_deserialize_json(data.decode('utf-8'))
        elif format == 'msgpack':
            return await self.safe_deserialize_msgpack(data)
        else:
            return {
                'success': False,
                'error_type': 'unknown_format',
                'format': format
            }
    
    async def sanitize_input(self, data: Any, max_size_bytes: int) -> Dict:
        """Sanitize input data to prevent overflow"""
        try:
            # Estimate size
            if isinstance(data, str):
                size = len(data.encode('utf-8'))
                if size > max_size_bytes:
                    return {
                        'sanitized': True,
                        'size_reduced': True,
                        'original_size': size,
                        'new_size': max_size_bytes,
                        'data': data[:max_size_bytes]
                    }
            elif isinstance(data, (list, dict)):
                # Simple size estimation
                size = len(str(data).encode('utf-8'))
                if size > max_size_bytes:
                    return {
                        'sanitized': True,
                        'rejected': True,
                        'reason': 'Data too large',
                        'size': size
                    }
            
            return {
                'sanitized': False,
                'data': data
            }
            
        except Exception as e:
            return {
                'sanitized': True,
                'error': str(e),
                'rejected': True
            }


def generate_corruption_report(
    data_type: str,
    corruption_type: CorruptionType,
    severity: str,
    recoverable: bool
) -> DataIntegrityReport:
    """Generate a data integrity report"""
    return DataIntegrityReport(
        timestamp=datetime.now(),
        data_type=data_type,
        corruption_detected=True,
        corruption_type=corruption_type,
        affected_fields=[],
        severity=severity,
        recoverable=recoverable,
        recovery_method=None,
        data_loss=not recoverable,
        checksum_before=None,
        checksum_after=None
    )