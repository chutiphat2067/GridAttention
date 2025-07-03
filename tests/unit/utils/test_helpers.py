"""
Unit tests for Helper Utilities.

Tests cover:
- Date and time utilities
- String manipulation and formatting
- File operations
- Data structure helpers
- Configuration helpers
- Logging utilities
- Retry and error handling
- Decorators and context managers
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import time
import asyncio
from typing import List, Dict, Optional, Any
import logging
from collections import defaultdict, OrderedDict

# GridAttention project imports
from utils.helpers import (
    DateTimeHelper,
    StringHelper,
    FileHelper,
    DataStructureHelper,
    ConfigHelper,
    LoggingHelper,
    RetryHelper,
    DecoratorHelper,
    ValidationHelper,
    ConversionHelper,
    CacheHelper,
    AsyncHelper
)


class TestDateTimeHelper:
    """Test cases for date and time utilities."""
    
    @pytest.fixture
    def dt_helper(self):
        """Create DateTimeHelper instance."""
        return DateTimeHelper(default_timezone='UTC')
        
    def test_timezone_conversion(self, dt_helper):
        """Test timezone conversion functions."""
        # UTC to other timezones
        utc_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        # Convert to EST
        est_time = dt_helper.convert_timezone(utc_time, 'US/Eastern')
        assert est_time.hour == 7  # 12 UTC = 7 EST (winter time)
        
        # Convert to Asia/Tokyo
        tokyo_time = dt_helper.convert_timezone(utc_time, 'Asia/Tokyo')
        assert tokyo_time.hour == 21  # 12 UTC = 21 JST
        
        # Naive datetime handling
        naive_time = datetime(2023, 1, 1, 12, 0, 0)
        utc_aware = dt_helper.make_aware(naive_time, 'UTC')
        assert utc_aware.tzinfo is not None
        
    def test_date_parsing(self, dt_helper):
        """Test flexible date parsing."""
        # Various date formats
        date_strings = [
            '2023-01-01',
            '01/01/2023',
            'Jan 1, 2023',
            '2023-01-01T12:00:00',
            '2023-01-01 12:00:00+00:00',
            '20230101',
            '1st January 2023'
        ]
        
        for date_str in date_strings:
            parsed = dt_helper.parse_date(date_str)
            assert isinstance(parsed, datetime)
            assert parsed.year == 2023
            assert parsed.month == 1
            assert parsed.day == 1
            
        # Invalid date
        with pytest.raises(ValueError):
            dt_helper.parse_date('not a date')
            
    def test_date_range_generation(self, dt_helper):
        """Test date range generation utilities."""
        # Daily range
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 5)
        
        daily_range = dt_helper.generate_date_range(
            start, end, frequency='D'
        )
        
        assert len(daily_range) == 5
        assert daily_range[0] == start
        assert daily_range[-1] == end
        
        # Business days only
        business_days = dt_helper.generate_business_days(
            start, end, holidays=[]
        )
        
        assert len(business_days) == 3  # Mon, Tue, Wed (1st is Sunday)
        
        # Custom holidays
        holidays = [datetime(2023, 1, 2)]  # Monday
        business_days_holiday = dt_helper.generate_business_days(
            start, end, holidays=holidays
        )
        
        assert len(business_days_holiday) == 2  # Only Tue, Wed
        
    def test_time_calculations(self, dt_helper):
        """Test time calculation utilities."""
        # Time difference
        time1 = datetime(2023, 1, 1, 10, 0, 0)
        time2 = datetime(2023, 1, 1, 14, 30, 45)
        
        diff = dt_helper.time_difference(time1, time2, unit='hours')
        assert diff == 4.5125  # 4 hours, 30 minutes, 45 seconds
        
        diff_minutes = dt_helper.time_difference(time1, time2, unit='minutes')
        assert diff_minutes == 270.75
        
        # Add time
        new_time = dt_helper.add_time(time1, hours=2, minutes=30)
        assert new_time == datetime(2023, 1, 1, 12, 30, 0)
        
        # Round time
        time_to_round = datetime(2023, 1, 1, 10, 37, 45)
        
        rounded_hour = dt_helper.round_time(time_to_round, 'hour')
        assert rounded_hour == datetime(2023, 1, 1, 11, 0, 0)
        
        rounded_15min = dt_helper.round_time(time_to_round, '15min')
        assert rounded_15min == datetime(2023, 1, 1, 10, 45, 0)
        
    def test_market_hours(self, dt_helper):
        """Test market hours utilities."""
        # Check if time is during market hours
        market_open = datetime(2023, 1, 2, 14, 30, 0, tzinfo=timezone.utc)  # 9:30 EST
        market_close = datetime(2023, 1, 2, 21, 0, 0, tzinfo=timezone.utc)  # 4:00 EST
        after_hours = datetime(2023, 1, 2, 22, 0, 0, tzinfo=timezone.utc)
        
        assert dt_helper.is_market_open(market_open, 'NYSE')
        assert dt_helper.is_market_open(market_close, 'NYSE')
        assert not dt_helper.is_market_open(after_hours, 'NYSE')
        
        # Weekend
        weekend = datetime(2023, 1, 7, 15, 0, 0, tzinfo=timezone.utc)  # Saturday
        assert not dt_helper.is_market_open(weekend, 'NYSE')
        
        # Next market open
        next_open = dt_helper.next_market_open(after_hours, 'NYSE')
        assert next_open.date() == datetime(2023, 1, 3).date()
        
    def test_time_formatting(self, dt_helper):
        """Test time formatting utilities."""
        dt = datetime(2023, 1, 1, 14, 30, 45, 123456)
        
        # Various formats
        assert dt_helper.format_datetime(dt, 'ISO') == '2023-01-01T14:30:45.123456'
        assert dt_helper.format_datetime(dt, 'US') == '01/01/2023 02:30:45 PM'
        assert dt_helper.format_datetime(dt, 'EU') == '01.01.2023 14:30:45'
        assert dt_helper.format_datetime(dt, 'FILENAME') == '20230101_143045'
        
        # Relative time
        now = datetime.now()
        assert dt_helper.format_relative(now - timedelta(seconds=30)) == '30 seconds ago'
        assert dt_helper.format_relative(now - timedelta(minutes=5)) == '5 minutes ago'
        assert dt_helper.format_relative(now - timedelta(hours=2)) == '2 hours ago'
        assert dt_helper.format_relative(now + timedelta(days=1)) == 'in 1 day'


class TestStringHelper:
    """Test cases for string manipulation utilities."""
    
    @pytest.fixture
    def str_helper(self):
        """Create StringHelper instance."""
        return StringHelper()
        
    def test_string_cleaning(self, str_helper):
        """Test string cleaning functions."""
        # Remove extra whitespace
        dirty_string = "  Hello   World  \n\t  "
        clean = str_helper.clean_whitespace(dirty_string)
        assert clean == "Hello World"
        
        # Remove special characters
        special_string = "Hello@#$World!123"
        alphanumeric = str_helper.keep_alphanumeric(special_string)
        assert alphanumeric == "HelloWorld123"
        
        # Normalize unicode
        unicode_string = "Héllö Wörld"
        normalized = str_helper.normalize_unicode(unicode_string)
        assert normalized == "Hello World"
        
    def test_string_validation(self, str_helper):
        """Test string validation functions."""
        # Email validation
        assert str_helper.is_valid_email("test@example.com")
        assert str_helper.is_valid_email("user.name+tag@example.co.uk")
        assert not str_helper.is_valid_email("invalid.email")
        assert not str_helper.is_valid_email("@example.com")
        
        # URL validation
        assert str_helper.is_valid_url("https://www.example.com")
        assert str_helper.is_valid_url("http://subdomain.example.com:8080/path")
        assert not str_helper.is_valid_url("not a url")
        assert not str_helper.is_valid_url("ftp://example.com")  # If only http/https allowed
        
        # Phone validation
        assert str_helper.is_valid_phone("+1-555-123-4567")
        assert str_helper.is_valid_phone("555-123-4567")
        assert not str_helper.is_valid_phone("123")
        
    def test_string_formatting(self, str_helper):
        """Test string formatting utilities."""
        # Title case with exceptions
        text = "the quick brown fox jumps over the lazy dog"
        title = str_helper.smart_title_case(text)
        assert title == "The Quick Brown Fox Jumps Over the Lazy Dog"
        
        # Camel case
        snake_case = "hello_world_example"
        camel = str_helper.to_camel_case(snake_case)
        assert camel == "helloWorldExample"
        
        # Snake case
        camel_case = "helloWorldExample"
        snake = str_helper.to_snake_case(camel_case)
        assert snake == "hello_world_example"
        
        # Truncate with ellipsis
        long_text = "This is a very long text that needs to be truncated"
        truncated = str_helper.truncate(long_text, max_length=20)
        assert truncated == "This is a very lo..."
        assert len(truncated) == 20
        
    def test_string_parsing(self, str_helper):
        """Test string parsing utilities."""
        # Extract numbers
        text = "The price is $123.45 and quantity is 10"
        numbers = str_helper.extract_numbers(text)
        assert numbers == [123.45, 10]
        
        # Extract emails
        text = "Contact us at info@example.com or support@example.org"
        emails = str_helper.extract_emails(text)
        assert emails == ["info@example.com", "support@example.org"]
        
        # Parse key-value pairs
        kv_string = "name=John age=30 city=NewYork"
        parsed = str_helper.parse_key_value(kv_string)
        assert parsed == {"name": "John", "age": "30", "city": "NewYork"}
        
    def test_string_encoding(self, str_helper):
        """Test string encoding/decoding utilities."""
        # Base64
        text = "Hello World!"
        encoded = str_helper.base64_encode(text)
        decoded = str_helper.base64_decode(encoded)
        assert decoded == text
        
        # URL encoding
        url_unsafe = "Hello World! @#$%"
        url_safe = str_helper.url_encode(url_unsafe)
        assert url_safe == "Hello+World%21+%40%23%24%25"
        
        decoded_url = str_helper.url_decode(url_safe)
        assert decoded_url == url_unsafe
        
    def test_string_comparison(self, str_helper):
        """Test string comparison utilities."""
        # Fuzzy matching
        str1 = "Hello World"
        str2 = "Helo World"  # Typo
        
        similarity = str_helper.fuzzy_match(str1, str2)
        assert 0.8 < similarity < 1.0  # High but not perfect
        
        # Levenshtein distance
        distance = str_helper.levenshtein_distance(str1, str2)
        assert distance == 1  # One character difference
        
        # Case-insensitive comparison
        assert str_helper.equals_ignore_case("Hello", "HELLO")
        assert not str_helper.equals_ignore_case("Hello", "World")


class TestFileHelper:
    """Test cases for file operation utilities."""
    
    @pytest.fixture
    def file_helper(self):
        """Create FileHelper instance."""
        return FileHelper()
        
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
            
    def test_file_operations(self, file_helper, temp_dir):
        """Test basic file operations."""
        # Write file
        test_file = temp_dir / "test.txt"
        content = "Hello World!"
        
        file_helper.write_file(test_file, content)
        assert test_file.exists()
        
        # Read file
        read_content = file_helper.read_file(test_file)
        assert read_content == content
        
        # Append to file
        file_helper.append_file(test_file, "\nNew Line")
        updated_content = file_helper.read_file(test_file)
        assert updated_content == "Hello World!\nNew Line"
        
        # Copy file
        copy_file = temp_dir / "copy.txt"
        file_helper.copy_file(test_file, copy_file)
        assert copy_file.exists()
        assert file_helper.read_file(copy_file) == updated_content
        
        # Move file
        moved_file = temp_dir / "moved.txt"
        file_helper.move_file(copy_file, moved_file)
        assert moved_file.exists()
        assert not copy_file.exists()
        
        # Delete file
        file_helper.delete_file(moved_file)
        assert not moved_file.exists()
        
    def test_json_operations(self, file_helper, temp_dir):
        """Test JSON file operations."""
        json_file = temp_dir / "data.json"
        data = {
            "name": "Test",
            "value": 123,
            "nested": {"key": "value"},
            "array": [1, 2, 3]
        }
        
        # Write JSON
        file_helper.write_json(json_file, data, indent=2)
        assert json_file.exists()
        
        # Read JSON
        loaded_data = file_helper.read_json(json_file)
        assert loaded_data == data
        
        # Update JSON
        file_helper.update_json(json_file, {"new_key": "new_value"})
        updated_data = file_helper.read_json(json_file)
        assert updated_data["new_key"] == "new_value"
        assert updated_data["name"] == "Test"  # Original data preserved
        
    def test_csv_operations(self, file_helper, temp_dir):
        """Test CSV file operations."""
        csv_file = temp_dir / "data.csv"
        
        # Write CSV
        data = [
            {"name": "Alice", "age": 30, "city": "New York"},
            {"name": "Bob", "age": 25, "city": "London"},
            {"name": "Charlie", "age": 35, "city": "Tokyo"}
        ]
        
        file_helper.write_csv(csv_file, data)
        assert csv_file.exists()
        
        # Read CSV
        df = file_helper.read_csv(csv_file)
        assert len(df) == 3
        assert list(df.columns) == ["name", "age", "city"]
        
        # Append to CSV
        new_data = [{"name": "David", "age": 28, "city": "Paris"}]
        file_helper.append_csv(csv_file, new_data)
        
        df_updated = file_helper.read_csv(csv_file)
        assert len(df_updated) == 4
        assert df_updated.iloc[-1]["name"] == "David"
        
    def test_directory_operations(self, file_helper, temp_dir):
        """Test directory operations."""
        # Create nested directories
        nested_dir = temp_dir / "level1" / "level2" / "level3"
        file_helper.create_directory(nested_dir, parents=True)
        assert nested_dir.exists()
        
        # List directory contents
        files = ["file1.txt", "file2.txt", "file3.csv", "file4.json"]
        for filename in files:
            (temp_dir / filename).touch()
            
        # List all files
        all_files = file_helper.list_files(temp_dir)
        assert len(all_files) >= 4
        
        # List with pattern
        txt_files = file_helper.list_files(temp_dir, pattern="*.txt")
        assert len(txt_files) == 2
        
        # Get directory size
        file_helper.write_file(temp_dir / "large.txt", "x" * 1000)
        dir_size = file_helper.get_directory_size(temp_dir)
        assert dir_size >= 1000
        
        # Clean directory
        file_helper.clean_directory(temp_dir, keep_days=0)
        remaining_files = file_helper.list_files(temp_dir)
        assert len(remaining_files) == 0
        
    def test_file_monitoring(self, file_helper, temp_dir):
        """Test file monitoring utilities."""
        test_file = temp_dir / "monitor.txt"
        
        # Get file info
        file_helper.write_file(test_file, "Initial content")
        file_info = file_helper.get_file_info(test_file)
        
        assert file_info["size"] > 0
        assert "created" in file_info
        assert "modified" in file_info
        assert file_info["extension"] == ".txt"
        
        # Check if file changed
        initial_modified = file_info["modified"]
        time.sleep(0.1)
        
        file_helper.write_file(test_file, "Updated content")
        assert file_helper.has_file_changed(test_file, initial_modified)
        
        # File hash
        hash1 = file_helper.get_file_hash(test_file)
        file_helper.write_file(test_file, "Different content")
        hash2 = file_helper.get_file_hash(test_file)
        
        assert hash1 != hash2
        
    def test_safe_file_operations(self, file_helper, temp_dir):
        """Test safe file operations with error handling."""
        # Safe write with backup
        important_file = temp_dir / "important.txt"
        file_helper.write_file(important_file, "Original content")
        
        file_helper.safe_write(
            important_file,
            "New content",
            create_backup=True
        )
        
        backup_files = file_helper.list_files(temp_dir, pattern="important.txt.backup*")
        assert len(backup_files) == 1
        
        # Atomic write
        atomic_file = temp_dir / "atomic.txt"
        file_helper.atomic_write(atomic_file, "Atomic content")
        assert file_helper.read_file(atomic_file) == "Atomic content"
        
        # File locking
        lock_file = temp_dir / "locked.txt"
        
        with file_helper.file_lock(lock_file) as locked:
            file_helper.write_file(lock_file, "Locked content")
            
            # Try to acquire lock again (should fail or wait)
            lock_acquired = file_helper.try_acquire_lock(lock_file, timeout=0.1)
            assert not lock_acquired


class TestDataStructureHelper:
    """Test cases for data structure manipulation utilities."""
    
    @pytest.fixture
    def ds_helper(self):
        """Create DataStructureHelper instance."""
        return DataStructureHelper()
        
    def test_dict_operations(self, ds_helper):
        """Test dictionary manipulation utilities."""
        # Deep merge
        dict1 = {"a": 1, "b": {"c": 2, "d": 3}, "e": [1, 2]}
        dict2 = {"b": {"c": 4, "f": 5}, "e": [3, 4], "g": 6}
        
        merged = ds_helper.deep_merge(dict1, dict2)
        assert merged["a"] == 1
        assert merged["b"]["c"] == 4  # Overwritten
        assert merged["b"]["d"] == 3  # Preserved
        assert merged["b"]["f"] == 5  # Added
        assert merged["e"] == [1, 2, 3, 4]  # Lists merged
        assert merged["g"] == 6
        
        # Flatten nested dict
        nested = {
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3
                }
            }
        }
        
        flattened = ds_helper.flatten_dict(nested)
        assert flattened == {"a": 1, "b.c": 2, "b.d.e": 3}
        
        # Unflatten dict
        unflattened = ds_helper.unflatten_dict(flattened)
        assert unflattened == nested
        
        # Get nested value
        value = ds_helper.get_nested_value(nested, "b.d.e")
        assert value == 3
        
        # Set nested value
        ds_helper.set_nested_value(nested, "b.d.f", 4)
        assert nested["b"]["d"]["f"] == 4
        
    def test_list_operations(self, ds_helper):
        """Test list manipulation utilities."""
        # Chunk list
        long_list = list(range(100))
        chunks = ds_helper.chunk_list(long_list, chunk_size=15)
        
        assert len(chunks) == 7  # 6 full chunks + 1 partial
        assert len(chunks[0]) == 15
        assert len(chunks[-1]) == 10  # Last chunk
        
        # Flatten nested lists
        nested_list = [[1, 2], [3, [4, 5]], [6, 7, [8, [9, 10]]]]
        flattened = ds_helper.flatten_list(nested_list)
        assert flattened == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # Remove duplicates preserving order
        duplicates = [1, 2, 3, 2, 4, 3, 5, 1]
        unique = ds_helper.remove_duplicates(duplicates, preserve_order=True)
        assert unique == [1, 2, 3, 4, 5]
        
        # Group by key
        items = [
            {"type": "A", "value": 1},
            {"type": "B", "value": 2},
            {"type": "A", "value": 3},
            {"type": "B", "value": 4}
        ]
        
        grouped = ds_helper.group_by(items, key="type")
        assert len(grouped["A"]) == 2
        assert len(grouped["B"]) == 2
        
    def test_set_operations(self, ds_helper):
        """Test set operation utilities."""
        set1 = {1, 2, 3, 4, 5}
        set2 = {4, 5, 6, 7, 8}
        
        # Set operations with details
        union_result = ds_helper.set_union_with_source(set1, set2)
        assert union_result == {
            1: "set1", 2: "set1", 3: "set1",
            4: "both", 5: "both",
            6: "set2", 7: "set2", 8: "set2"
        }
        
        # Symmetric difference with details
        diff_result = ds_helper.symmetric_difference_detailed(set1, set2)
        assert diff_result["only_in_set1"] == {1, 2, 3}
        assert diff_result["only_in_set2"] == {6, 7, 8}
        assert diff_result["in_both"] == {4, 5}
        
    def test_tree_operations(self, ds_helper):
        """Test tree structure operations."""
        # Build tree from parent-child relationships
        relationships = [
            {"id": 1, "parent_id": None, "name": "Root"},
            {"id": 2, "parent_id": 1, "name": "Child1"},
            {"id": 3, "parent_id": 1, "name": "Child2"},
            {"id": 4, "parent_id": 2, "name": "Grandchild1"},
            {"id": 5, "parent_id": 2, "name": "Grandchild2"}
        ]
        
        tree = ds_helper.build_tree(relationships)
        assert tree["id"] == 1
        assert len(tree["children"]) == 2
        assert len(tree["children"][0]["children"]) == 2
        
        # Traverse tree
        visited = []
        ds_helper.traverse_tree(
            tree,
            lambda node: visited.append(node["name"])
        )
        assert visited == ["Root", "Child1", "Grandchild1", "Grandchild2", "Child2"]
        
        # Find in tree
        found = ds_helper.find_in_tree(tree, lambda n: n["id"] == 4)
        assert found["name"] == "Grandchild1"
        
    def test_data_transformation(self, ds_helper):
        """Test data transformation utilities."""
        # Pivot data
        data = [
            {"date": "2023-01-01", "metric": "sales", "value": 100},
            {"date": "2023-01-01", "metric": "costs", "value": 80},
            {"date": "2023-01-02", "metric": "sales", "value": 120},
            {"date": "2023-01-02", "metric": "costs", "value": 90}
        ]
        
        pivoted = ds_helper.pivot_data(
            data,
            index="date",
            columns="metric",
            values="value"
        )
        
        assert pivoted["2023-01-01"]["sales"] == 100
        assert pivoted["2023-01-01"]["costs"] == 80
        
        # Transpose data
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        transposed = ds_helper.transpose_matrix(matrix)
        assert transposed == [[1, 4, 7], [2, 5, 8], [3, 6, 9]]


class TestConfigHelper:
    """Test cases for configuration utilities."""
    
    @pytest.fixture
    def config_helper(self):
        """Create ConfigHelper instance."""
        return ConfigHelper()
        
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        return {
            "app": {
                "name": "TradingBot",
                "version": "1.0.0",
                "debug": False
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "trading_db",
                "credentials": {
                    "username": "${DB_USER}",
                    "password": "${DB_PASS}"
                }
            },
            "trading": {
                "max_positions": 10,
                "risk_percentage": 0.02,
                "strategies": ["grid", "momentum"]
            }
        }
        
    def test_config_loading(self, config_helper, temp_dir, sample_config):
        """Test configuration loading from various sources."""
        # Load from JSON file
        json_file = temp_dir / "config.json"
        with open(json_file, 'w') as f:
            json.dump(sample_config, f)
            
        config_from_json = config_helper.load_config(json_file)
        assert config_from_json == sample_config
        
        # Load from YAML file
        yaml_file = temp_dir / "config.yaml"
        yaml_content = """
app:
  name: TradingBot
  version: 1.0.0
  debug: false
database:
  host: localhost
  port: 5432
"""
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
            
        config_from_yaml = config_helper.load_config(yaml_file)
        assert config_from_yaml["app"]["name"] == "TradingBot"
        
    def test_environment_substitution(self, config_helper, sample_config):
        """Test environment variable substitution."""
        # Set environment variables
        with patch.dict(os.environ, {"DB_USER": "trader", "DB_PASS": "secret123"}):
            resolved_config = config_helper.resolve_env_vars(sample_config)
            
            assert resolved_config["database"]["credentials"]["username"] == "trader"
            assert resolved_config["database"]["credentials"]["password"] == "secret123"
            
        # Missing environment variable
        with patch.dict(os.environ, {"DB_USER": "trader"}):
            with pytest.raises(KeyError):
                config_helper.resolve_env_vars(sample_config, raise_on_missing=True)
                
    def test_config_validation(self, config_helper, sample_config):
        """Test configuration validation."""
        # Define schema
        schema = {
            "app": {
                "name": {"type": "string", "required": True},
                "version": {"type": "string", "required": True},
                "debug": {"type": "boolean", "default": False}
            },
            "database": {
                "host": {"type": "string", "required": True},
                "port": {"type": "integer", "min": 1, "max": 65535},
                "name": {"type": "string", "required": True}
            },
            "trading": {
                "max_positions": {"type": "integer", "min": 1, "max": 100},
                "risk_percentage": {"type": "float", "min": 0, "max": 1}
            }
        }
        
        # Valid config
        is_valid, errors = config_helper.validate_config(sample_config, schema)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid config
        invalid_config = sample_config.copy()
        invalid_config["trading"]["max_positions"] = 200  # Exceeds max
        
        is_valid, errors = config_helper.validate_config(invalid_config, schema)
        assert not is_valid
        assert any("max_positions" in str(e) for e in errors)
        
    def test_config_merging(self, config_helper):
        """Test configuration merging with precedence."""
        # Base config
        base_config = {
            "app": {"name": "TradingBot", "version": "1.0.0"},
            "trading": {"max_positions": 10}
        }
        
        # Override config
        override_config = {
            "app": {"version": "2.0.0", "debug": True},
            "trading": {"risk_percentage": 0.03}
        }
        
        # Merge with override
        merged = config_helper.merge_configs(base_config, override_config)
        
        assert merged["app"]["name"] == "TradingBot"  # From base
        assert merged["app"]["version"] == "2.0.0"    # Overridden
        assert merged["app"]["debug"] is True          # New field
        assert merged["trading"]["max_positions"] == 10
        assert merged["trading"]["risk_percentage"] == 0.03
        
    def test_config_encryption(self, config_helper):
        """Test sensitive configuration encryption."""
        sensitive_config = {
            "api_keys": {
                "exchange": "secret_api_key_123",
                "webhook": "webhook_secret_456"
            }
        }
        
        # Encrypt sensitive fields
        encrypted = config_helper.encrypt_sensitive_fields(
            sensitive_config,
            fields_to_encrypt=["api_keys.exchange", "api_keys.webhook"],
            key="encryption_key_123"
        )
        
        assert encrypted["api_keys"]["exchange"] != "secret_api_key_123"
        assert encrypted["api_keys"]["exchange"].startswith("ENC:")
        
        # Decrypt
        decrypted = config_helper.decrypt_sensitive_fields(
            encrypted,
            fields_to_decrypt=["api_keys.exchange", "api_keys.webhook"],
            key="encryption_key_123"
        )
        
        assert decrypted == sensitive_config


class TestLoggingHelper:
    """Test cases for logging utilities."""
    
    @pytest.fixture
    def log_helper(self):
        """Create LoggingHelper instance."""
        return LoggingHelper()
        
    def test_logger_setup(self, log_helper, temp_dir):
        """Test logger setup and configuration."""
        log_file = temp_dir / "test.log"
        
        # Setup logger
        logger = log_helper.setup_logger(
            name="test_logger",
            level=logging.INFO,
            log_file=log_file,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Test logging
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        
        # Check log file
        assert log_file.exists()
        log_content = log_file.read_text()
        assert "Test info message" in log_content
        assert "Test warning message" in log_content
        assert "Test error message" in log_content
        
    def test_structured_logging(self, log_helper):
        """Test structured logging with context."""
        logger = log_helper.get_structured_logger("structured_test")
        
        # Log with context
        log_helper.log_with_context(
            logger,
            "Order executed",
            level="info",
            order_id="123",
            symbol="BTC/USDT",
            quantity=0.5,
            price=50000
        )
        
        # Test JSON formatter
        json_formatter = log_helper.get_json_formatter()
        
        # Create log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.order_id = "123"
        record.symbol = "BTC/USDT"
        
        formatted = json_formatter.format(record)
        parsed = json.loads(formatted)
        
        assert parsed["message"] == "Test message"
        assert parsed["order_id"] == "123"
        assert parsed["symbol"] == "BTC/USDT"
        
    def test_log_rotation(self, log_helper, temp_dir):
        """Test log rotation functionality."""
        log_file = temp_dir / "rotating.log"
        
        # Setup rotating logger
        logger = log_helper.setup_rotating_logger(
            name="rotating_test",
            log_file=log_file,
            max_bytes=1024,  # Small size to trigger rotation
            backup_count=3
        )
        
        # Write enough logs to trigger rotation
        for i in range(100):
            logger.info(f"Log message {i} - " + "x" * 50)
            
        # Check for rotated files
        log_files = list(temp_dir.glob("rotating.log*"))
        assert len(log_files) > 1  # Original + rotated files
        
    def test_performance_logging(self, log_helper):
        """Test performance timing decorator."""
        logger = log_helper.get_logger("performance_test")
        
        @log_helper.log_execution_time(logger)
        def slow_function():
            time.sleep(0.1)
            return "result"
            
        # Execute function
        result = slow_function()
        assert result == "result"
        
        # Test context manager
        with log_helper.log_duration(logger, "test_operation"):
            time.sleep(0.05)
            
    def test_error_logging(self, log_helper):
        """Test error logging with traceback."""
        logger = log_helper.get_logger("error_test")
        
        # Log exception
        try:
            1 / 0
        except ZeroDivisionError as e:
            log_helper.log_exception(
                logger,
                "Division error occurred",
                exc_info=e,
                context={"operation": "division", "values": [1, 0]}
            )
            
        # Test error aggregation
        error_stats = log_helper.aggregate_errors(logger)
        # Would need to implement error tracking in LoggingHelper


class TestRetryHelper:
    """Test cases for retry and error handling utilities."""
    
    @pytest.fixture
    def retry_helper(self):
        """Create RetryHelper instance."""
        return RetryHelper()
        
    def test_basic_retry(self, retry_helper):
        """Test basic retry functionality."""
        attempt_count = 0
        
        @retry_helper.retry(max_attempts=3, delay=0.01)
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Temporary error")
            return "success"
            
        result = flaky_function()
        assert result == "success"
        assert attempt_count == 3
        
    def test_retry_with_backoff(self, retry_helper):
        """Test retry with exponential backoff."""
        attempts = []
        
        @retry_helper.retry(
            max_attempts=4,
            delay=0.01,
            backoff_factor=2,
            max_delay=0.1
        )
        def backoff_function():
            attempts.append(time.time())
            if len(attempts) < 3:
                raise Exception("Retry needed")
            return "done"
            
        result = backoff_function()
        assert result == "done"
        assert len(attempts) == 3
        
        # Check backoff timing
        if len(attempts) > 2:
            delay1 = attempts[1] - attempts[0]
            delay2 = attempts[2] - attempts[1]
            assert delay2 > delay1  # Exponential backoff
            
    def test_retry_specific_exceptions(self, retry_helper):
        """Test retry only on specific exceptions."""
        @retry_helper.retry(
            max_attempts=3,
            retry_on=(ValueError, TypeError),
            delay=0.01
        )
        def selective_retry():
            if not hasattr(selective_retry, 'count'):
                selective_retry.count = 0
            selective_retry.count += 1
            
            if selective_retry.count == 1:
                raise ValueError("Retryable")
            elif selective_retry.count == 2:
                raise TypeError("Also retryable")
            elif selective_retry.count == 3:
                raise RuntimeError("Not retryable")
                
        with pytest.raises(RuntimeError):
            selective_retry()
            
    def test_async_retry(self, retry_helper):
        """Test async function retry."""
        attempt_count = 0
        
        @retry_helper.async_retry(max_attempts=3, delay=0.01)
        async def async_flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Async temporary error")
            return "async success"
            
        result = asyncio.run(async_flaky_function())
        assert result == "async success"
        assert attempt_count == 3
        
    def test_circuit_breaker(self, retry_helper):
        """Test circuit breaker pattern."""
        breaker = retry_helper.create_circuit_breaker(
            failure_threshold=3,
            recovery_timeout=0.1,
            expected_exception=Exception
        )
        
        @breaker
        def protected_function():
            if not hasattr(protected_function, 'count'):
                protected_function.count = 0
            protected_function.count += 1
            
            if protected_function.count <= 3:
                raise Exception("Service unavailable")
            return "service ok"
            
        # First 3 calls should fail and open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                protected_function()
                
        # Circuit should be open, calls fail immediately
        with pytest.raises(Exception) as exc_info:
            protected_function()
        assert "Circuit breaker is open" in str(exc_info.value)
        
        # Wait for recovery
        time.sleep(0.15)
        
        # Circuit should be half-open, next call succeeds
        result = protected_function()
        assert result == "service ok"


class TestDecoratorHelper:
    """Test cases for decorator utilities."""
    
    @pytest.fixture
    def decorator_helper(self):
        """Create DecoratorHelper instance."""
        return DecoratorHelper()
        
    def test_timing_decorator(self, decorator_helper):
        """Test function timing decorator."""
        execution_times = []
        
        @decorator_helper.time_it(callback=execution_times.append)
        def timed_function(duration):
            time.sleep(duration)
            return "done"
            
        result = timed_function(0.05)
        assert result == "done"
        assert len(execution_times) == 1
        assert 0.04 < execution_times[0] < 0.06
        
    def test_memoization_decorator(self, decorator_helper):
        """Test memoization decorator."""
        call_count = 0
        
        @decorator_helper.memoize(cache_size=128)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)
            return x + y
            
        # First call
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Cached call
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Not called again
        
        # Different arguments
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2
        
    def test_rate_limit_decorator(self, decorator_helper):
        """Test rate limiting decorator."""
        calls = []
        
        @decorator_helper.rate_limit(calls=2, period=0.1)
        def rate_limited_function():
            calls.append(time.time())
            return len(calls)
            
        # First two calls should succeed
        assert rate_limited_function() == 1
        assert rate_limited_function() == 2
        
        # Third call should be rate limited
        with pytest.raises(Exception) as exc_info:
            rate_limited_function()
        assert "Rate limit exceeded" in str(exc_info.value)
        
        # Wait for period to reset
        time.sleep(0.11)
        
        # Should work again
        assert rate_limited_function() == 3
        
    def test_validate_args_decorator(self, decorator_helper):
        """Test argument validation decorator."""
        @decorator_helper.validate_args(
            x=lambda v: v > 0,
            y=lambda v: isinstance(v, str)
        )
        def validated_function(x, y):
            return f"{y}: {x}"
            
        # Valid arguments
        result = validated_function(5, "Value")
        assert result == "Value: 5"
        
        # Invalid arguments
        with pytest.raises(ValueError):
            validated_function(-1, "Value")  # x must be > 0
            
        with pytest.raises(ValueError):
            validated_function(5, 123)  # y must be string
            
    def test_deprecated_decorator(self, decorator_helper):
        """Test deprecation warning decorator."""
        @decorator_helper.deprecated(
            message="Use new_function instead",
            version="2.0.0"
        )
        def old_function():
            return "old result"
            
        with pytest.warns(DeprecationWarning) as warnings:
            result = old_function()
            
        assert result == "old result"
        assert len(warnings) == 1
        assert "Use new_function instead" in str(warnings[0].message)


class TestCacheHelper:
    """Test cases for caching utilities."""
    
    @pytest.fixture
    def cache_helper(self):
        """Create CacheHelper instance."""
        return CacheHelper()
        
    def test_memory_cache(self, cache_helper):
        """Test in-memory caching."""
        cache = cache_helper.create_memory_cache(max_size=100)
        
        # Set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # TTL
        cache.set("key2", "value2", ttl=0.05)
        assert cache.get("key2") == "value2"
        
        time.sleep(0.06)
        assert cache.get("key2") is None  # Expired
        
        # Cache size limit
        for i in range(150):
            cache.set(f"key_{i}", f"value_{i}")
            
        assert len(cache) <= 100  # Size limited
        
    def test_disk_cache(self, cache_helper, temp_dir):
        """Test disk-based caching."""
        cache_dir = temp_dir / "cache"
        cache = cache_helper.create_disk_cache(cache_dir)
        
        # Store various types
        cache.set("string", "Hello World")
        cache.set("number", 42)
        cache.set("list", [1, 2, 3])
        cache.set("dict", {"a": 1, "b": 2})
        
        # Retrieve
        assert cache.get("string") == "Hello World"
        assert cache.get("number") == 42
        assert cache.get("list") == [1, 2, 3]
        assert cache.get("dict") == {"a": 1, "b": 2}
        
        # Persistence
        cache2 = cache_helper.create_disk_cache(cache_dir)
        assert cache2.get("string") == "Hello World"  # Persisted
        
    def test_cache_invalidation(self, cache_helper):
        """Test cache invalidation strategies."""
        cache = cache_helper.create_memory_cache()
        
        # Add items with tags
        cache.set("user_1", {"name": "Alice"}, tags=["user", "active"])
        cache.set("user_2", {"name": "Bob"}, tags=["user", "inactive"])
        cache.set("product_1", {"name": "Widget"}, tags=["product"])
        
        # Invalidate by tag
        cache.invalidate_by_tag("user")
        
        assert cache.get("user_1") is None
        assert cache.get("user_2") is None
        assert cache.get("product_1") is not None
        
        # Invalidate by pattern
        cache.set("temp_1", "value1")
        cache.set("temp_2", "value2")
        cache.set("permanent_1", "value3")
        
        cache.invalidate_by_pattern("temp_*")
        
        assert cache.get("temp_1") is None
        assert cache.get("temp_2") is None
        assert cache.get("permanent_1") is not None


class TestAsyncHelper:
    """Test cases for async utilities."""
    
    @pytest.fixture
    def async_helper(self):
        """Create AsyncHelper instance."""
        return AsyncHelper()
        
    @pytest.mark.asyncio
    async def test_async_timeout(self, async_helper):
        """Test async timeout functionality."""
        async def slow_operation():
            await asyncio.sleep(0.5)
            return "completed"
            
        # Should timeout
        with pytest.raises(asyncio.TimeoutError):
            result = await async_helper.with_timeout(slow_operation(), timeout=0.1)
            
        # Should complete
        async def fast_operation():
            await asyncio.sleep(0.01)
            return "fast"
            
        result = await async_helper.with_timeout(fast_operation(), timeout=0.1)
        assert result == "fast"
        
    @pytest.mark.asyncio
    async def test_async_retry(self, async_helper):
        """Test async retry functionality."""
        attempt_count = 0
        
        async def flaky_async_operation():
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count < 3:
                raise Exception("Temporary failure")
            return "success"
            
        result = await async_helper.retry_async(
            flaky_async_operation(),
            max_attempts=3,
            delay=0.01
        )
        
        assert result == "success"
        assert attempt_count == 3
        
    @pytest.mark.asyncio
    async def test_async_batch_processing(self, async_helper):
        """Test async batch processing."""
        async def process_item(item):
            await asyncio.sleep(0.01)
            return item * 2
            
        items = list(range(10))
        
        # Process with concurrency limit
        results = await async_helper.process_batch(
            items,
            process_item,
            max_concurrent=3
        )
        
        assert results == [i * 2 for i in items]
        
    @pytest.mark.asyncio
    async def test_async_rate_limiting(self, async_helper):
        """Test async rate limiting."""
        calls = []
        
        async def rate_limited_operation():
            calls.append(time.time())
            return len(calls)
            
        # Create rate limiter
        rate_limiter = async_helper.create_rate_limiter(
            rate=5,  # 5 calls
            period=0.1  # per 0.1 seconds
        )
        
        # Make rapid calls
        tasks = []
        for _ in range(10):
            task = rate_limiter.acquire(rate_limited_operation())
            tasks.append(task)
            
        results = await asyncio.gather(*tasks)
        
        # Check timing
        assert len(calls) == 10
        
        # Calls should be spread out
        for i in range(5, 10):
            assert calls[i] - calls[i-5] >= 0.09  # Close to 0.1 seconds