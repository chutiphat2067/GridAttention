"""
Unit tests for Data Validation Utilities.

Tests cover:
- Input data validation and sanitization
- Data type checking and conversion
- Range and constraint validation
- Missing data handling
- Data quality assessment
- Schema validation
- Time series data validation
- Cross-field validation
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import json
from unittest.mock import Mock, patch

# GridAttention project imports
from utils.data_validation import (
    DataValidator,
    ValidationRule,
    ValidationResult,
    SchemaValidator,
    TimeSeriesValidator,
    DataQualityChecker,
    DataSanitizer,
    ValidationError,
    DataType,
    ConstraintType
)


class TestValidationRule:
    """Test cases for ValidationRule configuration."""
    
    def test_rule_creation(self):
        """Test creating validation rules."""
        # Numeric range rule
        range_rule = ValidationRule(
            field='price',
            data_type=DataType.DECIMAL,
            constraints={
                'min': Decimal('0'),
                'max': Decimal('1000000'),
                'precision': 2
            },
            required=True
        )
        
        assert range_rule.field == 'price'
        assert range_rule.data_type == DataType.DECIMAL
        assert range_rule.required is True
        
    def test_constraint_types(self):
        """Test different constraint types."""
        # String pattern constraint
        pattern_rule = ValidationRule(
            field='symbol',
            data_type=DataType.STRING,
            constraints={
                'pattern': r'^[A-Z]{3,4}/[A-Z]{3,4}$',
                'min_length': 7,
                'max_length': 9
            }
        )
        
        # Enum constraint
        enum_rule = ValidationRule(
            field='order_type',
            data_type=DataType.STRING,
            constraints={
                'enum': ['market', 'limit', 'stop', 'stop_limit']
            }
        )
        
        # Custom validation function
        def custom_validator(value):
            return value % 100 == 0  # Must be multiple of 100
            
        custom_rule = ValidationRule(
            field='lot_size',
            data_type=DataType.INTEGER,
            constraints={
                'custom': custom_validator
            }
        )
        
        assert 'pattern' in pattern_rule.constraints
        assert 'enum' in enum_rule.constraints
        assert callable(custom_rule.constraints['custom'])
        
    def test_rule_validation(self):
        """Test rule application on values."""
        rule = ValidationRule(
            field='quantity',
            data_type=DataType.DECIMAL,
            constraints={
                'min': Decimal('0.001'),
                'max': Decimal('10000'),
                'precision': 8
            },
            required=True
        )
        
        # Valid values
        assert rule.validate(Decimal('1.5')).is_valid
        assert rule.validate(Decimal('0.001')).is_valid
        assert rule.validate(Decimal('10000')).is_valid
        
        # Invalid values
        assert not rule.validate(Decimal('0')).is_valid  # Below min
        assert not rule.validate(Decimal('10001')).is_valid  # Above max
        assert not rule.validate(None).is_valid  # Required but None
        assert not rule.validate('1.5').is_valid  # Wrong type


class TestDataValidator:
    """Test cases for the main DataValidator class."""
    
    @pytest.fixture
    def data_validator(self):
        """Create DataValidator instance with sample rules."""
        rules = [
            ValidationRule(
                field='symbol',
                data_type=DataType.STRING,
                constraints={'pattern': r'^[A-Z]+/[A-Z]+$'},
                required=True
            ),
            ValidationRule(
                field='price',
                data_type=DataType.DECIMAL,
                constraints={'min': Decimal('0'), 'max': Decimal('1000000')},
                required=True
            ),
            ValidationRule(
                field='quantity',
                data_type=DataType.DECIMAL,
                constraints={'min': Decimal('0.001')},
                required=True
            ),
            ValidationRule(
                field='timestamp',
                data_type=DataType.DATETIME,
                required=True
            ),
            ValidationRule(
                field='order_type',
                data_type=DataType.STRING,
                constraints={'enum': ['market', 'limit']},
                required=True
            )
        ]
        
        return DataValidator(rules)
        
    def test_single_record_validation(self, data_validator):
        """Test validating a single data record."""
        # Valid record
        valid_record = {
            'symbol': 'BTC/USDT',
            'price': Decimal('50000'),
            'quantity': Decimal('0.5'),
            'timestamp': datetime.now(),
            'order_type': 'limit'
        }
        
        result = data_validator.validate_record(valid_record)
        assert result.is_valid
        assert len(result.errors) == 0
        
        # Invalid record
        invalid_record = {
            'symbol': 'btc-usdt',  # Wrong format
            'price': Decimal('-100'),  # Negative
            'quantity': Decimal('0'),  # Below min
            'timestamp': 'not a datetime',  # Wrong type
            'order_type': 'stop'  # Not in enum
        }
        
        result = data_validator.validate_record(invalid_record)
        assert not result.is_valid
        assert len(result.errors) == 5
        
    def test_batch_validation(self, data_validator):
        """Test validating multiple records."""
        records = [
            {
                'symbol': 'BTC/USDT',
                'price': Decimal('50000'),
                'quantity': Decimal('0.5'),
                'timestamp': datetime.now(),
                'order_type': 'limit'
            },
            {
                'symbol': 'ETH/USDT',
                'price': Decimal('3000'),
                'quantity': Decimal('1.0'),
                'timestamp': datetime.now(),
                'order_type': 'market'
            },
            {
                'symbol': 'INVALID',  # Bad symbol
                'price': Decimal('100'),
                'quantity': Decimal('0.1'),
                'timestamp': datetime.now(),
                'order_type': 'limit'
            }
        ]
        
        results = data_validator.validate_batch(records)
        
        assert len(results) == 3
        assert results[0].is_valid
        assert results[1].is_valid
        assert not results[2].is_valid
        
        # Summary statistics
        summary = data_validator.get_validation_summary(results)
        assert summary['total'] == 3
        assert summary['valid'] == 2
        assert summary['invalid'] == 1
        assert summary['error_rate'] == 1/3
        
    def test_missing_field_handling(self, data_validator):
        """Test handling of missing required fields."""
        incomplete_record = {
            'symbol': 'BTC/USDT',
            'price': Decimal('50000')
            # Missing quantity, timestamp, order_type
        }
        
        result = data_validator.validate_record(incomplete_record)
        assert not result.is_valid
        
        missing_fields = [e for e in result.errors if e['type'] == 'missing_field']
        assert len(missing_fields) == 3
        
    def test_optional_field_validation(self):
        """Test validation with optional fields."""
        rules = [
            ValidationRule(
                field='symbol',
                data_type=DataType.STRING,
                required=True
            ),
            ValidationRule(
                field='comment',
                data_type=DataType.STRING,
                constraints={'max_length': 200},
                required=False
            )
        ]
        
        validator = DataValidator(rules)
        
        # Record without optional field
        record1 = {'symbol': 'BTC/USDT'}
        assert validator.validate_record(record1).is_valid
        
        # Record with valid optional field
        record2 = {'symbol': 'BTC/USDT', 'comment': 'Test order'}
        assert validator.validate_record(record2).is_valid
        
        # Record with invalid optional field
        record3 = {'symbol': 'BTC/USDT', 'comment': 'x' * 201}  # Too long
        assert not validator.validate_record(record3).is_valid
        
    def test_custom_validation_logic(self, data_validator):
        """Test custom validation functions."""
        # Add custom cross-field validation
        def validate_price_quantity(record):
            """Total value must be at least $10."""
            if 'price' in record and 'quantity' in record:
                total = record['price'] * record['quantity']
                return total >= Decimal('10'), f"Total value {total} is below minimum $10"
            return True, ""
            
        data_validator.add_custom_validator(validate_price_quantity)
        
        # Valid total value
        record1 = {
            'symbol': 'BTC/USDT',
            'price': Decimal('50000'),
            'quantity': Decimal('0.001'),  # Total = $50
            'timestamp': datetime.now(),
            'order_type': 'limit'
        }
        assert data_validator.validate_record(record1).is_valid
        
        # Invalid total value
        record2 = {
            'symbol': 'BTC/USDT',
            'price': Decimal('50000'),
            'quantity': Decimal('0.0001'),  # Total = $5
            'timestamp': datetime.now(),
            'order_type': 'limit'
        }
        result = data_validator.validate_record(record2)
        assert not result.is_valid
        assert any('below minimum' in str(e) for e in result.errors)


class TestSchemaValidator:
    """Test cases for schema-based validation."""
    
    @pytest.fixture
    def schema_validator(self):
        """Create SchemaValidator with sample schema."""
        schema = {
            'type': 'object',
            'properties': {
                'order': {
                    'type': 'object',
                    'properties': {
                        'id': {'type': 'string', 'pattern': '^[A-Z0-9]{8}$'},
                        'symbol': {'type': 'string'},
                        'side': {'type': 'string', 'enum': ['buy', 'sell']},
                        'price': {'type': 'number', 'minimum': 0},
                        'quantity': {'type': 'number', 'minimum': 0.001},
                        'timestamp': {'type': 'string', 'format': 'date-time'}
                    },
                    'required': ['id', 'symbol', 'side', 'price', 'quantity']
                },
                'metadata': {
                    'type': 'object',
                    'properties': {
                        'source': {'type': 'string'},
                        'version': {'type': 'number'}
                    }
                }
            },
            'required': ['order']
        }
        
        return SchemaValidator(schema)
        
    def test_valid_json_structure(self, schema_validator):
        """Test validation of valid JSON structure."""
        valid_data = {
            'order': {
                'id': 'ABC12345',
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'price': 50000,
                'quantity': 0.5,
                'timestamp': '2023-01-01T12:00:00Z'
            },
            'metadata': {
                'source': 'api',
                'version': 1.0
            }
        }
        
        result = schema_validator.validate(valid_data)
        assert result.is_valid
        
    def test_invalid_json_structure(self, schema_validator):
        """Test validation of invalid JSON structure."""
        # Missing required field
        invalid_data1 = {
            'order': {
                'id': 'ABC12345',
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'price': 50000
                # Missing quantity
            }
        }
        
        result1 = schema_validator.validate(invalid_data1)
        assert not result1.is_valid
        assert 'quantity' in str(result1.errors[0])
        
        # Invalid enum value
        invalid_data2 = {
            'order': {
                'id': 'ABC12345',
                'symbol': 'BTC/USDT',
                'side': 'hold',  # Not in enum
                'price': 50000,
                'quantity': 0.5
            }
        }
        
        result2 = schema_validator.validate(invalid_data2)
        assert not result2.is_valid
        assert 'enum' in str(result2.errors[0])
        
    def test_nested_schema_validation(self, schema_validator):
        """Test validation of nested structures."""
        complex_schema = {
            'type': 'object',
            'properties': {
                'portfolio': {
                    'type': 'object',
                    'properties': {
                        'positions': {
                            'type': 'array',
                            'items': {
                                'type': 'object',
                                'properties': {
                                    'symbol': {'type': 'string'},
                                    'quantity': {'type': 'number'},
                                    'entry_price': {'type': 'number'}
                                },
                                'required': ['symbol', 'quantity']
                            }
                        }
                    }
                }
            }
        }
        
        validator = SchemaValidator(complex_schema)
        
        valid_data = {
            'portfolio': {
                'positions': [
                    {'symbol': 'BTC/USDT', 'quantity': 0.5, 'entry_price': 50000},
                    {'symbol': 'ETH/USDT', 'quantity': 5.0, 'entry_price': 3000}
                ]
            }
        }
        
        assert validator.validate(valid_data).is_valid
        
    def test_schema_evolution(self, schema_validator):
        """Test handling schema version changes."""
        # Version 1 schema
        schema_v1 = {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'value': {'type': 'number'}
            },
            'required': ['name', 'value']
        }
        
        # Version 2 schema (backward compatible)
        schema_v2 = {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'value': {'type': 'number'},
                'description': {'type': 'string'}  # New optional field
            },
            'required': ['name', 'value']
        }
        
        validator_v1 = SchemaValidator(schema_v1)
        validator_v2 = SchemaValidator(schema_v2)
        
        # V1 data should be valid in both schemas
        v1_data = {'name': 'test', 'value': 123}
        assert validator_v1.validate(v1_data).is_valid
        assert validator_v2.validate(v1_data).is_valid
        
        # V2 data with new field
        v2_data = {'name': 'test', 'value': 123, 'description': 'test desc'}
        assert not validator_v1.validate(v2_data).is_valid  # Extra field in strict mode
        assert validator_v2.validate(v2_data).is_valid


class TestTimeSeriesValidator:
    """Test cases for time series data validation."""
    
    @pytest.fixture
    def ts_validator(self):
        """Create TimeSeriesValidator instance."""
        return TimeSeriesValidator(
            frequency='5min',
            max_gap_tolerance=3,  # Max 3 missing periods
            required_columns=['open', 'high', 'low', 'close', 'volume']
        )
        
    @pytest.fixture
    def sample_timeseries(self):
        """Create sample time series data."""
        dates = pd.date_range(start='2023-01-01 09:00', periods=100, freq='5min')
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': 50000 + np.random.randn(100) * 100,
            'high': 50100 + np.random.randn(100) * 100,
            'low': 49900 + np.random.randn(100) * 100,
            'close': 50000 + np.random.randn(100) * 100,
            'volume': np.random.randint(100, 1000, 100)
        })
        
        return data.set_index('timestamp')
        
    def test_frequency_validation(self, ts_validator, sample_timeseries):
        """Test time series frequency validation."""
        result = ts_validator.validate_frequency(sample_timeseries)
        
        assert result.is_valid
        assert result.metadata['detected_frequency'] == '5min'
        assert result.metadata['is_regular'] is True
        
        # Create irregular series
        irregular_series = sample_timeseries.copy()
        irregular_series = irregular_series.drop(irregular_series.index[10:15])
        
        result_irregular = ts_validator.validate_frequency(irregular_series)
        assert not result_irregular.is_valid
        assert 'gap' in str(result_irregular.errors[0])
        
    def test_missing_data_detection(self, ts_validator, sample_timeseries):
        """Test detection of missing data points."""
        # Remove some data points
        series_with_gaps = sample_timeseries.copy()
        series_with_gaps = series_with_gaps.drop(series_with_gaps.index[20:23])  # 3 missing
        
        gaps = ts_validator.find_gaps(series_with_gaps)
        
        assert len(gaps) == 1
        assert gaps[0]['start'] == sample_timeseries.index[20]
        assert gaps[0]['end'] == sample_timeseries.index[22]
        assert gaps[0]['periods'] == 3
        
    def test_ohlc_validation(self, ts_validator, sample_timeseries):
        """Test OHLC data consistency validation."""
        # Valid OHLC
        result = ts_validator.validate_ohlc(sample_timeseries)
        assert result.is_valid
        
        # Create invalid OHLC (high < low)
        invalid_ohlc = sample_timeseries.copy()
        invalid_ohlc.loc[invalid_ohlc.index[10], 'high'] = 49000
        invalid_ohlc.loc[invalid_ohlc.index[10], 'low'] = 51000
        
        result_invalid = ts_validator.validate_ohlc(invalid_ohlc)
        assert not result_invalid.is_valid
        assert 'high < low' in str(result_invalid.errors[0])
        
        # Close outside high/low range
        invalid_ohlc2 = sample_timeseries.copy()
        invalid_ohlc2.loc[invalid_ohlc2.index[20], 'close'] = 55000
        invalid_ohlc2.loc[invalid_ohlc2.index[20], 'high'] = 52000
        
        result_invalid2 = ts_validator.validate_ohlc(invalid_ohlc2)
        assert not result_invalid2.is_valid
        
    def test_volume_validation(self, ts_validator, sample_timeseries):
        """Test volume data validation."""
        # Add zero volume
        zero_volume = sample_timeseries.copy()
        zero_volume.loc[zero_volume.index[30], 'volume'] = 0
        
        result = ts_validator.validate_volume(
            zero_volume,
            allow_zero=False
        )
        
        assert not result.is_valid
        assert 'zero volume' in str(result.errors[0])
        
        # Negative volume
        negative_volume = sample_timeseries.copy()
        negative_volume.loc[negative_volume.index[40], 'volume'] = -100
        
        result_negative = ts_validator.validate_volume(negative_volume)
        assert not result_negative.is_valid
        assert 'negative volume' in str(result_negative.errors[0])
        
    def test_outlier_detection(self, ts_validator, sample_timeseries):
        """Test outlier detection in time series."""
        # Add outliers
        outlier_series = sample_timeseries.copy()
        outlier_series.loc[outlier_series.index[50], 'close'] = 100000  # 2x normal
        outlier_series.loc[outlier_series.index[60], 'volume'] = 50000  # 50x normal
        
        outliers = ts_validator.detect_outliers(
            outlier_series,
            method='zscore',
            threshold=3
        )
        
        assert len(outliers) >= 2
        assert any(o['column'] == 'close' and o['index'] == outlier_series.index[50] 
                  for o in outliers)
        assert any(o['column'] == 'volume' for o in outliers)
        
    def test_data_continuity(self, ts_validator):
        """Test data continuity validation."""
        # Create data with sudden jumps
        dates = pd.date_range(start='2023-01-01', periods=100, freq='5min')
        prices = np.ones(100) * 50000
        prices[50] = 25000  # 50% drop (flash crash)
        
        data = pd.DataFrame({
            'close': prices,
            'volume': np.ones(100) * 500
        }, index=dates)
        
        continuity_result = ts_validator.validate_continuity(
            data,
            max_pct_change=0.1  # Max 10% change
        )
        
        assert not continuity_result.is_valid
        assert len(continuity_result.errors) >= 1
        assert 'large price movement' in str(continuity_result.errors[0])


class TestDataQualityChecker:
    """Test cases for data quality assessment."""
    
    @pytest.fixture
    def quality_checker(self):
        """Create DataQualityChecker instance."""
        return DataQualityChecker()
        
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset with quality issues."""
        np.random.seed(42)
        
        # Create data with various quality issues
        n_rows = 1000
        data = pd.DataFrame({
            'id': range(n_rows),
            'value': np.random.normal(100, 20, n_rows),
            'category': np.random.choice(['A', 'B', 'C', None], n_rows),
            'timestamp': pd.date_range(start='2023-01-01', periods=n_rows, freq='1H'),
            'description': ['desc' + str(i) if i % 10 != 0 else None for i in range(n_rows)]
        })
        
        # Add duplicates
        data.loc[100:102, 'id'] = 50
        
        # Add more nulls
        data.loc[200:220, 'value'] = np.nan
        
        return data
        
    def test_completeness_check(self, quality_checker, sample_dataset):
        """Test data completeness assessment."""
        completeness = quality_checker.check_completeness(sample_dataset)
        
        assert 'overall_completeness' in completeness
        assert 'column_completeness' in completeness
        assert 'missing_patterns' in completeness
        
        # Check column-level completeness
        assert completeness['column_completeness']['id'] == 1.0
        assert completeness['column_completeness']['value'] < 1.0
        assert completeness['column_completeness']['description'] < 1.0
        
        # Check missing patterns
        assert len(completeness['missing_patterns']) > 0
        
    def test_uniqueness_check(self, quality_checker, sample_dataset):
        """Test uniqueness and duplicate detection."""
        uniqueness = quality_checker.check_uniqueness(
            sample_dataset,
            key_columns=['id']
        )
        
        assert 'has_duplicates' in uniqueness
        assert 'duplicate_count' in uniqueness
        assert 'duplicate_rows' in uniqueness
        
        assert uniqueness['has_duplicates'] is True
        assert uniqueness['duplicate_count'] > 0
        assert len(uniqueness['duplicate_rows']) > 0
        
    def test_consistency_check(self, quality_checker):
        """Test data consistency validation."""
        # Create inconsistent data
        data = pd.DataFrame({
            'product_id': [1, 2, 3, 1, 2],
            'product_name': ['Apple', 'Banana', 'Cherry', 'Apple', 'Bananna'],  # Typo
            'price': [1.5, 2.0, 3.0, 1.5, 2.0]
        })
        
        consistency = quality_checker.check_consistency(
            data,
            consistency_rules={
                'product_id': 'product_name'  # ID should map to consistent name
            }
        )
        
        assert not consistency['is_consistent']
        assert len(consistency['inconsistencies']) > 0
        assert any(i['field'] == 'product_name' for i in consistency['inconsistencies'])
        
    def test_accuracy_check(self, quality_checker):
        """Test data accuracy validation."""
        # Create data with accuracy issues
        data = pd.DataFrame({
            'age': [25, 30, 150, -5, 45],  # Invalid ages
            'email': ['test@example.com', 'invalid-email', 'user@domain.com', '', None],
            'phone': ['123-456-7890', '999-999-9999', 'not-a-phone', '555-1234', None]
        })
        
        accuracy_rules = {
            'age': {'min': 0, 'max': 120},
            'email': {'pattern': r'^[\w\.-]+@[\w\.-]+\.\w+$'},
            'phone': {'pattern': r'^\d{3}-\d{3}-\d{4}$'}
        }
        
        accuracy = quality_checker.check_accuracy(data, accuracy_rules)
        
        assert 'accuracy_score' in accuracy
        assert 'field_accuracy' in accuracy
        assert 'violations' in accuracy
        
        assert accuracy['field_accuracy']['age'] < 1.0
        assert accuracy['field_accuracy']['email'] < 1.0
        assert len(accuracy['violations']) > 0
        
    def test_timeliness_check(self, quality_checker):
        """Test data timeliness validation."""
        # Create data with varying freshness
        current_time = datetime.now()
        
        data = pd.DataFrame({
            'record_id': range(5),
            'last_updated': [
                current_time - timedelta(hours=1),
                current_time - timedelta(days=1),
                current_time - timedelta(days=7),
                current_time - timedelta(days=30),
                current_time - timedelta(days=365)
            ],
            'value': [100, 200, 300, 400, 500]
        })
        
        timeliness = quality_checker.check_timeliness(
            data,
            timestamp_column='last_updated',
            max_age_days=7
        )
        
        assert 'fresh_data_pct' in timeliness
        assert 'stale_records' in timeliness
        assert 'age_distribution' in timeliness
        
        assert timeliness['fresh_data_pct'] == 0.6  # 3 out of 5
        assert len(timeliness['stale_records']) == 2
        
    def test_quality_score_calculation(self, quality_checker, sample_dataset):
        """Test overall data quality score calculation."""
        quality_score = quality_checker.calculate_quality_score(
            sample_dataset,
            weights={
                'completeness': 0.3,
                'uniqueness': 0.2,
                'consistency': 0.2,
                'accuracy': 0.2,
                'timeliness': 0.1
            }
        )
        
        assert 'overall_score' in quality_score
        assert 'dimension_scores' in quality_score
        assert 'recommendations' in quality_score
        
        assert 0 <= quality_score['overall_score'] <= 1
        assert len(quality_score['dimension_scores']) == 5
        assert len(quality_score['recommendations']) > 0


class TestDataSanitizer:
    """Test cases for data sanitization."""
    
    @pytest.fixture
    def data_sanitizer(self):
        """Create DataSanitizer instance."""
        return DataSanitizer()
        
    def test_string_sanitization(self, data_sanitizer):
        """Test string data sanitization."""
        # Strings with issues
        dirty_strings = pd.Series([
            '  BTC/USDT  ',  # Extra spaces
            'eth/usdt',      # Wrong case
            'BNB/USDT\n',    # Newline
            'DOT/USDT\t',    # Tab
            None,            # Null
            '',              # Empty
            'XRP/USDT '      # Trailing space
        ])
        
        clean_strings = data_sanitizer.sanitize_strings(
            dirty_strings,
            strip_whitespace=True,
            uppercase=True,
            remove_special_chars=True
        )
        
        assert clean_strings[0] == 'BTC/USDT'
        assert clean_strings[1] == 'ETH/USDT'
        assert clean_strings[2] == 'BNB/USDT'
        assert pd.isna(clean_strings[4])  # None remains None
        assert clean_strings[5] == ''      # Empty remains empty
        
    def test_numeric_sanitization(self, data_sanitizer):
        """Test numeric data sanitization."""
        # Numbers with issues
        dirty_numbers = pd.Series([
            '123.45',      # String number
            123.45,        # Valid number
            '1,234.56',    # Comma separator
            '$100.00',     # Currency symbol
            'NaN',         # String NaN
            np.inf,        # Infinity
            -np.inf,       # Negative infinity
            None           # Null
        ])
        
        clean_numbers = data_sanitizer.sanitize_numeric(
            dirty_numbers,
            remove_currency=True,
            handle_infinity='nan',
            decimal_places=2
        )
        
        assert clean_numbers[0] == 123.45
        assert clean_numbers[1] == 123.45
        assert clean_numbers[2] == 1234.56
        assert clean_numbers[3] == 100.00
        assert pd.isna(clean_numbers[4])
        assert pd.isna(clean_numbers[5])  # Infinity -> NaN
        assert pd.isna(clean_numbers[6])  # -Infinity -> NaN
        
    def test_datetime_sanitization(self, data_sanitizer):
        """Test datetime data sanitization."""
        # Various datetime formats
        dirty_dates = pd.Series([
            '2023-01-01',
            '01/01/2023',
            '2023-01-01 12:00:00',
            '2023-01-01T12:00:00Z',
            'Jan 1, 2023',
            'invalid date',
            None
        ])
        
        clean_dates = data_sanitizer.sanitize_datetime(
            dirty_dates,
            target_timezone='UTC',
            format_output='%Y-%m-%d %H:%M:%S'
        )
        
        # First 5 should be successfully parsed
        for i in range(5):
            assert isinstance(clean_dates[i], datetime)
            
        # Invalid date should be None or NaT
        assert pd.isna(clean_dates[5])
        
    def test_outlier_handling(self, data_sanitizer):
        """Test outlier handling in sanitization."""
        # Data with outliers
        data = pd.Series([10, 12, 11, 13, 10, 11, 12, 1000, 11, 10])  # 1000 is outlier
        
        # Method 1: Capping
        capped_data = data_sanitizer.handle_outliers(
            data,
            method='cap',
            lower_percentile=5,
            upper_percentile=95
        )
        
        assert capped_data[7] < 1000  # Outlier capped
        assert capped_data[7] == capped_data[:7].max()  # Capped to max of others
        
        # Method 2: Remove
        removed_data = data_sanitizer.handle_outliers(
            data,
            method='remove',
            zscore_threshold=3
        )
        
        assert len(removed_data) < len(data)  # Outlier removed
        assert 1000 not in removed_data.values
        
        # Method 3: Transform
        transformed_data = data_sanitizer.handle_outliers(
            data,
            method='transform',
            transformation='log'
        )
        
        assert transformed_data[7] < np.log(1000)  # Log reduces impact
        
    def test_missing_value_imputation(self, data_sanitizer):
        """Test missing value imputation strategies."""
        # Data with missing values
        data = pd.DataFrame({
            'numeric': [1, 2, np.nan, 4, 5, np.nan, 7],
            'categorical': ['A', 'B', None, 'A', 'B', 'C', None],
            'temporal': pd.date_range('2023-01-01', periods=7, freq='D')
        })
        data.loc[2, 'temporal'] = pd.NaT
        data.loc[5, 'temporal'] = pd.NaT
        
        # Numeric imputation
        imputed_numeric = data_sanitizer.impute_missing(
            data['numeric'],
            method='mean'
        )
        assert not imputed_numeric.isna().any()
        assert imputed_numeric[2] == data['numeric'].mean()
        
        # Categorical imputation
        imputed_categorical = data_sanitizer.impute_missing(
            data['categorical'],
            method='mode'
        )
        assert not imputed_categorical.isna().any()
        
        # Forward fill for temporal
        imputed_temporal = data_sanitizer.impute_missing(
            data['temporal'],
            method='forward_fill'
        )
        assert not imputed_temporal.isna().any()
        
    def test_data_type_conversion(self, data_sanitizer):
        """Test safe data type conversion."""
        # Mixed type data
        mixed_data = pd.Series(['1', '2.5', '3', 'not_a_number', '5.0', None])
        
        # Convert to numeric with error handling
        numeric_data = data_sanitizer.safe_convert(
            mixed_data,
            target_type='numeric',
            errors='coerce'
        )
        
        assert numeric_data[0] == 1.0
        assert numeric_data[1] == 2.5
        assert pd.isna(numeric_data[3])  # 'not_a_number' -> NaN
        assert pd.isna(numeric_data[5])  # None -> NaN
        
        # Convert with custom error handling
        numeric_data_custom = data_sanitizer.safe_convert(
            mixed_data,
            target_type='numeric',
            errors='custom',
            error_value=-999
        )
        
        assert numeric_data_custom[3] == -999  # Custom error value


class TestCrossFieldValidation:
    """Test cases for cross-field validation logic."""
    
    def test_dependent_field_validation(self):
        """Test validation of dependent fields."""
        validator = DataValidator([])
        
        # Add cross-field rule
        def validate_stop_loss(record):
            """Stop loss must be below entry price for long positions."""
            if record.get('side') == 'long' and 'stop_loss' in record and 'entry_price' in record:
                return record['stop_loss'] < record['entry_price']
            return True
            
        validator.add_cross_field_validator(validate_stop_loss)
        
        # Valid long position
        valid_long = {
            'side': 'long',
            'entry_price': Decimal('50000'),
            'stop_loss': Decimal('49000')
        }
        assert validator.validate_record(valid_long).is_valid
        
        # Invalid long position
        invalid_long = {
            'side': 'long',
            'entry_price': Decimal('50000'),
            'stop_loss': Decimal('51000')  # Above entry
        }
        assert not validator.validate_record(invalid_long).is_valid
        
    def test_conditional_requirement_validation(self):
        """Test conditional field requirements."""
        validator = DataValidator([])
        
        # If order_type is 'limit', price is required
        def validate_limit_order(record):
            if record.get('order_type') == 'limit':
                return 'price' in record and record['price'] is not None
            return True
            
        validator.add_cross_field_validator(validate_limit_order)
        
        # Market order without price - valid
        market_order = {'order_type': 'market', 'quantity': Decimal('1.0')}
        assert validator.validate_record(market_order).is_valid
        
        # Limit order without price - invalid
        limit_order = {'order_type': 'limit', 'quantity': Decimal('1.0')}
        assert not validator.validate_record(limit_order).is_valid
        
        # Limit order with price - valid
        limit_order_valid = {
            'order_type': 'limit',
            'quantity': Decimal('1.0'),
            'price': Decimal('50000')
        }
        assert validator.validate_record(limit_order_valid).is_valid
        
    def test_sum_constraint_validation(self):
        """Test validation of sum constraints across fields."""
        validator = DataValidator([])
        
        # Portfolio weights must sum to 100%
        def validate_portfolio_weights(record):
            if 'portfolio' in record:
                weights = [p['weight'] for p in record['portfolio']]
                total = sum(weights)
                return abs(total - 1.0) < 0.001  # Allow small rounding error
            return True
            
        validator.add_cross_field_validator(validate_portfolio_weights)
        
        # Valid portfolio
        valid_portfolio = {
            'portfolio': [
                {'symbol': 'BTC', 'weight': 0.5},
                {'symbol': 'ETH', 'weight': 0.3},
                {'symbol': 'BNB', 'weight': 0.2}
            ]
        }
        assert validator.validate_record(valid_portfolio).is_valid
        
        # Invalid portfolio
        invalid_portfolio = {
            'portfolio': [
                {'symbol': 'BTC', 'weight': 0.6},
                {'symbol': 'ETH', 'weight': 0.3},
                {'symbol': 'BNB', 'weight': 0.3}  # Sum = 1.2
            ]
        }
        assert not validator.validate_record(invalid_portfolio).is_valid