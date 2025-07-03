"""
Data Protection Testing Suite for GridAttention Trading System
Tests encryption, data privacy, PII handling, GDPR compliance, and secure storage
"""

import pytest
import asyncio
import os
import json
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import boto3
from dataclasses import dataclass
import base64

# GridAttention imports - aligned with system structure
from src.security.data_protection import DataProtectionManager
from src.security.encryption_service import EncryptionService
from src.security.key_management import KeyManagementService
from src.security.pii_detector import PIIDetector
from src.security.data_masking import DataMaskingService
from src.security.secure_storage import SecureStorageService
from src.security.audit_logger import SecurityAuditLogger
from src.compliance.gdpr_manager import GDPRComplianceManager
from src.compliance.data_retention import DataRetentionManager

# Test utilities
from tests.utils.test_helpers import async_test, create_test_config
from tests.utils.security_helpers import (
    generate_test_data,
    create_pii_samples,
    generate_encryption_keys
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SensitiveData:
    """Container for sensitive data types"""
    user_id: str
    email: str
    phone: str
    ssn: str
    credit_card: str
    bank_account: str
    api_key: str
    password_hash: str
    ip_address: str
    location: Dict[str, float]


class TestEncryptionService:
    """Test data encryption and decryption"""
    
    @pytest.fixture
    def encryption_service(self):
        """Create encryption service"""
        config = create_test_config()
        config['encryption'] = {
            'algorithm': 'AES-256-GCM',
            'key_rotation_days': 90,
            'enable_key_derivation': True,
            'kdf_iterations': 100000
        }
        return EncryptionService(config)
    
    @pytest.fixture
    def key_management(self):
        """Create key management service"""
        config = create_test_config()
        config['kms'] = {
            'provider': 'aws',  # or 'hashicorp_vault', 'azure_keyvault'
            'master_key_alias': 'gridattention-master-key',
            'key_rotation_enabled': True,
            'audit_key_usage': True
        }
        return KeyManagementService(config)
    
    @async_test
    async def test_aes_encryption(self, encryption_service):
        """Test AES encryption and decryption"""
        # Test data
        sensitive_data = {
            'user_id': 'user_123',
            'api_key': 'sk_live_abcdef123456',
            'balance': 10000.50,
            'trades': [
                {'id': 1, 'amount': 100},
                {'id': 2, 'amount': 200}
            ]
        }
        
        # Encrypt data
        encrypted = await encryption_service.encrypt(
            data=json.dumps(sensitive_data),
            context={'purpose': 'storage'}
        )
        
        assert encrypted['ciphertext'] != json.dumps(sensitive_data)
        assert 'nonce' in encrypted
        assert 'tag' in encrypted
        assert encrypted['algorithm'] == 'AES-256-GCM'
        
        # Decrypt data
        decrypted_json = await encryption_service.decrypt(
            ciphertext=encrypted['ciphertext'],
            nonce=encrypted['nonce'],
            tag=encrypted['tag'],
            context={'purpose': 'storage'}
        )
        
        decrypted = json.loads(decrypted_json)
        assert decrypted == sensitive_data
    
    @async_test
    async def test_field_level_encryption(self, encryption_service):
        """Test field-level encryption for specific sensitive fields"""
        user_data = {
            'user_id': 'user_123',
            'username': 'trader1',  # Not encrypted
            'email': 'trader@example.com',  # Encrypted
            'phone': '+1234567890',  # Encrypted
            'preferences': {'theme': 'dark'},  # Not encrypted
            'ssn': '123-45-6789',  # Encrypted
            'created_at': '2024-01-01'  # Not encrypted
        }
        
        # Define sensitive fields
        sensitive_fields = ['email', 'phone', 'ssn']
        
        # Encrypt sensitive fields
        encrypted_data = await encryption_service.encrypt_fields(
            data=user_data,
            fields=sensitive_fields
        )
        
        # Verify non-sensitive fields are unchanged
        assert encrypted_data['username'] == user_data['username']
        assert encrypted_data['preferences'] == user_data['preferences']
        
        # Verify sensitive fields are encrypted
        for field in sensitive_fields:
            assert encrypted_data[field] != user_data[field]
            assert isinstance(encrypted_data[field], dict)
            assert 'ciphertext' in encrypted_data[field]
        
        # Decrypt fields
        decrypted_data = await encryption_service.decrypt_fields(
            data=encrypted_data,
            fields=sensitive_fields
        )
        
        assert decrypted_data == user_data
    
    @async_test
    async def test_key_rotation(self, encryption_service, key_management):
        """Test encryption key rotation"""
        # Encrypt with current key
        data = "sensitive information"
        encrypted_v1 = await encryption_service.encrypt(data)
        
        # Rotate keys
        rotation_result = await key_management.rotate_keys()
        assert rotation_result['success'] is True
        assert 'new_key_id' in rotation_result
        assert 'old_key_id' in rotation_result
        
        # Encrypt with new key
        encrypted_v2 = await encryption_service.encrypt(data)
        
        # Should have different key IDs
        assert encrypted_v2['key_id'] != encrypted_v1['key_id']
        
        # Should still be able to decrypt old data
        decrypted_v1 = await encryption_service.decrypt(
            ciphertext=encrypted_v1['ciphertext'],
            nonce=encrypted_v1['nonce'],
            tag=encrypted_v1['tag'],
            key_id=encrypted_v1['key_id']
        )
        
        assert decrypted_v1 == data
    
    @async_test
    async def test_envelope_encryption(self, encryption_service, key_management):
        """Test envelope encryption for large data"""
        # Generate large data (1MB)
        large_data = 'x' * (1024 * 1024)
        
        # Envelope encryption process:
        # 1. Generate data encryption key (DEK)
        dek = await key_management.generate_data_key()
        
        # 2. Encrypt data with DEK
        encrypted_data = await encryption_service.encrypt_with_key(
            data=large_data,
            key=dek['plaintext']
        )
        
        # 3. Encrypt DEK with master key (creating encrypted DEK)
        encrypted_dek = dek['ciphertext']
        
        # Store encrypted data and encrypted DEK together
        envelope = {
            'encrypted_data': encrypted_data,
            'encrypted_dek': encrypted_dek,
            'algorithm': 'AES-256-GCM'
        }
        
        # Decryption process:
        # 1. Decrypt DEK using master key
        decrypted_dek = await key_management.decrypt_data_key(encrypted_dek)
        
        # 2. Decrypt data using DEK
        decrypted_data = await encryption_service.decrypt_with_key(
            ciphertext=envelope['encrypted_data']['ciphertext'],
            key=decrypted_dek,
            nonce=envelope['encrypted_data']['nonce'],
            tag=envelope['encrypted_data']['tag']
        )
        
        assert decrypted_data == large_data


class TestPIIDetection:
    """Test Personal Identifiable Information (PII) detection"""
    
    @pytest.fixture
    def pii_detector(self):
        """Create PII detector"""
        config = create_test_config()
        config['pii'] = {
            'detection_enabled': True,
            'patterns': {
                'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                'ssn': r'\d{3}-\d{2}-\d{4}',
                'credit_card': r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}',
                'phone': r'[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[(]?[0-9]{1,3}[)]?[-\s\.]?[0-9]{3,4}[-\s\.]?[0-9]{3,4}'
            },
            'custom_dictionaries': ['names', 'addresses']
        }
        return PIIDetector(config)
    
    @async_test
    async def test_pii_detection_in_text(self, pii_detector):
        """Test PII detection in text data"""
        text = """
        Dear John Doe,
        Your account email is john.doe@example.com and your phone is +1-555-123-4567.
        Your SSN is 123-45-6789 and credit card ending in 4242.
        Your full card number is 4111-1111-1111-1234.
        You live at 123 Main St, New York, NY 10001.
        """
        
        pii_results = await pii_detector.scan_text(text)
        
        assert pii_results['contains_pii'] is True
        assert len(pii_results['findings']) >= 5
        
        # Check detected PII types
        pii_types = {f['type'] for f in pii_results['findings']}
        assert 'email' in pii_types
        assert 'phone' in pii_types
        assert 'ssn' in pii_types
        assert 'credit_card' in pii_types
        
        # Verify locations are correct
        email_finding = next(f for f in pii_results['findings'] if f['type'] == 'email')
        assert email_finding['value'] == 'john.doe@example.com'
        assert email_finding['start'] > 0
        assert email_finding['end'] > email_finding['start']
    
    @async_test
    async def test_pii_detection_in_json(self, pii_detector):
        """Test PII detection in JSON data"""
        json_data = {
            'user': {
                'id': 'user_123',
                'name': 'John Doe',
                'email': 'john@example.com',
                'phone': '+1-555-123-4567',
                'address': {
                    'street': '123 Main St',
                    'city': 'New York',
                    'zip': '10001'
                }
            },
            'payment': {
                'card_number': '4111111111111111',
                'cvv': '123',
                'expiry': '12/25'
            },
            'notes': 'Customer SSN: 123-45-6789 for verification'
        }
        
        pii_results = await pii_detector.scan_json(json_data)
        
        assert pii_results['contains_pii'] is True
        
        # Check detected paths
        pii_paths = {f['path'] for f in pii_results['findings']}
        assert 'user.email' in pii_paths
        assert 'user.phone' in pii_paths
        assert 'payment.card_number' in pii_paths
        assert 'notes' in pii_paths  # Contains SSN
    
    @async_test
    async def test_pii_detection_in_logs(self, pii_detector):
        """Test PII detection in log files"""
        log_entries = [
            "2024-01-15 10:30:45 INFO User john@example.com logged in from 192.168.1.100",
            "2024-01-15 10:31:02 ERROR Payment failed for card 4111-1111-1111-1111",
            "2024-01-15 10:31:15 DEBUG Processing order for SSN 123-45-6789",
            "2024-01-15 10:31:30 INFO API key sk_live_abcd1234efgh5678 was used"
        ]
        
        for log_entry in log_entries:
            result = await pii_detector.scan_text(log_entry)
            assert result['contains_pii'] is True, f"PII not detected in: {log_entry}"
        
        # Test log sanitization
        sanitized_logs = []
        for log_entry in log_entries:
            sanitized = await pii_detector.sanitize_text(log_entry)
            sanitized_logs.append(sanitized)
            assert 'john@example.com' not in sanitized
            assert '4111-1111-1111-1111' not in sanitized
            assert '123-45-6789' not in sanitized
            assert 'sk_live_' not in sanitized


class TestDataMasking:
    """Test data masking and anonymization"""
    
    @pytest.fixture
    def masking_service(self):
        """Create data masking service"""
        config = create_test_config()
        config['masking'] = {
            'default_strategy': 'hash',
            'preserve_format': True,
            'deterministic': True,  # Same input -> same output
            'salt': 'test_salt_123'
        }
        return DataMaskingService(config)
    
    @async_test
    async def test_email_masking(self, masking_service):
        """Test email address masking"""
        emails = [
            'john.doe@example.com',
            'jane_smith@company.org',
            'admin@gridattention.com'
        ]
        
        masked_emails = []
        for email in emails:
            masked = await masking_service.mask_email(email)
            masked_emails.append(masked)
            
            # Should preserve format
            assert '@' in masked
            assert '.' in masked.split('@')[1]
            
            # Should not be original
            assert masked != email
            
            # Should be consistent (deterministic)
            masked2 = await masking_service.mask_email(email)
            assert masked == masked2
        
        # Different emails should have different masks
        assert len(set(masked_emails)) == len(emails)
    
    @async_test
    async def test_credit_card_masking(self, masking_service):
        """Test credit card number masking"""
        credit_cards = [
            '4111111111111111',
            '5500-0000-0000-0004',
            '3400 0000 0000 009',
            '6011111111111117'
        ]
        
        for card in credit_cards:
            masked = await masking_service.mask_credit_card(card)
            
            # Should show only last 4 digits
            assert masked.count('*') >= 12
            assert masked[-4:].isdigit()
            
            # Should preserve format
            if '-' in card:
                assert '-' in masked
            elif ' ' in card:
                assert ' ' in masked
    
    @async_test
    async def test_ssn_masking(self, masking_service):
        """Test SSN masking"""
        ssns = [
            '123-45-6789',
            '987654321',
            '111-22-3333'
        ]
        
        for ssn in ssns:
            masked = await masking_service.mask_ssn(ssn)
            
            # Should show only last 4 digits
            if '-' in ssn:
                assert masked == '***-**-' + ssn[-4:]
            else:
                assert masked == '*****' + ssn[-4:]
    
    @async_test
    async def test_phone_masking(self, masking_service):
        """Test phone number masking"""
        phones = [
            '+1-555-123-4567',
            '(555) 123-4567',
            '5551234567',
            '+44 20 7123 4567'
        ]
        
        for phone in phones:
            masked = await masking_service.mask_phone(phone)
            
            # Should preserve country code if present
            if phone.startswith('+'):
                assert masked.startswith('+')
            
            # Should mask middle digits
            assert '*' in masked
            
            # Should show some digits
            assert any(c.isdigit() for c in masked)
    
    @async_test
    async def test_custom_masking_rules(self, masking_service):
        """Test custom masking rules"""
        # Add custom masking rule for API keys
        await masking_service.add_custom_rule(
            name='api_key',
            pattern=r'sk_[a-zA-Z]+_[a-zA-Z0-9]+',
            mask_function=lambda match: f"sk_{match.group().split('_')[1]}_{'*' * 12}"
        )
        
        text = "Your API key is sk_live_abcd1234efgh5678"
        masked = await masking_service.mask_text(text)
        
        assert 'sk_live_abcd1234efgh5678' not in masked
        assert 'sk_live_************' in masked


class TestSecureStorage:
    """Test secure data storage"""
    
    @pytest.fixture
    def secure_storage(self):
        """Create secure storage service"""
        config = create_test_config()
        config['storage'] = {
            'provider': 's3',  # or 'azure_blob', 'gcs'
            'bucket': 'gridattention-secure',
            'encryption': 'AES-256',
            'versioning': True,
            'access_logging': True
        }
        return SecureStorageService(config)
    
    @async_test
    async def test_encrypted_file_storage(self, secure_storage):
        """Test encrypted file storage"""
        # Create sensitive data
        sensitive_data = {
            'user_id': 'user_123',
            'trading_history': [
                {'date': '2024-01-01', 'profit': 1000},
                {'date': '2024-01-02', 'profit': -500}
            ],
            'api_keys': ['key1', 'key2']
        }
        
        file_content = json.dumps(sensitive_data).encode()
        
        # Store encrypted file
        result = await secure_storage.store_encrypted(
            key='users/user_123/trading_data.json',
            data=file_content,
            metadata={
                'content_type': 'application/json',
                'classification': 'confidential'
            }
        )
        
        assert result['success'] is True
        assert 'version_id' in result
        assert 'encryption' in result
        assert result['encryption']['algorithm'] == 'AES-256'
        
        # Retrieve and decrypt file
        retrieved = await secure_storage.retrieve_encrypted(
            key='users/user_123/trading_data.json'
        )
        
        assert retrieved['data'] == file_content
        assert retrieved['metadata']['classification'] == 'confidential'
    
    @async_test
    async def test_secure_database_storage(self, secure_storage):
        """Test secure database storage with encryption"""
        # Mock database connection
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            
            # Store encrypted data in database
            user_data = {
                'email': 'user@example.com',
                'phone': '+1234567890',
                'ssn': '123-45-6789'
            }
            
            await secure_storage.store_user_data(
                user_id='user_123',
                data=user_data,
                encrypt_fields=['email', 'phone', 'ssn']
            )
            
            # Verify encryption was applied
            mock_cursor.execute.assert_called()
            sql_call = mock_cursor.execute.call_args[0][0]
            
            # Should use parameterized queries
            assert '%s' in sql_call
            
            # Values should be encrypted
            values = mock_cursor.execute.call_args[0][1]
            assert values[1] != user_data['email']  # Encrypted
            assert values[2] != user_data['phone']  # Encrypted
            assert values[3] != user_data['ssn']    # Encrypted
    
    @async_test
    async def test_secure_key_value_storage(self, secure_storage):
        """Test secure key-value storage (Redis)"""
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.from_url.return_value = mock_redis_instance
            
            # Store sensitive session data
            session_data = {
                'user_id': 'user_123',
                'permissions': ['trade', 'read'],
                'ip_address': '192.168.1.100',
                'login_time': datetime.now().isoformat()
            }
            
            await secure_storage.store_session(
                session_id='sess_abc123',
                data=session_data,
                ttl=3600  # 1 hour
            )
            
            # Verify data was encrypted before storage
            mock_redis_instance.setex.assert_called()
            stored_value = mock_redis_instance.setex.call_args[0][2]
            
            # Should be encrypted
            assert stored_value != json.dumps(session_data)
            
            # Should be base64 encoded encrypted data
            try:
                base64.b64decode(stored_value)
                assert True
            except:
                assert False, "Stored value should be base64 encoded"


class TestGDPRCompliance:
    """Test GDPR compliance features"""
    
    @pytest.fixture
    def gdpr_manager(self):
        """Create GDPR compliance manager"""
        config = create_test_config()
        config['gdpr'] = {
            'enabled': True,
            'data_retention_days': 365,
            'deletion_grace_period_days': 30,
            'anonymization_enabled': True,
            'consent_required': True
        }
        return GDPRComplianceManager(config)
    
    @async_test
    async def test_right_to_access(self, gdpr_manager):
        """Test GDPR right to access (data portability)"""
        user_id = 'user_123'
        
        # Request all user data
        export_result = await gdpr_manager.export_user_data(
            user_id=user_id,
            format='json',
            include_derived_data=True
        )
        
        assert export_result['success'] is True
        assert 'export_id' in export_result
        assert 'download_url' in export_result
        assert export_result['format'] == 'json'
        
        # Verify exported data structure
        exported_data = export_result['data']
        assert 'personal_data' in exported_data
        assert 'trading_history' in exported_data
        assert 'account_settings' in exported_data
        assert 'consent_history' in exported_data
        
        # Verify audit log
        assert 'export_timestamp' in exported_data
        assert 'data_categories' in exported_data
    
    @async_test
    async def test_right_to_erasure(self, gdpr_manager):
        """Test GDPR right to erasure (right to be forgotten)"""
        user_id = 'user_123'
        
        # Request data deletion
        deletion_result = await gdpr_manager.delete_user_data(
            user_id=user_id,
            reason='User requested deletion',
            verify_identity=True
        )
        
        assert deletion_result['success'] is True
        assert deletion_result['deletion_id'] is not None
        assert deletion_result['scheduled_date'] is not None
        
        # Verify grace period
        scheduled_date = datetime.fromisoformat(deletion_result['scheduled_date'])
        grace_period = scheduled_date - datetime.now()
        assert grace_period.days >= 29  # 30 day grace period
        
        # Verify deletion scope
        assert 'personal_data' in deletion_result['data_to_delete']
        assert 'trading_history' not in deletion_result['data_to_delete']  # May need to keep for compliance
        
        # Test cancellation within grace period
        cancel_result = await gdpr_manager.cancel_deletion(
            deletion_id=deletion_result['deletion_id'],
            user_id=user_id
        )
        
        assert cancel_result['success'] is True
    
    @async_test
    async def test_consent_management(self, gdpr_manager):
        """Test GDPR consent management"""
        user_id = 'user_123'
        
        # Record consent
        consent_result = await gdpr_manager.record_consent(
            user_id=user_id,
            purpose='marketing',
            granted=True,
            method='explicit_checkbox',
            ip_address='192.168.1.100'
        )
        
        assert consent_result['success'] is True
        assert 'consent_id' in consent_result
        assert 'timestamp' in consent_result
        
        # Check consent status
        status = await gdpr_manager.check_consent(
            user_id=user_id,
            purpose='marketing'
        )
        
        assert status['has_consent'] is True
        assert status['consent_date'] is not None
        assert status['expiry_date'] is not None  # Consent expires
        
        # Withdraw consent
        withdrawal_result = await gdpr_manager.withdraw_consent(
            user_id=user_id,
            purpose='marketing'
        )
        
        assert withdrawal_result['success'] is True
        
        # Verify consent is withdrawn
        status_after = await gdpr_manager.check_consent(
            user_id=user_id,
            purpose='marketing'
        )
        
        assert status_after['has_consent'] is False
    
    @async_test
    async def test_data_anonymization(self, gdpr_manager):
        """Test data anonymization for analytics"""
        # Original user data
        user_data = {
            'user_id': 'user_123',
            'email': 'john.doe@example.com',
            'age': 35,
            'country': 'US',
            'trading_volume': 50000,
            'profit_loss': 2500,
            'created_date': '2023-01-15'
        }
        
        # Anonymize for analytics
        anonymized = await gdpr_manager.anonymize_for_analytics(
            data=user_data,
            preserve_fields=['age', 'country', 'trading_volume', 'profit_loss']
        )
        
        # Verify PII is removed
        assert 'user_id' not in anonymized or anonymized['user_id'] != user_data['user_id']
        assert 'email' not in anonymized
        
        # Verify analytical fields are preserved
        assert anonymized['age'] == user_data['age']
        assert anonymized['country'] == user_data['country']
        assert anonymized['trading_volume'] == user_data['trading_volume']
        
        # Verify k-anonymity
        assert 'age_group' in anonymized  # Generalized
        assert anonymized['age_group'] in ['30-40', '35-39']


class TestDataRetention:
    """Test data retention policies"""
    
    @pytest.fixture
    def retention_manager(self):
        """Create data retention manager"""
        config = create_test_config()
        config['retention'] = {
            'policies': {
                'user_data': 365 * 2,  # 2 years
                'trading_logs': 365 * 7,  # 7 years (regulatory)
                'session_data': 7,  # 7 days
                'temporary_files': 1,  # 1 day
                'audit_logs': 365 * 10  # 10 years
            },
            'enable_automatic_deletion': True,
            'deletion_batch_size': 1000
        }
        return DataRetentionManager(config)
    
    @async_test
    async def test_retention_policy_enforcement(self, retention_manager):
        """Test automatic data retention policy enforcement"""
        # Get data eligible for deletion
        eligible_data = await retention_manager.find_expired_data()
        
        categories = {item['category'] for item in eligible_data}
        
        # Each category should be checked
        for category in ['user_data', 'trading_logs', 'session_data', 'temporary_files']:
            if category in categories:
                category_items = [item for item in eligible_data if item['category'] == category]
                
                for item in category_items:
                    # Verify item is actually expired
                    age_days = (datetime.now() - item['created_date']).days
                    retention_days = retention_manager.policies[category]
                    assert age_days >= retention_days
    
    @async_test
    async def test_selective_data_deletion(self, retention_manager):
        """Test selective data deletion based on retention policies"""
        # Schedule deletion
        deletion_job = await retention_manager.schedule_deletion(
            category='session_data',
            older_than_days=7,
            dry_run=True  # Test mode
        )
        
        assert deletion_job['success'] is True
        assert deletion_job['total_items'] >= 0
        assert deletion_job['dry_run'] is True
        
        # Verify deletion plan
        if deletion_job['total_items'] > 0:
            assert 'deletion_plan' in deletion_job
            assert len(deletion_job['deletion_plan']) > 0
            
            # Each item should be older than retention period
            for item in deletion_job['deletion_plan']:
                age = (datetime.now() - datetime.fromisoformat(item['created_date'])).days
                assert age >= 7
    
    @async_test
    async def test_legal_hold(self, retention_manager):
        """Test legal hold prevents deletion"""
        user_id = 'user_123'
        
        # Place legal hold
        hold_result = await retention_manager.place_legal_hold(
            identifier=user_id,
            reason='Regulatory investigation',
            authorized_by='legal_team'
        )
        
        assert hold_result['success'] is True
        assert 'hold_id' in hold_result
        
        # Try to delete data under legal hold
        deletion_attempt = await retention_manager.delete_user_data(user_id)
        
        assert deletion_attempt['success'] is False
        assert 'legal hold' in deletion_attempt['error'].lower()
        
        # Remove legal hold
        release_result = await retention_manager.release_legal_hold(
            hold_id=hold_result['hold_id'],
            authorized_by='legal_team'
        )
        
        assert release_result['success'] is True


class TestDataBreachProtection:
    """Test data breach detection and protection"""
    
    @pytest.fixture
    def breach_detector(self):
        """Create data breach detector"""
        config = create_test_config()
        config['breach_detection'] = {
            'enable_monitoring': True,
            'alert_threshold': 100,  # Unusual access patterns
            'monitoring_window': 3600,  # 1 hour
            'alert_channels': ['email', 'sms', 'slack']
        }
        return DataBreachDetector(config)
    
    @async_test
    async def test_unusual_access_detection(self, breach_detector):
        """Test detection of unusual data access patterns"""
        user_id = 'user_123'
        
        # Simulate normal access pattern
        for i in range(10):
            await breach_detector.log_access(
                user_id=user_id,
                resource='profile',
                ip_address='192.168.1.100',
                timestamp=datetime.now() - timedelta(minutes=i*5)
            )
        
        # Simulate unusual access (many requests quickly)
        for i in range(150):
            await breach_detector.log_access(
                user_id=user_id,
                resource='all_user_data',
                ip_address='10.0.0.1',  # Different IP
                timestamp=datetime.now()
            )
        
        # Check for alerts
        alerts = await breach_detector.get_recent_alerts()
        
        assert len(alerts) > 0
        
        breach_alert = alerts[0]
        assert breach_alert['type'] == 'unusual_access_pattern'
        assert breach_alert['severity'] == 'high'
        assert breach_alert['user_id'] == user_id
        assert 'threshold_exceeded' in breach_alert['details']
    
    @async_test
    async def test_data_exfiltration_detection(self, breach_detector):
        """Test detection of potential data exfiltration"""
        # Simulate large data download
        download_event = {
            'user_id': 'user_123',
            'action': 'bulk_export',
            'data_size_mb': 500,  # Large export
            'destination_ip': '203.0.113.1',  # External IP
            'timestamp': datetime.now()
        }
        
        alert = await breach_detector.check_exfiltration(download_event)
        
        assert alert is not None
        assert alert['type'] == 'potential_exfiltration'
        assert alert['risk_score'] > 0.7  # High risk
        
        # Check factors
        assert 'large_volume' in alert['risk_factors']
        assert 'external_destination' in alert['risk_factors']


class TestDataPrivacyAudit:
    """Test data privacy audit logging"""
    
    @pytest.fixture
    def privacy_auditor(self):
        """Create privacy audit logger"""
        config = create_test_config()
        config['privacy_audit'] = {
            'enabled': True,
            'log_retention_days': 2555,  # 7 years
            'log_encryption': True,
            'tamper_proof': True
        }
        return PrivacyAuditLogger(config)
    
    @async_test
    async def test_data_access_logging(self, privacy_auditor):
        """Test logging of all data access"""
        # Log data access
        access_event = {
            'user_id': 'admin_123',
            'action': 'view_user_data',
            'target_user': 'user_456',
            'data_categories': ['personal', 'financial'],
            'purpose': 'customer_support',
            'ip_address': '192.168.1.50',
            'timestamp': datetime.now()
        }
        
        log_result = await privacy_auditor.log_access(access_event)
        
        assert log_result['success'] is True
        assert 'log_id' in log_result
        assert 'hash' in log_result  # For tamper detection
        
        # Verify log integrity
        verification = await privacy_auditor.verify_log_integrity(
            log_id=log_result['log_id']
        )
        
        assert verification['valid'] is True
        assert verification['tampered'] is False
    
    @async_test
    async def test_privacy_compliance_report(self, privacy_auditor):
        """Test generation of privacy compliance reports"""
        # Generate compliance report
        report = await privacy_auditor.generate_compliance_report(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            include_categories=['data_access', 'data_modification', 'data_deletion']
        )
        
        assert report['success'] is True
        assert 'summary' in report
        assert 'detailed_logs' in report
        
        summary = report['summary']
        assert 'total_access_events' in summary
        assert 'unique_users_accessed' in summary
        assert 'data_categories_accessed' in summary
        assert 'high_risk_events' in summary


# Helper Classes

class DataBreachDetector:
    """Detect potential data breaches"""
    
    def __init__(self, config):
        self.config = config
        self.access_logs = []
        self.alerts = []
    
    async def log_access(self, user_id: str, resource: str, ip_address: str, timestamp: datetime):
        """Log data access event"""
        self.access_logs.append({
            'user_id': user_id,
            'resource': resource,
            'ip_address': ip_address,
            'timestamp': timestamp
        })
        
        # Check for unusual patterns
        await self._check_patterns(user_id)
    
    async def _check_patterns(self, user_id: str):
        """Check for unusual access patterns"""
        # Count recent accesses
        window_start = datetime.now() - timedelta(seconds=self.config['breach_detection']['monitoring_window'])
        recent_accesses = [
            log for log in self.access_logs
            if log['user_id'] == user_id and log['timestamp'] > window_start
        ]
        
        if len(recent_accesses) > self.config['breach_detection']['alert_threshold']:
            self.alerts.append({
                'type': 'unusual_access_pattern',
                'severity': 'high',
                'user_id': user_id,
                'timestamp': datetime.now(),
                'details': {
                    'access_count': len(recent_accesses),
                    'threshold_exceeded': True
                }
            })
    
    async def get_recent_alerts(self) -> List[Dict]:
        """Get recent alerts"""
        return self.alerts
    
    async def check_exfiltration(self, event: Dict) -> Optional[Dict]:
        """Check for potential data exfiltration"""
        risk_score = 0
        risk_factors = []
        
        # Large data volume
        if event.get('data_size_mb', 0) > 100:
            risk_score += 0.4
            risk_factors.append('large_volume')
        
        # External destination
        if not event.get('destination_ip', '').startswith(('192.168.', '10.', '172.')):
            risk_score += 0.4
            risk_factors.append('external_destination')
        
        # Unusual time
        hour = event.get('timestamp', datetime.now()).hour
        if hour < 6 or hour > 22:
            risk_score += 0.2
            risk_factors.append('unusual_time')
        
        if risk_score > 0.7:
            return {
                'type': 'potential_exfiltration',
                'risk_score': risk_score,
                'risk_factors': risk_factors,
                'event': event
            }
        
        return None


class PrivacyAuditLogger:
    """Privacy audit logger with tamper protection"""
    
    def __init__(self, config):
        self.config = config
        self.logs = []
    
    async def log_access(self, event: Dict) -> Dict:
        """Log data access with tamper protection"""
        # Generate log entry
        log_entry = {
            'log_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            **event
        }
        
        # Calculate hash for tamper detection
        log_hash = hashlib.sha256(
            json.dumps(log_entry, sort_keys=True).encode()
        ).hexdigest()
        
        log_entry['hash'] = log_hash
        self.logs.append(log_entry)
        
        return {
            'success': True,
            'log_id': log_entry['log_id'],
            'hash': log_hash
        }
    
    async def verify_log_integrity(self, log_id: str) -> Dict:
        """Verify log hasn't been tampered with"""
        log_entry = next((log for log in self.logs if log['log_id'] == log_id), None)
        
        if not log_entry:
            return {'valid': False, 'error': 'Log not found'}
        
        # Recalculate hash
        stored_hash = log_entry['hash']
        log_copy = log_entry.copy()
        del log_copy['hash']
        
        calculated_hash = hashlib.sha256(
            json.dumps(log_copy, sort_keys=True).encode()
        ).hexdigest()
        
        return {
            'valid': calculated_hash == stored_hash,
            'tampered': calculated_hash != stored_hash
        }
    
    async def generate_compliance_report(self, start_date: datetime, end_date: datetime, 
                                       include_categories: List[str]) -> Dict:
        """Generate privacy compliance report"""
        # Filter logs by date range
        relevant_logs = [
            log for log in self.logs
            if start_date <= datetime.fromisoformat(log['timestamp']) <= end_date
        ]
        
        # Generate summary
        summary = {
            'total_access_events': len(relevant_logs),
            'unique_users_accessed': len(set(log.get('target_user', '') for log in relevant_logs)),
            'data_categories_accessed': list(set(
                cat for log in relevant_logs 
                for cat in log.get('data_categories', [])
            )),
            'high_risk_events': sum(1 for log in relevant_logs if log.get('risk_level') == 'high')
        }
        
        return {
            'success': True,
            'summary': summary,
            'detailed_logs': relevant_logs[:100]  # Limited for size
        }