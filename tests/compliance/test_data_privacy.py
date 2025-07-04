"""
Data privacy compliance tests for GridAttention trading system.

Ensures compliance with GDPR, CCPA, and other data protection regulations
for handling personal data, trading data, and sensitive information.
"""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import base64
from unittest.mock import Mock, patch, AsyncMock
import sqlite3
import re
from collections import defaultdict

# Import core components
from core.privacy_manager import PrivacyManager
from core.data_classifier import DataClassifier
from core.encryption_service import EncryptionService
from core.consent_manager import ConsentManager


class DataCategory(Enum):
    """Categories of data for privacy compliance"""
    # Personal Identifiable Information
    PII_DIRECT = "PII_DIRECT"  # Name, email, phone
    PII_INDIRECT = "PII_INDIRECT"  # IP address, device ID
    PII_SENSITIVE = "PII_SENSITIVE"  # Financial, health data
    
    # Trading Data
    TRADE_DATA = "TRADE_DATA"
    POSITION_DATA = "POSITION_DATA"
    ORDER_DATA = "ORDER_DATA"
    
    # Analytics
    BEHAVIORAL_DATA = "BEHAVIORAL_DATA"
    PERFORMANCE_DATA = "PERFORMANCE_DATA"
    
    # System
    TECHNICAL_DATA = "TECHNICAL_DATA"
    AUDIT_DATA = "AUDIT_DATA"


class PrivacyRight(Enum):
    """Data subject rights under privacy laws"""
    ACCESS = "ACCESS"  # Right to access
    RECTIFICATION = "RECTIFICATION"  # Right to correct
    ERASURE = "ERASURE"  # Right to be forgotten
    PORTABILITY = "PORTABILITY"  # Right to data portability
    RESTRICTION = "RESTRICTION"  # Right to restrict processing
    OBJECTION = "OBJECTION"  # Right to object
    AUTOMATED_DECISION = "AUTOMATED_DECISION"  # Rights related to automated decisions


@dataclass
class PersonalData:
    """Personal data record with privacy metadata"""
    data_id: str
    subject_id: str  # Data subject identifier
    category: DataCategory
    data: Dict[str, Any]
    collected_at: datetime
    purpose: str
    legal_basis: str
    retention_period: timedelta
    encrypted: bool = True
    anonymized: bool = False
    consents: List[str] = field(default_factory=list)


@dataclass
class PrivacyRequest:
    """Data subject privacy request"""
    request_id: str
    subject_id: str
    request_type: PrivacyRight
    requested_at: datetime
    details: Dict[str, Any]
    status: str = "PENDING"
    completed_at: Optional[datetime] = None
    response: Optional[Dict[str, Any]] = None


class TestDataPrivacy:
    """Test data privacy compliance"""
    
    @pytest.fixture
    async def privacy_manager(self):
        """Create privacy manager instance"""
        return PrivacyManager(
            encryption_enabled=True,
            anonymization_enabled=True,
            audit_logging=True,
            gdpr_mode=True,
            ccpa_mode=True
        )
    
    @pytest.fixture
    async def encryption_service(self):
        """Create encryption service"""
        return EncryptionService(
            algorithm='AES-256-GCM',
            key_rotation_days=90,
            use_hardware_security_module=False  # For testing
        )
    
    @pytest.fixture
    def sample_personal_data(self) -> List[PersonalData]:
        """Generate sample personal data"""
        data_records = []
        
        # Direct PII
        data_records.append(PersonalData(
            data_id='PII_001',
            subject_id='USER_001',
            category=DataCategory.PII_DIRECT,
            data={
                'first_name': 'John',
                'last_name': 'Doe',
                'email': 'john.doe@example.com',
                'phone': '+1234567890'
            },
            collected_at=datetime.now(timezone.utc),
            purpose='account_creation',
            legal_basis='consent',
            retention_period=timedelta(days=2555),  # 7 years
            consents=['marketing', 'analytics']
        ))
        
        # Trading data
        data_records.append(PersonalData(
            data_id='TRADE_001',
            subject_id='USER_001',
            category=DataCategory.TRADE_DATA,
            data={
                'trades': [
                    {'id': 'T001', 'symbol': 'BTC/USDT', 'quantity': '0.5', 'price': '50000'},
                    {'id': 'T002', 'symbol': 'ETH/USDT', 'quantity': '5.0', 'price': '3000'}
                ],
                'total_volume': '40000',
                'profit_loss': '2500'
            },
            collected_at=datetime.now(timezone.utc),
            purpose='service_provision',
            legal_basis='contract',
            retention_period=timedelta(days=2555)
        ))
        
        # Behavioral data
        data_records.append(PersonalData(
            data_id='BEHAV_001',
            subject_id='USER_001',
            category=DataCategory.BEHAVIORAL_DATA,
            data={
                'login_times': ['2024-01-01T10:00:00Z', '2024-01-02T09:30:00Z'],
                'features_used': ['grid_trading', 'analytics', 'reports'],
                'average_session_duration': '45 minutes'
            },
            collected_at=datetime.now(timezone.utc),
            purpose='service_improvement',
            legal_basis='legitimate_interest',
            retention_period=timedelta(days=365)
        ))
        
        return data_records
    
    @pytest.mark.asyncio
    async def test_data_encryption_at_rest(self, privacy_manager, encryption_service):
        """Test encryption of personal data at rest"""
        # Test data
        sensitive_data = {
            'account_number': '1234567890',
            'api_key': 'sk_live_abcdef123456',
            'balance': '100000.50',
            'ssn': '123-45-6789'  # Should never store, but testing encryption
        }
        
        # Encrypt data
        encrypted_data = await encryption_service.encrypt_data(
            data=json.dumps(sensitive_data),
            data_classification='SENSITIVE'
        )
        
        # Verify encryption
        assert encrypted_data['encrypted'] == True
        assert encrypted_data['algorithm'] == 'AES-256-GCM'
        assert encrypted_data['ciphertext'] != json.dumps(sensitive_data)
        assert 'nonce' in encrypted_data
        assert 'tag' in encrypted_data
        assert 'key_id' in encrypted_data
        
        # Test decryption
        decrypted_data = await encryption_service.decrypt_data(encrypted_data)
        assert json.loads(decrypted_data) == sensitive_data
        
        # Test key rotation
        old_key_id = encrypted_data['key_id']
        await encryption_service.rotate_encryption_keys()
        
        # Re-encrypt with new key
        new_encrypted = await encryption_service.encrypt_data(
            data=json.dumps(sensitive_data),
            data_classification='SENSITIVE'
        )
        
        assert new_encrypted['key_id'] != old_key_id
        
        # Should still be able to decrypt old data
        old_decrypted = await encryption_service.decrypt_data(encrypted_data)
        assert json.loads(old_decrypted) == sensitive_data
    
    @pytest.mark.asyncio
    async def test_data_anonymization(self, privacy_manager):
        """Test data anonymization techniques"""
        # Personal data to anonymize
        personal_data = {
            'user_id': 'USER_12345',
            'name': 'John Doe',
            'email': 'john.doe@example.com',
            'phone': '+1 (555) 123-4567',
            'ip_address': '192.168.1.100',
            'birth_date': '1990-01-15',
            'credit_card': '4111-1111-1111-1111',
            'trading_volume': '50000.00',
            'location': {'city': 'New York', 'country': 'USA'}
        }
        
        # Apply different anonymization techniques
        anonymized = await privacy_manager.anonymize_data(
            data=personal_data,
            techniques={
                'user_id': 'pseudonymize',
                'name': 'remove',
                'email': 'hash',
                'phone': 'mask',
                'ip_address': 'generalize',
                'birth_date': 'generalize_date',
                'credit_card': 'tokenize',
                'trading_volume': 'add_noise',
                'location': 'generalize_location'
            }
        )
        
        # Verify anonymization
        assert anonymized['user_id'] != personal_data['user_id']  # Pseudonymized
        assert 'name' not in anonymized  # Removed
        assert anonymized['email'] != personal_data['email']  # Hashed
        assert len(anonymized['email']) == 64  # SHA-256 hash
        assert '****' in anonymized['phone']  # Masked
        assert anonymized['ip_address'] == '192.168.1.0/24'  # Generalized
        assert anonymized['birth_date'] == '1990-01-01'  # Generalized to year
        assert anonymized['credit_card'].startswith('tok_')  # Tokenized
        assert anonymized['location'] == {'city': 'New York', 'country': 'USA', 'precision': 'city'}
        
        # Verify trading volume has noise but is still useful
        original_volume = float(personal_data['trading_volume'])
        anonymized_volume = float(anonymized['trading_volume'])
        assert abs(anonymized_volume - original_volume) / original_volume < 0.1  # Less than 10% noise
    
    @pytest.mark.asyncio
    async def test_consent_management(self, privacy_manager):
        """Test consent collection and management"""
        user_id = 'USER_TEST_001'
        
        # Record initial consents
        consents = {
            'marketing': {
                'granted': True,
                'timestamp': datetime.now(timezone.utc),
                'version': '1.0',
                'method': 'explicit_opt_in'
            },
            'analytics': {
                'granted': True,
                'timestamp': datetime.now(timezone.utc),
                'version': '1.0',
                'method': 'explicit_opt_in'
            },
            'third_party_sharing': {
                'granted': False,
                'timestamp': datetime.now(timezone.utc),
                'version': '1.0',
                'method': 'explicit_opt_out'
            }
        }
        
        # Store consents
        await privacy_manager.record_consent(user_id, consents)
        
        # Verify consent storage
        stored_consents = await privacy_manager.get_user_consents(user_id)
        assert stored_consents['marketing']['granted'] == True
        assert stored_consents['third_party_sharing']['granted'] == False
        
        # Test consent withdrawal
        await privacy_manager.withdraw_consent(
            user_id=user_id,
            consent_type='marketing',
            timestamp=datetime.now(timezone.utc)
        )
        
        # Verify withdrawal
        updated_consents = await privacy_manager.get_user_consents(user_id)
        assert updated_consents['marketing']['granted'] == False
        assert 'withdrawal_timestamp' in updated_consents['marketing']
        
        # Test consent-based data processing
        can_process = await privacy_manager.check_consent_for_processing(
            user_id=user_id,
            purpose='marketing',
            data_category=DataCategory.BEHAVIORAL_DATA
        )
        
        assert can_process == False  # Marketing consent withdrawn
        
        # Test consent audit trail
        consent_history = await privacy_manager.get_consent_history(user_id)
        assert len(consent_history) >= 4  # Initial consents + withdrawal
        
        # Verify all consent changes are logged
        for event in consent_history:
            assert 'timestamp' in event
            assert 'consent_type' in event
            assert 'action' in event  # 'granted' or 'withdrawn'
            assert 'version' in event
    
    @pytest.mark.asyncio
    async def test_right_to_access(self, privacy_manager, sample_personal_data):
        """Test GDPR right to access implementation"""
        user_id = 'USER_001'
        
        # Store sample data
        for data_record in sample_personal_data:
            await privacy_manager.store_personal_data(data_record)
        
        # Create access request
        access_request = PrivacyRequest(
            request_id='REQ_ACCESS_001',
            subject_id=user_id,
            request_type=PrivacyRight.ACCESS,
            requested_at=datetime.now(timezone.utc),
            details={'format': 'json', 'categories': 'all'}
        )
        
        # Process access request
        response = await privacy_manager.process_privacy_request(access_request)
        
        # Verify response
        assert response['status'] == 'COMPLETED'
        assert 'data_package' in response
        assert 'generated_at' in response
        assert 'categories' in response['data_package']
        
        data_package = response['data_package']
        
        # Verify all data categories are included
        assert DataCategory.PII_DIRECT.value in data_package['categories']
        assert DataCategory.TRADE_DATA.value in data_package['categories']
        assert DataCategory.BEHAVIORAL_DATA.value in data_package['categories']
        
        # Verify data is complete but secure
        pii_data = data_package['categories'][DataCategory.PII_DIRECT.value]
        assert 'email' in pii_data[0]['data']
        assert 'collected_at' in pii_data[0]
        assert 'purpose' in pii_data[0]
        assert 'retention_until' in pii_data[0]
        
        # Generate human-readable report
        readable_report = await privacy_manager.generate_access_report(
            user_id=user_id,
            format='pdf',
            include_metadata=True
        )
        
        assert readable_report['format'] == 'pdf'
        assert 'report_id' in readable_report
        assert 'generated_at' in readable_report
    
    @pytest.mark.asyncio
    async def test_right_to_erasure(self, privacy_manager, sample_personal_data):
        """Test GDPR right to erasure (right to be forgotten)"""
        user_id = 'USER_001'
        
        # Store sample data
        for data_record in sample_personal_data:
            await privacy_manager.store_personal_data(data_record)
        
        # Create erasure request
        erasure_request = PrivacyRequest(
            request_id='REQ_ERASURE_001',
            subject_id=user_id,
            request_type=PrivacyRight.ERASURE,
            requested_at=datetime.now(timezone.utc),
            details={
                'reason': 'user_request',
                'categories': [DataCategory.PII_DIRECT, DataCategory.BEHAVIORAL_DATA]
            }
        )
        
        # Check for data that cannot be erased (legal requirements)
        erasure_assessment = await privacy_manager.assess_erasure_request(erasure_request)
        
        assert 'can_erase' in erasure_assessment
        assert 'cannot_erase' in erasure_assessment
        assert 'reasons' in erasure_assessment
        
        # Trade data might need to be retained for regulatory compliance
        assert DataCategory.TRADE_DATA in erasure_assessment['cannot_erase']
        assert 'regulatory_requirement' in erasure_assessment['reasons'][DataCategory.TRADE_DATA]
        
        # Process erasure for allowed categories
        erasure_response = await privacy_manager.process_privacy_request(erasure_request)
        
        assert erasure_response['status'] == 'COMPLETED'
        assert 'erased_categories' in erasure_response
        assert DataCategory.PII_DIRECT.value in erasure_response['erased_categories']
        assert DataCategory.BEHAVIORAL_DATA.value in erasure_response['erased_categories']
        
        # Verify data is actually erased
        remaining_data = await privacy_manager.get_user_data(
            user_id=user_id,
            category=DataCategory.PII_DIRECT
        )
        
        assert len(remaining_data) == 0  # PII should be erased
        
        # Verify audit log of erasure
        erasure_log = await privacy_manager.get_erasure_log(user_id)
        assert len(erasure_log) > 0
        assert erasure_log[0]['action'] == 'ERASURE'
        assert 'categories_erased' in erasure_log[0]
        assert 'authorized_by' in erasure_log[0]
    
    @pytest.mark.asyncio
    async def test_data_portability(self, privacy_manager, sample_personal_data):
        """Test GDPR right to data portability"""
        user_id = 'USER_001'
        
        # Store sample data
        for data_record in sample_personal_data:
            await privacy_manager.store_personal_data(data_record)
        
        # Create portability request
        portability_request = PrivacyRequest(
            request_id='REQ_PORT_001',
            subject_id=user_id,
            request_type=PrivacyRight.PORTABILITY,
            requested_at=datetime.now(timezone.utc),
            details={
                'format': 'json',
                'include_derived_data': False,
                'destination': 'download'  # or 'transfer_to_controller'
            }
        )
        
        # Process portability request
        response = await privacy_manager.process_privacy_request(portability_request)
        
        assert response['status'] == 'COMPLETED'
        assert 'portable_data' in response
        assert 'format' in response
        assert response['format'] == 'json'
        
        portable_data = response['portable_data']
        
        # Verify data is machine-readable and structured
        assert isinstance(portable_data, dict)
        assert 'metadata' in portable_data
        assert 'data_categories' in portable_data
        
        # Verify only provided data is included (not derived/inferred)
        for category in portable_data['data_categories']:
            for record in portable_data['data_categories'][category]:
                assert record['legal_basis'] in ['consent', 'contract']
        
        # Test export in different formats
        csv_request = portability_request
        csv_request.details['format'] = 'csv'
        
        csv_response = await privacy_manager.process_privacy_request(csv_request)
        assert csv_response['format'] == 'csv'
        assert 'files' in csv_response  # Multiple CSV files for different categories
    
    @pytest.mark.asyncio
    async def test_automated_decision_making_rights(self, privacy_manager):
        """Test rights related to automated decision-making"""
        user_id = 'USER_001'
        
        # Record automated decisions
        automated_decisions = [
            {
                'decision_id': 'DEC_001',
                'timestamp': datetime.now(timezone.utc),
                'decision_type': 'trading_limit_adjustment',
                'algorithm': 'risk_assessment_v2',
                'inputs': {
                    'trading_history': 'positive',
                    'account_age': '2_years',
                    'verification_level': 'full'
                },
                'output': {
                    'new_limit': '100000',
                    'previous_limit': '50000',
                    'change': 'increase'
                },
                'impact': 'significant'
            },
            {
                'decision_id': 'DEC_002',
                'timestamp': datetime.now(timezone.utc),
                'decision_type': 'feature_access',
                'algorithm': 'ml_classifier_v1',
                'inputs': {
                    'usage_pattern': 'advanced',
                    'risk_score': 'low'
                },
                'output': {
                    'features_enabled': ['margin_trading', 'api_access']
                },
                'impact': 'significant'
            }
        ]
        
        for decision in automated_decisions:
            await privacy_manager.record_automated_decision(user_id, decision)
        
        # User requests information about automated decisions
        info_request = await privacy_manager.get_automated_decisions_info(
            user_id=user_id,
            include_logic=True
        )
        
        assert len(info_request['decisions']) == 2
        assert info_request['has_significant_decisions'] == True
        
        # User objects to automated decision
        objection = await privacy_manager.object_to_automated_decision(
            user_id=user_id,
            decision_id='DEC_001',
            reason='request_human_review'
        )
        
        assert objection['status'] == 'ACCEPTED'
        assert objection['action'] == 'HUMAN_REVIEW_SCHEDULED'
        assert 'review_id' in objection
        
        # Implement human review
        human_review = await privacy_manager.conduct_human_review(
            review_id=objection['review_id'],
            reviewer='compliance_officer_001',
            decision='OVERRIDE',
            new_outcome={'new_limit': '75000'},  # Compromise
            justification='Risk assessment too aggressive for user profile'
        )
        
        assert human_review['original_decision'] != human_review['final_decision']
        assert human_review['human_involved'] == True
    
    @pytest.mark.asyncio
    async def test_cross_border_data_transfer(self, privacy_manager):
        """Test compliance for cross-border data transfers"""
        # Define data transfer request
        transfer_request = {
            'source_country': 'EU',
            'destination_country': 'US',
            'data_categories': [DataCategory.PII_DIRECT, DataCategory.TRADE_DATA],
            'purpose': 'cloud_storage',
            'processor': 'AWS',
            'volume': 'bulk'
        }
        
        # Check transfer legality
        transfer_assessment = await privacy_manager.assess_data_transfer(transfer_request)
        
        assert 'legal_mechanism_required' in transfer_assessment
        assert 'available_mechanisms' in transfer_assessment
        
        # For EU->US transfers after Privacy Shield invalidation
        assert 'standard_contractual_clauses' in transfer_assessment['available_mechanisms']
        assert 'binding_corporate_rules' in transfer_assessment['available_mechanisms']
        
        # Implement Standard Contractual Clauses (SCCs)
        scc_implementation = await privacy_manager.implement_transfer_mechanism(
            mechanism='standard_contractual_clauses',
            transfer_details=transfer_request,
            additional_safeguards=['encryption', 'pseudonymization']
        )
        
        assert scc_implementation['status'] == 'APPROVED'
        assert 'contract_id' in scc_implementation
        assert 'safeguards_implemented' in scc_implementation
        
        # Log transfer for accountability
        transfer_log = await privacy_manager.log_data_transfer(
            transfer_id=scc_implementation['contract_id'],
            details=transfer_request,
            legal_basis=scc_implementation
        )
        
        assert transfer_log['logged'] == True
        assert 'transfer_record_id' in transfer_log
    
    @pytest.mark.asyncio
    async def test_data_breach_notification(self, privacy_manager):
        """Test data breach detection and notification procedures"""
        # Simulate detected breach
        breach_details = {
            'breach_id': 'BREACH_001',
            'detected_at': datetime.now(timezone.utc),
            'breach_type': 'unauthorized_access',
            'affected_systems': ['user_database', 'trading_history'],
            'estimated_records': 1500,
            'data_categories_affected': [
                DataCategory.PII_DIRECT,
                DataCategory.TRADE_DATA
            ],
            'attack_vector': 'compromised_credentials',
            'data_encrypted': True,
            'encryption_keys_compromised': False
        }
        
        # Assess breach severity
        breach_assessment = await privacy_manager.assess_data_breach(breach_details)
        
        assert 'severity' in breach_assessment
        assert 'notification_required' in breach_assessment
        assert 'risk_to_individuals' in breach_assessment
        
        # Under GDPR, notification required if high risk
        if breach_assessment['risk_to_individuals'] == 'HIGH':
            assert breach_assessment['notification_required'] == True
            assert breach_assessment['notification_deadline_hours'] == 72
        
        # Generate notifications
        notifications = await privacy_manager.generate_breach_notifications(
            breach_assessment=breach_assessment,
            breach_details=breach_details
        )
        
        assert 'regulatory_notification' in notifications
        assert 'user_notification' in notifications
        
        # Regulatory notification (to DPA)
        reg_notification = notifications['regulatory_notification']
        assert 'breach_description' in reg_notification
        assert 'affected_data_subjects' in reg_notification
        assert 'likely_consequences' in reg_notification
        assert 'measures_taken' in reg_notification
        assert 'dpo_contact' in reg_notification
        
        # User notification
        user_notification = notifications['user_notification']
        assert 'plain_language_description' in user_notification
        assert 'potential_consequences' in user_notification
        assert 'mitigation_advice' in user_notification
        assert 'support_contact' in user_notification
        
        # Track notification compliance
        notification_tracking = await privacy_manager.track_breach_notifications(
            breach_id=breach_details['breach_id'],
            notifications_sent={
                'regulatory': datetime.now(timezone.utc),
                'users': datetime.now(timezone.utc) + timedelta(hours=24)
            }
        )
        
        assert notification_tracking['compliant'] == True
        assert notification_tracking['regulatory_deadline_met'] == True
    
    @pytest.mark.asyncio
    async def test_privacy_by_design(self, privacy_manager):
        """Test privacy by design implementation"""
        # Test new feature privacy assessment
        new_feature = {
            'feature_name': 'social_trading',
            'description': 'Allow users to share and copy trades',
            'data_processing': {
                'user_profiles': True,
                'trading_history': True,
                'performance_metrics': True,
                'social_connections': True
            },
            'data_sharing': True,
            'third_party_involvement': False
        }
        
        # Conduct Privacy Impact Assessment (PIA)
        pia_result = await privacy_manager.conduct_privacy_impact_assessment(new_feature)
        
        assert 'risk_score' in pia_result
        assert 'identified_risks' in pia_result
        assert 'recommended_controls' in pia_result
        assert 'privacy_requirements' in pia_result
        
        # Verify high-risk areas identified
        risks = pia_result['identified_risks']
        assert any(risk['area'] == 'data_sharing' for risk in risks)
        assert any(risk['area'] == 'user_consent' for risk in risks)
        
        # Implement privacy controls
        privacy_controls = {
            'data_minimization': {
                'only_necessary_fields': True,
                'aggregation_where_possible': True
            },
            'consent_mechanism': {
                'granular_consent': True,
                'easy_withdrawal': True
            },
            'access_controls': {
                'role_based': True,
                'audit_logging': True
            },
            'anonymization': {
                'public_profiles': True,
                'performance_data': True
            }
        }
        
        implementation = await privacy_manager.implement_privacy_controls(
            feature=new_feature['feature_name'],
            controls=privacy_controls
        )
        
        assert implementation['status'] == 'IMPLEMENTED'
        assert all(control in implementation['active_controls'] 
                  for control in privacy_controls.keys())
    
    @pytest.mark.asyncio
    async def test_data_retention_and_deletion(self, privacy_manager):
        """Test data retention policies and automatic deletion"""
        # Configure retention policies
        retention_policies = {
            DataCategory.PII_DIRECT: timedelta(days=2555),  # 7 years
            DataCategory.TRADE_DATA: timedelta(days=2555),  # Regulatory requirement
            DataCategory.BEHAVIORAL_DATA: timedelta(days=365),  # 1 year
            DataCategory.TECHNICAL_DATA: timedelta(days=90)  # 3 months
        }
        
        await privacy_manager.configure_retention_policies(retention_policies)
        
        # Create test data with different ages
        test_data = []
        base_time = datetime.now(timezone.utc)
        
        # Old behavioral data (should be deleted)
        old_behavioral = PersonalData(
            data_id='OLD_BEHAV_001',
            subject_id='USER_001',
            category=DataCategory.BEHAVIORAL_DATA,
            data={'old_activity': 'data'},
            collected_at=base_time - timedelta(days=400),  # Over 1 year
            purpose='analytics',
            legal_basis='legitimate_interest',
            retention_period=timedelta(days=365)
        )
        test_data.append(old_behavioral)
        
        # Recent trade data (should be kept)
        recent_trade = PersonalData(
            data_id='RECENT_TRADE_001',
            subject_id='USER_001',
            category=DataCategory.TRADE_DATA,
            data={'trade': 'data'},
            collected_at=base_time - timedelta(days=30),
            purpose='service_provision',
            legal_basis='contract',
            retention_period=timedelta(days=2555)
        )
        test_data.append(recent_trade)
        
        # Store test data
        for data in test_data:
            await privacy_manager.store_personal_data(data)
        
        # Run retention policy enforcement
        deletion_report = await privacy_manager.enforce_retention_policies(
            dry_run=False,
            generate_report=True
        )
        
        assert deletion_report['total_reviewed'] >= 2
        assert deletion_report['total_deleted'] >= 1
        assert 'deleted_by_category' in deletion_report
        assert DataCategory.BEHAVIORAL_DATA.value in deletion_report['deleted_by_category']
        
        # Verify old data is deleted
        remaining_old = await privacy_manager.get_data_by_id('OLD_BEHAV_001')
        assert remaining_old is None
        
        # Verify recent data is retained
        remaining_recent = await privacy_manager.get_data_by_id('RECENT_TRADE_001')
        assert remaining_recent is not None
        
        # Test deletion holds for legal/investigation purposes
        legal_hold = await privacy_manager.place_legal_hold(
            data_categories=[DataCategory.TRADE_DATA],
            reason='regulatory_investigation',
            hold_until=base_time + timedelta(days=180)
        )
        
        assert legal_hold['hold_id'] is not None
        assert legal_hold['affected_categories'] == [DataCategory.TRADE_DATA]
        
        # Verify held data is not deleted
        held_data_report = await privacy_manager.enforce_retention_policies(
            dry_run=True,
            generate_report=True
        )
        
        assert held_data_report['skipped_due_to_hold'] > 0
    
    @pytest.mark.asyncio
    async def test_privacy_compliance_monitoring(self, privacy_manager):
        """Test ongoing privacy compliance monitoring"""
        # Configure compliance monitoring
        monitoring_config = {
            'consent_validity_check': True,
            'encryption_status_check': True,
            'access_log_review': True,
            'third_party_compliance': True,
            'retention_policy_adherence': True
        }
        
        await privacy_manager.configure_compliance_monitoring(monitoring_config)
        
        # Run compliance check
        compliance_report = await privacy_manager.run_compliance_check(
            check_depth='comprehensive',
            include_recommendations=True
        )
        
        # Verify all areas checked
        assert 'consent_compliance' in compliance_report
        assert 'encryption_compliance' in compliance_report
        assert 'access_control_compliance' in compliance_report
        assert 'retention_compliance' in compliance_report
        assert 'overall_score' in compliance_report
        
        # Check for specific compliance metrics
        consent_metrics = compliance_report['consent_compliance']
        assert 'valid_consents_percentage' in consent_metrics
        assert 'expired_consents' in consent_metrics
        assert 'missing_consents' in consent_metrics
        
        encryption_metrics = compliance_report['encryption_compliance']
        assert 'encrypted_data_percentage' in encryption_metrics
        assert 'weak_encryption_instances' in encryption_metrics
        assert 'key_rotation_status' in encryption_metrics
        
        # Generate compliance dashboard data
        dashboard_data = await privacy_manager.generate_compliance_dashboard()
        
        assert 'compliance_score' in dashboard_data
        assert 'trend_data' in dashboard_data
        assert 'risk_areas' in dashboard_data
        assert 'upcoming_tasks' in dashboard_data
        
        # Set up automated alerts
        alert_config = {
            'consent_expiry_warning': 30,  # Days before expiry
            'retention_deadline_warning': 7,  # Days before deletion
            'compliance_score_threshold': 0.95  # Alert if below 95%
        }
        
        alerts = await privacy_manager.check_compliance_alerts(alert_config)
        
        for alert in alerts:
            assert 'alert_type' in alert
            assert 'severity' in alert
            assert 'affected_items' in alert
            assert 'recommended_action' in alert


class TestPrivacyRegulations:
    """Test specific privacy regulation compliance"""
    
    @pytest.mark.asyncio
    async def test_gdpr_specific_requirements(self, privacy_manager):
        """Test GDPR-specific compliance requirements"""
        # Test lawful basis recording
        processing_activity = {
            'activity_id': 'PROC_001',
            'name': 'User Analytics',
            'purpose': 'Service improvement',
            'legal_basis': 'legitimate_interest',
            'data_categories': [DataCategory.BEHAVIORAL_DATA],
            'retention_period': timedelta(days=365),
            'recipients': ['internal_analytics_team'],
            'international_transfers': False
        }
        
        # Record processing activity (Article 30 requirement)
        await privacy_manager.record_processing_activity(processing_activity)
        
        # Conduct legitimate interest assessment
        lia_result = await privacy_manager.conduct_legitimate_interest_assessment(
            processing_activity=processing_activity,
            necessity_test='Service cannot be improved without usage analytics',
            balancing_test={
                'business_interest': 'high',
                'user_impact': 'low',
                'data_sensitivity': 'low',
                'user_expectation': 'reasonable'
            }
        )
        
        assert lia_result['assessment_result'] in ['APPROVED', 'REJECTED']
        assert 'justification' in lia_result
        
        # Test Data Protection Officer (DPO) requirements
        dpo_tasks = await privacy_manager.get_dpo_tasks()
        assert 'privacy_assessments_pending' in dpo_tasks
        assert 'breach_investigations' in dpo_tasks
        assert 'user_requests_pending' in dpo_tasks
    
    @pytest.mark.asyncio
    async def test_ccpa_specific_requirements(self, privacy_manager):
        """Test CCPA-specific compliance requirements"""
        # Test opt-out of sale
        opt_out_request = {
            'user_id': 'CA_USER_001',
            'request_type': 'opt_out_of_sale',
            'timestamp': datetime.now(timezone.utc)
        }
        
        opt_out_result = await privacy_manager.process_ccpa_opt_out(opt_out_request)
        
        assert opt_out_result['status'] == 'COMPLETED'
        assert opt_out_result['sale_blocked'] == True
        
        # Test financial incentive disclosure
        incentive_program = {
            'program_name': 'Premium Analytics',
            'data_used': [DataCategory.TRADE_DATA, DataCategory.BEHAVIORAL_DATA],
            'value_provided': 'Advanced trading insights',
            'data_value_calculation': 'Based on improved trading performance'
        }
        
        disclosure = await privacy_manager.generate_incentive_disclosure(incentive_program)
        
        assert 'good_faith_estimate' in disclosure
        assert 'calculation_method' in disclosure
        assert 'opt_in_required' in disclosure


if __name__ == "__main__":
    pytest.main([__file__, "-v"])