"""
Authentication Security Testing Suite for GridAttention Trading System
Tests user authentication, API key management, session handling, and access control
"""

import pytest
import asyncio
import jwt
import hashlib
import hmac
import base64
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from unittest.mock import Mock, AsyncMock, patch
import logging
from dataclasses import dataclass
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import pyotp

# GridAttention imports - aligned with system structure
from src.grid_attention_layer import GridAttentionLayer
from src.auth.authentication_manager import AuthenticationManager
from src.auth.api_key_manager import APIKeyManager
from src.auth.session_manager import SessionManager
from src.auth.token_manager import TokenManager
from src.auth.mfa_manager import MFAManager
from src.auth.permission_manager import PermissionManager

# Test utilities
from tests.utils.test_helpers import async_test, create_test_config
from tests.utils.security_helpers import (
    generate_test_user,
    generate_api_key,
    create_test_token
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AuthTestUser:
    """Test user data"""
    username: str
    password: str
    email: str
    api_key: str
    secret_key: str
    permissions: List[str]
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None


class TestUserAuthentication:
    """Test user authentication mechanisms"""
    
    @pytest.fixture
    def auth_manager(self):
        """Create authentication manager"""
        config = create_test_config()
        config['auth'] = {
            'max_login_attempts': 5,
            'lockout_duration': 300,  # 5 minutes
            'password_min_length': 12,
            'require_special_chars': True,
            'require_numbers': True,
            'require_uppercase': True,
            'session_timeout': 3600,  # 1 hour
            'jwt_expiry': 86400  # 24 hours
        }
        return AuthenticationManager(config)
    
    @pytest.fixture
    def test_users(self):
        """Create test users"""
        return [
            AuthTestUser(
                username="trader1",
                password="SecureP@ssw0rd123!",
                email="trader1@test.com",
                api_key="test_api_key_1",
                secret_key="test_secret_key_1",
                permissions=["trade", "read"]
            ),
            AuthTestUser(
                username="admin1",
                password="AdminP@ssw0rd456!",
                email="admin@test.com",
                api_key="test_api_key_2",
                secret_key="test_secret_key_2",
                permissions=["trade", "read", "admin"]
            ),
            AuthTestUser(
                username="readonly1",
                password="ReadOnlyP@ssw0rd789!",
                email="readonly@test.com",
                api_key="test_api_key_3",
                secret_key="test_secret_key_3",
                permissions=["read"]
            )
        ]
    
    @async_test
    async def test_user_registration(self, auth_manager):
        """Test secure user registration"""
        # Valid registration
        user_data = {
            'username': 'newtrader',
            'password': 'SecureNewP@ssw0rd123!',
            'email': 'newtrader@test.com',
            'terms_accepted': True
        }
        
        result = await auth_manager.register_user(user_data)
        
        assert result['success'] is True
        assert 'user_id' in result
        assert result['username'] == user_data['username']
        
        # Verify password is hashed
        stored_user = await auth_manager.get_user(result['user_id'])
        assert stored_user['password_hash'] != user_data['password']
        assert bcrypt.checkpw(
            user_data['password'].encode('utf-8'),
            stored_user['password_hash'].encode('utf-8')
        )
    
    @async_test
    async def test_password_requirements(self, auth_manager):
        """Test password strength requirements"""
        test_cases = [
            ('short', False),  # Too short
            ('NoSpecialChar123', False),  # No special character
            ('nouppercasechar123!', False),  # No uppercase
            ('NoNumbers!', False),  # No numbers
            ('ValidP@ssw0rd123!', True),  # Valid password
            ('Another$ecure1Pass', True),  # Another valid password
        ]
        
        for password, should_pass in test_cases:
            result = await auth_manager.validate_password(password)
            assert result['valid'] == should_pass, f"Password '{password}' validation failed"
            
            if not should_pass:
                assert len(result['errors']) > 0
                logger.info(f"Password '{password}' errors: {result['errors']}")
    
    @async_test
    async def test_login_authentication(self, auth_manager, test_users):
        """Test user login authentication"""
        # Register test user
        user = test_users[0]
        await auth_manager.register_user({
            'username': user.username,
            'password': user.password,
            'email': user.email
        })
        
        # Successful login
        login_result = await auth_manager.authenticate(
            username=user.username,
            password=user.password
        )
        
        assert login_result['success'] is True
        assert 'access_token' in login_result
        assert 'refresh_token' in login_result
        assert login_result['username'] == user.username
        
        # Failed login - wrong password
        failed_result = await auth_manager.authenticate(
            username=user.username,
            password='WrongPassword123!'
        )
        
        assert failed_result['success'] is False
        assert failed_result['error'] == 'Invalid credentials'
    
    @async_test
    async def test_brute_force_protection(self, auth_manager, test_users):
        """Test protection against brute force attacks"""
        user = test_users[0]
        await auth_manager.register_user({
            'username': user.username,
            'password': user.password,
            'email': user.email
        })
        
        # Attempt multiple failed logins
        for i in range(6):  # Max attempts is 5
            result = await auth_manager.authenticate(
                username=user.username,
                password='WrongPassword!'
            )
            
            if i < 5:
                assert result['success'] is False
                assert result['error'] == 'Invalid credentials'
                assert result.get('attempts_remaining', 0) == 5 - i - 1
            else:
                # Account should be locked
                assert result['success'] is False
                assert result['error'] == 'Account locked due to too many failed attempts'
                assert result.get('lockout_duration', 0) > 0
        
        # Verify login still fails with correct password during lockout
        locked_result = await auth_manager.authenticate(
            username=user.username,
            password=user.password
        )
        
        assert locked_result['success'] is False
        assert 'locked' in locked_result['error'].lower()
    
    @async_test
    async def test_session_management(self, auth_manager, test_users):
        """Test secure session management"""
        user = test_users[0]
        await auth_manager.register_user({
            'username': user.username,
            'password': user.password,
            'email': user.email
        })
        
        # Login to create session
        login_result = await auth_manager.authenticate(
            username=user.username,
            password=user.password
        )
        
        session_token = login_result['session_token']
        
        # Verify session is valid
        session_valid = await auth_manager.validate_session(session_token)
        assert session_valid['valid'] is True
        assert session_valid['username'] == user.username
        
        # Test session expiry
        with patch('time.time', return_value=time.time() + 3700):  # 1 hour + 100 seconds
            expired_result = await auth_manager.validate_session(session_token)
            assert expired_result['valid'] is False
            assert expired_result['error'] == 'Session expired'
        
        # Test session logout
        await auth_manager.logout(session_token)
        
        logout_result = await auth_manager.validate_session(session_token)
        assert logout_result['valid'] is False
        assert logout_result['error'] == 'Invalid session'


class TestAPIKeyAuthentication:
    """Test API key authentication"""
    
    @pytest.fixture
    def api_key_manager(self):
        """Create API key manager"""
        config = create_test_config()
        config['api_key'] = {
            'min_length': 32,
            'max_keys_per_user': 5,
            'key_expiry_days': 365,
            'require_ip_whitelist': True,
            'rate_limit_per_minute': 60
        }
        return APIKeyManager(config)
    
    @async_test
    async def test_api_key_generation(self, api_key_manager):
        """Test secure API key generation"""
        user_id = "test_user_123"
        
        # Generate API key
        result = await api_key_manager.generate_api_key(
            user_id=user_id,
            name="Trading Bot Key",
            permissions=["trade", "read"],
            ip_whitelist=["192.168.1.1", "10.0.0.1"]
        )
        
        assert result['success'] is True
        assert 'api_key' in result
        assert 'api_secret' in result
        assert len(result['api_key']) >= 32
        assert len(result['api_secret']) >= 64
        
        # Verify key format
        assert result['api_key'].startswith('gak_')  # GridAttention API Key
        assert all(c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' 
                  for c in result['api_key'][4:])
    
    @async_test
    async def test_api_key_validation(self, api_key_manager):
        """Test API key validation"""
        user_id = "test_user_123"
        
        # Generate key
        key_result = await api_key_manager.generate_api_key(
            user_id=user_id,
            name="Test Key"
        )
        
        api_key = key_result['api_key']
        api_secret = key_result['api_secret']
        
        # Test valid authentication
        auth_result = await api_key_manager.authenticate_api_key(
            api_key=api_key,
            api_secret=api_secret,
            request_ip="192.168.1.1"
        )
        
        assert auth_result['success'] is True
        assert auth_result['user_id'] == user_id
        
        # Test invalid secret
        invalid_result = await api_key_manager.authenticate_api_key(
            api_key=api_key,
            api_secret="wrong_secret",
            request_ip="192.168.1.1"
        )
        
        assert invalid_result['success'] is False
        assert invalid_result['error'] == 'Invalid API credentials'
    
    @async_test
    async def test_api_key_ip_whitelist(self, api_key_manager):
        """Test IP whitelist enforcement"""
        user_id = "test_user_123"
        
        # Generate key with IP whitelist
        key_result = await api_key_manager.generate_api_key(
            user_id=user_id,
            name="Restricted Key",
            ip_whitelist=["192.168.1.1", "192.168.1.2"]
        )
        
        api_key = key_result['api_key']
        api_secret = key_result['api_secret']
        
        # Test from whitelisted IP
        valid_ip_result = await api_key_manager.authenticate_api_key(
            api_key=api_key,
            api_secret=api_secret,
            request_ip="192.168.1.1"
        )
        
        assert valid_ip_result['success'] is True
        
        # Test from non-whitelisted IP
        invalid_ip_result = await api_key_manager.authenticate_api_key(
            api_key=api_key,
            api_secret=api_secret,
            request_ip="10.0.0.1"
        )
        
        assert invalid_ip_result['success'] is False
        assert 'IP not whitelisted' in invalid_ip_result['error']
    
    @async_test
    async def test_api_key_rate_limiting(self, api_key_manager):
        """Test API key rate limiting"""
        user_id = "test_user_123"
        
        # Generate key
        key_result = await api_key_manager.generate_api_key(
            user_id=user_id,
            name="Rate Limited Key"
        )
        
        api_key = key_result['api_key']
        api_secret = key_result['api_secret']
        
        # Make requests up to rate limit
        for i in range(60):  # Rate limit is 60 per minute
            result = await api_key_manager.check_rate_limit(api_key)
            assert result['allowed'] is True
            assert result['remaining'] == 59 - i
        
        # Next request should be rate limited
        limited_result = await api_key_manager.check_rate_limit(api_key)
        assert limited_result['allowed'] is False
        assert limited_result['retry_after'] > 0
        assert 'Rate limit exceeded' in limited_result['error']
    
    @async_test
    async def test_api_key_revocation(self, api_key_manager):
        """Test API key revocation"""
        user_id = "test_user_123"
        
        # Generate key
        key_result = await api_key_manager.generate_api_key(
            user_id=user_id,
            name="Revocable Key"
        )
        
        api_key = key_result['api_key']
        api_secret = key_result['api_secret']
        
        # Key should work initially
        valid_result = await api_key_manager.authenticate_api_key(
            api_key=api_key,
            api_secret=api_secret,
            request_ip="192.168.1.1"
        )
        assert valid_result['success'] is True
        
        # Revoke key
        revoke_result = await api_key_manager.revoke_api_key(
            api_key=api_key,
            user_id=user_id,
            reason="Security breach"
        )
        assert revoke_result['success'] is True
        
        # Key should no longer work
        revoked_result = await api_key_manager.authenticate_api_key(
            api_key=api_key,
            api_secret=api_secret,
            request_ip="192.168.1.1"
        )
        assert revoked_result['success'] is False
        assert 'revoked' in revoked_result['error'].lower()


class TestJWTAuthentication:
    """Test JWT token authentication"""
    
    @pytest.fixture
    def token_manager(self):
        """Create token manager"""
        config = create_test_config()
        config['jwt'] = {
            'secret_key': secrets.token_urlsafe(32),
            'algorithm': 'HS256',
            'access_token_expiry': 3600,  # 1 hour
            'refresh_token_expiry': 604800,  # 7 days
            'issuer': 'gridattention',
            'audience': 'gridattention-api'
        }
        return TokenManager(config)
    
    @async_test
    async def test_jwt_token_generation(self, token_manager):
        """Test JWT token generation"""
        user_data = {
            'user_id': 'test_user_123',
            'username': 'testuser',
            'permissions': ['trade', 'read']
        }
        
        # Generate tokens
        tokens = await token_manager.generate_tokens(user_data)
        
        assert 'access_token' in tokens
        assert 'refresh_token' in tokens
        assert tokens['token_type'] == 'Bearer'
        assert tokens['expires_in'] == 3600
        
        # Decode and verify access token
        decoded = jwt.decode(
            tokens['access_token'],
            token_manager.secret_key,
            algorithms=[token_manager.algorithm],
            audience=token_manager.audience,
            issuer=token_manager.issuer
        )
        
        assert decoded['user_id'] == user_data['user_id']
        assert decoded['username'] == user_data['username']
        assert decoded['permissions'] == user_data['permissions']
        assert 'exp' in decoded
        assert 'iat' in decoded
        assert 'jti' in decoded  # JWT ID for tracking
    
    @async_test
    async def test_jwt_token_validation(self, token_manager):
        """Test JWT token validation"""
        user_data = {
            'user_id': 'test_user_123',
            'username': 'testuser',
            'permissions': ['trade', 'read']
        }
        
        tokens = await token_manager.generate_tokens(user_data)
        
        # Validate valid token
        valid_result = await token_manager.validate_token(tokens['access_token'])
        assert valid_result['valid'] is True
        assert valid_result['user_id'] == user_data['user_id']
        
        # Test invalid token
        invalid_result = await token_manager.validate_token('invalid.token.here')
        assert invalid_result['valid'] is False
        assert 'error' in invalid_result
        
        # Test expired token
        expired_token = jwt.encode(
            {
                'user_id': user_data['user_id'],
                'exp': datetime.utcnow() - timedelta(hours=1)
            },
            token_manager.secret_key,
            algorithm=token_manager.algorithm
        )
        
        expired_result = await token_manager.validate_token(expired_token)
        assert expired_result['valid'] is False
        assert 'expired' in expired_result['error'].lower()
    
    @async_test
    async def test_jwt_refresh_token(self, token_manager):
        """Test JWT refresh token flow"""
        user_data = {
            'user_id': 'test_user_123',
            'username': 'testuser',
            'permissions': ['trade', 'read']
        }
        
        # Generate initial tokens
        tokens = await token_manager.generate_tokens(user_data)
        refresh_token = tokens['refresh_token']
        
        # Use refresh token to get new access token
        refresh_result = await token_manager.refresh_access_token(refresh_token)
        
        assert refresh_result['success'] is True
        assert 'access_token' in refresh_result
        assert refresh_result['access_token'] != tokens['access_token']
        
        # Verify new access token is valid
        valid_result = await token_manager.validate_token(refresh_result['access_token'])
        assert valid_result['valid'] is True
        
        # Test refresh token rotation
        if token_manager.rotate_refresh_tokens:
            assert 'refresh_token' in refresh_result
            assert refresh_result['refresh_token'] != refresh_token
    
    @async_test
    async def test_jwt_token_blacklist(self, token_manager):
        """Test JWT token blacklisting"""
        user_data = {
            'user_id': 'test_user_123',
            'username': 'testuser'
        }
        
        tokens = await token_manager.generate_tokens(user_data)
        access_token = tokens['access_token']
        
        # Token should be valid initially
        valid_result = await token_manager.validate_token(access_token)
        assert valid_result['valid'] is True
        
        # Blacklist token
        blacklist_result = await token_manager.blacklist_token(
            access_token,
            reason="User logout"
        )
        assert blacklist_result['success'] is True
        
        # Token should now be invalid
        blacklisted_result = await token_manager.validate_token(access_token)
        assert blacklisted_result['valid'] is False
        assert 'blacklisted' in blacklisted_result['error'].lower()


class TestMultiFactorAuthentication:
    """Test multi-factor authentication"""
    
    @pytest.fixture
    def mfa_manager(self):
        """Create MFA manager"""
        config = create_test_config()
        config['mfa'] = {
            'totp_window': 30,  # seconds
            'totp_skew': 1,  # Allow 1 window skew
            'backup_codes_count': 10,
            'sms_timeout': 300,  # 5 minutes
            'email_timeout': 600  # 10 minutes
        }
        return MFAManager(config)
    
    @async_test
    async def test_totp_setup(self, mfa_manager):
        """Test TOTP (Time-based One-Time Password) setup"""
        user_id = "test_user_123"
        
        # Generate TOTP secret
        setup_result = await mfa_manager.setup_totp(user_id)
        
        assert setup_result['success'] is True
        assert 'secret' in setup_result
        assert 'qr_code' in setup_result
        assert 'backup_codes' in setup_result
        assert len(setup_result['backup_codes']) == 10
        
        # Verify secret format
        secret = setup_result['secret']
        assert len(secret) == 32  # Base32 encoded
        assert secret.isupper()
        
        # Verify QR code contains correct data
        assert f'otpauth://totp/GridAttention:{user_id}' in setup_result['qr_code']
    
    @async_test
    async def test_totp_verification(self, mfa_manager):
        """Test TOTP verification"""
        user_id = "test_user_123"
        
        # Setup TOTP
        setup_result = await mfa_manager.setup_totp(user_id)
        secret = setup_result['secret']
        
        # Generate valid TOTP code
        totp = pyotp.TOTP(secret)
        valid_code = totp.now()
        
        # Verify valid code
        verify_result = await mfa_manager.verify_totp(
            user_id=user_id,
            code=valid_code
        )
        
        assert verify_result['success'] is True
        assert verify_result['method'] == 'totp'
        
        # Test invalid code
        invalid_result = await mfa_manager.verify_totp(
            user_id=user_id,
            code="000000"
        )
        
        assert invalid_result['success'] is False
        assert invalid_result['error'] == 'Invalid TOTP code'
        
        # Test code reuse prevention
        reuse_result = await mfa_manager.verify_totp(
            user_id=user_id,
            code=valid_code
        )
        
        assert reuse_result['success'] is False
        assert 'already used' in reuse_result['error'].lower()
    
    @async_test
    async def test_backup_codes(self, mfa_manager):
        """Test backup code functionality"""
        user_id = "test_user_123"
        
        # Setup MFA with backup codes
        setup_result = await mfa_manager.setup_totp(user_id)
        backup_codes = setup_result['backup_codes']
        
        # Use a backup code
        backup_result = await mfa_manager.verify_backup_code(
            user_id=user_id,
            code=backup_codes[0]
        )
        
        assert backup_result['success'] is True
        assert backup_result['method'] == 'backup_code'
        assert backup_result['remaining_codes'] == 9
        
        # Verify code can't be reused
        reuse_result = await mfa_manager.verify_backup_code(
            user_id=user_id,
            code=backup_codes[0]
        )
        
        assert reuse_result['success'] is False
        assert 'Invalid or already used' in reuse_result['error']
    
    @async_test
    async def test_sms_mfa(self, mfa_manager):
        """Test SMS-based MFA"""
        user_id = "test_user_123"
        phone_number = "+1234567890"
        
        # Send SMS code
        with patch('src.auth.mfa_manager.send_sms') as mock_sms:
            mock_sms.return_value = {'success': True}
            
            send_result = await mfa_manager.send_sms_code(
                user_id=user_id,
                phone_number=phone_number
            )
            
            assert send_result['success'] is True
            assert 'code_sent' in send_result
            
            # Get the sent code from mock
            sent_code = mock_sms.call_args[0][1]
            assert len(sent_code) == 6
            assert sent_code.isdigit()
        
        # Verify SMS code
        verify_result = await mfa_manager.verify_sms_code(
            user_id=user_id,
            code=sent_code
        )
        
        assert verify_result['success'] is True
        assert verify_result['method'] == 'sms'


class TestSecurityHeaders:
    """Test security headers and CORS"""
    
    @async_test
    async def test_security_headers(self):
        """Test that security headers are properly set"""
        grid_attention = GridAttentionLayer(create_test_config())
        
        # Simulate HTTP request
        request_headers = {
            'Origin': 'https://example.com',
            'User-Agent': 'TestClient/1.0'
        }
        
        response = await grid_attention.handle_request(
            method='GET',
            path='/api/status',
            headers=request_headers
        )
        
        # Verify security headers
        assert response.headers['X-Content-Type-Options'] == 'nosniff'
        assert response.headers['X-Frame-Options'] == 'DENY'
        assert response.headers['X-XSS-Protection'] == '1; mode=block'
        assert response.headers['Strict-Transport-Security'] == 'max-age=31536000; includeSubDomains'
        assert 'Server' not in response.headers  # Don't expose server info
        
        # Verify CSP header
        csp = response.headers.get('Content-Security-Policy', '')
        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp
        assert "style-src 'self' 'unsafe-inline'" in csp
    
    @async_test
    async def test_cors_configuration(self):
        """Test CORS configuration"""
        config = create_test_config()
        config['cors'] = {
            'allowed_origins': ['https://app.gridattention.com', 'https://test.gridattention.com'],
            'allowed_methods': ['GET', 'POST', 'PUT', 'DELETE'],
            'allowed_headers': ['Content-Type', 'Authorization'],
            'expose_headers': ['X-Request-ID'],
            'max_age': 86400,
            'credentials': True
        }
        
        grid_attention = GridAttentionLayer(config)
        
        # Test allowed origin
        response = await grid_attention.handle_request(
            method='OPTIONS',
            path='/api/trade',
            headers={'Origin': 'https://app.gridattention.com'}
        )
        
        assert response.headers['Access-Control-Allow-Origin'] == 'https://app.gridattention.com'
        assert response.headers['Access-Control-Allow-Methods'] == 'GET, POST, PUT, DELETE'
        assert response.headers['Access-Control-Allow-Credentials'] == 'true'
        assert response.headers['Access-Control-Max-Age'] == '86400'
        
        # Test disallowed origin
        response = await grid_attention.handle_request(
            method='OPTIONS',
            path='/api/trade',
            headers={'Origin': 'https://evil.com'}
        )
        
        assert 'Access-Control-Allow-Origin' not in response.headers


class TestOAuthIntegration:
    """Test OAuth 2.0 integration"""
    
    @pytest.fixture
    def oauth_manager(self):
        """Create OAuth manager"""
        config = create_test_config()
        config['oauth'] = {
            'providers': {
                'google': {
                    'client_id': 'test_google_client_id',
                    'client_secret': 'test_google_client_secret',
                    'redirect_uri': 'https://api.gridattention.com/auth/google/callback'
                },
                'github': {
                    'client_id': 'test_github_client_id',
                    'client_secret': 'test_github_client_secret',
                    'redirect_uri': 'https://api.gridattention.com/auth/github/callback'
                }
            }
        }
        return OAuthManager(config)
    
    @async_test
    async def test_oauth_flow(self, oauth_manager):
        """Test OAuth authentication flow"""
        provider = 'google'
        
        # Generate authorization URL
        auth_url_result = await oauth_manager.get_authorization_url(
            provider=provider,
            state=secrets.token_urlsafe(32)
        )
        
        assert auth_url_result['success'] is True
        assert 'authorization_url' in auth_url_result
        assert 'state' in auth_url_result
        
        # Verify URL contains required parameters
        auth_url = auth_url_result['authorization_url']
        assert 'client_id=test_google_client_id' in auth_url
        assert 'redirect_uri=' in auth_url
        assert 'state=' in auth_url
        assert 'response_type=code' in auth_url
        
        # Simulate callback with authorization code
        with patch('src.auth.oauth_manager.exchange_code_for_token') as mock_exchange:
            mock_exchange.return_value = {
                'access_token': 'test_access_token',
                'refresh_token': 'test_refresh_token',
                'expires_in': 3600
            }
            
            callback_result = await oauth_manager.handle_callback(
                provider=provider,
                code='test_auth_code',
                state=auth_url_result['state']
            )
            
            assert callback_result['success'] is True
            assert 'access_token' in callback_result
            assert 'user_info' in callback_result


# Helper Classes

class OAuthManager:
    """OAuth 2.0 manager"""
    
    def __init__(self, config):
        self.config = config
        self.providers = config['oauth']['providers']
    
    async def get_authorization_url(self, provider: str, state: str) -> Dict:
        """Generate OAuth authorization URL"""
        if provider not in self.providers:
            return {'success': False, 'error': 'Invalid provider'}
        
        provider_config = self.providers[provider]
        
        # Build authorization URL
        base_url = {
            'google': 'https://accounts.google.com/o/oauth2/v2/auth',
            'github': 'https://github.com/login/oauth/authorize'
        }[provider]
        
        params = {
            'client_id': provider_config['client_id'],
            'redirect_uri': provider_config['redirect_uri'],
            'response_type': 'code',
            'state': state,
            'scope': 'email profile' if provider == 'google' else 'user:email'
        }
        
        query_string = '&'.join(f'{k}={v}' for k, v in params.items())
        authorization_url = f'{base_url}?{query_string}'
        
        return {
            'success': True,
            'authorization_url': authorization_url,
            'state': state
        }
    
    async def handle_callback(self, provider: str, code: str, state: str) -> Dict:
        """Handle OAuth callback"""
        # Exchange code for token
        token_data = await self.exchange_code_for_token(provider, code)
        
        # Get user info
        user_info = await self.get_user_info(provider, token_data['access_token'])
        
        return {
            'success': True,
            'access_token': token_data['access_token'],
            'user_info': user_info
        }
    
    async def exchange_code_for_token(self, provider: str, code: str) -> Dict:
        """Exchange authorization code for access token"""
        # This would make actual HTTP request in production
        pass
    
    async def get_user_info(self, provider: str, access_token: str) -> Dict:
        """Get user info from OAuth provider"""
        # This would make actual HTTP request in production
        return {
            'id': 'oauth_user_123',
            'email': 'user@example.com',
            'name': 'Test User'
        }