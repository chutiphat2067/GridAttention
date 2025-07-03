"""
API Security Testing Suite for GridAttention Trading System
Tests API endpoints, rate limiting, authentication, CORS, request validation, and response security
"""

import pytest
import asyncio
import json
import time
import hashlib
import hmac
import base64
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import logging
import aiohttp
from aiohttp import web
import jwt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# GridAttention imports - aligned with system structure
from src.api.api_server import GridAttentionAPI
from src.api.middleware.auth_middleware import AuthMiddleware
from src.api.middleware.rate_limit_middleware import RateLimitMiddleware
from src.api.middleware.cors_middleware import CORSMiddleware
from src.api.middleware.security_middleware import SecurityMiddleware
from src.api.middleware.validation_middleware import ValidationMiddleware
from src.security.api_security import APISecurityManager
from src.security.request_validator import RequestValidator
from src.security.response_sanitizer import ResponseSanitizer

# Test utilities
from tests.utils.test_helpers import async_test, create_test_config
from tests.utils.api_test_client import TestAPIClient
from tests.utils.security_helpers import (
    generate_test_token,
    create_malicious_payloads,
    generate_api_keys
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAPIAuthentication:
    """Test API authentication mechanisms"""
    
    @pytest.fixture
    async def api_server(self):
        """Create test API server"""
        config = create_test_config()
        config['api'] = {
            'host': '127.0.0.1',
            'port': 8888,
            'enable_auth': True,
            'enable_rate_limit': True,
            'enable_cors': True,
            'enable_https': True
        }
        server = GridAttentionAPI(config)
        await server.setup()
        yield server
        await server.cleanup()
    
    @pytest.fixture
    def test_client(self, api_server):
        """Create test client"""
        return TestAPIClient(api_server)
    
    @async_test
    async def test_bearer_token_authentication(self, test_client):
        """Test Bearer token authentication"""
        # Valid token
        valid_token = generate_test_token({
            'user_id': 'test_user',
            'permissions': ['trade:read', 'trade:create']
        })
        
        response = await test_client.get(
            '/api/v1/orders',
            headers={'Authorization': f'Bearer {valid_token}'}
        )
        
        assert response.status == 200
        
        # Invalid token
        invalid_token = 'invalid.token.here'
        
        response = await test_client.get(
            '/api/v1/orders',
            headers={'Authorization': f'Bearer {invalid_token}'}
        )
        
        assert response.status == 401
        assert response.json['error'] == 'Invalid token'
        
        # Expired token
        expired_token = generate_test_token(
            {'user_id': 'test_user'},
            expires_delta=timedelta(hours=-1)
        )
        
        response = await test_client.get(
            '/api/v1/orders',
            headers={'Authorization': f'Bearer {expired_token}'}
        )
        
        assert response.status == 401
        assert 'expired' in response.json['error'].lower()
    
    @async_test
    async def test_api_key_authentication(self, test_client):
        """Test API key authentication"""
        # Generate valid API key
        api_key, api_secret = generate_api_keys()
        
        # Create request signature
        timestamp = str(int(time.time()))
        method = 'GET'
        path = '/api/v1/account/balance'
        body = ''
        
        message = f"{timestamp}{method}{path}{body}"
        signature = hmac.new(
            api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            'X-API-Key': api_key,
            'X-API-Signature': signature,
            'X-API-Timestamp': timestamp
        }
        
        response = await test_client.get(path, headers=headers)
        assert response.status == 200
        
        # Test with invalid signature
        headers['X-API-Signature'] = 'invalid_signature'
        response = await test_client.get(path, headers=headers)
        
        assert response.status == 401
        assert 'Invalid signature' in response.json['error']
        
        # Test with expired timestamp (older than 5 minutes)
        old_timestamp = str(int(time.time()) - 400)
        headers['X-API-Timestamp'] = old_timestamp
        
        response = await test_client.get(path, headers=headers)
        assert response.status == 401
        assert 'Request expired' in response.json['error']
    
    @async_test
    async def test_oauth2_authentication(self, test_client):
        """Test OAuth 2.0 authentication flow"""
        # Test authorization code flow
        # Step 1: Get authorization URL
        response = await test_client.get(
            '/api/v1/auth/oauth/authorize',
            params={
                'client_id': 'test_client',
                'redirect_uri': 'http://localhost:3000/callback',
                'response_type': 'code',
                'scope': 'trade:read trade:create',
                'state': 'random_state_123'
            }
        )
        
        assert response.status == 302  # Redirect
        location = response.headers.get('Location')
        assert 'code=' in location
        
        # Extract authorization code
        auth_code = location.split('code=')[1].split('&')[0]
        
        # Step 2: Exchange code for token
        response = await test_client.post(
            '/api/v1/auth/oauth/token',
            json={
                'grant_type': 'authorization_code',
                'code': auth_code,
                'client_id': 'test_client',
                'client_secret': 'test_secret',
                'redirect_uri': 'http://localhost:3000/callback'
            }
        )
        
        assert response.status == 200
        token_data = response.json
        assert 'access_token' in token_data
        assert 'refresh_token' in token_data
        assert token_data['token_type'] == 'Bearer'
    
    @async_test
    async def test_mutual_tls_authentication(self, test_client):
        """Test mutual TLS (mTLS) authentication"""
        # Create client with certificate
        client_cert = 'path/to/client.crt'
        client_key = 'path/to/client.key'
        
        # Mock SSL context
        with patch('ssl.create_default_context') as mock_ssl:
            mock_context = MagicMock()
            mock_ssl.return_value = mock_context
            
            response = await test_client.get(
                '/api/v1/orders',
                ssl_cert=(client_cert, client_key)
            )
            
            # Verify certificate was loaded
            mock_context.load_cert_chain.assert_called_with(
                certfile=client_cert,
                keyfile=client_key
            )
    
    @async_test
    async def test_multi_factor_authentication(self, test_client):
        """Test multi-factor authentication requirement"""
        # Login with username/password
        response = await test_client.post(
            '/api/v1/auth/login',
            json={
                'username': 'test_user',
                'password': 'SecurePassword123!'
            }
        )
        
        assert response.status == 200
        assert response.json['requires_mfa'] is True
        mfa_token = response.json['mfa_token']
        
        # Submit MFA code
        response = await test_client.post(
            '/api/v1/auth/mfa/verify',
            json={
                'mfa_token': mfa_token,
                'code': '123456'  # TOTP code
            }
        )
        
        assert response.status == 200
        assert 'access_token' in response.json


class TestAPIRateLimiting:
    """Test API rate limiting"""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter"""
        config = create_test_config()
        config['rate_limit'] = {
            'default_limit': 60,  # per minute
            'burst_limit': 10,
            'window_seconds': 60,
            'by_endpoint': {
                '/api/v1/orders': 30,
                '/api/v1/trades': 100,
                '/api/v1/auth/login': 5
            }
        }
        return RateLimitMiddleware(config)
    
    @async_test
    async def test_basic_rate_limiting(self, test_client):
        """Test basic rate limiting"""
        # Make requests up to limit
        for i in range(60):
            response = await test_client.get('/api/v1/account/info')
            assert response.status == 200
            assert 'X-RateLimit-Limit' in response.headers
            assert 'X-RateLimit-Remaining' in response.headers
            assert int(response.headers['X-RateLimit-Remaining']) == 59 - i
        
        # Next request should be rate limited
        response = await test_client.get('/api/v1/account/info')
        assert response.status == 429  # Too Many Requests
        assert 'X-RateLimit-Reset' in response.headers
        
        error = response.json
        assert error['error'] == 'Rate limit exceeded'
        assert 'retry_after' in error
    
    @async_test
    async def test_endpoint_specific_limits(self, test_client):
        """Test endpoint-specific rate limits"""
        # Login endpoint has limit of 5
        for i in range(5):
            response = await test_client.post(
                '/api/v1/auth/login',
                json={'username': 'test', 'password': 'wrong'}
            )
            assert response.status in [200, 401]
        
        # 6th attempt should be rate limited
        response = await test_client.post(
            '/api/v1/auth/login',
            json={'username': 'test', 'password': 'wrong'}
        )
        assert response.status == 429
    
    @async_test
    async def test_distributed_rate_limiting(self, test_client):
        """Test distributed rate limiting with Redis"""
        # Mock Redis client
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.from_url.return_value = mock_redis_instance
            
            # Simulate distributed counter
            mock_redis_instance.incr.return_value = 1
            mock_redis_instance.expire.return_value = True
            
            response = await test_client.get('/api/v1/orders')
            
            # Verify Redis was used
            mock_redis_instance.incr.assert_called()
            key_used = mock_redis_instance.incr.call_args[0][0]
            assert 'rate_limit' in key_used
    
    @async_test
    async def test_rate_limit_bypass_for_privileged_users(self, test_client):
        """Test rate limit bypass for privileged API keys"""
        # Create privileged API key
        api_key, api_secret = generate_api_keys(privileged=True)
        
        # Make many requests
        for i in range(100):
            response = await test_client.get(
                '/api/v1/orders',
                headers=create_api_headers(api_key, api_secret, 'GET', '/api/v1/orders')
            )
            assert response.status == 200
            # Should have unlimited rate limit
            assert response.headers.get('X-RateLimit-Limit') == 'unlimited'


class TestAPICORS:
    """Test CORS (Cross-Origin Resource Sharing) configuration"""
    
    @pytest.fixture
    def cors_middleware(self):
        """Create CORS middleware"""
        config = create_test_config()
        config['cors'] = {
            'allowed_origins': [
                'https://app.gridattention.com',
                'https://test.gridattention.com',
                'http://localhost:3000'  # Development
            ],
            'allowed_methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
            'allowed_headers': ['Content-Type', 'Authorization', 'X-API-Key'],
            'expose_headers': ['X-Request-ID', 'X-RateLimit-Remaining'],
            'max_age': 86400,
            'credentials': True
        }
        return CORSMiddleware(config)
    
    @async_test
    async def test_cors_preflight_request(self, test_client):
        """Test CORS preflight request handling"""
        response = await test_client.options(
            '/api/v1/orders',
            headers={
                'Origin': 'https://app.gridattention.com',
                'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'Content-Type, Authorization'
            }
        )
        
        assert response.status == 200
        assert response.headers['Access-Control-Allow-Origin'] == 'https://app.gridattention.com'
        assert 'POST' in response.headers['Access-Control-Allow-Methods']
        assert 'Authorization' in response.headers['Access-Control-Allow-Headers']
        assert response.headers['Access-Control-Allow-Credentials'] == 'true'
        assert response.headers['Access-Control-Max-Age'] == '86400'
    
    @async_test
    async def test_cors_actual_request(self, test_client):
        """Test CORS headers on actual requests"""
        valid_token = generate_test_token({'user_id': 'test'})
        
        response = await test_client.get(
            '/api/v1/orders',
            headers={
                'Origin': 'https://app.gridattention.com',
                'Authorization': f'Bearer {valid_token}'
            }
        )
        
        assert response.status == 200
        assert response.headers['Access-Control-Allow-Origin'] == 'https://app.gridattention.com'
        assert response.headers['Access-Control-Allow-Credentials'] == 'true'
        assert 'X-Request-ID' in response.headers['Access-Control-Expose-Headers']
    
    @async_test
    async def test_cors_disallowed_origin(self, test_client):
        """Test CORS rejection of disallowed origins"""
        response = await test_client.get(
            '/api/v1/orders',
            headers={
                'Origin': 'https://evil.com',
                'Authorization': 'Bearer valid_token'
            }
        )
        
        # Should not include CORS headers for disallowed origin
        assert 'Access-Control-Allow-Origin' not in response.headers
    
    @async_test
    async def test_cors_wildcard_subdomain(self, test_client):
        """Test CORS with wildcard subdomain matching"""
        # Configure to allow *.gridattention.com
        response = await test_client.get(
            '/api/v1/orders',
            headers={
                'Origin': 'https://staging.gridattention.com'
            }
        )
        
        # Should be allowed if wildcard matching is enabled
        # This depends on configuration
        pass


class TestAPISecurityHeaders:
    """Test security headers in API responses"""
    
    @async_test
    async def test_security_headers_presence(self, test_client):
        """Test that all security headers are present"""
        response = await test_client.get('/api/v1/status')
        
        # Required security headers
        required_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',
            'Content-Security-Policy': "default-src 'self'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }
        
        for header, expected_value in required_headers.items():
            assert header in response.headers
            if expected_value:
                assert response.headers[header] == expected_value
        
        # Should not expose server information
        assert 'Server' not in response.headers
        assert 'X-Powered-By' not in response.headers
    
    @async_test
    async def test_content_type_validation(self, test_client):
        """Test Content-Type validation and enforcement"""
        # POST without Content-Type
        response = await test_client.post(
            '/api/v1/orders',
            data='{"symbol": "BTC/USDT"}',
            headers={}
        )
        
        assert response.status == 400
        assert 'Content-Type required' in response.json['error']
        
        # POST with wrong Content-Type
        response = await test_client.post(
            '/api/v1/orders',
            data='{"symbol": "BTC/USDT"}',
            headers={'Content-Type': 'text/plain'}
        )
        
        assert response.status == 400
        assert 'Invalid Content-Type' in response.json['error']
        
        # POST with correct Content-Type
        response = await test_client.post(
            '/api/v1/orders',
            json={'symbol': 'BTC/USDT', 'side': 'buy', 'quantity': 0.1},
            headers={'Authorization': 'Bearer valid_token'}
        )
        
        assert response.status in [200, 201]


class TestAPIRequestValidation:
    """Test API request validation"""
    
    @pytest.fixture
    def request_validator(self):
        """Create request validator"""
        config = create_test_config()
        return RequestValidator(config)
    
    @async_test
    async def test_request_size_limits(self, test_client):
        """Test request size limits"""
        # Create large payload (over 1MB limit)
        large_data = 'x' * (1024 * 1024 + 1)
        
        response = await test_client.post(
            '/api/v1/orders/bulk',
            data=large_data,
            headers={'Content-Type': 'application/json'}
        )
        
        assert response.status == 413  # Payload Too Large
        assert 'Request too large' in response.json['error']
    
    @async_test
    async def test_parameter_validation(self, test_client):
        """Test query parameter validation"""
        # Valid parameters
        response = await test_client.get(
            '/api/v1/orders',
            params={
                'symbol': 'BTC/USDT',
                'limit': 100,
                'offset': 0,
                'status': 'open'
            }
        )
        assert response.status == 200
        
        # Invalid parameters
        invalid_params = [
            {'limit': 'abc'},  # Should be integer
            {'limit': -1},  # Negative not allowed
            {'limit': 10000},  # Too large
            {'offset': -10},  # Negative offset
            {'symbol': 'INVALID SYMBOL'},  # Invalid format
            {'status': 'invalid_status'},  # Invalid enum
            {'sort': 'id; DROP TABLE orders;--'},  # SQL injection
        ]
        
        for params in invalid_params:
            response = await test_client.get('/api/v1/orders', params=params)
            assert response.status == 400
            assert 'Invalid parameter' in response.json['error']
    
    @async_test
    async def test_request_id_tracking(self, test_client):
        """Test request ID generation and tracking"""
        response = await test_client.get('/api/v1/status')
        
        # Should have request ID header
        assert 'X-Request-ID' in response.headers
        request_id = response.headers['X-Request-ID']
        
        # Should be valid UUID
        assert len(request_id) == 36
        assert request_id.count('-') == 4
        
        # Error responses should include request ID
        response = await test_client.get('/api/v1/invalid_endpoint')
        assert response.status == 404
        assert 'request_id' in response.json
    
    @async_test
    async def test_json_payload_validation(self, test_client):
        """Test JSON payload validation"""
        # Malformed JSON
        response = await test_client.post(
            '/api/v1/orders',
            data='{invalid json}',
            headers={'Content-Type': 'application/json'}
        )
        
        assert response.status == 400
        assert 'Invalid JSON' in response.json['error']
        
        # Deeply nested JSON (DoS prevention)
        deeply_nested = {'a': {}}
        current = deeply_nested['a']
        for i in range(100):
            current['b'] = {}
            current = current['b']
        
        response = await test_client.post(
            '/api/v1/orders',
            json=deeply_nested
        )
        
        assert response.status == 400
        assert 'JSON too deeply nested' in response.json['error']


class TestAPIResponseSecurity:
    """Test API response security"""
    
    @pytest.fixture
    def response_sanitizer(self):
        """Create response sanitizer"""
        return ResponseSanitizer(create_test_config())
    
    @async_test
    async def test_sensitive_data_filtering(self, test_client):
        """Test filtering of sensitive data from responses"""
        # Get user profile (contains sensitive data)
        response = await test_client.get(
            '/api/v1/user/profile',
            headers={'Authorization': 'Bearer valid_token'}
        )
        
        assert response.status == 200
        user_data = response.json
        
        # Should not contain sensitive fields
        assert 'password' not in user_data
        assert 'password_hash' not in user_data
        assert 'api_secret' not in user_data
        assert 'mfa_secret' not in user_data
        assert 'session_tokens' not in user_data
        
        # Should contain safe fields
        assert 'user_id' in user_data
        assert 'username' in user_data
        assert 'email' in user_data
    
    @async_test
    async def test_error_message_sanitization(self, test_client):
        """Test that error messages don't leak sensitive information"""
        # Database error simulation
        with patch('src.database.query', side_effect=Exception("Connection to database 'prod_db' on host '10.0.0.5' failed")):
            response = await test_client.get('/api/v1/orders')
            
            assert response.status == 500
            error = response.json
            
            # Should not contain internal details
            assert '10.0.0.5' not in error['error']
            assert 'prod_db' not in error['error']
            assert error['error'] == 'Internal server error'
            assert 'request_id' in error  # For debugging
    
    @async_test
    async def test_response_compression(self, test_client):
        """Test response compression for large payloads"""
        # Request with Accept-Encoding
        response = await test_client.get(
            '/api/v1/orders/history',  # Large response
            headers={
                'Accept-Encoding': 'gzip, deflate',
                'Authorization': 'Bearer valid_token'
            }
        )
        
        assert response.status == 200
        assert response.headers.get('Content-Encoding') in ['gzip', 'deflate']
        
        # Verify content is actually compressed
        assert len(response.content) < len(response.json)  # Compressed size < JSON size
    
    @async_test
    async def test_json_hijacking_prevention(self, test_client):
        """Test prevention of JSON hijacking attacks"""
        # Array response should be wrapped or prefixed
        response = await test_client.get(
            '/api/v1/orders',
            headers={'Authorization': 'Bearer valid_token'}
        )
        
        assert response.status == 200
        
        # Response should not be a direct array
        content = response.text
        assert not content.startswith('[')
        
        # Should be wrapped in object or have prefix
        data = response.json
        assert isinstance(data, dict)
        assert 'data' in data or 'orders' in data


class TestAPIVersioning:
    """Test API versioning security"""
    
    @async_test
    async def test_version_in_url(self, test_client):
        """Test URL-based versioning"""
        # Current version
        response = await test_client.get('/api/v1/status')
        assert response.status == 200
        
        # Future version (not yet available)
        response = await test_client.get('/api/v2/status')
        assert response.status == 404
        
        # No version
        response = await test_client.get('/api/status')
        assert response.status == 404
    
    @async_test
    async def test_version_in_header(self, test_client):
        """Test header-based versioning"""
        # With version header
        response = await test_client.get(
            '/api/orders',
            headers={'API-Version': '1.0'}
        )
        assert response.status == 200
        
        # Unsupported version
        response = await test_client.get(
            '/api/orders',
            headers={'API-Version': '99.0'}
        )
        assert response.status == 400
        assert 'Unsupported API version' in response.json['error']
    
    @async_test
    async def test_deprecation_warnings(self, test_client):
        """Test deprecation warnings for old versions"""
        # Using older endpoint
        response = await test_client.get(
            '/api/v1/orders/list',  # Deprecated in favor of /api/v1/orders
            headers={'Authorization': 'Bearer valid_token'}
        )
        
        assert response.status == 200
        assert 'Sunset' in response.headers  # RFC 8594
        assert 'Deprecation' in response.headers
        assert response.headers['Deprecation'] == 'true'
        
        # Warning in response
        assert 'warnings' in response.json
        assert any('deprecated' in w.lower() for w in response.json['warnings'])


class TestWebSocketSecurity:
    """Test WebSocket API security"""
    
    @async_test
    async def test_websocket_authentication(self, test_client):
        """Test WebSocket connection authentication"""
        # Connect without authentication
        try:
            ws = await test_client.ws_connect('/ws/v1/stream')
            await ws.close()
            assert False, "Should not connect without auth"
        except Exception as e:
            assert '401' in str(e) or 'Unauthorized' in str(e)
        
        # Connect with valid token
        valid_token = generate_test_token({'user_id': 'test'})
        ws = await test_client.ws_connect(
            '/ws/v1/stream',
            headers={'Authorization': f'Bearer {valid_token}'}
        )
        
        # Should receive welcome message
        msg = await ws.receive_json()
        assert msg['type'] == 'welcome'
        assert 'connection_id' in msg
        
        await ws.close()
    
    @async_test
    async def test_websocket_message_validation(self, test_client):
        """Test WebSocket message validation"""
        valid_token = generate_test_token({'user_id': 'test'})
        ws = await test_client.ws_connect(
            '/ws/v1/stream',
            headers={'Authorization': f'Bearer {valid_token}'}
        )
        
        # Send valid message
        await ws.send_json({
            'type': 'subscribe',
            'channel': 'orders',
            'symbol': 'BTC/USDT'
        })
        
        response = await ws.receive_json()
        assert response['type'] == 'subscribed'
        
        # Send invalid message
        await ws.send_json({
            'type': 'invalid_type',
            'data': '<script>alert(1)</script>'
        })
        
        response = await ws.receive_json()
        assert response['type'] == 'error'
        assert 'Invalid message type' in response['error']
        
        await ws.close()
    
    @async_test
    async def test_websocket_rate_limiting(self, test_client):
        """Test WebSocket message rate limiting"""
        valid_token = generate_test_token({'user_id': 'test'})
        ws = await test_client.ws_connect(
            '/ws/v1/stream',
            headers={'Authorization': f'Bearer {valid_token}'}
        )
        
        # Send many messages quickly
        for i in range(100):
            await ws.send_json({
                'type': 'ping',
                'id': i
            })
        
        # Should receive rate limit error
        response = await ws.receive_json()
        assert response['type'] == 'error'
        assert 'rate limit' in response['error'].lower()
        
        # Connection might be closed
        assert ws.closed or await ws.receive_json() == {'type': 'close'}


class TestAPIAuditLogging:
    """Test API audit logging"""
    
    @async_test
    async def test_successful_request_logging(self, test_client):
        """Test logging of successful API requests"""
        with patch('src.audit.logger.log') as mock_log:
            response = await test_client.get(
                '/api/v1/orders',
                headers={'Authorization': 'Bearer valid_token'}
            )
            
            assert response.status == 200
            
            # Verify audit log was created
            mock_log.assert_called()
            log_entry = mock_log.call_args[0][0]
            
            assert log_entry['event'] == 'api_request'
            assert log_entry['method'] == 'GET'
            assert log_entry['path'] == '/api/v1/orders'
            assert log_entry['status'] == 200
            assert 'user_id' in log_entry
            assert 'ip_address' in log_entry
            assert 'request_id' in log_entry
    
    @async_test
    async def test_failed_authentication_logging(self, test_client):
        """Test logging of failed authentication attempts"""
        with patch('src.audit.logger.log') as mock_log:
            response = await test_client.get(
                '/api/v1/orders',
                headers={'Authorization': 'Bearer invalid_token'}
            )
            
            assert response.status == 401
            
            # Verify security event was logged
            mock_log.assert_called()
            log_entry = mock_log.call_args[0][0]
            
            assert log_entry['event'] == 'auth_failure'
            assert log_entry['reason'] == 'invalid_token'
            assert 'ip_address' in log_entry
            assert 'user_agent' in log_entry
    
    @async_test
    async def test_suspicious_activity_logging(self, test_client):
        """Test logging of suspicious activities"""
        with patch('src.audit.logger.log') as mock_log:
            # SQL injection attempt
            response = await test_client.get(
                '/api/v1/orders',
                params={'symbol': "BTC'; DROP TABLE orders;--"}
            )
            
            # Verify security alert was logged
            security_logs = [
                call[0][0] for call in mock_log.call_args_list
                if call[0][0].get('event') == 'security_alert'
            ]
            
            assert len(security_logs) > 0
            alert = security_logs[0]
            assert alert['threat_type'] == 'sql_injection'
            assert 'payload' in alert
            assert 'ip_address' in alert


# Helper functions

def create_api_headers(api_key: str, api_secret: str, method: str, path: str, body: str = '') -> Dict[str, str]:
    """Create API request headers with signature"""
    timestamp = str(int(time.time()))
    message = f"{timestamp}{method}{path}{body}"
    signature = hmac.new(
        api_secret.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return {
        'X-API-Key': api_key,
        'X-API-Signature': signature,
        'X-API-Timestamp': timestamp
    }


async def simulate_dos_attack(test_client, endpoint: str, requests_per_second: int = 1000):
    """Simulate DoS attack for testing"""
    start_time = time.time()
    responses = []
    
    async def send_request():
        try:
            response = await test_client.get(endpoint)
            return response.status
        except:
            return 'error'
    
    # Send many concurrent requests
    tasks = []
    for _ in range(requests_per_second):
        tasks.append(asyncio.create_task(send_request()))
    
    results = await asyncio.gather(*tasks)
    duration = time.time() - start_time
    
    return {
        'total_requests': len(results),
        'successful': results.count(200),
        'rate_limited': results.count(429),
        'errors': results.count('error'),
        'duration': duration,
        'rps': len(results) / duration
    }