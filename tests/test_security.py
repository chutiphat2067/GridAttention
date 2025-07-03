"""
Security Tests for GridAttention Trading System
Tests authentication, authorization, input validation, and security measures
"""

import asyncio
import pytest
import hmac
import hashlib
import secrets
from unittest.mock import Mock, patch
import json
from datetime import datetime, timedelta


class TestAPISecurity:
    """Test API security measures"""
    
    @pytest.fixture
    def api_client(self):
        """Create test API client"""
        from api_security import SecureAPIClient
        return SecureAPIClient()
    
    async def test_api_key_encryption(self, api_client):
        """Test API keys are properly encrypted"""
        # API keys should never be stored in plain text
        api_key = "test_api_key_12345"
        api_secret = "test_secret_67890"
        
        # Store credentials
        await api_client.store_credentials(api_key, api_secret)
        
        # Verify they're encrypted
        stored_data = await api_client.get_stored_credentials()
        assert stored_data['api_key'] != api_key
        assert stored_data['api_secret'] != api_secret
        
        # Verify can decrypt correctly
        decrypted = await api_client.get_credentials()
        assert decrypted['api_key'] == api_key
        assert decrypted['api_secret'] == api_secret
    
    async def test_request_signing(self, api_client):
        """Test request signature generation"""
        # All requests should be signed
        request_data = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 0.01,
            'timestamp': int(datetime.now().timestamp() * 1000)
        }
        
        signature = await api_client.sign_request(request_data)
        
        # Signature should be deterministic
        signature2 = await api_client.sign_request(request_data)
        assert signature == signature2
        
        # Different data should produce different signature
        request_data['amount'] = 0.02
        signature3 = await api_client.sign_request(request_data)
        assert signature != signature3
    
    async def test_rate_limiting(self, api_client):
        """Test rate limiting protection"""
        # Should enforce rate limits
        rate_limit = 10  # 10 requests per second
        
        start_time = asyncio.get_event_loop().time()
        request_times = []
        
        for i in range(15):
            allowed = await api_client.check_rate_limit()
            if allowed:
                request_times.append(asyncio.get_event_loop().time())
                await api_client.record_request()
        
        # Should not exceed rate limit
        assert len(request_times) <= rate_limit
        
        # Wait and try again
        await asyncio.sleep(1)
        allowed = await api_client.check_rate_limit()
        assert allowed


class TestInputValidation:
    """Test input validation and sanitization"""
    
    async def test_order_validation(self):
        """Test order parameter validation"""
        from execution_engine import ExecutionEngine
        
        engine = ExecutionEngine({})
        
        # Valid order
        valid_order = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 0.01,
            'price': 50000
        }
        
        is_valid = await engine.validate_order(valid_order)
        assert is_valid
        
        # Invalid orders
        invalid_orders = [
            # SQL injection attempt
            {'symbol': "BTC/USDT'; DROP TABLE orders;--", 'side': 'buy', 'amount': 0.01},
            # XSS attempt
            {'symbol': '<script>alert("xss")</script>', 'side': 'buy', 'amount': 0.01},
            # Invalid amount
            {'symbol': 'BTC/USDT', 'side': 'buy', 'amount': -0.01},
            # Invalid side
            {'symbol': 'BTC/USDT', 'side': 'hack', 'amount': 0.01},
            # Missing required fields
            {'symbol': 'BTC/USDT', 'amount': 0.01},
        ]
        
        for invalid_order in invalid_orders:
            is_valid = await engine.validate_order(invalid_order)
            assert not is_valid
    
    async def test_config_validation(self):
        """Test configuration file validation"""
        from config_validator import validate_config
        
        # Valid config
        valid_config = {
            'risk_management': {
                'max_position_size': 0.05,
                'max_daily_loss': 0.02
            }
        }
        
        is_valid = validate_config(valid_config)
        assert is_valid
        
        # Attempt to inject malicious values
        malicious_configs = [
            # Command injection
            {'risk_management': {'max_position_size': '$(rm -rf /)'}},
            # Path traversal
            {'checkpoint_dir': '../../../etc/passwd'},
            # Excessive values
            {'risk_management': {'max_position_size': 10.0}},  # 1000% position
        ]
        
        for config in malicious_configs:
            is_valid = validate_config(config)
            assert not is_valid


class TestAuthenticationAuthorization:
    """Test authentication and authorization"""
    
    async def test_session_management(self):
        """Test secure session handling"""
        from auth_manager import AuthManager
        
        auth = AuthManager()
        
        # Create session
        user_id = "test_user_123"
        session_token = await auth.create_session(user_id)
        
        # Token should be cryptographically secure
        assert len(session_token) >= 32
        assert session_token != user_id
        
        # Verify session
        verified_user = await auth.verify_session(session_token)
        assert verified_user == user_id
        
        # Session expiry
        await auth.set_session_expiry(session_token, timedelta(seconds=1))
        await asyncio.sleep(2)
        
        expired_user = await auth.verify_session(session_token)
        assert expired_user is None
    
    async def test_permission_levels(self):
        """Test authorization levels"""
        from auth_manager import AuthManager, PermissionLevel
        
        auth = AuthManager()
        
        # Different permission levels
        users = {
            'admin': PermissionLevel.ADMIN,
            'trader': PermissionLevel.TRADER,
            'viewer': PermissionLevel.VIEWER
        }
        
        for user_id, level in users.items():
            await auth.set_permission_level(user_id, level)
        
        # Test permissions
        assert await auth.can_execute_trades('admin')
        assert await auth.can_execute_trades('trader')
        assert not await auth.can_execute_trades('viewer')
        
        assert await auth.can_modify_config('admin')
        assert not await auth.can_modify_config('trader')
        assert not await auth.can_modify_config('viewer')


class TestDataProtection:
    """Test sensitive data protection"""
    
    async def test_log_sanitization(self):
        """Test that sensitive data is not logged"""
        from logger_security import SecureLogger
        
        logger = SecureLogger()
        
        # Log message with sensitive data
        message = {
            'action': 'order_placed',
            'api_key': 'secret_key_12345',
            'api_secret': 'super_secret_67890',
            'order_id': 'ORD123'
        }
        
        sanitized = await logger.sanitize_log(message)
        
        # Sensitive data should be masked
        assert 'secret_key_12345' not in str(sanitized)
        assert 'super_secret_67890' not in str(sanitized)
        assert sanitized['api_key'] == '***REDACTED***'
        assert sanitized['api_secret'] == '***REDACTED***'
        assert sanitized['order_id'] == 'ORD123'  # Non-sensitive data preserved
    
    async def test_memory_clearing(self):
        """Test secure memory clearing"""
        from security_utils import SecureString
        
        # Create secure string
        sensitive_data = SecureString("my_api_secret_key")
        
        # Use the data
        assert sensitive_data.get() == "my_api_secret_key"
        
        # Clear from memory
        sensitive_data.clear()
        
        # Should be cleared
        assert sensitive_data.get() != "my_api_secret_key"
        assert len(sensitive_data.get()) == 0


class TestNetworkSecurity:
    """Test network security measures"""
    
    async def test_ssl_verification(self):
        """Test SSL certificate verification"""
        from network_client import SecureNetworkClient
        
        client = SecureNetworkClient()
        
        # Should verify SSL certificates
        with pytest.raises(Exception) as exc_info:
            await client.connect("https://self-signed.badssl.com/")
        
        assert "certificate" in str(exc_info.value).lower()
        
        # Valid SSL should work
        response = await client.connect("https://example.com/")
        assert response is not None
    
    async def test_connection_encryption(self):
        """Test all connections are encrypted"""
        from network_monitor import NetworkMonitor
        
        monitor = NetworkMonitor()
        
        # Monitor outgoing connections
        connections = await monitor.get_active_connections()
        
        for conn in connections:
            # All trading connections should use encryption
            if conn['port'] in [80, 8080]:  # HTTP ports
                assert conn['service'] not in ['trading', 'api', 'websocket']
            
            # Trading connections should use secure ports
            if conn['service'] == 'trading':
                assert conn['port'] in [443, 8443]  # HTTPS ports
                assert conn['encrypted'] is True


# Test runner
async def run_security_tests():
    """Run all security tests"""
    print("🔒 Running Security Tests...")
    
    test_classes = [
        TestAPISecurity,
        TestInputValidation,
        TestAuthenticationAuthorization,
        TestDataProtection,
        TestNetworkSecurity
    ]
    
    results = []
    
    for test_class in test_classes:
        print(f"\nTesting {test_class.__name__}...")
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_')]
        
        for method_name in test_methods:
            try:
                method = getattr(test_instance, method_name)
                if asyncio.iscoroutinefunction(method):
                    # Handle fixtures if needed
                    if 'api_client' in method_name:
                        api_client = test_instance.api_client()
                        await method(api_client)
                    else:
                        await method()
                else:
                    method()
                    
                print(f"  ✅ {method_name}")
                results.append((method_name, True))
                
            except Exception as e:
                print(f"  ❌ {method_name}: {e}")
                results.append((method_name, False))
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n📊 Security Test Results: {passed}/{total} passed")
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_security_tests())
    exit(0 if success else 1)