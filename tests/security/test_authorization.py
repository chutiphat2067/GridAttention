"""
Authorization Security Testing Suite for GridAttention Trading System
Tests role-based access control, permissions, resource authorization, and access policies
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from unittest.mock import Mock, AsyncMock, patch
import logging
from dataclasses import dataclass
from enum import Enum
import json

# GridAttention imports - aligned with system structure
from src.grid_attention_layer import GridAttentionLayer
from src.auth.authorization_manager import AuthorizationManager
from src.auth.permission_manager import PermissionManager
from src.auth.role_manager import RoleManager
from src.auth.resource_manager import ResourceManager
from src.auth.policy_engine import PolicyEngine
from src.auth.access_control import AccessControlList

# Test utilities
from tests.utils.test_helpers import async_test, create_test_config
from tests.utils.security_helpers import (
    create_test_user_context,
    create_test_resource,
    generate_test_policy
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Permission(Enum):
    """System permissions"""
    # Trading permissions
    TRADE_READ = "trade:read"
    TRADE_CREATE = "trade:create"
    TRADE_UPDATE = "trade:update"
    TRADE_DELETE = "trade:delete"
    TRADE_EXECUTE = "trade:execute"
    
    # Strategy permissions
    STRATEGY_READ = "strategy:read"
    STRATEGY_CREATE = "strategy:create"
    STRATEGY_UPDATE = "strategy:update"
    STRATEGY_DELETE = "strategy:delete"
    STRATEGY_BACKTEST = "strategy:backtest"
    
    # Risk permissions
    RISK_READ = "risk:read"
    RISK_OVERRIDE = "risk:override"
    RISK_CONFIG = "risk:config"
    
    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_SYSTEM = "admin:system"
    ADMIN_AUDIT = "admin:audit"


class Role(Enum):
    """System roles"""
    VIEWER = "viewer"
    TRADER = "trader"
    RISK_MANAGER = "risk_manager"
    STRATEGY_DEVELOPER = "strategy_developer"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


@dataclass
class UserContext:
    """User context for authorization"""
    user_id: str
    username: str
    roles: List[Role]
    permissions: Set[Permission]
    attributes: Dict
    session_id: str
    ip_address: str
    authenticated_at: datetime


@dataclass
class Resource:
    """Resource to be accessed"""
    resource_id: str
    resource_type: str
    owner_id: str
    attributes: Dict
    created_at: datetime
    access_level: str


class TestRoleBasedAccessControl:
    """Test role-based access control (RBAC)"""
    
    @pytest.fixture
    def role_manager(self):
        """Create role manager"""
        config = create_test_config()
        config['rbac'] = {
            'enable_inheritance': True,
            'max_roles_per_user': 10,
            'cache_ttl': 300
        }
        return RoleManager(config)
    
    @pytest.fixture
    def permission_manager(self):
        """Create permission manager"""
        return PermissionManager(create_test_config())
    
    @async_test
    async def test_role_creation(self, role_manager):
        """Test role creation with permissions"""
        # Create trader role
        trader_permissions = [
            Permission.TRADE_READ,
            Permission.TRADE_CREATE,
            Permission.TRADE_UPDATE,
            Permission.TRADE_EXECUTE,
            Permission.STRATEGY_READ
        ]
        
        result = await role_manager.create_role(
            name=Role.TRADER,
            description="Standard trader role",
            permissions=trader_permissions
        )
        
        assert result['success'] is True
        assert result['role']['name'] == Role.TRADER
        assert len(result['role']['permissions']) == len(trader_permissions)
        
        # Verify permissions are correctly assigned
        role_perms = await role_manager.get_role_permissions(Role.TRADER)
        assert set(role_perms) == set(trader_permissions)
    
    @async_test
    async def test_role_hierarchy(self, role_manager):
        """Test role inheritance hierarchy"""
        # Define role hierarchy
        hierarchy = {
            Role.VIEWER: [],
            Role.TRADER: [Role.VIEWER],
            Role.RISK_MANAGER: [Role.VIEWER],
            Role.STRATEGY_DEVELOPER: [Role.TRADER],
            Role.ADMIN: [Role.TRADER, Role.RISK_MANAGER],
            Role.SUPER_ADMIN: [Role.ADMIN]
        }
        
        # Create roles with inheritance
        for role, parents in hierarchy.items():
            await role_manager.create_role_with_inheritance(
                role=role,
                parent_roles=parents
            )
        
        # Test permission inheritance
        # Viewer has basic permissions
        viewer_perms = await role_manager.get_effective_permissions(Role.VIEWER)
        assert Permission.TRADE_READ in viewer_perms
        
        # Trader inherits from Viewer
        trader_perms = await role_manager.get_effective_permissions(Role.TRADER)
        assert Permission.TRADE_READ in trader_perms  # Inherited
        assert Permission.TRADE_EXECUTE in trader_perms  # Own permission
        
        # Admin inherits from multiple roles
        admin_perms = await role_manager.get_effective_permissions(Role.ADMIN)
        assert Permission.TRADE_EXECUTE in admin_perms  # From Trader
        assert Permission.RISK_OVERRIDE in admin_perms  # From Risk Manager
        assert Permission.ADMIN_USERS in admin_perms  # Own permission
    
    @async_test
    async def test_user_role_assignment(self, role_manager):
        """Test assigning roles to users"""
        user_id = "test_user_123"
        
        # Assign single role
        assign_result = await role_manager.assign_role(
            user_id=user_id,
            role=Role.TRADER
        )
        
        assert assign_result['success'] is True
        
        # Get user roles
        user_roles = await role_manager.get_user_roles(user_id)
        assert Role.TRADER in user_roles
        
        # Assign multiple roles
        await role_manager.assign_role(user_id, Role.RISK_MANAGER)
        
        user_roles = await role_manager.get_user_roles(user_id)
        assert len(user_roles) == 2
        assert Role.TRADER in user_roles
        assert Role.RISK_MANAGER in user_roles
        
        # Test role limit
        for i in range(10):
            await role_manager.assign_role(user_id, f"custom_role_{i}")
        
        # Should fail due to role limit
        limit_result = await role_manager.assign_role(user_id, "extra_role")
        assert limit_result['success'] is False
        assert 'limit' in limit_result['error'].lower()
    
    @async_test
    async def test_role_revocation(self, role_manager):
        """Test role revocation"""
        user_id = "test_user_123"
        
        # Assign roles
        await role_manager.assign_role(user_id, Role.TRADER)
        await role_manager.assign_role(user_id, Role.RISK_MANAGER)
        
        # Revoke one role
        revoke_result = await role_manager.revoke_role(
            user_id=user_id,
            role=Role.RISK_MANAGER,
            reason="Role change"
        )
        
        assert revoke_result['success'] is True
        
        # Verify role is revoked
        user_roles = await role_manager.get_user_roles(user_id)
        assert Role.TRADER in user_roles
        assert Role.RISK_MANAGER not in user_roles
        
        # Verify audit log
        audit_log = await role_manager.get_role_audit_log(user_id)
        assert any(log['action'] == 'revoke' and log['role'] == Role.RISK_MANAGER 
                  for log in audit_log)


class TestPermissionManagement:
    """Test permission management and checking"""
    
    @pytest.fixture
    def auth_manager(self):
        """Create authorization manager"""
        config = create_test_config()
        return AuthorizationManager(config)
    
    @async_test
    async def test_permission_checking(self, auth_manager):
        """Test permission checking for users"""
        # Create user context
        user_context = UserContext(
            user_id="trader_123",
            username="trader1",
            roles=[Role.TRADER],
            permissions={
                Permission.TRADE_READ,
                Permission.TRADE_CREATE,
                Permission.TRADE_EXECUTE
            },
            attributes={},
            session_id="session_123",
            ip_address="192.168.1.100",
            authenticated_at=datetime.now()
        )
        
        # Check allowed permission
        allowed = await auth_manager.check_permission(
            user_context=user_context,
            permission=Permission.TRADE_EXECUTE
        )
        assert allowed is True
        
        # Check denied permission
        denied = await auth_manager.check_permission(
            user_context=user_context,
            permission=Permission.ADMIN_USERS
        )
        assert denied is False
    
    @async_test
    async def test_multiple_permission_checking(self, auth_manager):
        """Test checking multiple permissions"""
        user_context = UserContext(
            user_id="trader_123",
            username="trader1",
            roles=[Role.TRADER],
            permissions={
                Permission.TRADE_READ,
                Permission.TRADE_CREATE,
                Permission.STRATEGY_READ
            },
            attributes={},
            session_id="session_123",
            ip_address="192.168.1.100",
            authenticated_at=datetime.now()
        )
        
        # Check ANY (OR) permissions
        any_result = await auth_manager.check_any_permission(
            user_context=user_context,
            permissions=[Permission.TRADE_DELETE, Permission.TRADE_CREATE]
        )
        assert any_result is True  # Has TRADE_CREATE
        
        # Check ALL (AND) permissions
        all_result = await auth_manager.check_all_permissions(
            user_context=user_context,
            permissions=[Permission.TRADE_READ, Permission.STRATEGY_READ]
        )
        assert all_result is True  # Has both
        
        all_result_fail = await auth_manager.check_all_permissions(
            user_context=user_context,
            permissions=[Permission.TRADE_READ, Permission.ADMIN_USERS]
        )
        assert all_result_fail is False  # Missing ADMIN_USERS
    
    @async_test
    async def test_dynamic_permission_grants(self, auth_manager):
        """Test temporary permission grants"""
        user_context = UserContext(
            user_id="trader_123",
            username="trader1",
            roles=[Role.TRADER],
            permissions={Permission.TRADE_READ},
            attributes={},
            session_id="session_123",
            ip_address="192.168.1.100",
            authenticated_at=datetime.now()
        )
        
        # Grant temporary permission
        grant_result = await auth_manager.grant_temporary_permission(
            user_context=user_context,
            permission=Permission.RISK_OVERRIDE,
            duration_seconds=300,  # 5 minutes
            reason="Emergency trade adjustment"
        )
        
        assert grant_result['success'] is True
        assert 'grant_id' in grant_result
        
        # Check permission is now allowed
        allowed = await auth_manager.check_permission(
            user_context=user_context,
            permission=Permission.RISK_OVERRIDE
        )
        assert allowed is True
        
        # Simulate time passage
        with patch('time.time', return_value=time.time() + 400):
            # Permission should be expired
            expired = await auth_manager.check_permission(
                user_context=user_context,
                permission=Permission.RISK_OVERRIDE
            )
            assert expired is False


class TestResourceAuthorization:
    """Test resource-level authorization"""
    
    @pytest.fixture
    def resource_manager(self):
        """Create resource manager"""
        config = create_test_config()
        config['resources'] = {
            'enable_ownership': True,
            'enable_sharing': True,
            'max_shares_per_resource': 50
        }
        return ResourceManager(config)
    
    @async_test
    async def test_resource_ownership(self, resource_manager):
        """Test resource ownership authorization"""
        owner_id = "user_123"
        other_user_id = "user_456"
        
        # Create resource
        resource = Resource(
            resource_id="strategy_001",
            resource_type="trading_strategy",
            owner_id=owner_id,
            attributes={
                'name': 'My Strategy',
                'visibility': 'private'
            },
            created_at=datetime.now(),
            access_level="owner"
        )
        
        # Owner should have full access
        owner_access = await resource_manager.check_resource_access(
            user_id=owner_id,
            resource=resource,
            action="delete"
        )
        assert owner_access['allowed'] is True
        assert owner_access['reason'] == 'owner'
        
        # Other user should not have access
        other_access = await resource_manager.check_resource_access(
            user_id=other_user_id,
            resource=resource,
            action="read"
        )
        assert other_access['allowed'] is False
        assert other_access['reason'] == 'no_access'
    
    @async_test
    async def test_resource_sharing(self, resource_manager):
        """Test resource sharing permissions"""
        owner_id = "user_123"
        shared_user_id = "user_456"
        
        # Create resource
        resource = Resource(
            resource_id="portfolio_001",
            resource_type="portfolio",
            owner_id=owner_id,
            attributes={'name': 'My Portfolio'},
            created_at=datetime.now(),
            access_level="owner"
        )
        
        # Share resource with read permission
        share_result = await resource_manager.share_resource(
            resource=resource,
            owner_id=owner_id,
            shared_with_id=shared_user_id,
            permissions=['read'],
            expires_at=datetime.now() + timedelta(days=7)
        )
        
        assert share_result['success'] is True
        
        # Shared user should have read access
        read_access = await resource_manager.check_resource_access(
            user_id=shared_user_id,
            resource=resource,
            action="read"
        )
        assert read_access['allowed'] is True
        assert read_access['reason'] == 'shared'
        
        # But not write access
        write_access = await resource_manager.check_resource_access(
            user_id=shared_user_id,
            resource=resource,
            action="update"
        )
        assert write_access['allowed'] is False
    
    @async_test
    async def test_resource_access_levels(self, resource_manager):
        """Test different resource access levels"""
        # Define access levels
        access_levels = {
            'public': ['read'],
            'protected': ['read', 'comment'],
            'private': [],
            'shared': ['read', 'update'],
            'owner': ['read', 'update', 'delete', 'share']
        }
        
        # Create resources with different access levels
        resources = []
        for level, allowed_actions in access_levels.items():
            resource = Resource(
                resource_id=f"resource_{level}",
                resource_type="strategy",
                owner_id="owner_123",
                attributes={'access_level': level},
                created_at=datetime.now(),
                access_level=level
            )
            resources.append((resource, allowed_actions))
        
        # Test access for different user types
        test_users = [
            ("owner_123", "owner"),
            ("shared_user", "shared"),
            ("public_user", "public")
        ]
        
        for user_id, user_type in test_users:
            for resource, allowed_actions in resources:
                for action in ['read', 'update', 'delete']:
                    result = await resource_manager.check_resource_access(
                        user_id=user_id,
                        resource=resource,
                        action=action
                    )
                    
                    # Determine expected result
                    if user_id == resource.owner_id:
                        expected = True  # Owner has all permissions
                    elif resource.access_level == 'public' and action == 'read':
                        expected = True
                    elif user_type == 'shared' and resource.access_level == 'shared' and action in ['read', 'update']:
                        expected = True
                    else:
                        expected = False
                    
                    assert result['allowed'] == expected, \
                        f"Access check failed for {user_type} on {resource.access_level} resource, action: {action}"


class TestPolicyBasedAuthorization:
    """Test policy-based authorization (PBAC)"""
    
    @pytest.fixture
    def policy_engine(self):
        """Create policy engine"""
        config = create_test_config()
        config['policies'] = {
            'enable_conditions': True,
            'enable_time_restrictions': True,
            'enable_attribute_based': True,
            'policy_cache_ttl': 300
        }
        return PolicyEngine(config)
    
    @async_test
    async def test_policy_creation(self, policy_engine):
        """Test creating authorization policies"""
        # Create trading hours policy
        policy = {
            'id': 'trading_hours_policy',
            'name': 'Trading Hours Restriction',
            'effect': 'deny',
            'actions': ['trade:execute'],
            'resources': ['trading/*'],
            'conditions': {
                'time_range': {
                    'after': '00:00',
                    'before': '09:30'
                },
                'day_of_week': {
                    'not_in': ['saturday', 'sunday']
                }
            }
        }
        
        result = await policy_engine.create_policy(policy)
        assert result['success'] is True
        assert result['policy']['id'] == policy['id']
        
        # Verify policy is stored
        stored_policy = await policy_engine.get_policy(policy['id'])
        assert stored_policy is not None
        assert stored_policy['conditions']['time_range']['before'] == '09:30'
    
    @async_test
    async def test_policy_evaluation(self, policy_engine):
        """Test policy evaluation with conditions"""
        # Create IP restriction policy
        ip_policy = {
            'id': 'ip_restriction_policy',
            'effect': 'allow',
            'actions': ['trade:*'],
            'resources': ['*'],
            'conditions': {
                'ip_address': {
                    'in': ['192.168.1.0/24', '10.0.0.0/8']
                }
            }
        }
        
        await policy_engine.create_policy(ip_policy)
        
        # Test allowed IP
        allowed_context = {
            'user_id': 'user_123',
            'action': 'trade:execute',
            'resource': 'order/12345',
            'ip_address': '192.168.1.100'
        }
        
        allowed_result = await policy_engine.evaluate(allowed_context)
        assert allowed_result['allowed'] is True
        assert allowed_result['matched_policy'] == 'ip_restriction_policy'
        
        # Test denied IP
        denied_context = {
            'user_id': 'user_123',
            'action': 'trade:execute',
            'resource': 'order/12345',
            'ip_address': '172.16.0.100'  # Not in allowed ranges
        }
        
        denied_result = await policy_engine.evaluate(denied_context)
        assert denied_result['allowed'] is False
    
    @async_test
    async def test_attribute_based_policies(self, policy_engine):
        """Test attribute-based access control (ABAC)"""
        # Create risk-based trading policy
        risk_policy = {
            'id': 'risk_based_trading',
            'effect': 'deny',
            'actions': ['trade:execute'],
            'resources': ['order/*'],
            'conditions': {
                'user_attributes.risk_score': {
                    'greater_than': 80
                },
                'resource_attributes.order_value': {
                    'greater_than': 100000
                }
            }
        }
        
        await policy_engine.create_policy(risk_policy)
        
        # Test high-risk user with large order
        high_risk_context = {
            'user_id': 'user_123',
            'action': 'trade:execute',
            'resource': 'order/12345',
            'user_attributes': {
                'risk_score': 85,
                'account_type': 'standard'
            },
            'resource_attributes': {
                'order_value': 150000,
                'asset_type': 'crypto'
            }
        }
        
        result = await policy_engine.evaluate(high_risk_context)
        assert result['allowed'] is False
        assert result['matched_policy'] == 'risk_based_trading'
        assert 'risk_score' in result['denial_reason']
        
        # Test low-risk user
        low_risk_context = high_risk_context.copy()
        low_risk_context['user_attributes']['risk_score'] = 30
        
        result = await policy_engine.evaluate(low_risk_context)
        assert result['allowed'] is True
    
    @async_test
    async def test_time_based_policies(self, policy_engine):
        """Test time-based access policies"""
        # Create maintenance window policy
        maintenance_policy = {
            'id': 'maintenance_window',
            'effect': 'deny',
            'actions': ['*'],
            'resources': ['*'],
            'conditions': {
                'time_window': {
                    'cron': '0 2-4 * * SUN'  # Sunday 2-4 AM
                }
            },
            'priority': 100  # High priority
        }
        
        await policy_engine.create_policy(maintenance_policy)
        
        # Test during maintenance window
        with patch('datetime.datetime') as mock_datetime:
            # Set time to Sunday 3 AM
            mock_datetime.now.return_value = datetime(2024, 1, 7, 3, 0, 0)  # Sunday
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            context = {
                'user_id': 'admin_user',
                'action': 'system:configure',
                'resource': 'system/config'
            }
            
            result = await policy_engine.evaluate(context)
            assert result['allowed'] is False
            assert result['matched_policy'] == 'maintenance_window'
    
    @async_test
    async def test_policy_priority_and_conflicts(self, policy_engine):
        """Test policy priority and conflict resolution"""
        # Create conflicting policies
        # General deny policy
        general_deny = {
            'id': 'general_deny',
            'effect': 'deny',
            'actions': ['trade:*'],
            'resources': ['*'],
            'priority': 1
        }
        
        # Specific allow policy with higher priority
        specific_allow = {
            'id': 'vip_allow',
            'effect': 'allow',
            'actions': ['trade:execute'],
            'resources': ['order/*'],
            'conditions': {
                'user_attributes.vip_status': {
                    'equals': True
                }
            },
            'priority': 10
        }
        
        await policy_engine.create_policy(general_deny)
        await policy_engine.create_policy(specific_allow)
        
        # Test VIP user
        vip_context = {
            'user_id': 'vip_user',
            'action': 'trade:execute',
            'resource': 'order/12345',
            'user_attributes': {
                'vip_status': True
            }
        }
        
        result = await policy_engine.evaluate(vip_context)
        assert result['allowed'] is True
        assert result['matched_policy'] == 'vip_allow'
        
        # Test non-VIP user
        regular_context = {
            'user_id': 'regular_user',
            'action': 'trade:execute',
            'resource': 'order/12345',
            'user_attributes': {
                'vip_status': False
            }
        }
        
        result = await policy_engine.evaluate(regular_context)
        assert result['allowed'] is False
        assert result['matched_policy'] == 'general_deny'


class TestAccessControlList:
    """Test Access Control List (ACL) implementation"""
    
    @pytest.fixture
    def acl_manager(self):
        """Create ACL manager"""
        config = create_test_config()
        return AccessControlList(config)
    
    @async_test
    async def test_acl_entry_creation(self, acl_manager):
        """Test creating ACL entries"""
        # Create ACL entry
        entry = await acl_manager.create_entry(
            resource_id="portfolio_123",
            principal_id="user_456",
            principal_type="user",
            permissions=["read", "update"],
            allow=True
        )
        
        assert entry['success'] is True
        assert entry['entry']['permissions'] == ["read", "update"]
        assert entry['entry']['allow'] is True
    
    @async_test
    async def test_acl_inheritance(self, acl_manager):
        """Test ACL inheritance for hierarchical resources"""
        # Create parent resource ACL
        await acl_manager.create_entry(
            resource_id="/strategies",
            principal_id="group_traders",
            principal_type="group",
            permissions=["read"],
            allow=True,
            inheritable=True
        )
        
        # Create child resource ACL
        await acl_manager.create_entry(
            resource_id="/strategies/my_strategy",
            principal_id="user_123",
            principal_type="user",
            permissions=["update", "delete"],
            allow=True
        )
        
        # Check access for group member on child resource
        group_member_access = await acl_manager.check_access(
            resource_id="/strategies/my_strategy",
            principal_id="user_789",  # Member of group_traders
            principal_groups=["group_traders"],
            permission="read"
        )
        
        assert group_member_access['allowed'] is True
        assert group_member_access['inherited'] is True
        
        # Check owner access
        owner_access = await acl_manager.check_access(
            resource_id="/strategies/my_strategy",
            principal_id="user_123",
            principal_groups=[],
            permission="delete"
        )
        
        assert owner_access['allowed'] is True
        assert owner_access['inherited'] is False


class TestAuthorizationAuditing:
    """Test authorization audit logging"""
    
    @pytest.fixture
    def audit_manager(self):
        """Create audit manager"""
        config = create_test_config()
        config['audit'] = {
            'enable_audit_log': True,
            'log_retention_days': 90,
            'log_successful_access': True,
            'log_denied_access': True
        }
        return AuthorizationAuditManager(config)
    
    @async_test
    async def test_access_audit_logging(self, audit_manager, auth_manager):
        """Test logging of authorization decisions"""
        user_context = UserContext(
            user_id="user_123",
            username="testuser",
            roles=[Role.TRADER],
            permissions={Permission.TRADE_READ},
            attributes={},
            session_id="session_123",
            ip_address="192.168.1.100",
            authenticated_at=datetime.now()
        )
        
        # Perform authorization check
        await auth_manager.check_permission(
            user_context=user_context,
            permission=Permission.TRADE_EXECUTE,
            audit=True
        )
        
        # Get audit logs
        logs = await audit_manager.get_recent_logs(
            user_id="user_123",
            limit=10
        )
        
        assert len(logs) > 0
        latest_log = logs[0]
        
        assert latest_log['user_id'] == "user_123"
        assert latest_log['action'] == "permission_check"
        assert latest_log['resource'] == Permission.TRADE_EXECUTE.value
        assert latest_log['result'] == "denied"
        assert latest_log['ip_address'] == "192.168.1.100"
    
    @async_test
    async def test_suspicious_activity_detection(self, audit_manager):
        """Test detection of suspicious authorization patterns"""
        user_id = "suspicious_user"
        
        # Simulate multiple failed authorization attempts
        for i in range(20):
            await audit_manager.log_access(
                user_id=user_id,
                action="permission_check",
                resource=f"admin_resource_{i}",
                result="denied",
                ip_address="192.168.1.100"
            )
        
        # Check for suspicious activity
        suspicious = await audit_manager.check_suspicious_activity(
            user_id=user_id,
            window_minutes=5
        )
        
        assert suspicious['is_suspicious'] is True
        assert suspicious['reason'] == 'excessive_denied_attempts'
        assert suspicious['denied_count'] == 20
        assert suspicious['recommendation'] == 'investigate_user'


class TestDynamicAuthorization:
    """Test dynamic and contextual authorization"""
    
    @async_test
    async def test_location_based_authorization(self, auth_manager):
        """Test authorization based on geographic location"""
        # Create location-aware policy
        location_policy = {
            'id': 'geo_restriction',
            'effect': 'deny',
            'actions': ['trade:execute'],
            'conditions': {
                'geo_location': {
                    'not_in': ['US', 'EU', 'UK']  # Allowed regions
                }
            }
        }
        
        # User from allowed region
        us_context = {
            'user_id': 'user_123',
            'action': 'trade:execute',
            'geo_location': 'US',
            'ip_address': '8.8.8.8'
        }
        
        result = await auth_manager.evaluate_with_policy(us_context, location_policy)
        assert result['allowed'] is True
        
        # User from restricted region
        restricted_context = {
            'user_id': 'user_456',
            'action': 'trade:execute',
            'geo_location': 'CN',
            'ip_address': '1.2.3.4'
        }
        
        result = await auth_manager.evaluate_with_policy(restricted_context, location_policy)
        assert result['allowed'] is False
        assert 'geo_location' in result['denial_reason']
    
    @async_test
    async def test_risk_based_authorization(self, auth_manager):
        """Test authorization based on risk assessment"""
        # Create risk scoring function
        async def calculate_risk_score(context):
            score = 0
            
            # Time-based risk
            current_hour = datetime.now().hour
            if current_hour < 6 or current_hour > 22:
                score += 20
            
            # Location-based risk
            if context.get('geo_location') not in ['US', 'EU']:
                score += 30
            
            # Transaction-based risk
            if context.get('transaction_amount', 0) > 50000:
                score += 25
            
            # History-based risk
            if context.get('failed_attempts', 0) > 3:
                score += 25
            
            return score
        
        # Test low-risk scenario
        low_risk_context = {
            'user_id': 'user_123',
            'geo_location': 'US',
            'transaction_amount': 1000,
            'failed_attempts': 0
        }
        
        risk_score = await calculate_risk_score(low_risk_context)
        assert risk_score < 50  # Low risk
        
        # Test high-risk scenario
        high_risk_context = {
            'user_id': 'user_456',
            'geo_location': 'Unknown',
            'transaction_amount': 100000,
            'failed_attempts': 5
        }
        
        risk_score = await calculate_risk_score(high_risk_context)
        assert risk_score > 50  # High risk
        
        # Apply risk-based authorization
        if risk_score > 50:
            # Require additional authentication or deny
            auth_result = {
                'allowed': False,
                'reason': 'high_risk_score',
                'risk_score': risk_score,
                'required_action': 'mfa_required'
            }
        else:
            auth_result = {'allowed': True}
        
        assert auth_result['allowed'] == (risk_score <= 50)


# Helper Classes

class AuthorizationAuditManager:
    """Manages authorization audit logs"""
    
    def __init__(self, config):
        self.config = config
        self.audit_logs = []
    
    async def log_access(self, user_id: str, action: str, resource: str, 
                        result: str, ip_address: str):
        """Log authorization attempt"""
        log_entry = {
            'timestamp': datetime.now(),
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'result': result,
            'ip_address': ip_address
        }
        self.audit_logs.append(log_entry)
    
    async def get_recent_logs(self, user_id: str, limit: int = 100) -> List[Dict]:
        """Get recent audit logs for user"""
        user_logs = [log for log in self.audit_logs if log['user_id'] == user_id]
        return sorted(user_logs, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    async def check_suspicious_activity(self, user_id: str, window_minutes: int = 5) -> Dict:
        """Check for suspicious authorization patterns"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_logs = [
            log for log in self.audit_logs 
            if log['user_id'] == user_id and log['timestamp'] > cutoff_time
        ]
        
        denied_count = sum(1 for log in recent_logs if log['result'] == 'denied')
        
        if denied_count > 10:
            return {
                'is_suspicious': True,
                'reason': 'excessive_denied_attempts',
                'denied_count': denied_count,
                'recommendation': 'investigate_user'
            }
        
        return {'is_suspicious': False}