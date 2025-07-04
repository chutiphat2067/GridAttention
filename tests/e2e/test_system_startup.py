"""
System startup end-to-end tests for GridAttention trading system.

Tests the complete system initialization process, dependency management,
configuration loading, health checks, and readiness validation.
"""

import pytest
import asyncio
import os
import sys
import json
import yaml
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import psutil
import aiohttp
import aioredis
import motor.motor_asyncio
from unittest.mock import Mock, patch, AsyncMock
import logging

# Import all system components
from core.system_manager import SystemManager
from core.config_loader import ConfigLoader
from core.dependency_manager import DependencyManager
from core.health_checker import HealthChecker
from core.service_registry import ServiceRegistry
from core.connection_pool import ConnectionPool
from core.cache_manager import CacheManager
from core.database_manager import DatabaseManager


class StartupPhase(Enum):
    """System startup phases"""
    INITIALIZATION = "initialization"
    CONFIGURATION = "configuration"
    DEPENDENCIES = "dependencies"
    CONNECTIONS = "connections"
    SERVICES = "services"
    VALIDATION = "validation"
    WARMUP = "warmup"
    READY = "ready"


class ServiceStatus(Enum):
    """Service status during startup"""
    NOT_STARTED = "not_started"
    STARTING = "starting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    FAILED = "failed"


@dataclass
class StartupConfig:
    """System startup configuration"""
    config_path: str
    environment: str  # development, staging, production
    services_to_start: List[str]
    startup_timeout_seconds: int = 300
    health_check_retries: int = 5
    parallel_startup: bool = True
    fail_fast: bool = True
    enable_warmup: bool = True


@dataclass
class StartupResult:
    """Result of system startup"""
    success: bool
    phase_completed: StartupPhase
    duration: timedelta
    services_started: List[str]
    services_failed: List[str]
    health_checks: Dict[str, bool]
    warnings: List[str]
    errors: List[str]
    system_ready: bool


@dataclass
class ServiceDependency:
    """Service dependency definition"""
    service: str
    depends_on: List[str]
    required: bool = True
    startup_order: int = 0
    health_check_endpoint: Optional[str] = None
    timeout_seconds: int = 60


class TestSystemStartup:
    """Test system startup procedures"""
    
    @pytest.fixture
    async def system_manager(self):
        """Create system manager for startup orchestration"""
        return SystemManager(
            enable_auto_recovery=True,
            max_startup_attempts=3,
            startup_parallelism=5
        )
    
    @pytest.fixture
    async def config_loader(self):
        """Create configuration loader"""
        return ConfigLoader(
            config_sources=['file', 'env', 'remote'],
            validation_enabled=True,
            schema_path='config/schemas'
        )
    
    @pytest.fixture
    async def dependency_manager(self):
        """Create dependency manager"""
        return DependencyManager(
            enable_circular_detection=True,
            resolve_conflicts=True
        )
    
    @pytest.fixture
    async def health_checker(self):
        """Create health checker"""
        return HealthChecker(
            check_interval_seconds=10,
            timeout_seconds=5,
            enable_detailed_checks=True
        )
    
    @pytest.fixture
    def test_config_file(self, tmp_path):
        """Create test configuration file"""
        config_data = {
            'system': {
                'name': 'GridAttention',
                'version': '1.0.0',
                'environment': 'test'
            },
            'services': {
                'market_data': {
                    'enabled': True,
                    'host': 'localhost',
                    'port': 8001,
                    'dependencies': []
                },
                'trading_engine': {
                    'enabled': True,
                    'host': 'localhost',
                    'port': 8002,
                    'dependencies': ['market_data', 'risk_management']
                },
                'risk_management': {
                    'enabled': True,
                    'host': 'localhost',
                    'port': 8003,
                    'dependencies': ['database']
                },
                'database': {
                    'enabled': True,
                    'type': 'postgresql',
                    'host': 'localhost',
                    'port': 5432,
                    'dependencies': []
                },
                'cache': {
                    'enabled': True,
                    'type': 'redis',
                    'host': 'localhost',
                    'port': 6379,
                    'dependencies': []
                },
                'api': {
                    'enabled': True,
                    'host': '0.0.0.0',
                    'port': 8000,
                    'dependencies': ['trading_engine', 'market_data']
                }
            },
            'connections': {
                'exchange': {
                    'type': 'websocket',
                    'urls': ['wss://exchange1.com', 'wss://exchange2.com'],
                    'reconnect': True,
                    'heartbeat_interval': 30
                },
                'database': {
                    'pool_size': 20,
                    'max_overflow': 10,
                    'timeout': 30
                }
            },
            'monitoring': {
                'metrics': {
                    'enabled': True,
                    'port': 9090
                },
                'logging': {
                    'level': 'INFO',
                    'handlers': ['console', 'file']
                }
            }
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        return str(config_file)
    
    @pytest.mark.asyncio
    async def test_basic_system_startup(self, system_manager, config_loader, test_config_file):
        """Test basic system startup sequence"""
        # Create startup configuration
        startup_config = StartupConfig(
            config_path=test_config_file,
            environment='test',
            services_to_start=['market_data', 'risk_management', 'trading_engine'],
            startup_timeout_seconds=60
        )
        
        # Execute startup
        startup_result = await self._execute_startup(
            system_manager=system_manager,
            config_loader=config_loader,
            startup_config=startup_config
        )
        
        # Verify successful startup
        assert startup_result.success == True
        assert startup_result.phase_completed == StartupPhase.READY
        assert startup_result.system_ready == True
        
        # Check all requested services started
        for service in startup_config.services_to_start:
            assert service in startup_result.services_started
        
        # No services should have failed
        assert len(startup_result.services_failed) == 0
        
        # Health checks should pass
        for service, healthy in startup_result.health_checks.items():
            assert healthy == True, f"Service {service} is not healthy"
        
        # Startup should complete within timeout
        assert startup_result.duration.total_seconds() < startup_config.startup_timeout_seconds
    
    async def _execute_startup(
        self,
        system_manager: SystemManager,
        config_loader: ConfigLoader,
        startup_config: StartupConfig
    ) -> StartupResult:
        """Execute system startup sequence"""
        start_time = datetime.now(timezone.utc)
        services_started = []
        services_failed = []
        health_checks = {}
        warnings = []
        errors = []
        current_phase = StartupPhase.INITIALIZATION
        
        try:
            # Phase 1: Initialization
            await self._run_initialization_phase(system_manager)
            current_phase = StartupPhase.CONFIGURATION
            
            # Phase 2: Load Configuration
            config = await self._run_configuration_phase(
                config_loader=config_loader,
                config_path=startup_config.config_path,
                environment=startup_config.environment
            )
            current_phase = StartupPhase.DEPENDENCIES
            
            # Phase 3: Resolve Dependencies
            dependency_graph = await self._run_dependency_phase(
                config=config,
                services=startup_config.services_to_start
            )
            current_phase = StartupPhase.CONNECTIONS
            
            # Phase 4: Establish Connections
            connections = await self._run_connection_phase(
                config=config,
                required_connections=['database', 'cache', 'exchange']
            )
            current_phase = StartupPhase.SERVICES
            
            # Phase 5: Start Services
            service_results = await self._run_service_startup_phase(
                system_manager=system_manager,
                config=config,
                dependency_graph=dependency_graph,
                connections=connections,
                parallel=startup_config.parallel_startup
            )
            
            services_started = service_results['started']
            services_failed = service_results['failed']
            
            if services_failed and startup_config.fail_fast:
                raise Exception(f"Failed to start services: {services_failed}")
            
            current_phase = StartupPhase.VALIDATION
            
            # Phase 6: Validation
            validation_results = await self._run_validation_phase(
                system_manager=system_manager,
                services_started=services_started,
                config=config
            )
            
            health_checks = validation_results['health_checks']
            warnings.extend(validation_results.get('warnings', []))
            
            # Phase 7: Warmup (optional)
            if startup_config.enable_warmup:
                current_phase = StartupPhase.WARMUP
                await self._run_warmup_phase(
                    system_manager=system_manager,
                    services=services_started
                )
            
            current_phase = StartupPhase.READY
            success = True
            system_ready = all(health_checks.values())
            
        except Exception as e:
            success = False
            system_ready = False
            errors.append(str(e))
            logging.error(f"Startup failed in phase {current_phase}: {e}")
        
        end_time = datetime.now(timezone.utc)
        
        return StartupResult(
            success=success,
            phase_completed=current_phase,
            duration=end_time - start_time,
            services_started=services_started,
            services_failed=services_failed,
            health_checks=health_checks,
            warnings=warnings,
            errors=errors,
            system_ready=system_ready
        )
    
    async def _run_initialization_phase(self, system_manager: SystemManager):
        """Run system initialization phase"""
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create necessary directories
        required_dirs = ['logs', 'data', 'cache', 'tmp']
        for dir_name in required_dirs:
            Path(dir_name).mkdir(exist_ok=True)
        
        # Initialize system manager
        await system_manager.initialize()
        
        # Set up signal handlers
        await system_manager.setup_signal_handlers()
        
        logging.info("Initialization phase completed")
    
    async def _run_configuration_phase(
        self,
        config_loader: ConfigLoader,
        config_path: str,
        environment: str
    ) -> Dict[str, Any]:
        """Load and validate configuration"""
        # Load base configuration
        config = await config_loader.load_file(config_path)
        
        # Load environment-specific overrides
        env_config = await config_loader.load_environment_config(environment)
        config = config_loader.merge_configs(config, env_config)
        
        # Load secrets
        secrets = await config_loader.load_secrets()
        config['secrets'] = secrets
        
        # Validate configuration
        validation_result = await config_loader.validate_config(config)
        if not validation_result['valid']:
            raise ValueError(f"Invalid configuration: {validation_result['errors']}")
        
        logging.info("Configuration loaded and validated")
        return config
    
    async def _run_dependency_phase(
        self,
        config: Dict[str, Any],
        services: List[str]
    ) -> Dict[str, ServiceDependency]:
        """Resolve service dependencies"""
        dependency_graph = {}
        
        # Build dependency graph
        for service in services:
            service_config = config['services'].get(service, {})
            dependencies = service_config.get('dependencies', [])
            
            dependency_graph[service] = ServiceDependency(
                service=service,
                depends_on=dependencies,
                required=True,
                startup_order=0,
                health_check_endpoint=f"http://localhost:{service_config.get('port', 8000)}/health"
            )
        
        # Detect circular dependencies
        circular = self._detect_circular_dependencies(dependency_graph)
        if circular:
            raise ValueError(f"Circular dependencies detected: {circular}")
        
        # Calculate startup order
        startup_order = self._calculate_startup_order(dependency_graph)
        for service, order in startup_order.items():
            dependency_graph[service].startup_order = order
        
        logging.info(f"Dependency resolution completed. Startup order: {startup_order}")
        return dependency_graph
    
    def _detect_circular_dependencies(
        self,
        graph: Dict[str, ServiceDependency]
    ) -> List[List[str]]:
        """Detect circular dependencies in service graph"""
        visited = set()
        rec_stack = set()
        circular_paths = []
        
        def dfs(service: str, path: List[str]) -> bool:
            visited.add(service)
            rec_stack.add(service)
            path.append(service)
            
            if service in graph:
                for dep in graph[service].depends_on:
                    if dep not in visited:
                        if dfs(dep, path.copy()):
                            return True
                    elif dep in rec_stack:
                        # Found circular dependency
                        cycle_start = path.index(dep)
                        circular_paths.append(path[cycle_start:] + [dep])
                        return True
            
            path.pop()
            rec_stack.remove(service)
            return False
        
        for service in graph:
            if service not in visited:
                dfs(service, [])
        
        return circular_paths
    
    def _calculate_startup_order(
        self,
        graph: Dict[str, ServiceDependency]
    ) -> Dict[str, int]:
        """Calculate service startup order using topological sort"""
        in_degree = {service: 0 for service in graph}
        
        # Calculate in-degrees
        for service, deps in graph.items():
            for dep in deps.depends_on:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        # Topological sort
        queue = [service for service, degree in in_degree.items() if degree == 0]
        startup_order = {}
        order = 0
        
        while queue:
            level_services = queue.copy()
            queue.clear()
            
            for service in level_services:
                startup_order[service] = order
                
                # Reduce in-degree for dependent services
                for s, deps in graph.items():
                    if service in deps.depends_on:
                        in_degree[s] -= 1
                        if in_degree[s] == 0:
                            queue.append(s)
            
            order += 1
        
        return startup_order
    
    async def _run_connection_phase(
        self,
        config: Dict[str, Any],
        required_connections: List[str]
    ) -> Dict[str, Any]:
        """Establish required connections"""
        connections = {}
        
        # Database connection
        if 'database' in required_connections:
            db_config = config['services']['database']
            try:
                db_client = motor.motor_asyncio.AsyncIOMotorClient(
                    f"mongodb://{db_config['host']}:{db_config['port']}"
                )
                # Test connection
                await db_client.admin.command('ping')
                connections['database'] = db_client
                logging.info("Database connection established")
            except Exception as e:
                logging.error(f"Failed to connect to database: {e}")
                if 'database' in [c for c in required_connections]:
                    raise
        
        # Cache connection
        if 'cache' in required_connections:
            cache_config = config['services']['cache']
            try:
                redis_client = await aioredis.create_redis_pool(
                    f"redis://{cache_config['host']}:{cache_config['port']}"
                )
                # Test connection
                await redis_client.ping()
                connections['cache'] = redis_client
                logging.info("Cache connection established")
            except Exception as e:
                logging.error(f"Failed to connect to cache: {e}")
                if 'cache' in [c for c in required_connections]:
                    raise
        
        # Exchange connections
        if 'exchange' in required_connections:
            exchange_config = config['connections']['exchange']
            exchange_connections = []
            
            for url in exchange_config['urls']:
                try:
                    # Mock exchange connection for testing
                    conn = {'url': url, 'connected': True, 'last_heartbeat': datetime.now()}
                    exchange_connections.append(conn)
                    logging.info(f"Exchange connection established: {url}")
                except Exception as e:
                    logging.error(f"Failed to connect to exchange {url}: {e}")
            
            if exchange_connections:
                connections['exchange'] = exchange_connections
            elif 'exchange' in [c for c in required_connections]:
                raise Exception("No exchange connections available")
        
        return connections
    
    async def _run_service_startup_phase(
        self,
        system_manager: SystemManager,
        config: Dict[str, Any],
        dependency_graph: Dict[str, ServiceDependency],
        connections: Dict[str, Any],
        parallel: bool = True
    ) -> Dict[str, List[str]]:
        """Start services in dependency order"""
        services_started = []
        services_failed = []
        
        # Group services by startup order
        order_groups = {}
        for service, dep in dependency_graph.items():
            order = dep.startup_order
            if order not in order_groups:
                order_groups[order] = []
            order_groups[order].append(service)
        
        # Start services in order
        for order in sorted(order_groups.keys()):
            services_in_order = order_groups[order]
            
            if parallel and len(services_in_order) > 1:
                # Start services in parallel
                tasks = []
                for service in services_in_order:
                    task = self._start_service(
                        system_manager=system_manager,
                        service=service,
                        config=config,
                        connections=connections
                    )
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for service, result in zip(services_in_order, results):
                    if isinstance(result, Exception):
                        logging.error(f"Failed to start {service}: {result}")
                        services_failed.append(service)
                    else:
                        services_started.append(service)
            else:
                # Start services sequentially
                for service in services_in_order:
                    try:
                        await self._start_service(
                            system_manager=system_manager,
                            service=service,
                            config=config,
                            connections=connections
                        )
                        services_started.append(service)
                    except Exception as e:
                        logging.error(f"Failed to start {service}: {e}")
                        services_failed.append(service)
        
        return {
            'started': services_started,
            'failed': services_failed
        }
    
    async def _start_service(
        self,
        system_manager: SystemManager,
        service: str,
        config: Dict[str, Any],
        connections: Dict[str, Any]
    ):
        """Start individual service"""
        service_config = config['services'][service]
        
        # Create service instance based on type
        if service == 'market_data':
            from core.market_data import MarketDataService
            service_instance = MarketDataService(
                config=service_config,
                connections=connections
            )
        elif service == 'trading_engine':
            from core.trading_engine import TradingEngineService
            service_instance = TradingEngineService(
                config=service_config,
                connections=connections
            )
        elif service == 'risk_management':
            from core.risk_management import RiskManagementService
            service_instance = RiskManagementService(
                config=service_config,
                connections=connections
            )
        else:
            # Generic service
            service_instance = Mock()
            service_instance.start = AsyncMock()
            service_instance.health_check = AsyncMock(return_value={'healthy': True})
        
        # Start the service
        await service_instance.start()
        
        # Register with system manager
        await system_manager.register_service(service, service_instance)
        
        logging.info(f"Service started: {service}")
    
    async def _run_validation_phase(
        self,
        system_manager: SystemManager,
        services_started: List[str],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate system readiness"""
        health_checks = {}
        warnings = []
        
        # Check each service health
        for service in services_started:
            try:
                service_instance = await system_manager.get_service(service)
                if hasattr(service_instance, 'health_check'):
                    health_result = await service_instance.health_check()
                    health_checks[service] = health_result.get('healthy', False)
                else:
                    # Basic health check
                    health_checks[service] = True
            except Exception as e:
                logging.error(f"Health check failed for {service}: {e}")
                health_checks[service] = False
        
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        if cpu_percent > 80:
            warnings.append(f"High CPU usage: {cpu_percent}%")
        if memory_percent > 80:
            warnings.append(f"High memory usage: {memory_percent}%")
        if disk_percent > 90:
            warnings.append(f"Low disk space: {disk_percent}% used")
        
        # Validate critical services
        critical_services = ['trading_engine', 'risk_management']
        for service in critical_services:
            if service in services_started and not health_checks.get(service, False):
                warnings.append(f"Critical service unhealthy: {service}")
        
        return {
            'health_checks': health_checks,
            'warnings': warnings,
            'system_metrics': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_percent': disk_percent
            }
        }
    
    async def _run_warmup_phase(
        self,
        system_manager: SystemManager,
        services: List[str]
    ):
        """Run service warmup procedures"""
        warmup_tasks = []
        
        for service in services:
            service_instance = await system_manager.get_service(service)
            
            if hasattr(service_instance, 'warmup'):
                # Service has warmup procedure
                task = service_instance.warmup()
                warmup_tasks.append(task)
            elif service == 'trading_engine':
                # Custom warmup for trading engine
                task = self._warmup_trading_engine(service_instance)
                warmup_tasks.append(task)
            elif service == 'market_data':
                # Custom warmup for market data
                task = self._warmup_market_data(service_instance)
                warmup_tasks.append(task)
        
        if warmup_tasks:
            await asyncio.gather(*warmup_tasks, return_exceptions=True)
        
        logging.info("Warmup phase completed")
    
    async def _warmup_trading_engine(self, trading_engine):
        """Warmup trading engine with test orders"""
        # Load historical data
        await trading_engine.load_historical_data(days=7)
        
        # Run backtests
        await trading_engine.run_warmup_backtest()
        
        # Verify order routing
        test_order = {
            'symbol': 'BTC/USDT',
            'side': 'BUY',
            'quantity': 0.001,
            'type': 'LIMIT',
            'price': 50000
        }
        result = await trading_engine.validate_order(test_order)
        
        logging.info("Trading engine warmup completed")
    
    async def _warmup_market_data(self, market_data):
        """Warmup market data connections"""
        # Subscribe to test symbols
        test_symbols = ['BTC/USDT', 'ETH/USDT']
        for symbol in test_symbols:
            await market_data.subscribe(symbol)
        
        # Wait for initial data
        await asyncio.sleep(2)
        
        # Verify data flow
        for symbol in test_symbols:
            data = await market_data.get_latest(symbol)
            if not data:
                raise Exception(f"No data received for {symbol}")
        
        logging.info("Market data warmup completed")
    
    @pytest.mark.asyncio
    async def test_startup_with_failures(self, system_manager, config_loader, test_config_file):
        """Test system startup with service failures"""
        # Modify config to include failing service
        with open(test_config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        config['services']['failing_service'] = {
            'enabled': True,
            'host': 'localhost',
            'port': 9999,
            'dependencies': [],
            'fail_on_start': True  # Will cause failure
        }
        
        with open(test_config_file, 'w') as f:
            yaml.dump(config, f)
        
        startup_config = StartupConfig(
            config_path=test_config_file,
            environment='test',
            services_to_start=['market_data', 'failing_service'],
            fail_fast=False  # Don't fail entire startup
        )
        
        # Mock the failing service
        async def mock_start_service(system_manager, service, config, connections):
            if service == 'failing_service':
                raise Exception("Service failed to start")
            return await self._start_service(system_manager, service, config, connections)
        
        # Patch the start service method
        original_start = self._start_service
        self._start_service = mock_start_service
        
        try:
            startup_result = await self._execute_startup(
                system_manager=system_manager,
                config_loader=config_loader,
                startup_config=startup_config
            )
            
            # System should start but with failures
            assert startup_result.success == True  # Because fail_fast=False
            assert 'market_data' in startup_result.services_started
            assert 'failing_service' in startup_result.services_failed
            assert len(startup_result.errors) == 0  # No fatal errors
            
            # Health check should show failing service as unhealthy
            assert startup_result.health_checks.get('failing_service', True) == False
            
        finally:
            self._start_service = original_start
    
    @pytest.mark.asyncio
    async def test_startup_dependency_order(self, system_manager, config_loader, test_config_file):
        """Test services start in correct dependency order"""
        startup_config = StartupConfig(
            config_path=test_config_file,
            environment='test',
            services_to_start=['api', 'trading_engine', 'risk_management', 'database'],
            parallel_startup=False  # Force sequential to verify order
        )
        
        # Track startup order
        startup_order = []
        
        async def track_start_service(system_manager, service, config, connections):
            startup_order.append(service)
            return await self._start_service(system_manager, service, config, connections)
        
        # Patch to track order
        original_start = self._start_service
        self._start_service = track_start_service
        
        try:
            startup_result = await self._execute_startup(
                system_manager=system_manager,
                config_loader=config_loader,
                startup_config=startup_config
            )
            
            assert startup_result.success == True
            
            # Verify dependency order
            # database should start before risk_management
            assert startup_order.index('database') < startup_order.index('risk_management')
            
            # risk_management should start before trading_engine
            assert startup_order.index('risk_management') < startup_order.index('trading_engine')
            
            # trading_engine should start before api
            assert startup_order.index('trading_engine') < startup_order.index('api')
            
        finally:
            self._start_service = original_start
    
    @pytest.mark.asyncio
    async def test_startup_timeout(self, system_manager, config_loader, test_config_file):
        """Test startup timeout handling"""
        startup_config = StartupConfig(
            config_path=test_config_file,
            environment='test',
            services_to_start=['market_data'],
            startup_timeout_seconds=1  # Very short timeout
        )
        
        # Mock slow service startup
        async def slow_start_service(system_manager, service, config, connections):
            await asyncio.sleep(2)  # Longer than timeout
            return await self._start_service(system_manager, service, config, connections)
        
        original_start = self._start_service
        self._start_service = slow_start_service
        
        try:
            # Use timeout wrapper
            startup_task = self._execute_startup(
                system_manager=system_manager,
                config_loader=config_loader,
                startup_config=startup_config
            )
            
            startup_result = await asyncio.wait_for(
                startup_task,
                timeout=startup_config.startup_timeout_seconds + 0.5
            )
            
            # Should not reach here due to timeout
            assert False, "Startup should have timed out"
            
        except asyncio.TimeoutError:
            # Expected behavior
            pass
        finally:
            self._start_service = original_start
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self, system_manager, config_loader):
        """Test configuration validation during startup"""
        # Create invalid config
        invalid_config = {
            'system': {
                'name': 'GridAttention'
                # Missing required 'version' field
            },
            'services': {
                'invalid_service': {
                    'enabled': True
                    # Missing required fields
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            invalid_config_path = f.name
        
        try:
            startup_config = StartupConfig(
                config_path=invalid_config_path,
                environment='test',
                services_to_start=['invalid_service']
            )
            
            # Mock validation to actually validate
            async def mock_validate_config(config):
                errors = []
                if 'version' not in config.get('system', {}):
                    errors.append("Missing system.version")
                if 'invalid_service' in config.get('services', {}):
                    service_config = config['services']['invalid_service']
                    if 'host' not in service_config:
                        errors.append("Missing services.invalid_service.host")
                    if 'port' not in service_config:
                        errors.append("Missing services.invalid_service.port")
                
                return {
                    'valid': len(errors) == 0,
                    'errors': errors
                }
            
            config_loader.validate_config = mock_validate_config
            
            startup_result = await self._execute_startup(
                system_manager=system_manager,
                config_loader=config_loader,
                startup_config=startup_config
            )
            
            # Startup should fail due to invalid config
            assert startup_result.success == False
            assert startup_result.phase_completed == StartupPhase.CONFIGURATION
            assert len(startup_result.errors) > 0
            assert any('Invalid configuration' in error for error in startup_result.errors)
            
        finally:
            os.unlink(invalid_config_path)
    
    @pytest.mark.asyncio
    async def test_health_check_retries(self, system_manager, config_loader, test_config_file, health_checker):
        """Test health check retry mechanism"""
        startup_config = StartupConfig(
            config_path=test_config_file,
            environment='test',
            services_to_start=['market_data'],
            health_check_retries=3
        )
        
        # Mock health check to fail initially then succeed
        health_check_attempts = 0
        
        async def mock_health_check():
            nonlocal health_check_attempts
            health_check_attempts += 1
            
            if health_check_attempts < 3:
                return {'healthy': False, 'reason': 'Still initializing'}
            else:
                return {'healthy': True}
        
        # Start service
        startup_result = await self._execute_startup(
            system_manager=system_manager,
            config_loader=config_loader,
            startup_config=startup_config
        )
        
        # Mock the health check after service is started
        service = await system_manager.get_service('market_data')
        if service:
            service.health_check = mock_health_check
            
            # Run health checks with retries
            for i in range(startup_config.health_check_retries):
                health_result = await service.health_check()
                if health_result['healthy']:
                    break
                await asyncio.sleep(1)
            
            assert health_check_attempts == 3
            assert health_result['healthy'] == True
    
    @pytest.mark.asyncio
    async def test_warmup_phase(self, system_manager, config_loader, test_config_file):
        """Test system warmup phase"""
        startup_config = StartupConfig(
            config_path=test_config_file,
            environment='test',
            services_to_start=['trading_engine', 'market_data'],
            enable_warmup=True
        )
        
        # Track warmup execution
        warmup_executed = {
            'trading_engine': False,
            'market_data': False
        }
        
        # Mock warmup methods
        async def mock_warmup_trading_engine(engine):
            warmup_executed['trading_engine'] = True
            await asyncio.sleep(0.1)
        
        async def mock_warmup_market_data(market_data):
            warmup_executed['market_data'] = True
            await asyncio.sleep(0.1)
        
        # Patch warmup methods
        original_warmup_te = self._warmup_trading_engine
        original_warmup_md = self._warmup_market_data
        
        self._warmup_trading_engine = mock_warmup_trading_engine
        self._warmup_market_data = mock_warmup_market_data
        
        try:
            startup_result = await self._execute_startup(
                system_manager=system_manager,
                config_loader=config_loader,
                startup_config=startup_config
            )
            
            assert startup_result.success == True
            assert startup_result.phase_completed == StartupPhase.READY
            
            # Verify warmup was executed
            assert warmup_executed['trading_engine'] == True
            assert warmup_executed['market_data'] == True
            
        finally:
            self._warmup_trading_engine = original_warmup_te
            self._warmup_market_data = original_warmup_md
    
    @pytest.mark.asyncio
    async def test_parallel_startup_performance(self, system_manager, config_loader, test_config_file):
        """Test performance improvement with parallel startup"""
        services = ['service1', 'service2', 'service3', 'service4']
        
        # Add test services to config
        with open(test_config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        for service in services:
            config['services'][service] = {
                'enabled': True,
                'host': 'localhost',
                'port': 8000 + len(services),
                'dependencies': []  # No dependencies, can start in parallel
            }
        
        with open(test_config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Mock service startup with delay
        async def mock_slow_start_service(system_manager, service, config, connections):
            await asyncio.sleep(1)  # Each service takes 1 second
            return await self._start_service(system_manager, service, config, connections)
        
        original_start = self._start_service
        self._start_service = mock_slow_start_service
        
        try:
            # Test sequential startup
            sequential_config = StartupConfig(
                config_path=test_config_file,
                environment='test',
                services_to_start=services,
                parallel_startup=False
            )
            
            start_time = datetime.now()
            sequential_result = await self._execute_startup(
                system_manager=system_manager,
                config_loader=config_loader,
                startup_config=sequential_config
            )
            sequential_duration = (datetime.now() - start_time).total_seconds()
            
            # Test parallel startup
            parallel_config = StartupConfig(
                config_path=test_config_file,
                environment='test',
                services_to_start=services,
                parallel_startup=True
            )
            
            start_time = datetime.now()
            parallel_result = await self._execute_startup(
                system_manager=system_manager,
                config_loader=config_loader,
                startup_config=parallel_config
            )
            parallel_duration = (datetime.now() - start_time).total_seconds()
            
            # Both should succeed
            assert sequential_result.success == True
            assert parallel_result.success == True
            
            # Parallel should be significantly faster
            assert parallel_duration < sequential_duration / 2  # At least 2x faster
            
        finally:
            self._start_service = original_start


class TestStartupRecovery:
    """Test startup recovery mechanisms"""
    
    @pytest.mark.asyncio
    async def test_partial_startup_recovery(self, system_manager, config_loader, test_config_file):
        """Test recovery from partial startup failure"""
        startup_config = StartupConfig(
            config_path=test_config_file,
            environment='test',
            services_to_start=['database', 'cache', 'trading_engine'],
            fail_fast=False
        )
        
        # Track which services started
        started_services = set()
        
        # Mock to fail trading_engine on first attempt
        attempt_count = 0
        
        async def mock_start_with_retry(system_manager, service, config, connections):
            nonlocal attempt_count
            
            if service == 'trading_engine' and attempt_count == 0:
                attempt_count += 1
                raise Exception("Temporary failure")
            
            started_services.add(service)
            return await TestSystemStartup()._start_service(system_manager, service, config, connections)
        
        # First attempt - partial failure
        test_instance = TestSystemStartup()
        test_instance._start_service = mock_start_with_retry
        
        first_result = await test_instance._execute_startup(
            system_manager=system_manager,
            config_loader=config_loader,
            startup_config=startup_config
        )
        
        # Should have started some services but not all
        assert 'database' in first_result.services_started
        assert 'cache' in first_result.services_started
        assert 'trading_engine' in first_result.services_failed
        
        # Attempt recovery
        recovery_config = StartupConfig(
            config_path=test_config_file,
            environment='test',
            services_to_start=['trading_engine'],  # Only retry failed service
            fail_fast=True
        )
        
        recovery_result = await test_instance._execute_startup(
            system_manager=system_manager,
            config_loader=config_loader,
            startup_config=recovery_config
        )
        
        # Recovery should succeed
        assert recovery_result.success == True
        assert 'trading_engine' in recovery_result.services_started
    
    @pytest.mark.asyncio
    async def test_startup_state_persistence(self, system_manager, config_loader, test_config_file):
        """Test startup state is persisted for recovery"""
        startup_config = StartupConfig(
            config_path=test_config_file,
            environment='test',
            services_to_start=['market_data', 'risk_management']
        )
        
        # Create state file path
        state_file = Path('startup_state.json')
        
        # Mock state persistence
        async def save_startup_state(phase: StartupPhase, services_started: List[str], services_failed: List[str]):
            state = {
                'phase': phase.value,
                'services_started': services_started,
                'services_failed': services_failed,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            with open(state_file, 'w') as f:
                json.dump(state, f)
        
        async def load_startup_state():
            if state_file.exists():
                with open(state_file, 'r') as f:
                    return json.load(f)
            return None
        
        try:
            # Run startup
            test_instance = TestSystemStartup()
            
            # Save state after each phase
            original_phases = [
                test_instance._run_initialization_phase,
                test_instance._run_configuration_phase,
                test_instance._run_dependency_phase,
                test_instance._run_connection_phase,
                test_instance._run_service_startup_phase
            ]
            
            # Add state saving to each phase
            for phase_method in original_phases:
                original_method = phase_method
                
                async def wrapped_method(*args, **kwargs):
                    result = await original_method(*args, **kwargs)
                    # Save state after phase
                    await save_startup_state(
                        StartupPhase.SERVICES,  # Example phase
                        [],  # Would track actual services
                        []
                    )
                    return result
                
                phase_method = wrapped_method
            
            startup_result = await test_instance._execute_startup(
                system_manager=system_manager,
                config_loader=config_loader,
                startup_config=startup_config
            )
            
            # Load and verify state
            saved_state = await load_startup_state()
            assert saved_state is not None
            assert 'phase' in saved_state
            assert 'services_started' in saved_state
            assert 'timestamp' in saved_state
            
        finally:
            if state_file.exists():
                state_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])