"""
Error recovery and resilience utilities for GridAttention
"""
import asyncio
import logging
import time
from typing import Any, Callable, Optional, Dict, TypeVar
from functools import wraps
import random
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
    async def call(self, func: Callable) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
                
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func()
            else:
                result = func()
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
            
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset the circuit"""
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
        )


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,)
):
    """Decorator for retry with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        logger.error(f"Final attempt failed for {func.__name__}: {e}")
                        raise
                        
                    # Calculate next delay
                    if jitter:
                        delay = delay * (0.5 + random.random())
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}. "
                        f"Retrying in {delay:.2f}s... Error: {e}"
                    )
                    
                    await asyncio.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)
                    
            raise last_exception
            
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        raise
                        
                    if jitter:
                        delay = delay * (0.5 + random.random())
                        
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    time.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)
                    
            raise last_exception
            
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


class ResilientConnection:
    """Base class for resilient network connections"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = config.get('max_reconnect_attempts', 10)
        self.reconnect_delay = config.get('reconnect_delay', 5)
        self.circuit_breaker = CircuitBreaker()
        self._reconnect_task = None
        
    async def connect(self):
        """Establish connection with retry logic"""
        @retry_with_backoff(max_attempts=3)
        async def _connect():
            await self._do_connect()
            self.is_connected = True
            self.reconnect_attempts = 0
            logger.info(f"Connected to {self.__class__.__name__}")
            
        await _connect()
        
    async def _do_connect(self):
        """Actual connection logic - override in subclass"""
        raise NotImplementedError
        
    async def disconnect(self):
        """Disconnect and cleanup"""
        if self._reconnect_task:
            self._reconnect_task.cancel()
            await asyncio.gather(self._reconnect_task, return_exceptions=True)
            
        await self._do_disconnect()
        self.is_connected = False
        
    async def _do_disconnect(self):
        """Actual disconnection logic - override in subclass"""
        raise NotImplementedError
        
    async def ensure_connected(self):
        """Ensure connection is established"""
        if not self.is_connected:
            await self.connect()
            
    async def execute_with_reconnect(self, operation: Callable):
        """Execute operation with automatic reconnection"""
        try:
            return await self.circuit_breaker.call(
                lambda: self._execute_operation(operation)
            )
        except Exception as e:
            if not self.is_connected:
                await self._schedule_reconnect()
            raise
            
    async def _execute_operation(self, operation: Callable):
        """Execute operation with connection check"""
        await self.ensure_connected()
        if asyncio.iscoroutinefunction(operation):
            return await operation()
        else:
            return operation()
        
    async def _schedule_reconnect(self):
        """Schedule reconnection attempt"""
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())
            
    async def _reconnect_loop(self):
        """Reconnection loop with exponential backoff"""
        delay = self.reconnect_delay
        
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                logger.info(f"Attempting reconnection ({self.reconnect_attempts + 1}/{self.max_reconnect_attempts})")
                await self.connect()
                return
            except Exception as e:
                self.reconnect_attempts += 1
                logger.error(f"Reconnection failed: {e}")
                
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 300)  # Max 5 minutes
                    
        logger.error("Max reconnection attempts reached")