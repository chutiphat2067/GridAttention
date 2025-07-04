#!/usr/bin/env python3
"""
Async Helpers for GridAttention Trading System
Provides specialized utilities for testing asynchronous code
"""

import asyncio
import time
import functools
import inspect
import logging
import sys
from typing import (
    Dict, List, Optional, Any, Callable, Union, TypeVar, 
    Coroutine, AsyncIterator, AsyncGenerator, Awaitable, Set
)
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from collections import defaultdict, deque
import weakref
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import numpy as np
from unittest.mock import Mock, AsyncMock, MagicMock


logger = logging.getLogger(__name__)
T = TypeVar('T')


# Async Event Loop Management

class AsyncTestRunner:
    """Manages async test execution with proper setup and teardown"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._original_loop: Optional[asyncio.AbstractEventLoop] = None
        
    def __enter__(self):
        """Setup new event loop"""
        # Save original loop if exists
        try:
            self._original_loop = asyncio.get_event_loop()
        except RuntimeError:
            self._original_loop = None
        
        # Create new loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        if self.debug:
            self.loop.set_debug(True)
            
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup event loop"""
        # Cancel all tasks
        pending = asyncio.all_tasks(self.loop)
        for task in pending:
            task.cancel()
        
        # Run until all tasks are cancelled
        if pending:
            self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        
        # Close loop
        self.loop.close()
        
        # Restore original loop
        if self._original_loop:
            asyncio.set_event_loop(self._original_loop)
    
    def run(self, coro: Coroutine) -> Any:
        """Run coroutine in the test loop"""
        return self.loop.run_until_complete(coro)


def run_async_test(coro: Coroutine, timeout: float = 30.0, debug: bool = False) -> Any:
    """Run async test with proper setup and teardown"""
    with AsyncTestRunner(debug=debug) as runner:
        return runner.run(asyncio.wait_for(coro, timeout=timeout))


# Async Decorators

def async_timeout(seconds: float = 30.0):
    """Decorator to add timeout to async functions"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
        return wrapper
    return decorator


def async_retry(
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator to retry async functions"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < retries - 1:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        raise
            
            raise last_exception
        return wrapper
    return decorator


def async_cached(ttl: float = 60.0):
    """Decorator to cache async function results"""
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_times = {}
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            
            # Check cache
            if key in cache:
                if time.time() - cache_times[key] < ttl:
                    return cache[key]
            
            # Call function
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache[key] = result
            cache_times[key] = time.time()
            
            return result
        
        wrapper.clear_cache = lambda: (cache.clear(), cache_times.clear())
        return wrapper
    return decorator


# Async Context Managers

@asynccontextmanager
async def async_timer(name: str = "Operation"):
    """Async context manager to time operations"""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        logger.info(f"{name} took {elapsed:.3f} seconds")


@asynccontextmanager
async def async_timeout_context(seconds: float):
    """Async context manager for timeout"""
    async with asyncio.timeout(seconds):
        yield


@asynccontextmanager
async def async_suppress(*exceptions):
    """Async context manager to suppress exceptions"""
    try:
        yield
    except exceptions:
        pass


@asynccontextmanager
async def async_lock_context(lock: asyncio.Lock):
    """Async context manager for lock with timeout"""
    acquired = False
    try:
        acquired = await lock.acquire()
        yield
    finally:
        if acquired:
            lock.release()


# Async Synchronization Primitives

class AsyncBarrier:
    """Async barrier for synchronizing multiple coroutines"""
    
    def __init__(self, parties: int):
        self.parties = parties
        self.count = 0
        self.event = asyncio.Event()
        self.lock = asyncio.Lock()
        self.broken = False
        
    async def wait(self, timeout: Optional[float] = None):
        """Wait at barrier"""
        if self.broken:
            raise RuntimeError("Barrier is broken")
        
        async with self.lock:
            self.count += 1
            if self.count >= self.parties:
                self.event.set()
        
        try:
            await asyncio.wait_for(self.event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self.broken = True
            raise
        
    def reset(self):
        """Reset barrier"""
        self.count = 0
        self.event.clear()
        self.broken = False


class AsyncLatch:
    """Async countdown latch"""
    
    def __init__(self, count: int):
        self.count = count
        self.event = asyncio.Event()
        self.lock = asyncio.Lock()
        
    async def count_down(self):
        """Count down the latch"""
        async with self.lock:
            self.count -= 1
            if self.count <= 0:
                self.event.set()
    
    async def wait(self, timeout: Optional[float] = None):
        """Wait for latch to reach zero"""
        await asyncio.wait_for(self.event.wait(), timeout=timeout)


class AsyncThrottle:
    """Async rate limiter/throttle"""
    
    def __init__(self, rate_limit: int, per_seconds: float = 1.0):
        self.rate_limit = rate_limit
        self.per_seconds = per_seconds
        self.semaphore = asyncio.Semaphore(rate_limit)
        self.reset_task = None
        
    async def __aenter__(self):
        await self.acquire()
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Release will happen after delay
        pass
        
    async def acquire(self):
        """Acquire throttle permit"""
        await self.semaphore.acquire()
        
        # Schedule release
        asyncio.create_task(self._delayed_release())
    
    async def _delayed_release(self):
        """Release permit after delay"""
        await asyncio.sleep(self.per_seconds)
        self.semaphore.release()


# Async Testing Utilities

class AsyncTestCase:
    """Base class for async test cases"""
    
    def __init__(self):
        self.tasks: List[asyncio.Task] = []
        self.cleanup_callbacks: List[Callable] = []
        
    async def setUp(self):
        """Setup test case"""
        pass
        
    async def tearDown(self):
        """Teardown test case"""
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for cancellation
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Run cleanup callbacks
        for callback in self.cleanup_callbacks:
            if asyncio.iscoroutinefunction(callback):
                await callback()
            else:
                callback()
    
    def create_task(self, coro: Coroutine) -> asyncio.Task:
        """Create and track task"""
        task = asyncio.create_task(coro)
        self.tasks.append(task)
        return task
    
    def add_cleanup(self, callback: Callable):
        """Add cleanup callback"""
        self.cleanup_callbacks.append(callback)


class AsyncMockManager:
    """Manager for async mocks"""
    
    def __init__(self):
        self.mocks: Dict[str, AsyncMock] = {}
        
    def create_mock(self, name: str, **kwargs) -> AsyncMock:
        """Create and track async mock"""
        mock = AsyncMock(**kwargs)
        self.mocks[name] = mock
        return mock
    
    def get_mock(self, name: str) -> Optional[AsyncMock]:
        """Get mock by name"""
        return self.mocks.get(name)
    
    def reset_all(self):
        """Reset all mocks"""
        for mock in self.mocks.values():
            mock.reset_mock()
    
    def assert_all_awaited(self):
        """Assert all mocks were awaited"""
        for name, mock in self.mocks.items():
            mock.assert_awaited()


# Async Wait Utilities

async def wait_for_predicate(
    predicate: Callable[[], bool],
    timeout: float = 5.0,
    interval: float = 0.1,
    message: str = "Predicate not satisfied"
) -> bool:
    """Wait for predicate to become true"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if predicate():
            return True
        await asyncio.sleep(interval)
    
    raise TimeoutError(f"{message} after {timeout}s")


async def wait_for_all(
    *awaitables: Awaitable,
    timeout: Optional[float] = None,
    return_exceptions: bool = False
) -> List[Any]:
    """Wait for all awaitables with optional timeout"""
    if timeout:
        return await asyncio.wait_for(
            asyncio.gather(*awaitables, return_exceptions=return_exceptions),
            timeout=timeout
        )
    else:
        return await asyncio.gather(*awaitables, return_exceptions=return_exceptions)


async def wait_for_any(
    *awaitables: Awaitable,
    timeout: Optional[float] = None
) -> Tuple[Set[asyncio.Task], Set[asyncio.Task]]:
    """Wait for any awaitable to complete"""
    tasks = [asyncio.create_task(a) for a in awaitables]
    
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED,
        timeout=timeout
    )
    
    # Cancel pending tasks
    for task in pending:
        task.cancel()
    
    return done, pending


async def wait_with_progress(
    awaitables: List[Awaitable],
    callback: Optional[Callable[[int, int], None]] = None
) -> List[Any]:
    """Wait for awaitables with progress callback"""
    total = len(awaitables)
    completed = 0
    results = [None] * total
    
    async def wait_with_index(index: int, awaitable: Awaitable):
        nonlocal completed
        result = await awaitable
        results[index] = result
        completed += 1
        if callback:
            callback(completed, total)
        return result
    
    await asyncio.gather(
        *[wait_with_index(i, a) for i, a in enumerate(awaitables)]
    )
    
    return results


# Async Generators and Iterators

async def async_range(start: int, stop: int, step: int = 1) -> AsyncIterator[int]:
    """Async version of range"""
    current = start
    while current < stop:
        yield current
        await asyncio.sleep(0)  # Allow other tasks to run
        current += step


async def async_enumerate(
    async_iterable: AsyncIterator[T],
    start: int = 0
) -> AsyncIterator[Tuple[int, T]]:
    """Async version of enumerate"""
    index = start
    async for item in async_iterable:
        yield index, item
        index += 1


async def async_zip(
    *async_iterables: AsyncIterator
) -> AsyncIterator[Tuple]:
    """Async version of zip"""
    iterators = [aiter(it) for it in async_iterables]
    
    while True:
        values = []
        for it in iterators:
            try:
                value = await anext(it)
                values.append(value)
            except StopAsyncIteration:
                return
        
        yield tuple(values)


async def async_filter(
    predicate: Callable[[T], bool],
    async_iterable: AsyncIterator[T]
) -> AsyncIterator[T]:
    """Async version of filter"""
    async for item in async_iterable:
        if predicate(item):
            yield item


async def async_map(
    func: Callable[[T], Any],
    async_iterable: AsyncIterator[T]
) -> AsyncIterator[Any]:
    """Async version of map"""
    async for item in async_iterable:
        if asyncio.iscoroutinefunction(func):
            yield await func(item)
        else:
            yield func(item)


# Async Queue Utilities

class AsyncPriorityQueue:
    """Async priority queue with additional features"""
    
    def __init__(self, maxsize: int = 0):
        self.queue = asyncio.PriorityQueue(maxsize=maxsize)
        self.item_count = 0
        
    async def put(self, item: Any, priority: int = 0):
        """Put item with priority"""
        await self.queue.put((priority, self.item_count, item))
        self.item_count += 1
    
    async def get(self) -> Any:
        """Get highest priority item"""
        _, _, item = await self.queue.get()
        return item
    
    def qsize(self) -> int:
        """Get queue size"""
        return self.queue.qsize()
    
    async def clear(self):
        """Clear the queue"""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class AsyncBatchQueue:
    """Queue that returns items in batches"""
    
    def __init__(self, batch_size: int = 10, timeout: float = 1.0):
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue: asyncio.Queue = asyncio.Queue()
        
    async def put(self, item: Any):
        """Put item in queue"""
        await self.queue.put(item)
    
    async def get_batch(self) -> List[Any]:
        """Get batch of items"""
        batch = []
        deadline = time.time() + self.timeout
        
        while len(batch) < self.batch_size and time.time() < deadline:
            try:
                timeout_remaining = max(0, deadline - time.time())
                item = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=timeout_remaining
                )
                batch.append(item)
            except asyncio.TimeoutError:
                break
        
        return batch


# Async Stream Processing

class AsyncStreamProcessor:
    """Process async streams with transformations"""
    
    def __init__(self, source: AsyncIterator[T]):
        self.source = source
        self.transformations = []
        
    def map(self, func: Callable[[T], Any]) -> 'AsyncStreamProcessor':
        """Add map transformation"""
        self.transformations.append(('map', func))
        return self
    
    def filter(self, predicate: Callable[[T], bool]) -> 'AsyncStreamProcessor':
        """Add filter transformation"""
        self.transformations.append(('filter', predicate))
        return self
    
    def batch(self, size: int) -> 'AsyncStreamProcessor':
        """Add batch transformation"""
        self.transformations.append(('batch', size))
        return self
    
    def throttle(self, delay: float) -> 'AsyncStreamProcessor':
        """Add throttle transformation"""
        self.transformations.append(('throttle', delay))
        return self
    
    async def collect(self) -> List[Any]:
        """Collect all results"""
        results = []
        async for item in self:
            results.append(item)
        return results
    
    def __aiter__(self):
        return self._process()
    
    async def _process(self):
        """Process stream with transformations"""
        stream = self.source
        
        for transform_type, param in self.transformations:
            if transform_type == 'map':
                stream = async_map(param, stream)
            elif transform_type == 'filter':
                stream = async_filter(param, stream)
            elif transform_type == 'batch':
                stream = self._batch_stream(stream, param)
            elif transform_type == 'throttle':
                stream = self._throttle_stream(stream, param)
        
        async for item in stream:
            yield item
    
    async def _batch_stream(self, stream: AsyncIterator, size: int) -> AsyncIterator[List]:
        """Batch items from stream"""
        batch = []
        async for item in stream:
            batch.append(item)
            if len(batch) >= size:
                yield batch
                batch = []
        
        if batch:
            yield batch
    
    async def _throttle_stream(self, stream: AsyncIterator, delay: float) -> AsyncIterator:
        """Throttle items from stream"""
        async for item in stream:
            yield item
            await asyncio.sleep(delay)


# Async Testing Fixtures

class AsyncFixture:
    """Base class for async fixtures"""
    
    async def setup(self):
        """Setup fixture"""
        pass
    
    async def teardown(self):
        """Teardown fixture"""
        pass
    
    async def __aenter__(self):
        await self.setup()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.teardown()


class AsyncWebServerFixture(AsyncFixture):
    """Fixture for testing with async web server"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.app = None
        self.runner = None
        self.site = None
        
    async def setup(self):
        """Start web server"""
        from aiohttp import web
        
        self.app = web.Application()
        self.setup_routes(self.app)
        
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, 'localhost', self.port)
        await self.site.start()
    
    async def teardown(self):
        """Stop web server"""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
    
    def setup_routes(self, app):
        """Setup test routes"""
        async def hello(request):
            return web.Response(text="Hello, World!")
        
        app.router.add_get('/', hello)
    
    @property
    def url(self) -> str:
        """Get server URL"""
        return f"http://localhost:{self.port}"


class AsyncWebSocketFixture(AsyncFixture):
    """Fixture for WebSocket testing"""
    
    def __init__(self, url: str):
        self.url = url
        self.session = None
        self.ws = None
        
    async def setup(self):
        """Connect to WebSocket"""
        self.session = aiohttp.ClientSession()
        self.ws = await self.session.ws_connect(self.url)
    
    async def teardown(self):
        """Disconnect from WebSocket"""
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()
    
    async def send(self, data: Union[str, bytes, dict]):
        """Send data"""
        if isinstance(data, dict):
            await self.ws.send_json(data)
        elif isinstance(data, bytes):
            await self.ws.send_bytes(data)
        else:
            await self.ws.send_str(str(data))
    
    async def receive(self, timeout: float = 5.0) -> Any:
        """Receive data"""
        msg = await asyncio.wait_for(self.ws.receive(), timeout=timeout)
        
        if msg.type == aiohttp.WSMsgType.TEXT:
            return msg.data
        elif msg.type == aiohttp.WSMsgType.BINARY:
            return msg.data
        elif msg.type == aiohttp.WSMsgType.ERROR:
            raise Exception(f"WebSocket error: {msg}")
        elif msg.type == aiohttp.WSMsgType.CLOSE:
            raise Exception("WebSocket closed")


# Async Performance Testing

class AsyncLoadTester:
    """Async load testing utility"""
    
    def __init__(self, target_func: Callable, concurrency: int = 10):
        self.target_func = target_func
        self.concurrency = concurrency
        self.results = []
        self.errors = []
        
    async def run(
        self,
        total_requests: int,
        duration: Optional[float] = None,
        ramp_up: float = 0.0
    ) -> Dict[str, Any]:
        """Run load test"""
        start_time = time.time()
        semaphore = asyncio.Semaphore(self.concurrency)
        
        async def worker(request_id: int):
            # Ramp up delay
            if ramp_up > 0:
                delay = (request_id / total_requests) * ramp_up
                await asyncio.sleep(delay)
            
            async with semaphore:
                request_start = time.time()
                try:
                    if asyncio.iscoroutinefunction(self.target_func):
                        result = await self.target_func()
                    else:
                        result = self.target_func()
                    
                    self.results.append({
                        'request_id': request_id,
                        'duration': time.time() - request_start,
                        'success': True,
                        'result': result
                    })
                except Exception as e:
                    self.errors.append({
                        'request_id': request_id,
                        'duration': time.time() - request_start,
                        'error': str(e)
                    })
        
        # Create tasks
        if duration:
            # Run for duration
            end_time = time.time() + duration
            request_id = 0
            tasks = []
            
            while time.time() < end_time:
                task = asyncio.create_task(worker(request_id))
                tasks.append(task)
                request_id += 1
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.001)
            
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Run fixed number of requests
            tasks = [worker(i) for i in range(total_requests)]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate statistics
        total_time = time.time() - start_time
        successful_requests = len(self.results)
        failed_requests = len(self.errors)
        
        if self.results:
            durations = [r['duration'] for r in self.results]
            avg_duration = np.mean(durations)
            min_duration = np.min(durations)
            max_duration = np.max(durations)
            p50_duration = np.percentile(durations, 50)
            p95_duration = np.percentile(durations, 95)
            p99_duration = np.percentile(durations, 99)
        else:
            avg_duration = min_duration = max_duration = 0
            p50_duration = p95_duration = p99_duration = 0
        
        return {
            'total_requests': successful_requests + failed_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'total_time': total_time,
            'requests_per_second': (successful_requests + failed_requests) / total_time,
            'average_duration': avg_duration,
            'min_duration': min_duration,
            'max_duration': max_duration,
            'p50_duration': p50_duration,
            'p95_duration': p95_duration,
            'p99_duration': p99_duration,
            'errors': self.errors[:10]  # First 10 errors
        }


# Async Debugging Utilities

class AsyncDebugger:
    """Utilities for debugging async code"""
    
    @staticmethod
    def print_tasks():
        """Print all running tasks"""
        tasks = asyncio.all_tasks()
        print(f"\n=== Active Tasks ({len(tasks)}) ===")
        for task in tasks:
            print(f"Task {task.get_name()}: {task}")
            if hasattr(task, 'get_stack'):
                stack = task.get_stack()
                if stack:
                    print(f"  Stack: {stack[0]}")
    
    @staticmethod
    async def monitor_event_loop(interval: float = 1.0):
        """Monitor event loop performance"""
        while True:
            loop = asyncio.get_event_loop()
            
            # Get loop statistics
            tasks = asyncio.all_tasks(loop)
            pending = sum(1 for t in tasks if not t.done())
            
            print(f"Event Loop Stats: {pending} pending tasks")
            
            await asyncio.sleep(interval)
    
    @staticmethod
    def trace_coroutine(coro: Coroutine) -> Coroutine:
        """Trace coroutine execution"""
        @functools.wraps(coro)
        async def wrapper(*args, **kwargs):
            coro_name = coro.__name__ if hasattr(coro, '__name__') else str(coro)
            print(f"[TRACE] Starting: {coro_name}")
            start_time = time.time()
            
            try:
                result = await coro(*args, **kwargs)
                elapsed = time.time() - start_time
                print(f"[TRACE] Completed: {coro_name} ({elapsed:.3f}s)")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"[TRACE] Failed: {coro_name} ({elapsed:.3f}s) - {e}")
                raise
        
        return wrapper


# Helper Functions

async def async_run_in_executor(
    func: Callable,
    *args,
    executor: Optional[ThreadPoolExecutor] = None
) -> Any:
    """Run sync function in executor"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)


async def async_sleep_until(target_time: datetime):
    """Sleep until specific time"""
    now = datetime.now()
    if target_time > now:
        delay = (target_time - now).total_seconds()
        await asyncio.sleep(delay)


def create_async_mock(
    return_value: Any = None,
    side_effect: Any = None,
    **kwargs
) -> AsyncMock:
    """Create configured async mock"""
    mock = AsyncMock(**kwargs)
    
    if return_value is not None:
        mock.return_value = return_value
    
    if side_effect is not None:
        mock.side_effect = side_effect
    
    return mock


# Example Usage

if __name__ == "__main__":
    async def example_tests():
        # Test with timeout
        @async_timeout(5.0)
        async def test_api_call():
            await asyncio.sleep(1)
            return "Success"
        
        # Test with retry
        @async_retry(retries=3, delay=0.5)
        async def flaky_operation():
            if random.random() < 0.7:
                raise Exception("Random failure")
            return "Success"
        
        # Load testing
        async def target_operation():
            await asyncio.sleep(random.uniform(0.01, 0.1))
            return "OK"
        
        tester = AsyncLoadTester(target_operation, concurrency=50)
        results = await tester.run(total_requests=1000)
        
        print(f"Load Test Results:")
        print(f"  Total: {results['total_requests']}")
        print(f"  RPS: {results['requests_per_second']:.2f}")
        print(f"  Avg Duration: {results['average_duration']:.3f}s")
        print(f"  P95 Duration: {results['p95_duration']:.3f}s")
        
        # Stream processing
        async def data_source():
            for i in range(100):
                yield i
                await asyncio.sleep(0.01)
        
        processor = AsyncStreamProcessor(data_source())
        results = await processor \
            .filter(lambda x: x % 2 == 0) \
            .map(lambda x: x * 2) \
            .batch(10) \
            .collect()
        
        print(f"\nProcessed {len(results)} batches")
    
    # Run examples
    run_async_test(example_tests())