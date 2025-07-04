#!/usr/bin/env python3
"""
Mock Database for GridAttention Trading System
Provides a realistic database simulation with async operations and persistence
"""

import asyncio
import json
import pickle
import sqlite3
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, OrderedDict
from pathlib import Path
import pandas as pd
import numpy as np
from decimal import Decimal
import threading
import time
import aiosqlite
from contextlib import asynccontextmanager


logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Database types"""
    MEMORY = "memory"
    SQLITE = "sqlite"
    MOCK = "mock"


class CollectionType(Enum):
    """Collection/Table types"""
    TRADES = "trades"
    ORDERS = "orders"
    POSITIONS = "positions"
    MARKET_DATA = "market_data"
    PERFORMANCE = "performance"
    SYSTEM_STATE = "system_state"
    ATTENTION_WEIGHTS = "attention_weights"
    GRID_STATE = "grid_state"
    RISK_METRICS = "risk_metrics"
    LOGS = "logs"


@dataclass
class QueryResult:
    """Database query result"""
    data: List[Dict[str, Any]]
    count: int
    execution_time_ms: float
    query: str = ""
    
    @property
    def is_empty(self) -> bool:
        return self.count == 0
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data) if self.data else pd.DataFrame()


@dataclass
class Transaction:
    """Database transaction"""
    transaction_id: str
    operations: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_operation(self, operation: str, collection: str, data: Any):
        """Add operation to transaction"""
        self.operations.append({
            "operation": operation,
            "collection": collection,
            "data": data,
            "timestamp": datetime.now()
        })


class MockDatabase:
    """Mock database implementation"""
    
    def __init__(
        self,
        db_type: DatabaseType = DatabaseType.MEMORY,
        db_path: Optional[str] = None,
        enable_persistence: bool = False
    ):
        self.db_type = db_type
        self.db_path = db_path or ":memory:"
        self.enable_persistence = enable_persistence
        
        # In-memory storage
        self.collections: Dict[str, OrderedDict] = defaultdict(OrderedDict)
        self.indexes: Dict[str, Dict[str, Dict]] = defaultdict(lambda: defaultdict(dict))
        
        # Transaction management
        self.transactions: Dict[str, Transaction] = {}
        self.active_transaction: Optional[Transaction] = None
        
        # Connection pool
        self.connection_pool: List[Any] = []
        self.max_connections = 10
        
        # Query cache
        self.query_cache: OrderedDict = OrderedDict()
        self.cache_size = 100
        
        # Locks
        self.write_lock = threading.Lock()
        self.read_locks: Dict[str, threading.RLock] = defaultdict(threading.RLock)
        
        # Statistics
        self.stats = {
            "queries": 0,
            "writes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_execution_time": 0
        }
        
        # Initialize database
        self._initialize_database()
        
        # Start background tasks
        self.running = True
        self._start_background_tasks()
    
    def _initialize_database(self):
        """Initialize database structure"""
        # Create collections
        for collection in CollectionType:
            self.collections[collection.value] = OrderedDict()
        
        # Create indexes
        self._create_index(CollectionType.TRADES.value, "symbol")
        self._create_index(CollectionType.TRADES.value, "timestamp")
        self._create_index(CollectionType.ORDERS.value, "order_id")
        self._create_index(CollectionType.ORDERS.value, "symbol")
        self._create_index(CollectionType.POSITIONS.value, "symbol")
        self._create_index(CollectionType.MARKET_DATA.value, "symbol")
        self._create_index(CollectionType.MARKET_DATA.value, "timestamp")
        
        # Load persisted data if enabled
        if self.enable_persistence and Path(self.db_path).exists():
            self._load_from_disk()
    
    def _start_background_tasks(self):
        """Start background tasks"""
        # Periodic persistence
        if self.enable_persistence:
            threading.Thread(target=self._persistence_loop, daemon=True).start()
        
        # Cache cleanup
        threading.Thread(target=self._cache_cleanup_loop, daemon=True).start()
    
    def _persistence_loop(self):
        """Periodically persist data to disk"""
        while self.running:
            try:
                time.sleep(60)  # Persist every minute
                self._save_to_disk()
            except Exception as e:
                logger.error(f"Persistence error: {e}")
    
    def _cache_cleanup_loop(self):
        """Clean up old cache entries"""
        while self.running:
            try:
                time.sleep(300)  # Clean every 5 minutes
                self._cleanup_cache()
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    # Core Database Operations
    
    async def connect(self) -> 'MockDatabaseConnection':
        """Create database connection"""
        if len(self.connection_pool) < self.max_connections:
            conn = MockDatabaseConnection(self)
            self.connection_pool.append(conn)
            return conn
        else:
            # Wait for available connection
            await asyncio.sleep(0.1)
            return await self.connect()
    
    async def insert_one(
        self,
        collection: str,
        document: Dict[str, Any]
    ) -> str:
        """Insert single document"""
        start_time = time.time()
        
        with self.write_lock:
            # Generate ID if not exists
            if "_id" not in document:
                document["_id"] = str(uuid.uuid4())
            
            # Add timestamps
            document["created_at"] = document.get("created_at", datetime.now())
            document["updated_at"] = datetime.now()
            
            # Store document
            self.collections[collection][document["_id"]] = document.copy()
            
            # Update indexes
            self._update_indexes(collection, document)
            
            # Track stats
            self.stats["writes"] += 1
            
            # Clear cache for collection
            self._invalidate_cache(collection)
            
            # Add to transaction if active
            if self.active_transaction:
                self.active_transaction.add_operation("insert", collection, document)
        
        execution_time = (time.time() - start_time) * 1000
        self.stats["total_execution_time"] += execution_time
        
        return document["_id"]
    
    async def insert_many(
        self,
        collection: str,
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """Insert multiple documents"""
        ids = []
        for doc in documents:
            doc_id = await self.insert_one(collection, doc)
            ids.append(doc_id)
        return ids
    
    async def find_one(
        self,
        collection: str,
        query: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Find single document"""
        results = await self.find(collection, query, limit=1)
        return results.data[0] if results.data else None
    
    async def find(
        self,
        collection: str,
        query: Dict[str, Any] = None,
        projection: Optional[List[str]] = None,
        sort: Optional[List[Tuple[str, int]]] = None,
        limit: Optional[int] = None,
        skip: Optional[int] = None
    ) -> QueryResult:
        """Find documents"""
        start_time = time.time()
        query = query or {}
        
        # Check cache
        cache_key = self._get_cache_key(collection, query, projection, sort, limit, skip)
        if cache_key in self.query_cache:
            self.stats["cache_hits"] += 1
            return self.query_cache[cache_key]
        
        self.stats["cache_misses"] += 1
        
        with self.read_locks[collection]:
            # Get all documents
            documents = list(self.collections[collection].values())
            
            # Apply query filter
            if query:
                documents = self._apply_query(documents, query)
            
            # Apply sort
            if sort:
                for field, order in reversed(sort):
                    documents.sort(key=lambda x: x.get(field), reverse=(order == -1))
            
            # Apply skip and limit
            if skip:
                documents = documents[skip:]
            if limit:
                documents = documents[:limit]
            
            # Apply projection
            if projection:
                documents = [
                    {k: v for k, v in doc.items() if k in projection}
                    for doc in documents
                ]
            
            # Create result
            result = QueryResult(
                data=documents,
                count=len(documents),
                execution_time_ms=(time.time() - start_time) * 1000,
                query=str(query)
            )
            
            # Cache result
            self.query_cache[cache_key] = result
            if len(self.query_cache) > self.cache_size:
                self.query_cache.popitem(last=False)
            
            # Track stats
            self.stats["queries"] += 1
            self.stats["total_execution_time"] += result.execution_time_ms
            
            return result
    
    async def update_one(
        self,
        collection: str,
        query: Dict[str, Any],
        update: Dict[str, Any]
    ) -> bool:
        """Update single document"""
        doc = await self.find_one(collection, query)
        if not doc:
            return False
        
        with self.write_lock:
            # Apply update
            if "$set" in update:
                doc.update(update["$set"])
            elif "$inc" in update:
                for field, value in update["$inc"].items():
                    doc[field] = doc.get(field, 0) + value
            else:
                doc.update(update)
            
            # Update timestamp
            doc["updated_at"] = datetime.now()
            
            # Store updated document
            self.collections[collection][doc["_id"]] = doc
            
            # Update indexes
            self._update_indexes(collection, doc)
            
            # Clear cache
            self._invalidate_cache(collection)
            
            # Track stats
            self.stats["writes"] += 1
            
            # Add to transaction
            if self.active_transaction:
                self.active_transaction.add_operation("update", collection, doc)
        
        return True
    
    async def update_many(
        self,
        collection: str,
        query: Dict[str, Any],
        update: Dict[str, Any]
    ) -> int:
        """Update multiple documents"""
        results = await self.find(collection, query)
        updated = 0
        
        for doc in results.data:
            if await self.update_one(collection, {"_id": doc["_id"]}, update):
                updated += 1
        
        return updated
    
    async def delete_one(
        self,
        collection: str,
        query: Dict[str, Any]
    ) -> bool:
        """Delete single document"""
        doc = await self.find_one(collection, query)
        if not doc:
            return False
        
        with self.write_lock:
            # Remove from collection
            del self.collections[collection][doc["_id"]]
            
            # Remove from indexes
            self._remove_from_indexes(collection, doc)
            
            # Clear cache
            self._invalidate_cache(collection)
            
            # Track stats
            self.stats["writes"] += 1
            
            # Add to transaction
            if self.active_transaction:
                self.active_transaction.add_operation("delete", collection, doc)
        
        return True
    
    async def delete_many(
        self,
        collection: str,
        query: Dict[str, Any]
    ) -> int:
        """Delete multiple documents"""
        results = await self.find(collection, query)
        deleted = 0
        
        for doc in results.data:
            if await self.delete_one(collection, {"_id": doc["_id"]}):
                deleted += 1
        
        return deleted
    
    async def count(
        self,
        collection: str,
        query: Dict[str, Any] = None
    ) -> int:
        """Count documents"""
        results = await self.find(collection, query)
        return results.count
    
    # Aggregation Operations
    
    async def aggregate(
        self,
        collection: str,
        pipeline: List[Dict[str, Any]]
    ) -> QueryResult:
        """Aggregation pipeline"""
        start_time = time.time()
        
        # Start with all documents
        documents = list(self.collections[collection].values())
        
        # Process pipeline stages
        for stage in pipeline:
            if "$match" in stage:
                documents = self._apply_query(documents, stage["$match"])
            
            elif "$group" in stage:
                documents = self._apply_group(documents, stage["$group"])
            
            elif "$sort" in stage:
                for field, order in stage["$sort"].items():
                    documents.sort(key=lambda x: x.get(field), reverse=(order == -1))
            
            elif "$limit" in stage:
                documents = documents[:stage["$limit"]]
            
            elif "$skip" in stage:
                documents = documents[stage["$skip"]:]
            
            elif "$project" in stage:
                projection = stage["$project"]
                documents = [
                    {k: v for k, v in doc.items() if projection.get(k, 0) == 1}
                    for doc in documents
                ]
        
        return QueryResult(
            data=documents,
            count=len(documents),
            execution_time_ms=(time.time() - start_time) * 1000,
            query=str(pipeline)
        )
    
    # Time Series Operations
    
    async def insert_time_series(
        self,
        collection: str,
        symbol: str,
        data: pd.DataFrame
    ) -> int:
        """Insert time series data"""
        documents = []
        
        for _, row in data.iterrows():
            doc = row.to_dict()
            doc["symbol"] = symbol
            doc["timestamp"] = row.name if isinstance(row.name, datetime) else datetime.now()
            documents.append(doc)
        
        ids = await self.insert_many(collection, documents)
        return len(ids)
    
    async def get_time_series(
        self,
        collection: str,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get time series data"""
        query = {
            "symbol": symbol,
            "timestamp": {
                "$gte": start_time,
                "$lte": end_time
            }
        }
        
        results = await self.find(
            collection,
            query,
            projection=fields,
            sort=[("timestamp", 1)]
        )
        
        if results.data:
            df = pd.DataFrame(results.data)
            df.set_index("timestamp", inplace=True)
            return df
        
        return pd.DataFrame()
    
    # Transaction Operations
    
    async def begin_transaction(self) -> Transaction:
        """Begin transaction"""
        transaction = Transaction(
            transaction_id=str(uuid.uuid4()),
            status="active"
        )
        
        self.transactions[transaction.transaction_id] = transaction
        self.active_transaction = transaction
        
        return transaction
    
    async def commit_transaction(self, transaction_id: str) -> bool:
        """Commit transaction"""
        if transaction_id not in self.transactions:
            return False
        
        transaction = self.transactions[transaction_id]
        transaction.status = "committed"
        
        if self.active_transaction == transaction:
            self.active_transaction = None
        
        # Persist if enabled
        if self.enable_persistence:
            self._save_to_disk()
        
        return True
    
    async def rollback_transaction(self, transaction_id: str) -> bool:
        """Rollback transaction"""
        if transaction_id not in self.transactions:
            return False
        
        transaction = self.transactions[transaction_id]
        
        # Reverse operations
        for op in reversed(transaction.operations):
            if op["operation"] == "insert":
                await self.delete_one(op["collection"], {"_id": op["data"]["_id"]})
            elif op["operation"] == "delete":
                await self.insert_one(op["collection"], op["data"])
            # Note: Updates are more complex and would need before/after states
        
        transaction.status = "rolled_back"
        
        if self.active_transaction == transaction:
            self.active_transaction = None
        
        return True
    
    # Backup and Restore
    
    async def backup(self, backup_path: str) -> bool:
        """Backup database"""
        try:
            backup_data = {
                "collections": dict(self.collections),
                "indexes": dict(self.indexes),
                "stats": self.stats,
                "timestamp": datetime.now()
            }
            
            with open(backup_path, "wb") as f:
                pickle.dump(backup_data, f)
            
            return True
        except Exception as e:
            logger.error(f"Backup error: {e}")
            return False
    
    async def restore(self, backup_path: str) -> bool:
        """Restore database from backup"""
        try:
            with open(backup_path, "rb") as f:
                backup_data = pickle.load(f)
            
            self.collections = defaultdict(OrderedDict, backup_data["collections"])
            self.indexes = defaultdict(lambda: defaultdict(dict), backup_data["indexes"])
            self.stats = backup_data["stats"]
            
            # Clear cache
            self.query_cache.clear()
            
            return True
        except Exception as e:
            logger.error(f"Restore error: {e}")
            return False
    
    # Index Operations
    
    def _create_index(self, collection: str, field: str):
        """Create index on field"""
        # Build index from existing documents
        for doc_id, doc in self.collections[collection].items():
            if field in doc:
                value = doc[field]
                if value not in self.indexes[collection][field]:
                    self.indexes[collection][field][value] = set()
                self.indexes[collection][field][value].add(doc_id)
    
    def _update_indexes(self, collection: str, document: Dict[str, Any]):
        """Update indexes for document"""
        doc_id = document["_id"]
        
        for field, index in self.indexes[collection].items():
            if field in document:
                value = document[field]
                if value not in index:
                    index[value] = set()
                index[value].add(doc_id)
    
    def _remove_from_indexes(self, collection: str, document: Dict[str, Any]):
        """Remove document from indexes"""
        doc_id = document["_id"]
        
        for field, index in self.indexes[collection].items():
            if field in document:
                value = document[field]
                if value in index:
                    index[value].discard(doc_id)
                    if not index[value]:
                        del index[value]
    
    # Query Operations
    
    def _apply_query(self, documents: List[Dict], query: Dict) -> List[Dict]:
        """Apply query filter to documents"""
        results = []
        
        for doc in documents:
            if self._match_document(doc, query):
                results.append(doc)
        
        return results
    
    def _match_document(self, doc: Dict, query: Dict) -> bool:
        """Check if document matches query"""
        for field, condition in query.items():
            if field == "$or":
                # OR condition
                if not any(self._match_document(doc, sub_query) for sub_query in condition):
                    return False
            elif field == "$and":
                # AND condition
                if not all(self._match_document(doc, sub_query) for sub_query in condition):
                    return False
            elif isinstance(condition, dict):
                # Operators
                doc_value = doc.get(field)
                
                for op, value in condition.items():
                    if op == "$eq" and doc_value != value:
                        return False
                    elif op == "$ne" and doc_value == value:
                        return False
                    elif op == "$gt" and not (doc_value and doc_value > value):
                        return False
                    elif op == "$gte" and not (doc_value and doc_value >= value):
                        return False
                    elif op == "$lt" and not (doc_value and doc_value < value):
                        return False
                    elif op == "$lte" and not (doc_value and doc_value <= value):
                        return False
                    elif op == "$in" and doc_value not in value:
                        return False
                    elif op == "$nin" and doc_value in value:
                        return False
                    elif op == "$exists":
                        exists = field in doc
                        if value and not exists:
                            return False
                        elif not value and exists:
                            return False
            else:
                # Direct equality
                if doc.get(field) != condition:
                    return False
        
        return True
    
    def _apply_group(self, documents: List[Dict], group_spec: Dict) -> List[Dict]:
        """Apply group aggregation"""
        groups = defaultdict(list)
        
        # Group documents
        group_id = group_spec.get("_id")
        if isinstance(group_id, str) and group_id.startswith("$"):
            # Group by field
            field = group_id[1:]
            for doc in documents:
                key = doc.get(field)
                groups[key].append(doc)
        else:
            # Single group
            groups[None] = documents
        
        # Apply aggregations
        results = []
        for key, group_docs in groups.items():
            result = {"_id": key}
            
            for field, agg in group_spec.items():
                if field == "_id":
                    continue
                
                if isinstance(agg, dict):
                    if "$sum" in agg:
                        if agg["$sum"] == 1:
                            result[field] = len(group_docs)
                        else:
                            sum_field = agg["$sum"][1:] if isinstance(agg["$sum"], str) else None
                            if sum_field:
                                result[field] = sum(doc.get(sum_field, 0) for doc in group_docs)
                    elif "$avg" in agg:
                        avg_field = agg["$avg"][1:]
                        values = [doc.get(avg_field, 0) for doc in group_docs if avg_field in doc]
                        result[field] = sum(values) / len(values) if values else 0
                    elif "$max" in agg:
                        max_field = agg["$max"][1:]
                        values = [doc.get(max_field) for doc in group_docs if max_field in doc]
                        result[field] = max(values) if values else None
                    elif "$min" in agg:
                        min_field = agg["$min"][1:]
                        values = [doc.get(min_field) for doc in group_docs if min_field in doc]
                        result[field] = min(values) if values else None
            
            results.append(result)
        
        return results
    
    # Cache Operations
    
    def _get_cache_key(
        self,
        collection: str,
        query: Dict,
        projection: Optional[List[str]],
        sort: Optional[List[Tuple[str, int]]],
        limit: Optional[int],
        skip: Optional[int]
    ) -> str:
        """Generate cache key"""
        key_parts = [
            collection,
            json.dumps(query, sort_keys=True),
            str(projection),
            str(sort),
            str(limit),
            str(skip)
        ]
        return "|".join(key_parts)
    
    def _invalidate_cache(self, collection: str):
        """Invalidate cache for collection"""
        keys_to_remove = [
            key for key in self.query_cache
            if key.startswith(collection + "|")
        ]
        for key in keys_to_remove:
            del self.query_cache[key]
    
    def _cleanup_cache(self):
        """Clean up old cache entries"""
        # Keep only recent entries
        while len(self.query_cache) > self.cache_size:
            self.query_cache.popitem(last=False)
    
    # Persistence Operations
    
    def _save_to_disk(self):
        """Save data to disk"""
        if not self.enable_persistence:
            return
        
        try:
            data = {
                "collections": dict(self.collections),
                "indexes": dict(self.indexes),
                "stats": self.stats
            }
            
            # Save to temporary file first
            temp_path = f"{self.db_path}.tmp"
            with open(temp_path, "wb") as f:
                pickle.dump(data, f)
            
            # Atomic rename
            Path(temp_path).replace(self.db_path)
            
        except Exception as e:
            logger.error(f"Save to disk error: {e}")
    
    def _load_from_disk(self):
        """Load data from disk"""
        try:
            with open(self.db_path, "rb") as f:
                data = pickle.load(f)
            
            self.collections = defaultdict(OrderedDict, data.get("collections", {}))
            self.indexes = defaultdict(lambda: defaultdict(dict), data.get("indexes", {}))
            self.stats = data.get("stats", self.stats)
            
        except Exception as e:
            logger.error(f"Load from disk error: {e}")
    
    # Statistics and Monitoring
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        collection_stats = {}
        
        for name, collection in self.collections.items():
            collection_stats[name] = {
                "count": len(collection),
                "indexes": list(self.indexes[name].keys())
            }
        
        return {
            "collections": collection_stats,
            "total_documents": sum(len(c) for c in self.collections.values()),
            "cache_size": len(self.query_cache),
            "transactions": len(self.transactions),
            "connections": len(self.connection_pool),
            "performance": self.stats
        }
    
    def shutdown(self):
        """Shutdown database"""
        self.running = False
        
        # Save final state
        if self.enable_persistence:
            self._save_to_disk()
        
        # Close connections
        for conn in self.connection_pool:
            conn.close()


class MockDatabaseConnection:
    """Mock database connection"""
    
    def __init__(self, database: MockDatabase):
        self.database = database
        self.connection_id = str(uuid.uuid4())
        self.is_open = True
    
    async def execute(self, operation: str, *args, **kwargs):
        """Execute database operation"""
        if not self.is_open:
            raise Exception("Connection closed")
        
        # Map operations to database methods
        if hasattr(self.database, operation):
            method = getattr(self.database, operation)
            return await method(*args, **kwargs)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def close(self):
        """Close connection"""
        self.is_open = False
        if self in self.database.connection_pool:
            self.database.connection_pool.remove(self)


class MockDatabasePool:
    """Connection pool for mock database"""
    
    def __init__(self, database: MockDatabase, min_size: int = 1, max_size: int = 10):
        self.database = database
        self.min_size = min_size
        self.max_size = max_size
        self.connections: List[MockDatabaseConnection] = []
        self.available: asyncio.Queue = asyncio.Queue()
        
        # Initialize minimum connections
        for _ in range(min_size):
            conn = MockDatabaseConnection(database)
            self.connections.append(conn)
            self.available.put_nowait(conn)
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire connection from pool"""
        # Get available connection
        if self.available.empty() and len(self.connections) < self.max_size:
            # Create new connection
            conn = MockDatabaseConnection(self.database)
            self.connections.append(conn)
        else:
            # Wait for available connection
            conn = await self.available.get()
        
        try:
            yield conn
        finally:
            # Return to pool
            if conn.is_open:
                await self.available.put(conn)
    
    async def close(self):
        """Close all connections"""
        for conn in self.connections:
            conn.close()


# Specialized Collections

class TimeSeriesCollection:
    """Specialized collection for time series data"""
    
    def __init__(self, database: MockDatabase, collection_name: str):
        self.database = database
        self.collection_name = collection_name
    
    async def insert_tick(self, symbol: str, price: float, volume: float, timestamp: Optional[datetime] = None):
        """Insert single tick"""
        doc = {
            "symbol": symbol,
            "price": price,
            "volume": volume,
            "timestamp": timestamp or datetime.now()
        }
        return await self.database.insert_one(self.collection_name, doc)
    
    async def insert_candle(
        self,
        symbol: str,
        timeframe: str,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float,
        timestamp: datetime
    ):
        """Insert candle data"""
        doc = {
            "symbol": symbol,
            "timeframe": timeframe,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "timestamp": timestamp
        }
        return await self.database.insert_one(self.collection_name, doc)
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol"""
        result = await self.database.find(
            self.collection_name,
            {"symbol": symbol},
            projection=["price", "timestamp"],
            sort=[("timestamp", -1)],
            limit=1
        )
        
        if result.data:
            return result.data[0]["price"]
        return None
    
    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Get OHLCV data"""
        return await self.database.get_time_series(
            self.collection_name,
            symbol,
            start_time,
            end_time,
            fields=["open", "high", "low", "close", "volume"]
        )


# Helper Functions

def create_mock_database(
    db_type: DatabaseType = DatabaseType.MEMORY,
    enable_persistence: bool = False
) -> MockDatabase:
    """Create mock database instance"""
    return MockDatabase(db_type=db_type, enable_persistence=enable_persistence)


def create_test_database() -> MockDatabase:
    """Create test database with sample data"""
    db = MockDatabase()
    
    # Add sample data
    asyncio.run(_populate_test_data(db))
    
    return db


async def _populate_test_data(db: MockDatabase):
    """Populate database with test data"""
    # Add sample trades
    trades = [
        {
            "symbol": "BTC/USDT",
            "side": "buy",
            "price": 45000,
            "quantity": 0.1,
            "timestamp": datetime.now() - timedelta(hours=i)
        }
        for i in range(10)
    ]
    await db.insert_many(CollectionType.TRADES.value, trades)
    
    # Add sample positions
    positions = [
        {
            "symbol": "BTC/USDT",
            "side": "long",
            "quantity": 0.5,
            "entry_price": 44500,
            "current_price": 45000,
            "pnl": 250
        }
    ]
    await db.insert_many(CollectionType.POSITIONS.value, positions)


# Context Manager Support

@asynccontextmanager
async def mock_database_session(db: MockDatabase):
    """Database session context manager"""
    conn = await db.connect()
    try:
        yield conn
    finally:
        conn.close()


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Create database
        db = create_mock_database(enable_persistence=True)
        
        # Insert document
        doc_id = await db.insert_one(
            CollectionType.TRADES.value,
            {
                "symbol": "BTC/USDT",
                "price": 45000,
                "quantity": 0.1,
                "side": "buy"
            }
        )
        print(f"Inserted document: {doc_id}")
        
        # Find documents
        results = await db.find(
            CollectionType.TRADES.value,
            {"symbol": "BTC/USDT"},
            sort=[("timestamp", -1)],
            limit=10
        )
        print(f"Found {results.count} trades")
        
        # Aggregation
        pipeline = [
            {"$match": {"symbol": "BTC/USDT"}},
            {"$group": {
                "_id": "$side",
                "total_volume": {"$sum": "$quantity"},
                "avg_price": {"$avg": "$price"}
            }}
        ]
        agg_results = await db.aggregate(CollectionType.TRADES.value, pipeline)
        print(f"Aggregation results: {agg_results.data}")
        
        # Get stats
        stats = await db.get_stats()
        print(f"Database stats: {json.dumps(stats, indent=2)}")
        
        # Shutdown
        db.shutdown()
    
    asyncio.run(main())