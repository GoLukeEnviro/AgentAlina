#!/usr/bin/env python3
"""
State Management & Persistence System for AgentAlina
Implements robust state management with Redis and PostgreSQL for conversation
results, context persistence, and distributed state coordination.
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StateType(Enum):
    """Types of state that can be managed."""
    CONVERSATION = "conversation"
    CONTEXT = "context"
    SESSION = "session"
    AGENT = "agent"
    WORKFLOW = "workflow"
    CACHE = "cache"

class PersistenceLevel(Enum):
    """Levels of persistence for state data."""
    MEMORY_ONLY = "memory_only"  # Redis only, temporary
    PERSISTENT = "persistent"    # PostgreSQL, permanent
    HYBRID = "hybrid"           # Both Redis and PostgreSQL

@dataclass
class StateEntry:
    """Represents a state entry in the system."""
    id: str
    state_type: StateType
    key: str
    value: Any
    persistence_level: PersistenceLevel
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    version: int = 1
    checksum: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.checksum is None:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for data integrity."""
        data_str = json.dumps(self.value, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['state_type'] = self.state_type.value
        data['persistence_level'] = self.persistence_level.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateEntry':
        """Create StateEntry from dictionary."""
        data['state_type'] = StateType(data['state_type'])
        data['persistence_level'] = PersistenceLevel(data['persistence_level'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if data.get('expires_at'):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        return cls(**data)

class RedisStateManager:
    """Manages state in Redis for fast access and temporary storage."""
    
    def __init__(self, redis_client, key_prefix: str = "agentAlina:state"):
        self.redis_client = redis_client
        self.key_prefix = key_prefix
        self.metrics = {
            'operations': 0,
            'hits': 0,
            'misses': 0,
            'errors': 0,
            'evictions': 0
        }
    
    def _make_key(self, state_type: StateType, key: str) -> str:
        """Create Redis key with proper namespacing."""
        return f"{self.key_prefix}:{state_type.value}:{key}"
    
    async def store(self, entry: StateEntry) -> bool:
        """Store state entry in Redis."""
        try:
            redis_key = self._make_key(entry.state_type, entry.key)
            data = entry.to_dict()
            
            # Set expiration if specified
            if entry.expires_at:
                ttl = int((entry.expires_at - datetime.now()).total_seconds())
                if ttl > 0:
                    self.redis_client.setex(redis_key, ttl, json.dumps(data))
                else:
                    logger.warning(f"Entry {entry.id} already expired, not storing")
                    return False
            else:
                self.redis_client.set(redis_key, json.dumps(data))
            
            # Store in index for efficient querying
            index_key = f"{self.key_prefix}:index:{entry.state_type.value}"
            self.redis_client.sadd(index_key, entry.key)
            
            self.metrics['operations'] += 1
            logger.debug(f"Stored state entry: {entry.id}")
            return True
        
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error storing state entry {entry.id}: {e}")
            return False
    
    async def retrieve(self, state_type: StateType, key: str) -> Optional[StateEntry]:
        """Retrieve state entry from Redis."""
        try:
            redis_key = self._make_key(state_type, key)
            data = self.redis_client.get(redis_key)
            
            if data is None:
                self.metrics['misses'] += 1
                return None
            
            entry_data = json.loads(data)
            entry = StateEntry.from_dict(entry_data)
            
            self.metrics['hits'] += 1
            self.metrics['operations'] += 1
            return entry
        
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error retrieving state entry {state_type.value}:{key}: {e}")
            return None
    
    async def delete(self, state_type: StateType, key: str) -> bool:
        """Delete state entry from Redis."""
        try:
            redis_key = self._make_key(state_type, key)
            result = self.redis_client.delete(redis_key)
            
            # Remove from index
            index_key = f"{self.key_prefix}:index:{state_type.value}"
            self.redis_client.srem(index_key, key)
            
            self.metrics['operations'] += 1
            return result > 0
        
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error deleting state entry {state_type.value}:{key}: {e}")
            return False
    
    async def list_keys(self, state_type: StateType) -> List[str]:
        """List all keys for a given state type."""
        try:
            index_key = f"{self.key_prefix}:index:{state_type.value}"
            keys = self.redis_client.smembers(index_key)
            return list(keys)
        
        except Exception as e:
            logger.error(f"Error listing keys for {state_type.value}: {e}")
            return []
    
    async def cleanup_expired(self) -> int:
        """Clean up expired entries and return count of cleaned entries."""
        cleaned = 0
        try:
            for state_type in StateType:
                keys = await self.list_keys(state_type)
                for key in keys:
                    entry = await self.retrieve(state_type, key)
                    if entry and entry.expires_at and entry.expires_at < datetime.now():
                        await self.delete(state_type, key)
                        cleaned += 1
                        self.metrics['evictions'] += 1
        
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        return cleaned
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get Redis state manager metrics."""
        return self.metrics.copy()

class PostgreSQLStateManager:
    """Manages persistent state in PostgreSQL."""
    
    def __init__(self, db_connection):
        self.db_connection = db_connection
        self.metrics = {
            'operations': 0,
            'inserts': 0,
            'updates': 0,
            'deletes': 0,
            'queries': 0,
            'errors': 0
        }
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Ensure required tables exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS state_entries (
            id VARCHAR(255) PRIMARY KEY,
            state_type VARCHAR(50) NOT NULL,
            key VARCHAR(255) NOT NULL,
            value JSONB NOT NULL,
            persistence_level VARCHAR(50) NOT NULL,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            expires_at TIMESTAMP,
            metadata JSONB,
            version INTEGER DEFAULT 1,
            checksum VARCHAR(32),
            UNIQUE(state_type, key)
        );
        
        CREATE INDEX IF NOT EXISTS idx_state_entries_type_key ON state_entries(state_type, key);
        CREATE INDEX IF NOT EXISTS idx_state_entries_expires_at ON state_entries(expires_at);
        CREATE INDEX IF NOT EXISTS idx_state_entries_created_at ON state_entries(created_at);
        """
        
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute(create_table_sql)
                self.db_connection.commit()
                logger.info("PostgreSQL state tables ensured")
        except Exception as e:
            logger.error(f"Error creating state tables: {e}")
            self.db_connection.rollback()
    
    async def store(self, entry: StateEntry) -> bool:
        """Store state entry in PostgreSQL."""
        try:
            upsert_sql = """
            INSERT INTO state_entries 
            (id, state_type, key, value, persistence_level, created_at, updated_at, expires_at, metadata, version, checksum)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (state_type, key) 
            DO UPDATE SET
                value = EXCLUDED.value,
                updated_at = EXCLUDED.updated_at,
                expires_at = EXCLUDED.expires_at,
                metadata = EXCLUDED.metadata,
                version = state_entries.version + 1,
                checksum = EXCLUDED.checksum
            """
            
            with self.db_connection.cursor() as cursor:
                cursor.execute(upsert_sql, (
                    entry.id,
                    entry.state_type.value,
                    entry.key,
                    json.dumps(entry.value),
                    entry.persistence_level.value,
                    entry.created_at,
                    entry.updated_at,
                    entry.expires_at,
                    json.dumps(entry.metadata),
                    entry.version,
                    entry.checksum
                ))
                
                self.db_connection.commit()
                
                if cursor.rowcount > 0:
                    self.metrics['inserts'] += 1
                else:
                    self.metrics['updates'] += 1
                
                self.metrics['operations'] += 1
                logger.debug(f"Stored state entry in PostgreSQL: {entry.id}")
                return True
        
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error storing state entry {entry.id} in PostgreSQL: {e}")
            self.db_connection.rollback()
            return False
    
    async def retrieve(self, state_type: StateType, key: str) -> Optional[StateEntry]:
        """Retrieve state entry from PostgreSQL."""
        try:
            select_sql = """
            SELECT id, state_type, key, value, persistence_level, created_at, updated_at, 
                   expires_at, metadata, version, checksum
            FROM state_entries 
            WHERE state_type = %s AND key = %s
            AND (expires_at IS NULL OR expires_at > NOW())
            """
            
            with self.db_connection.cursor() as cursor:
                cursor.execute(select_sql, (state_type.value, key))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                entry_data = {
                    'id': row[0],
                    'state_type': row[1],
                    'key': row[2],
                    'value': json.loads(row[3]),
                    'persistence_level': row[4],
                    'created_at': row[5].isoformat(),
                    'updated_at': row[6].isoformat(),
                    'expires_at': row[7].isoformat() if row[7] else None,
                    'metadata': json.loads(row[8]) if row[8] else {},
                    'version': row[9],
                    'checksum': row[10]
                }
                
                self.metrics['queries'] += 1
                self.metrics['operations'] += 1
                return StateEntry.from_dict(entry_data)
        
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error retrieving state entry {state_type.value}:{key} from PostgreSQL: {e}")
            return None
    
    async def delete(self, state_type: StateType, key: str) -> bool:
        """Delete state entry from PostgreSQL."""
        try:
            delete_sql = "DELETE FROM state_entries WHERE state_type = %s AND key = %s"
            
            with self.db_connection.cursor() as cursor:
                cursor.execute(delete_sql, (state_type.value, key))
                self.db_connection.commit()
                
                self.metrics['deletes'] += 1
                self.metrics['operations'] += 1
                return cursor.rowcount > 0
        
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error deleting state entry {state_type.value}:{key} from PostgreSQL: {e}")
            self.db_connection.rollback()
            return False
    
    async def cleanup_expired(self) -> int:
        """Clean up expired entries from PostgreSQL."""
        try:
            delete_sql = "DELETE FROM state_entries WHERE expires_at IS NOT NULL AND expires_at <= NOW()"
            
            with self.db_connection.cursor() as cursor:
                cursor.execute(delete_sql)
                self.db_connection.commit()
                
                cleaned = cursor.rowcount
                self.metrics['deletes'] += cleaned
                return cleaned
        
        except Exception as e:
            logger.error(f"Error cleaning up expired entries: {e}")
            self.db_connection.rollback()
            return 0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get PostgreSQL state manager metrics."""
        return self.metrics.copy()

class StateManagementSystem:
    """Main state management system coordinating Redis and PostgreSQL."""
    
    def __init__(self, redis_client=None, db_connection=None):
        self.redis_manager = RedisStateManager(redis_client) if redis_client else None
        self.postgres_manager = PostgreSQLStateManager(db_connection) if db_connection else None
        self.metrics = {
            'total_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'fallback_to_postgres': 0
        }
        
        # Start cleanup task
        self._cleanup_task = None
        if self.redis_manager or self.postgres_manager:
            self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start periodic cleanup task."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(3600)  # Run every hour
                    await self.cleanup_expired()
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def store(self, state_type: StateType, key: str, value: Any, 
                   persistence_level: PersistenceLevel = PersistenceLevel.HYBRID,
                   expires_in: Optional[timedelta] = None,
                   metadata: Dict[str, Any] = None) -> bool:
        """Store state with specified persistence level."""
        entry_id = str(uuid.uuid4())
        now = datetime.now()
        expires_at = now + expires_in if expires_in else None
        
        entry = StateEntry(
            id=entry_id,
            state_type=state_type,
            key=key,
            value=value,
            persistence_level=persistence_level,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
            metadata=metadata or {}
        )
        
        success = True
        
        # Store based on persistence level
        if persistence_level in [PersistenceLevel.MEMORY_ONLY, PersistenceLevel.HYBRID]:
            if self.redis_manager:
                redis_success = await self.redis_manager.store(entry)
                if not redis_success:
                    success = False
                    logger.warning(f"Failed to store in Redis: {key}")
        
        if persistence_level in [PersistenceLevel.PERSISTENT, PersistenceLevel.HYBRID]:
            if self.postgres_manager:
                postgres_success = await self.postgres_manager.store(entry)
                if not postgres_success:
                    success = False
                    logger.warning(f"Failed to store in PostgreSQL: {key}")
        
        self.metrics['total_operations'] += 1
        return success
    
    async def retrieve(self, state_type: StateType, key: str) -> Optional[StateEntry]:
        """Retrieve state with fallback strategy."""
        self.metrics['total_operations'] += 1
        
        # Try Redis first (faster)
        if self.redis_manager:
            entry = await self.redis_manager.retrieve(state_type, key)
            if entry:
                self.metrics['cache_hits'] += 1
                return entry
            else:
                self.metrics['cache_misses'] += 1
        
        # Fallback to PostgreSQL
        if self.postgres_manager:
            entry = await self.postgres_manager.retrieve(state_type, key)
            if entry:
                self.metrics['fallback_to_postgres'] += 1
                
                # Store back in Redis for future access
                if self.redis_manager and entry.persistence_level in [PersistenceLevel.HYBRID, PersistenceLevel.MEMORY_ONLY]:
                    await self.redis_manager.store(entry)
                
                return entry
        
        return None
    
    async def delete(self, state_type: StateType, key: str) -> bool:
        """Delete state from all storage layers."""
        redis_success = True
        postgres_success = True
        
        if self.redis_manager:
            redis_success = await self.redis_manager.delete(state_type, key)
        
        if self.postgres_manager:
            postgres_success = await self.postgres_manager.delete(state_type, key)
        
        self.metrics['total_operations'] += 1
        return redis_success and postgres_success
    
    async def store_conversation(self, conversation_id: str, messages: List[Dict[str, Any]], 
                               context: Dict[str, Any] = None) -> bool:
        """Store conversation with context."""
        conversation_data = {
            'messages': messages,
            'context': context or {},
            'message_count': len(messages),
            'last_activity': datetime.now().isoformat()
        }
        
        return await self.store(
            StateType.CONVERSATION,
            conversation_id,
            conversation_data,
            PersistenceLevel.PERSISTENT,
            metadata={'type': 'conversation', 'message_count': len(messages)}
        )
    
    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve conversation data."""
        entry = await self.retrieve(StateType.CONVERSATION, conversation_id)
        return entry.value if entry else None
    
    async def store_session(self, session_id: str, session_data: Dict[str, Any], 
                           expires_in: timedelta = timedelta(hours=24)) -> bool:
        """Store session data with expiration."""
        return await self.store(
            StateType.SESSION,
            session_id,
            session_data,
            PersistenceLevel.MEMORY_ONLY,
            expires_in=expires_in,
            metadata={'type': 'session'}
        )
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data."""
        entry = await self.retrieve(StateType.SESSION, session_id)
        return entry.value if entry else None
    
    async def cleanup_expired(self) -> Dict[str, int]:
        """Clean up expired entries from all storage layers."""
        results = {'redis': 0, 'postgres': 0}
        
        if self.redis_manager:
            results['redis'] = await self.redis_manager.cleanup_expired()
        
        if self.postgres_manager:
            results['postgres'] = await self.postgres_manager.cleanup_expired()
        
        logger.info(f"Cleanup completed: {results}")
        return results
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all components."""
        metrics = {
            'system': self.metrics.copy(),
            'redis': None,
            'postgres': None
        }
        
        if self.redis_manager:
            metrics['redis'] = self.redis_manager.get_metrics()
        
        if self.postgres_manager:
            metrics['postgres'] = self.postgres_manager.get_metrics()
        
        return metrics
    
    async def shutdown(self):
        """Gracefully shutdown the state management system."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("State management system shutdown complete")

if __name__ == "__main__":
    # Example usage
    import redis
    import psycopg2
    
    async def example_usage():
        # Initialize connections
        try:
            redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            redis_client.ping()
        except:
            redis_client = None
            logger.warning("Redis not available")
        
        try:
            db_connection = psycopg2.connect(
                host='localhost',
                database='agentAlina',
                user='postgres',
                password='password'
            )
        except:
            db_connection = None
            logger.warning("PostgreSQL not available")
        
        # Initialize state management system
        state_system = StateManagementSystem(redis_client, db_connection)
        
        # Example operations
        conversation_id = "conv_123"
        messages = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'}
        ]
        
        # Store conversation
        await state_system.store_conversation(conversation_id, messages)
        
        # Retrieve conversation
        conversation = await state_system.get_conversation(conversation_id)
        print("Retrieved conversation:", conversation)
        
        # Store session
        session_id = "session_456"
        session_data = {'user_id': 'user123', 'preferences': {'theme': 'dark'}}
        await state_system.store_session(session_id, session_data)
        
        # Get metrics
        metrics = state_system.get_comprehensive_metrics()
        print("System metrics:", json.dumps(metrics, indent=2))
        
        # Cleanup
        await state_system.shutdown()
    
    # Run example
    asyncio.run(example_usage())