import psycopg2
import redis
import aiohttp
import asyncio
import logging
import os
from psycopg2.extras import Json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_USER = os.getenv("POSTGRES_USER", "alina")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "securepassword")
POSTGRES_DB = os.getenv("POSTGRES_DB", "knowledge_graph")

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")

# Initialize connections
def get_db_connection():
    """Create a connection to PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database=POSTGRES_DB
        )
        return conn
    except Exception as e:
        logger.error(f"Error connecting to PostgreSQL: {e}")
        return None

def get_redis_client():
    """Create a connection to Redis cache."""
    try:
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        return client
    except Exception as e:
        logger.error(f"Error connecting to Redis: {e}")
        return None

class KnowledgeGraph:
    def __init__(self):
        self.db_conn = get_db_connection()
        self.redis_client = get_redis_client()
        self.create_tables()

    def create_tables(self):
        """Create necessary tables if they don't exist."""
        if not self.db_conn:
            return
        
        try:
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS nodes (
                        id SERIAL PRIMARY KEY,
                        name TEXT NOT NULL,
                        type TEXT NOT NULL,
                        data JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE TABLE IF NOT EXISTS edges (
                        id SERIAL PRIMARY KEY,
                        source_id INTEGER REFERENCES nodes(id),
                        target_id INTEGER REFERENCES nodes(id),
                        relation TEXT NOT NULL,
                        data JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                self.db_conn.commit()
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            self.db_conn.rollback()

    def add_node(self, name, node_type, data=None):
        """Add a node to the knowledge graph."""
        if not self.db_conn:
            return None
        
        try:
            with self.db_conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO nodes (name, type, data) VALUES (%s, %s, %s) RETURNING id",
                    (name, node_type, Json(data) if data else None)
                )
                node_id = cur.fetchone()[0]
                self.db_conn.commit()
                logger.info(f"Added node {name} with ID {node_id}")
                return node_id
        except Exception as e:
            logger.error(f"Error adding node {name}: {e}")
            self.db_conn.rollback()
            return None

    def add_edge(self, source_id, target_id, relation, data=None):
        """Add an edge between two nodes in the knowledge graph."""
        if not self.db_conn:
            return None
        
        try:
            with self.db_conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO edges (source_id, target_id, relation, data) VALUES (%s, %s, %s, %s) RETURNING id",
                    (source_id, target_id, relation, Json(data) if data else None)
                )
                edge_id = cur.fetchone()[0]
                self.db_conn.commit()
                logger.info(f"Added edge {source_id} -> {target_id} with ID {edge_id}")
                return edge_id
        except Exception as e:
            logger.error(f"Error adding edge {source_id} -> {target_id}: {e}")
            self.db_conn.rollback()
            return None

    def get_context(self, node_id, depth=1):
        """Retrieve context for a node up to a specified depth of relationships."""
        if not self.db_conn:
            return None
        
        try:
            with self.db_conn.cursor() as cur:
                # Simple context retrieval (expand as needed)
                cur.execute("SELECT * FROM nodes WHERE id = %s", (node_id,))
                node = cur.fetchone()
                return node
        except Exception as e:
            logger.error(f"Error retrieving context for node {node_id}: {e}")
            return None

    def cache_context(self, key, context_data, ttl=3600):
        """Cache context data in Redis with a time-to-live."""
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.setex(key, ttl, str(context_data))
            logger.info(f"Cached context with key {key}")
            return True
        except Exception as e:
            logger.error(f"Error caching context with key {key}: {e}")
            return False

def main():
    """Main function to run the memory service."""
    logger.info("Starting Memory Service...")
    kg = KnowledgeGraph()
    
    if not kg.db_conn or not kg.redis_client:
        logger.error("Failed to initialize database or cache. Exiting.")
        return
    
    logger.info("Memory Service initialized. Listening for requests...")
    
    # Placeholder for HTTP server or other request handling mechanism
    while True:
        # Simulate service running
        asyncio.sleep(60)

if __name__ == "__main__":
    main()
