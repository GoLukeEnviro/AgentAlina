#!/usr/bin/env python3
"""
AgentAlina Main Application
Orchestrator for all microservices and agents
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentAlinaOrchestrator:
    """Main orchestrator for AgentAlina system"""
    
    def __init__(self):
        self.services = {}
        self.agents = {}
        
    async def initialize_services(self):
        """Initialize all microservices"""
        logger.info("Initializing AgentAlina services...")
        
        # Check environment variables
        required_env_vars = [
            'POSTGRES_HOST', 'POSTGRES_PORT', 'POSTGRES_DB',
            'POSTGRES_USER', 'POSTGRES_PASSWORD',
            'REDIS_HOST', 'REDIS_PORT'
        ]
        
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            logger.warning(f"Missing environment variables: {missing_vars}")
            logger.info("Using default values for missing variables")
        
        # Initialize memory service
        try:
            from services.memory.src.main import KnowledgeGraph
            self.services['memory'] = KnowledgeGraph()
            logger.info("Memory service initialized")
        except ImportError as e:
            logger.error(f"Failed to import memory service: {e}")
        
        # Initialize agents
        await self.initialize_agents()
        
    async def initialize_agents(self):
        """Initialize all AI agents"""
        logger.info("Initializing AI agents...")
        
        agent_modules = [
            'knowledge_graph_collector',
            'monitoring_observability_system',
            'optimizer_framework_system',
            'state_management_system'
        ]
        
        for agent_name in agent_modules:
            try:
                module_path = f"agents.{agent_name}"
                module = __import__(module_path, fromlist=[agent_name])
                self.agents[agent_name] = module
                logger.info(f"Agent {agent_name} loaded")
            except ImportError as e:
                logger.error(f"Failed to load agent {agent_name}: {e}")
    
    async def health_check(self):
        """Perform health check on all services"""
        logger.info("Performing health check...")
        
        # Check database connectivity
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=os.getenv('POSTGRES_HOST', 'postgres'),
                port=os.getenv('POSTGRES_PORT', '5432'),
                user=os.getenv('POSTGRES_USER', 'agentalina_user'),
                password=os.getenv('POSTGRES_PASSWORD', 'agentalina_password'),
                database=os.getenv('POSTGRES_DB', 'agentalina')
            )
            conn.close()
            logger.info("✓ PostgreSQL connection successful")
        except Exception as e:
            logger.error(f"✗ PostgreSQL connection failed: {e}")
        
        # Check Redis connectivity
        try:
            import redis
            r = redis.Redis(
                host=os.getenv('REDIS_HOST', 'redis'),
                port=int(os.getenv('REDIS_PORT', '6379'))
            )
            r.ping()
            logger.info("✓ Redis connection successful")
        except Exception as e:
            logger.error(f"✗ Redis connection failed: {e}")
        
        return True
    
    async def start_web_server(self):
        """Start the main web server"""
        try:
            from aiohttp import web, web_runner
            
            app = web.Application()
            
            # Add routes
            app.router.add_get('/', self.handle_root)
            app.router.add_get('/health', self.handle_health)
            app.router.add_get('/status', self.handle_status)
            
            # Start server
            runner = web_runner.AppRunner(app)
            await runner.setup()
            
            site = web_runner.TCPSite(runner, '0.0.0.0', 8000)
            await site.start()
            
            logger.info("🚀 AgentAlina web server started on http://0.0.0.0:8000")
            
            # Keep the server running
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Failed to start web server: {e}")
            raise
    
    async def handle_root(self, request):
        """Handle root endpoint"""
        from aiohttp import web
        return web.json_response({
            'message': 'AgentAlina is running!',
            'version': '1.0.0',
            'services': list(self.services.keys()),
            'agents': list(self.agents.keys())
        })
    
    async def handle_health(self, request):
        """Handle health check endpoint"""
        from aiohttp import web
        healthy = await self.health_check()
        return web.json_response({'status': 'healthy' if healthy else 'unhealthy'})
    
    async def handle_status(self, request):
        """Handle status endpoint"""
        from aiohttp import web
        return web.json_response({
            'services': {
                name: 'running' for name in self.services.keys()
            },
            'agents': {
                name: 'loaded' for name in self.agents.keys()
            }
        })
    
    async def run(self):
        """Main run method"""
        logger.info("🤖 Starting AgentAlina...")
        
        # Initialize all components
        await self.initialize_services()
        
        # Perform initial health check
        await self.health_check()
        
        # Start web server
        await self.start_web_server()

async def main():
    """Main entry point"""
    orchestrator = AgentAlinaOrchestrator()
    await orchestrator.run()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("AgentAlina shutdown requested")
    except Exception as e:
        logger.error(f"AgentAlina crashed: {e}")
        sys.exit(1)