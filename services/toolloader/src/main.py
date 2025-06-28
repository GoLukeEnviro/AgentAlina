import requests
import aiohttp
import asyncio
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MCP client for plugin-mcp
MCP_PLUGIN_ENDPOINT = "http://plugin-mcp:8004"

async def fetch_plugins():
    """Fetch available plugins from MCP plugin server."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{MCP_PLUGIN_ENDPOINT}/list_plugins") as response:
                if response.status == 200:
                    plugins = await response.json()
                    logger.info(f"Available plugins: {plugins}")
                    return plugins
                else:
                    logger.error(f"Failed to fetch plugins: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching plugins: {e}")
            return []

async def install_plugin(plugin_name, version="latest"):
    """Install a plugin using MCP plugin server."""
    async with aiohttp.ClientSession() as session:
        payload = {"name": plugin_name, "version": version}
        try:
            async with session.post(f"{MCP_PLUGIN_ENDPOINT}/install_plugin", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Installed plugin {plugin_name}: {result}")
                    return result
                else:
                    logger.error(f"Failed to install plugin {plugin_name}: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error installing plugin {plugin_name}: {e}")
            return None

def main():
    """Main function to run the toolloader service."""
    logger.info("Starting Toolloader Service...")
    
    # Example: Fetch and install plugins on startup
    loop = asyncio.get_event_loop()
    plugins = loop.run_until_complete(fetch_plugins())
    
    for plugin in plugins:
        if plugin.get("auto_install", False):
            loop.run_until_complete(install_plugin(plugin["name"]))
    
    logger.info("Toolloader Service initialized. Listening for requests...")
    
    # Placeholder for HTTP server or other request handling mechanism
    while True:
        # Simulate service running
        asyncio.sleep(60)

if __name__ == "__main__":
    main()
