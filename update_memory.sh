#!/bin/bash

echo "Starting update of Memory service for Agent Alina..."

# Stop and remove existing Memory container if it exists
echo "Stopping and removing existing Memory container..."
docker stop docker_memory_1 2>/dev/null
docker rm docker_memory_1 2>/dev/null

# Rebuild the Memory image
echo "Building Memory service..."
docker build -t docker_memory /root/AgentAlina/services/memory

# Start the Memory container
echo "Starting Memory service..."
docker run -d --name docker_memory_1 --network internal docker_memory

echo "Memory service update completed successfully!"
