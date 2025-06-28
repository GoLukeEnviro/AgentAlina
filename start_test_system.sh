#!/bin/bash

echo "Starting test system for Agent Alina..."

# Activate virtual environment
echo "Activating virtual environment..."
source /root/AgentAlina/.venv/bin/activate

# Start Docker containers for test services
echo "Starting Docker containers for test services..."
docker-compose -f /root/AgentAlina/test-config/docker-compose.test.yml up -d

echo "Test system started successfully!"
