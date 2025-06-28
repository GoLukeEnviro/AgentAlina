#!/bin/bash

# Deployment script for Agent Alina
echo "Starting deployment of Agent Alina..."

# Navigate to project directory (assuming script is run from project root)
cd "$(dirname "$0")" || { echo "Project directory not found!"; exit 1; }

# Pull latest changes from repository (if applicable)
echo "Pulling latest changes from Git..."
if git pull origin main; then
    echo "Git pull successful."
else
    echo "Failed to pull latest changes or not a git repository. Continuing anyway..."
fi

# Pull latest Docker images
echo "Pulling latest Docker images..."
docker-compose -f ./docker/docker-compose.yml pull || { echo "Failed to pull Docker images!"; exit 1; }

# Start services with Docker Compose
echo "Starting services with Docker Compose..."
docker-compose -f ./docker/docker-compose.yml up -d --remove-orphans || { echo "Failed to start services!"; exit 1; }

# Run post-deployment smoke tests
echo "Running post-deployment smoke tests..."
# Placeholder for smoke tests (expand as needed)
docker-compose -f ./docker/docker-compose.yml ps || { echo "Failed to list running services!"; exit 1; }

echo "Deployment completed successfully!"
