#!/bin/bash

echo "Starting manual deployment of Agent Alina..."

# Stop and remove existing containers if they exist
echo "Stopping and removing existing containers..."
docker stop docker_ollama-server_1 docker_postgres_1 docker_redis_1 docker_prometheus_1 docker_grafana_1 docker_toolloader_1 docker_memory_1 docker_trading_1 docker_monitor_1 docker_optimizer_1 2>/dev/null
docker rm docker_ollama-server_1 docker_postgres_1 docker_redis_1 docker_prometheus_1 docker_grafana_1 docker_toolloader_1 docker_memory_1 docker_trading_1 docker_monitor_1 docker_optimizer_1 2>/dev/null

# Create networks if they don't exist
echo "Creating networks..."
docker network create internal 2>/dev/null
docker network create monitor-net 2>/dev/null

# Start containers manually
echo "Starting Ollama Server..."
docker run -d --name docker_ollama-server_1 -p 11434:11434 -v ollama_data:/root/.ollama -e OLLAMA_HOST=0.0.0.0 ollama/ollama:latest

echo "Starting Postgres..."
docker run -d --name docker_postgres_1 -p 5432:5432 -e POSTGRES_USER=alina -e POSTGRES_PASSWORD=securepassword -e POSTGRES_DB=knowledge_graph -v postgres_data:/var/lib/postgresql/data postgres:15

echo "Starting Redis..."
docker run -d --name docker_redis_1 -p 6379:6379 -v redis_data:/data redis:7

echo "Starting Prometheus..."
docker run -d --name docker_prometheus_1 -p 9090:9090 -v /root/AgentAlina/infra/prometheus.yml:/etc/prometheus/prometheus.yml -v /root/AgentAlina/infra/alert_rules.yml:/etc/prometheus/alert_rules.yml -v prometheus_data:/prometheus prom/prometheus:latest

echo "Starting Grafana..."
docker run -d --name docker_grafana_1 -p 3000:3000 -e GF_SECURITY_ADMIN_PASSWORD=admin123 -v grafana_data:/var/lib/grafana grafana/grafana:latest

echo "Building and starting Toolloader..."
docker build -t docker_toolloader /root/AgentAlina/services/toolloader
docker run -d --name docker_toolloader_1 --network internal docker_toolloader

echo "Building and starting Memory..."
docker build -t docker_memory /root/AgentAlina/services/memory
docker run -d --name docker_memory_1 --network internal docker_memory

echo "Building and starting Trading..."
docker build -t docker_trading /root/AgentAlina/services/trading
docker run -d --name docker_trading_1 --network internal docker_trading

echo "Building and starting Monitor..."
docker build -t docker_monitor /root/AgentAlina/services/monitor
docker run -d --name docker_monitor_1 --network internal --network monitor-net docker_monitor

echo "Building and starting Optimizer..."
docker build -t docker_optimizer /root/AgentAlina/services/optimizer
docker run -d --name docker_optimizer_1 --network internal docker_optimizer

echo "Deployment completed successfully!"
