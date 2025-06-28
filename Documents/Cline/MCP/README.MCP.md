# AgentAlina MCP Server

A comprehensive Model Context Protocol (MCP) server implementation for AgentAlina, providing integrated tools for plugin management, memory operations, trading automation, system monitoring, and performance optimization.

## 🚀 Features

### Core Tools
- **Toolloader**: Dynamic plugin loading and execution
- **Memory**: Knowledge graph storage with PostgreSQL and Redis caching
- **Trading**: Automated trading bot management and threshold control
- **Monitor**: System metrics collection and resource limit management
- **Optimizer**: Performance evaluation and optimization recommendations

### Architecture
- **Protocol**: MCP (Model Context Protocol) via stdio
- **Database**: PostgreSQL for persistent storage
- **Cache**: Redis for real-time data and session management
- **Integration**: Seamless connection with existing AgentAlina services

## 📋 Prerequisites

- Docker and Docker Compose
- Python 3.8+
- PostgreSQL 15+
- Redis 7+

## 🛠 Installation & Setup

### 1. Clone and Navigate
```bash
cd /root/AgentAlina/services/alina-mcp
```

### 2. Build and Start Services
```bash
docker-compose up --build -d
```

### 3. Verify Installation
```bash
docker-compose logs alina-mcp
```

## 🧪 Testing

### Run Test Suite
```bash
python3 test_server.py
```

### Manual Testing
```bash
# Start the server
python3 src/server.py

# Send test requests via stdin
echo '{"method": "list_tools"}' | python3 src/server.py
```

## Verwendung

Der MCP Server kommuniziert über stdio mit dem JSON-RPC 2.0 Protokoll.

### Verfügbare Tools

#### 1. Toolloader
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "call_tool",
  "params": {
    "name": "toolloader",
    "arguments": {
      "plugin_url": "https://example.com/plugin.py",
      "execution_params": {}
    }
  }
}
```

#### 2. Memory
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "call_tool",
  "params": {
    "name": "memory",
    "arguments": {
      "action": "store",
      "key": "user_preference",
      "value": {"theme": "dark", "language": "de"}
    }
  }
}
```

#### 3. Trading
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "call_tool",
  "params": {
    "name": "trading",
    "arguments": {
      "bot_id": "btc_scalper",
      "action": "start"
    }
  }
}
```

#### 4. Monitor
```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "call_tool",
  "params": {
    "name": "monitor",
    "arguments": {
      "metric_name": "cpu_usage",
      "metric_value": 75.5
    }
  }
}
```

#### 5. Optimizer
```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "call_tool",
  "params": {
    "name": "optimizer",
    "arguments": {
      "output_id": "response_123",
      "score": 0.85,
      "adjustments": {
        "temperature": 0.7,
        "max_tokens": 1000
      }
    }
  }
}
```

## Integration mit AgentAlina

Der MCP Server ist Teil des AgentAlina Docker-Compose Setups und wird automatisch mit den anderen Services gestartet.

```yaml
# In docker-compose.yml
alina-mcp:
  build: ./services/alina-mcp
  container_name: alina-mcp-server
  networks:
    - alina-network
  volumes:
    - ./logs:/app/logs
  environment:
    - LOG_LEVEL=INFO
```

## Entwicklung

### Tests ausführen
```bash
python3 -m pytest tests/
```

### Logging
Der Server loggt alle Aktivitäten nach stderr. Log-Level kann über die Umgebungsvariable `LOG_LEVEL` gesteuert werden.

## Architektur

Der MCP Server implementiert das Model Context Protocol und stellt eine einheitliche Schnittstelle für verschiedene AgentAlina-Funktionalitäten bereit:

```
┌─────────────────┐
│   MCP Client    │
│  (AgentAlina)   │
└─────────┬───────┘
          │ JSON-RPC 2.0
          │ über stdio
┌─────────▼───────┐
│  Alina MCP      │
│    Server       │
├─────────────────┤
│ • Toolloader    │
│ • Memory        │
│ • Trading       │
│ • Monitor       │
│ • Optimizer     │
└─────────────────┘
```

## Lizenz

MIT License - siehe LICENSE Datei für Details.