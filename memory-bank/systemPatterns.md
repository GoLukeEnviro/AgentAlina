# System Patterns for AgentAlina

## System Architecture
AgentAlina is designed with a modular and distributed architecture to facilitate autonomous operation, efficient resource usage, and continuous self-optimization. The architecture is structured around a central orchestrator (ALINA) that manages multiple agents, tools, and LLM inference tasks. Key components include:

- **Central Orchestrator (ALINA)**: The core component responsible for coordinating all activities, managing agent lifecycles, tool execution, and LLM calls. It operates as the decision-making hub, ensuring tasks are delegated and executed efficiently.
- **Agent Framework**: A collection of specialized agents for tasks such as crypto-trading, monitoring, and optimization. Agents operate concurrently under the orchestrator's control, with defined limits to prevent resource overutilization.
- **Tool Ecosystem**: Modular tools (e.g., toolloader, memory, trading) that can be dynamically loaded and executed. Tools are designed as plugins to extend system functionality without core modifications.
- **Memory Graph**: A persistent knowledge store using Postgres for long-term data and Redis for caching, enabling context retrieval and logging of operational states and decisions.
- **LLM Inference Layer**: Utilizes Ollama for local hosting of quantized models (4-/8-bit) to perform inference tasks efficiently on CPU-constrained hardware (8 vCPU/48 GB RAM).
- **Deployment Container**: The entire system is encapsulated in Docker containers managed by Docker-Compose, with defined CPU shares, memory limits, and network isolation for security and resource control.

## Key Technical Decisions
- **Local Inference with Ollama**: Chosen for its ability to host LLMs locally with quantized models, reducing latency and dependency on external APIs while optimizing for CPU inference.
- **Framework Selection**: Integration of LangChain/LlamaIndex for structured workflow orchestration, Agent Squad for lightweight multi-agent coordination, and AutoGen for implementing self-optimization through meta-prompt loops.
- **Quantization Strategy**: Adoption of 4-/8-bit quantization to balance model performance with hardware constraints, ensuring efficient inference on limited resources.
- **Concurrency Control**: Limiting to 4 parallel agents to prevent overloading the system, with adjustable parameters based on performance monitoring.
- **Persistent Memory**: Use of Postgres for durable storage and Redis for fast caching to maintain a knowledge graph, ensuring data persistence across system resets.
- **Docker Deployment**: Utilizing Docker-Compose for deployment to enforce resource limits, enable CPU-pinning, and provide network isolation for security.

## Design Patterns in Use
- **Modular Design**: Clear separation of system, tools, memory, and execution components to allow independent updates and scalability. Each module has defined interfaces for interaction with the orchestrator.
- **Plugin Architecture**: Tools and plugins are loaded dynamically via the toolloader, enabling extensibility without modifying the core system.
- **Feedback Loop Pattern**: Implemented for self-optimization, where output quality is evaluated, and prompts or hyperparameters are adjusted based on scoring heuristics.
- **Observer Pattern**: Used by the monitoring agent to track system metrics (latency, resource usage) and trigger alerts or adjustments through Prometheus/Grafana integration.
- **Retry and Fallback Mechanism**: Error handling pattern that retries operations on timeout (up to 2 attempts) and falls back to a safe mode on exceptions, ensuring system resilience.
- **State Persistence**: Maintaining operational state in the memory graph to recover context after resets, critical for continuity given the system's memory reset characteristic.

## Component Relationships
- **Orchestrator to Agents**: The orchestrator delegates tasks to agents, monitors their execution, and aggregates results. Agents report status and metrics back to the orchestrator for decision-making.
- **Agents to Tools**: Agents utilize tools for specific functionalities (e.g., trading agents use the trading tool for executing CCXT bots), with tools acting as reusable components across agents.
- **Tools to Memory**: Tools log actions and retrieve context from the memory graph, ensuring all operations are documented and can leverage historical data for decision-making.
- **Memory to LLM Inference**: The memory graph provides context to LLM calls, enhancing response relevance through Retrieval-Augmented Generation (RAG) before inference.
- **Monitoring to Orchestrator**: The monitoring agent feeds performance metrics and alerts to the orchestrator, which may trigger optimization actions or resource adjustments.
- **Docker to System Components**: Docker containers encapsulate all components, enforcing resource limits and providing a consistent deployment environment managed by Docker-Compose.

## Critical Implementation Paths
- **Initialization Path**: System startup involves loading the configuration (`config.yaml`), initializing the memory graph from Postgres/Redis, loading plugins via toolloader, and starting agents as per the concurrency limit (4).
- **Task Execution Path**: A task is received by the orchestrator, delegated to an appropriate agent, which uses relevant tools and LLM inference (via Ollama) with context from the memory graph, logs results, and reports back.
- **Self-Optimization Path**: Post-task, the optimizer agent evaluates output quality (score < 0.7 triggers adjustments), refines prompts or switches models, and schedules hourly reviews for continuous improvement.
- **Error Handling Path**: On error, the system logs the issue to memory, retries if applicable (up to 2 times on timeout), alerts the monitor, and falls back to safe mode if unresolved, ensuring operational continuity.
- **Monitoring Path**: Continuous metric collection (latency, token costs) by the monitor agent, exported to Prometheus/Grafana, with alerts triggering orchestrator actions for resource or performance optimization.
