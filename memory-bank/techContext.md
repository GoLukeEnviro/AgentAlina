# Technical Context for AgentAlina

## Technologies Used
- **Ollama**: A local inference server for hosting Large Language Models (LLMs) with support for GPTQ quantization (4-/8-bit) to enable efficient CPU-based inference on constrained hardware.
- **LangChain/LlamaIndex**: Frameworks for workflow orchestration and data pipeline management, facilitating structured interactions with LLMs and external data sources for enhanced response generation.
- **Agent Squad**: A lightweight, open-source framework (formerly Multi-Agent Orchestrator) for coordinating multiple agents, ensuring efficient task delegation and communication.
- **AutoGen**: A framework for implementing agentic self-optimization and meta-prompt loops, allowing the system to refine its operations based on performance feedback.
- **Docker and Docker-Compose**: Containerization technologies for deploying and managing the AgentAlina system, enforcing resource limits, CPU-pinning, and network isolation for security and consistency.
- **Postgres**: A relational database for persistent storage of the knowledge graph, maintaining long-term context and operational logs.
- **Redis**: An in-memory data store used for caching to speed up context retrieval and reduce latency in memory graph operations.
- **Prometheus and Grafana**: Monitoring tools for collecting and visualizing system metrics (latency, token costs, resource usage), enabling performance analysis and alerting.
- **CCXT**: A library for crypto-trading operations, integrated within the trading tool to execute bots and monitor profit & loss (P&L) streams.
- **Python 3.11**: The primary programming language for scripting and implementing agent logic, tool functionalities, and system orchestration.
- **Node.js and npm**: Used for frontend development and potentially for specific tool or agent implementations requiring JavaScript-based libraries.

## Development Setup
- **Hardware Environment**: The system is designed to operate on a setup with 8 vCPU and 48 GB RAM, optimized for CPU inference with quantized LLM models to manage resource constraints effectively.
- **Container Configuration**: Docker containers are configured with specific CPU shares (2048) and RAM limits (48 GB) as defined in `config.yaml`, managed via Docker-Compose for deployment.
- **Development Tools**: Utilizes standard development tools like git for version control (pending setup for code deployment), and shell scripting for automation tasks within the Linux environment (Ubuntu-based).
- **Initial Directory Structure**: Established at `/root/AgentAlina` with subdirectories for configuration (`config.yaml`), memory bank documentation (`memory-bank/`), and future directories for tools, agents, and deployment scripts.
- **Quantization and Batching**: LLM models are quantized to 4-/8-bit to reduce memory footprint and inference time, with batching and parallelization strategies to handle multiple requests efficiently using fair-share scheduling across CPU cores.

## Technical Constraints
- **Resource Limitations**: The system must operate within the bounds of 8 vCPU and 48 GB RAM, necessitating careful management of concurrency (limited to 4 parallel agents) and model size through quantization.
- **CPU-Based Inference**: Lacking GPU support, all LLM inference must be optimized for CPU execution, impacting model selection and processing speed, hence the reliance on quantized models.
- **Local Hosting**: All critical components (LLM inference, data storage) must be hosted locally to minimize latency and avoid external dependencies, aligning with the use of Ollama and self-contained Docker environments.
- **Memory Reset Characteristic**: Due to the system's design (memory resets between sessions), comprehensive documentation in the Memory Bank is critical for maintaining continuity and operational context.
- **Security Requirements**: Implementation of SSH key access, token rotation, and Docker network isolation to secure operations, requiring additional configuration and monitoring overhead.

## Dependencies
- **Core Dependencies**: Docker and Docker-Compose for containerization, Python 3.11 for scripting, Node.js/npm for potential frontend or tool development, and git for future code deployment.
- **LLM and Framework Dependencies**: Ollama for local inference, LangChain/LlamaIndex for orchestration (pending installation), Agent Squad for agent coordination (pending setup), and AutoGen for self-optimization (pending integration).
- **Storage Dependencies**: Postgres for persistent storage and Redis for caching, both to be set up within Docker containers for the memory graph.
- **Monitoring Dependencies**: Prometheus and Grafana for metrics collection and visualization, to be integrated for performance monitoring and alerting.
- **Trading Dependencies**: CCXT library for crypto-trading operations, to be installed and configured within the trading tool context.

## Tool Usage Patterns
- **Toolloader**: Dynamically loads plugins via SSH/HTTP, requiring secure access configurations and validation checks before execution to prevent malicious code injection.
- **Memory Tool**: Interfaces with Postgres and Redis to store and retrieve context, following a pattern of logging all significant actions and decisions for historical reference and recovery post-reset.
- **Trading Tool**: Utilizes CCXT for bot execution and P&L monitoring, with a pattern of real-time alert generation on threshold breaches, feeding into the orchestrator for action.
- **Monitor Tool**: Exports metrics to Prometheus/Grafana, adhering to a continuous observation pattern where latency or resource spikes trigger alerts or automatic resource adjustments.
- **Optimizer Tool**: Evaluates output quality using scoring functions, following a feedback loop pattern to refine prompts or switch models if scores fall below thresholds (e.g., < 0.7), with hourly optimization schedules.
- **Docker-Compose Deployment**: Usage pattern involves defining service units with CPU-pinning and resource limits, ensuring consistent deployment and operational environments across restarts or resets.
