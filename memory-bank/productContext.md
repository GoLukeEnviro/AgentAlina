# Product Context for AgentAlina

## Why AgentAlina Exists
AgentAlina is developed to serve as a central orchestrator for managing distributed agents, tools, and LLM (Large Language Model) calls, with a specific focus on crypto-trading operations. The system addresses the need for an autonomous, efficient, and self-optimizing framework that can operate on constrained hardware while maintaining high performance and reliability. It aims to streamline complex workflows involving multiple agents and tools, ensuring seamless integration and operation without constant human oversight.

## Problems It Solves
- **Complexity Management**: Handling multiple agents, tools, and LLM interactions in a coordinated manner, reducing the complexity of managing disparate systems.
- **Resource Efficiency**: Operating effectively on limited hardware (8 vCPU/48 GB RAM) by using quantized models and optimized inference techniques, thus solving the problem of high resource demands typical of LLM systems.
- **Performance Optimization**: Continuously improving system performance through self-optimization loops that adjust prompts and hyperparameters based on output quality and resource usage metrics.
- **Autonomous Operation**: Enabling a hands-off approach to crypto-trading and other agent-based tasks by providing robust error handling and autonomous decision-making capabilities.
- **Data and Context Persistence**: Maintaining a persistent knowledge graph for context retrieval and status logging, ensuring continuity across sessions and system resets.

## How It Should Work
AgentAlina operates as a modular system with clear separation of concerns:
- **System Core**: Acts as the central orchestrator (ALINA), managing the lifecycle of agents, tools, and LLM calls.
- **Tool Integration**: Dynamically loads and executes plugins for various functionalities like trading, monitoring, and optimization.
- **Memory Management**: Utilizes a persistent storage system (Postgres with Redis caching) to maintain a knowledge graph for context and historical data.
- **Execution Control**: Manages concurrency, CPU shares, and RAM limits within a Docker environment to ensure efficient resource allocation.
- **Self-Optimization**: Implements feedback loops to evaluate output quality, adjust operational parameters, and monitor system metrics for continuous improvement.
- **Error Handling**: Provides robust mechanisms to retry operations, log issues, and fall back to safe modes during failures.

The system initializes by loading plugins, setting up the memory graph, and starting operational agents like trading bots. It continuously monitors performance and adjusts based on predefined criteria and real-time feedback.

## User Experience Goals
- **Transparency**: Users should have clear visibility into the system's operations through detailed logging and monitoring metrics exported to Prometheus/Grafana.
- **Minimal Intervention**: The system should require minimal user input after initial setup, operating autonomously with self-optimization and error recovery mechanisms.
- **Reliability**: Users should trust that AgentAlina maintains operational stability, handles errors gracefully, and delivers consistent performance in crypto-trading and other tasks.
- **Ease of Extension**: The modular design should allow users to easily add new tools or agents by updating configurations or loading new plugins without deep system knowledge.
- **Documentation and Continuity**: Comprehensive documentation within the Memory Bank ensures that users (and the system itself post-reset) can quickly understand the project state, goals, and operational patterns.
