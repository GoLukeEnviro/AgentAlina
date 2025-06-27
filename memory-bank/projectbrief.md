# Project Brief for AgentAlina

## Overview
AgentAlina is a central orchestrator for distributed agents, tools, LLM calls, and crypto-trading. The system is designed to operate autonomously on an 8 vCPU/48 GB RAM setup using Ollama models with quantized LLMs for efficient CPU inference. It dynamically loads plugins and implements self-optimization loops to continuously improve performance by evaluating quality, adjusting prompts, and monitoring resources.

## Core Requirements
- **Local LLM Hosting**: Utilize Ollama as a locally hosted inference server supporting GPTQ-quantized models for CPU efficiency.
- **Framework Integration**: Incorporate LangChain/LlamaIndex for workflow orchestration, Agent Squad for agent coordination, and AutoGen for self-optimization and meta-prompt loops.
- **Hardware Optimization**: Implement 4-/8-bit model quantization, batching, parallelization, and container tuning with Docker resource limits.
- **Self-Optimization**: Include feedback loops for quality assessment, monitoring agents for performance metrics, and dynamic retrieval for RAG (Retrieval-Augmented Generation).
- **Modular Agent Structure**: Define clear system, tool, and memory sections for each agent, with robust error handling and resource management.
- **Deployment**: Use Docker-Compose for deployment with CPU-pinning, GitHub Actions for CI/CD, and ensure security through SSH key access, token rotation, and network isolation.

## Goals
- **Autonomous Operation**: Ensure AgentAlina can operate independently with minimal human intervention.
- **Performance Efficiency**: Optimize resource usage for CPU-based inference on constrained hardware.
- **Continuous Improvement**: Establish mechanisms for self-optimization to enhance output quality over time.
- **Robustness**: Build a fault-tolerant system capable of handling errors gracefully and maintaining operational stability.
- **Documentation**: Maintain a comprehensive Memory Bank to ensure continuity and clarity across sessions and resets.

## Scope
This project focuses on setting up the infrastructure for AgentAlina, integrating specified frameworks, and establishing the initial configuration for tools, memory, and execution parameters. Future expansions may include additional plugins, trading strategies, and advanced optimization techniques.
