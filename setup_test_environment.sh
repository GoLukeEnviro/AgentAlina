#!/bin/bash

echo "Starting setup of test environment for Agent Alina..."

# Create and activate virtual environment
echo "Creating and activating virtual environment..."
python3 -m venv /root/AgentAlina/.venv
source /root/AgentAlina/.venv/bin/activate

# Ensure pip is installed and updated in the virtual environment
echo "Ensuring pip is installed and updated..."
python3 -m ensurepip
pip install --upgrade pip

# Install test dependencies for Python
echo "Installing Python test dependencies..."
pip install pytest pytest-cov pytest-xdist tox testcontainers

# Install JavaScript test tools with a compatible Node.js version
echo "Installing JavaScript test tools..."
nvm install 20 || echo "nvm not found, continuing with current Node.js version"
npm install --save-dev jest vitest playwright cypress

# Install pre-commit for formatting and linting
echo "Installing pre-commit for formatting and linting..."
pip install pre-commit
pre-commit install || echo "pre-commit installation failed, continuing setup"

echo "Test environment setup completed successfully!"
