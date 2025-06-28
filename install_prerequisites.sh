#!/bin/bash

echo "Installing prerequisites for test environment setup..."

# Update package list and install python3-venv
echo "Updating package list and installing python3-venv..."
apt update
apt install -y python3-venv python3-pip

# Install Node.js and npm if not already installed
echo "Installing Node.js and npm..."
apt install -y nodejs npm

echo "Prerequisites installation completed successfully!"
