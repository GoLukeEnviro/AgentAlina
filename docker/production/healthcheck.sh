#!/bin/bash
set -e

# Check if the application is running and responding to health checks
if ! curl -f http://localhost:8000/health; then
    echo "Health check failed"
    exit 1
fi

echo "Health check passed"
exit 0
