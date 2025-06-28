#!/bin/bash
set -e

# Activate virtual environment
source /app/venv/bin/activate

# Run database migrations if needed
# python manage.py migrate

# Collect static files if needed
# python manage.py collectstatic --noinput

# Additional setup commands can be added here
echo "Starting AgentAlina application..."

# Execute the command passed to docker run
exec "$@"
