FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir prometheus-client aiohttp requests

# Copy source code
COPY src/ .

# Command to run the service
CMD ["python", "main.py"]
