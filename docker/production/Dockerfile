# Base image with Python and necessary tools
FROM python:3.10-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    APP_HOME=/app \
    VENV_PATH=/app/venv

# Set working directory
WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    automake \
    git-core \
    autoconf \
    make \
    patch \
    cmake \
    libssl-dev \
    grep \
    binutils \
    libc6 \
    libbz2-dev \
    subversion \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Build stage for compiling dependencies if needed
FROM base AS builder
WORKDIR $APP_HOME
# Add here any build steps for custom dependencies if needed

# Final stage
FROM base AS final
WORKDIR $APP_HOME

# Copy application code
COPY . .

# Copy built dependencies from builder stage if any
COPY --from=builder $APP_HOME $APP_HOME

# Create a non-root user for security
RUN useradd -m -u 1000 appuser
USER appuser

# Expose port for the application
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["python", "healthcheck.py"]

# Entrypoint script
COPY docker/production/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Command to run the application
CMD ["python", "main.py"]
