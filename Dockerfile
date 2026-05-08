FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/

# Install the project and dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Create data and checkpoint directories
RUN mkdir -p /app/data /app/checkpoints

# Default command runs the full pipeline
CMD ["bash", "scripts/run_pipeline.sh"]
