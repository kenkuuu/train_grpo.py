# GRPO Trainer Docker Image
# =========================

FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04

LABEL maintainer="kossisoroyce"
LABEL description="GRPO Trainer - Advanced GRPO Training Framework for LLM Fine-tuning"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Install PyTorch for CUDA 12.0 (must match host driver)
RUN pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install the package (exclude flash-attn; installed separately below)
RUN pip install -e ".[dev,deepspeed]"

# Install flash-attn separately (requires --no-build-isolation for CUDA compilation)
RUN pip install flash-attn --no-build-isolation || echo "Flash attention installation skipped"

# Create directories for outputs and data
RUN mkdir -p /app/outputs /app/data

# Set default command
CMD ["grpo-train", "--help"]

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import grpo_trainer; print('OK')" || exit 1
