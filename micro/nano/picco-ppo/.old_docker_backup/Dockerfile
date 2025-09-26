# Lightweight Dockerfile for PPO Trading
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install \
    numpy==1.24.3 \
    pandas==2.1.1 \
    duckdb==0.9.1 \
    matplotlib==3.7.2 \
    tqdm==4.66.1

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copy application code
COPY . /app

# Create necessary directories
RUN mkdir -p /app/models /app/tensorboard /app/logs /app/results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TF_CPP_MIN_LOG_LEVEL=2

# Default command
CMD ["python", "train.py"]