FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python packages
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    duckdb \
    matplotlib

# Copy the nano directory
COPY . /app/

# Create output directory
RUN mkdir -p /app/output

CMD ["python3", "swing_state_tracker.py"]
