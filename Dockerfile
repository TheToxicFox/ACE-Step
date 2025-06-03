#Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8006 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    curl \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ENV Music_URL=http://127.0.0.1:8006

# Set working directory
WORKDIR /app

COPY . /app

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir hf_transfer peft && \
    pip3 install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
RUN pip3 install --no-cache-dir .

# Ensure target directories for volumes exist and have correct initial ownership
RUN mkdir -p /app/outputs /app/checkpoints  /app/logs

# Expose the port the app runs on
EXPOSE 8006

VOLUME [ "/app/checkpoints", "/app/outputs", "/app/logs" ]

# Command to run the application with GPU support
CMD ["python3", "infer-api.py", "--host", "0.0.0.0", "--port", "8006"]
