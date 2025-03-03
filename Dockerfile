# Use a lightweight Python 3.10 base image
FROM python:3.10-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    wget \
    python3-numpy \
    build-essential \
    pkg-config \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA tools for nvrtc.so
RUN cd /tmp && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | sed 's/\.//')/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-nvrtc-dev-11-8 && \
    rm -rf /var/lib/apt/lists/* && \
    rm -f cuda-keyring_1.0-1_all.deb

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

# Copy project files
COPY app/ ./app/

# Set environment variables
ENV PYTHONPATH=/app
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV HF_HOME=/app/huggingface
ENV PYANNOTE_CACHE=/app/huggingface/pyannote

# Create cache directories
RUN mkdir -p /app/huggingface /app/huggingface/pyannote

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 