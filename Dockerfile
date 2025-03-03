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
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Add NVIDIA repository - using the Ubuntu 20.04 repo which is compatible with debian bullseye
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update

# Install CUDA tools for nvrtc.so
RUN apt-get install -y --no-install-recommends \
    cuda-nvrtc-11-8 \
    libcudnn8 \
    && rm -rf /var/lib/apt/lists/*

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
# Add environment variables for diarization parameters
ENV DIARIZATION_MAX_RETRIES=3
ENV DIARIZATION_DEFAULT_MIN_SPEAKERS=1
ENV DIARIZATION_DEFAULT_MAX_SPEAKERS=8
ENV CUDA_LAUNCH_BLOCKING=1

# Create cache directories
RUN mkdir -p /app/huggingface /app/huggingface/pyannote

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 