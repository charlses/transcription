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
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

# Copy project files
COPY app/ ./app/

# Create a script to initialize models at startup for better reliability
RUN echo '#!/bin/bash\n\
# Create cache directories\n\
mkdir -p /app/huggingface /app/pyannote_cache\n\
\n\
# Check if model exists to prevent re-downloading on every restart\n\
if [ ! -d "/app/huggingface/hub/models--pyannote--speaker-diarization-3.1" ]; then\n\
  echo "First run: Downloading diarization model..."\n\
  # Get token from environment\n\
  if [ -n "$HF_TOKEN" ]; then\n\
    # Create token file for non-interactive auth\n\
    mkdir -p /root/.cache/huggingface\n\
    echo $HF_TOKEN > /root/.cache/huggingface/token\n\
    \n\
    # Download the model using explicit token auth to avoid interactive prompts\n\
    python -c "\
import os\n\
from pyannote.audio import Pipeline\n\
print(\"Authenticating with HuggingFace token...\")\n\
token = os.environ.get(\"HF_TOKEN\")\n\
try:\n\
    print(\"Attempting to download speaker diarization model...\")\n\
    pipeline = Pipeline.from_pretrained(\"pyannote/speaker-diarization-3.1\", use_auth_token=token)\n\
    print(\"Diarization model downloaded successfully\")\n\
except Exception as e:\n\
    print(f\"Error downloading model: {e}\")\n\
    print(\"Please ensure you have accepted the user conditions at:\")\n\
    print(\"https://huggingface.co/pyannote/speaker-diarization-3.1\")\n\
    print(\"https://huggingface.co/pyannote/segmentation-3.0\")\n\
"\n\
  else\n\
    echo "WARNING: HF_TOKEN not set. Model download will fail."\n\
  fi\n\
else\n\
  echo "Diarization model already downloaded."\n\
fi\n\
\n\
# Start the application\n\
exec "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV HF_HOME=/app/huggingface
ENV TRANSFORMERS_CACHE=/app/huggingface/transformers
ENV PYANNOTE_CACHE=/app/pyannote_cache

# Create necessary directories
RUN mkdir -p /app/huggingface /app/pyannote_cache

# Expose FastAPI port
EXPOSE 8000

# Use our entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]

# Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 