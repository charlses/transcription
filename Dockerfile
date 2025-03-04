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
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch separately to avoid timeouts
RUN pip install torch==2.1.1 torchaudio==2.1.1 --extra-index-url https://download.pytorch.org/whl/cu118

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies in separate commands for better caching and reduced timeout risk
RUN pip install "fastapi==0.104.1" "uvicorn==0.24.0" "python-multipart==0.0.6" "python-dotenv==1.0.0"
RUN pip install "numpy==1.26.2" "scipy>=1.10.0" "requests==2.31.0" "omegaconf>=2.3.0" "protobuf>=3.20.0"
RUN pip install "ffmpeg-python>=0.2.0" "huggingface_hub>=0.12.0" "transformers>=4.27.0" "torchmetrics>=0.11.4"

# Install WhisperX and dependencies
RUN pip install "openai-whisper==20231117" "faster-whisper==1.1.0" "whisperx==3.3.1" "deepmultilingualpunctuation" "ctranslate2>=3.17.1"

# Install audio-related dependencies
RUN pip install "pyannote.audio>=3.1.1" "pyannote.core>=5.0.0" "silero-vad>=1.0.0" "pyparsing>=3.1.1"

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
# Initialize WhisperX models if needed\n\
if [ ! -d "/app/huggingface/hub/models--guillaumekln--faster-whisper-large-v3" ]; then\n\
  echo "First run: Pre-downloading WhisperX models..."\n\
  # Get token from environment\n\
  if [ -n "$HF_TOKEN" ]; then\n\
    # Pre-download WhisperX models to avoid on-demand downloads\n\
    python -c "\
import os\n\
import whisperx\n\
print(\"Pre-downloading WhisperX models...\")\n\
try:\n\
    print(\"Downloading ASR model...\")\n\
    whisperx.load_model(\"large-v3\", \"cpu\", compute_type=\"int8\")\n\
    print(\"Downloading alignment model...\")\n\
    whisperx.load_align_model(language_code=\"de\", device=\"cpu\")\n\
    print(\"WhisperX models downloaded successfully\")\n\
except Exception as e:\n\
    print(f\"Error downloading WhisperX models: {e}\")\n\
"\n\
  else\n\
    echo "WARNING: HF_TOKEN not set. WhisperX model download may fail."\n\
  fi\n\
else\n\
  echo "WhisperX models already downloaded."\n\
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
ENV WHISPERX_MODEL_SIZE=large-v3
ENV WHISPERX_LANGUAGE=de
ENV WHISPERX_COMPUTE_TYPE=float16

# Create necessary directories
RUN mkdir -p /app/huggingface /app/pyannote_cache

# Expose FastAPI port
EXPOSE 8000

# Use our entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]

# Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 