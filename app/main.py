import os
import uuid
import tempfile
import shutil
import torch
import subprocess
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from .whisperx_transcription import WhisperXTranscription
from .whisperx_config_transcription import WhisperXConfigurableTranscription, TranscriptionConfig
import aiohttp
import asyncio
import logging
import time
import platform
import sys
import fastapi
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("transcription")

# Create output directory for processed files
OUTPUT_DIR = Path("/app/output")  # This will be mounted to your local machine
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Whisper Transcription API",
    description="API for transcribing audio using OpenAI's Whisper model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize WhisperX transcription handler
cuda_available = torch.cuda.is_available()
if cuda_available:
    logger.info("CUDA is available. Initializing WhisperX with GPU acceleration.")
else:
    logger.warning("CUDA is not available! WhisperX will run much slower on CPU.")
    
whisperx_transcriber = WhisperXTranscription(use_gpu=True)  # Always request GPU, will fallback to CPU if not available

# Initialize configurable WhisperX transcription handler
configurable_whisperx_transcriber = WhisperXConfigurableTranscription(use_gpu=True)  # Use GPU if available, fallback to CPU

# Load models before starting the server
logger.info("Loading WhisperX models...")
try:
    whisperx_transcriber._load_model()  # Call the correct method name
    logger.info("WhisperX models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load WhisperX models: {str(e)}", exc_info=True)
    raise RuntimeError("Failed to load WhisperX models. Check logs for details.")

# Verify CUDA availability at startup
CUDA_AVAILABLE = cuda_available
if CUDA_AVAILABLE:
    logger.info(f"CUDA is available. Found device: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("CUDA is not available. Running on CPU mode.")

class TranscriptionRequest(BaseModel):
    recording_remote: str
    recording_local: str

class ConfigurableTranscriptionRequest(BaseModel):
    recording_remote: str
    recording_local: str
    config: TranscriptionConfig

async def download_file(url: str, path: str):
    """Download a file from URL to local path"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download file from {url}")
            
            with open(path, 'wb') as f:
                while True:
                    chunk = await response.content.read(1024)
                    if not chunk:
                        break
                    f.write(chunk)

async def process_audio_file(input_path: str, output_path: str) -> str:
    """
    Process audio file using ffmpeg to ensure correct format and headers.
    Converts the input file to MP3 format with standard parameters.
    
    Args:
        input_path: Path to input audio file
        output_path: Path where processed file should be saved
        
    Returns:
        Path to the processed file
    """
    try:
        # FFmpeg command to process the audio file
        # -y: Overwrite output file if it exists
        # -i: Input file
        # -acodec libmp3lame: Use LAME MP3 encoder
        # -b:a 128k: Set bitrate to 128kbps (good quality for speech)
        # -ar 16000: Set sample rate to 16kHz (good for speech)
        # -ac 1: Convert to mono
        # -q:a 2: Set quality to high (0-9, where 0 is highest)
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-acodec', 'libmp3lame',
            '-b:a', '128k',
            '-ar', '16000',
            '-ac', '1',
            '-q:a', '2',
            output_path
        ]
        
        # Run ffmpeg command
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise Exception(f"FFmpeg processing failed: {error_msg}")
            
        return output_path
        
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing audio file: {str(e)}")

async def save_processed_file(processed_path: str, original_filename: str) -> str:
    """
    Save the processed file to the output directory with a descriptive name.
    
    Args:
        processed_path: Path to the processed file
        original_filename: Original filename for reference
        
    Returns:
        Path to the saved file
    """
    try:
        # Create a descriptive filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{Path(original_filename).stem}_processed.mp3"
        output_path = OUTPUT_DIR / filename
        
        # Copy the file to the output directory
        shutil.copy2(processed_path, output_path)
        logger.info(f"Saved processed file to: {output_path}")
        
        return str(output_path)
    except Exception as e:
        logger.error(f"Error saving processed file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error saving processed file: {str(e)}")

@app.post("/transcribe")
async def transcribe(
    background_tasks: BackgroundTasks,
    request: TranscriptionRequest
):
    """
    Transcribe two audio files from provided URLs
    """
    # Create a unique job ID
    job_id = str(uuid.uuid4())
    
    # Create temp directory for this job
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Define paths for downloaded files
        remote_path = os.path.join(temp_dir, f"recording_remote_{job_id}.webm")
        local_path = os.path.join(temp_dir, f"recording_local_{job_id}.webm")
        
        # Download files from URLs
        logger.info(f"Downloading remote recording to {remote_path}")
        await download_file(request.recording_remote, remote_path)
        
        logger.info(f"Downloading local recording to {local_path}")
        await download_file(request.recording_local, local_path)
        
        # Process the audio files
        processed_remote_path = os.path.join(temp_dir, f"processed_remote_{job_id}.wav")
        processed_local_path = os.path.join(temp_dir, f"processed_local_{job_id}.wav")
        
        logger.info("Processing remote recording")
        await process_audio_file(remote_path, processed_remote_path)
        
        logger.info("Processing local recording")
        await process_audio_file(local_path, processed_local_path)
        
        # Transcribe both recordings
        logger.info("Starting remote recording transcription")
        transcript_remote = await whisperx_transcriber.transcribe_audio(processed_remote_path, "client")
        
        logger.info("Starting local recording transcription")
        transcript_local = await whisperx_transcriber.transcribe_audio(processed_local_path, "agent")
        
        # Combine and sort segments by start time
        all_segments = []
        
        # Add remote segments with client tag
        for segment in transcript_remote["segments"]:
            segment["speaker"] = "client"
            all_segments.append(segment)
            
        # Add local segments with agent tag
        for segment in transcript_local["segments"]:
            segment["speaker"] = "agent"
            all_segments.append(segment)
        
        # Sort segments by start time
        all_segments.sort(key=lambda x: x["start"])
        
        # Create combined transcript text
        combined_text = " ".join(segment["text"] for segment in all_segments)
        
        # Cleanup temp files in the background
        background_tasks.add_task(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        
        return {
            "transcript_remote": transcript_remote,
            "transcript_local": transcript_local,
            "transcript_combined": {
                "segments": all_segments,
                "text": combined_text
            }
        }
        
    except Exception as e:
        # Ensure temp directory is cleaned up even if an error occurs
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/transcribe/configurable")
async def transcribe_configurable(
    background_tasks: BackgroundTasks,
    request: ConfigurableTranscriptionRequest
):
    """
    Transcribe two audio files with configurable WhisperX parameters
    
    This endpoint allows for customizing the WhisperX transcription parameters
    including temperature, VAD settings, beam size, and more.
    """
    # Create a unique job ID
    job_id = str(uuid.uuid4())
    
    # Create temp directory for this job
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Define paths for downloaded files
        remote_path = os.path.join(temp_dir, f"recording_remote_{job_id}.webm")
        local_path = os.path.join(temp_dir, f"recording_local_{job_id}.webm")
        
        # Download files from URLs
        logger.info(f"Downloading remote recording to {remote_path}")
        await download_file(request.recording_remote, remote_path)
        
        logger.info(f"Downloading local recording to {local_path}")
        await download_file(request.recording_local, local_path)
        
        # Process the audio files with ffmpeg
        processed_remote_path = os.path.join(temp_dir, f"processed_remote_{job_id}.mp3")
        processed_local_path = os.path.join(temp_dir, f"processed_local_{job_id}.mp3")
        
        logger.info("Processing remote recording with ffmpeg")
        await process_audio_file(remote_path, processed_remote_path)
        
        logger.info("Processing local recording with ffmpeg")
        await process_audio_file(local_path, processed_local_path)
        
        # Save processed files to output directory
        saved_remote_path = await save_processed_file(processed_remote_path, f"remote_{job_id}.webm")
        saved_local_path = await save_processed_file(processed_local_path, f"local_{job_id}.webm")
        
        # Log configuration settings
        logger.info(f"Using custom transcription configuration: language={request.config.language}, "
                   f"temperature={request.config.temperature}, vad_filter={request.config.vad_filter}")
        
        # Transcribe both recordings with the provided configuration
        result = await configurable_whisperx_transcriber.transcribe_audio_combined(
            processed_local_path, 
            processed_remote_path, 
            request.config
        )
        
        # Add saved file paths to the result
        result["processed_files"] = {
            "remote": saved_remote_path,
            "local": saved_local_path
        }
        
        # Cleanup temp files in the background
        background_tasks.add_task(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        
        return result
        
    except Exception as e:
        # Ensure temp directory is cleaned up even if an error occurs
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.error(f"Error processing audio with custom configuration: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    """Root endpoint to check API status"""
    logger.info("Status check requested")
    
    # Collect system information
    system_info = {
        "status": "online",
        "cuda_available": CUDA_AVAILABLE,
        "version": "1.0.0",
        "timestamp": time.time()
    }
    
    # Add GPU information if available
    if CUDA_AVAILABLE:
        try:
            gpu_info = {
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device()
            }
            
            # Get memory information
            gpu_info["total_memory"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gpu_info["allocated_memory"] = torch.cuda.memory_allocated(0) / (1024 ** 3)
            gpu_info["cached_memory"] = torch.cuda.memory_reserved(0) / (1024 ** 3)
            
            system_info["gpu"] = gpu_info
            logger.info(f"Status check - GPU info: {gpu_info['device_name']}, "
                       f"Memory: {gpu_info['allocated_memory']:.2f}GB / {gpu_info['total_memory']:.2f}GB")
        except Exception as e:
            logger.warning(f"Failed to get detailed GPU information: {e}")
            system_info["gpu_error"] = str(e)
    
    # Add Python and package versions
    try:
        versions = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "fastapi": fastapi.__version__
        }
        
        try:
            import whisper
            versions["whisper"] = whisper.__version__
        except (ImportError, AttributeError):
            versions["whisper"] = "unknown"
            
        try:
            import pyannote
            versions["pyannote"] = pyannote.__version__
        except (ImportError, AttributeError):
            versions["pyannote"] = "unknown"
            
        system_info["versions"] = versions
    except Exception as e:
        logger.warning(f"Failed to get version information: {e}")
    
    logger.info("Status check completed")
    return system_info 