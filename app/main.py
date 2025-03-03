from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import torch
from .utils import download_audio, transcribe_audio
import logging
from pydantic import BaseModel
from typing import List, Optional, Dict
import uuid
import time
import platform
import sys
import fastapi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define request and response models
class TranscriptionRequest(BaseModel):
    lead_id: str
    call_id: str
    email: str
    email_2: Optional[str] = None
    email_3: Optional[str] = None
    email_4: Optional[str] = None
    audio_url: str
    diarize: bool = True
    num_speakers: Optional[int] = None

class TranscriptionResponse(BaseModel):
    lead_id: str
    call_id: str
    transcript_full: str
    transcript_segments: List[Dict]
    email: str
    email_2: Optional[str] = None
    email_3: Optional[str] = None
    email_4: Optional[str] = None
    audio_url: str
    num_speakers: Optional[int] = None

# Initialize FastAPI app
app = FastAPI(
    title="Whisper Transcription API",
    description="API for transcribing audio using OpenAI's Whisper model",
    version="1.0.0"
)

# Verify CUDA availability at startup
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    logger.info(f"CUDA is available. Found device: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("CUDA is not available. Running on CPU mode.")

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

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(request: TranscriptionRequest):
    """
    Transcribe audio from the provided request data
    
    Args:
        request (TranscriptionRequest): Request containing lead and audio information
    
    Returns:
        TranscriptionResponse: Response containing transcription and original data
    """
    request_id = str(uuid.uuid4())[:8]  # Generate a short request ID for tracking
    logger.info(f"[{request_id}] Received transcription request: lead_id={request.lead_id}, call_id={request.call_id}")
    logger.info(f"[{request_id}] Audio URL: {request.audio_url}")
    logger.info(f"[{request_id}] Diarization enabled: {request.diarize}, Num speakers: {request.num_speakers}")
    
    start_time = time.time()
    
    try:
        # Download the audio file
        logger.info(f"[{request_id}] Initiating audio download")
        download_start = time.time()
        audio_path = await download_audio(request.audio_url)
        download_time = time.time() - download_start
        logger.info(f"[{request_id}] Audio downloaded in {download_time:.2f}s: {audio_path}")
        
        # Transcribe the audio
        logger.info(f"[{request_id}] Starting transcription process with GPU={CUDA_AVAILABLE}")
        transcription_start = time.time()
        transcription_result = await transcribe_audio(
            audio_path, 
            use_gpu=CUDA_AVAILABLE,
            diarize=request.diarize,
            num_speakers=request.num_speakers
        )
        transcription_time = time.time() - transcription_start
        
        num_segments = len(transcription_result["segments"])
        text_length = len(transcription_result["text"])
        logger.info(f"[{request_id}] Transcription completed in {transcription_time:.2f}s: {num_segments} segments, {text_length} characters")
        
        if request.diarize:
            num_speakers = transcription_result.get("num_speakers", 0)
            logger.info(f"[{request_id}] Diarization results: {num_speakers} speakers detected")
            
            # Log speaker distribution
            if num_speakers > 0:
                speaker_counts = {}
                for segment in transcription_result["segments"]:
                    speaker = segment.get("speaker", "unknown")
                    if speaker not in speaker_counts:
                        speaker_counts[speaker] = 0
                    speaker_counts[speaker] += 1
                
                for speaker, count in speaker_counts.items():
                    percentage = count / num_segments * 100
                    logger.info(f"[{request_id}] Speaker {speaker}: {count} segments ({percentage:.1f}%)")
        
        # Create response
        logger.info(f"[{request_id}] Assembling response")
        response = TranscriptionResponse(
            lead_id=request.lead_id,
            call_id=request.call_id,
            transcript_full=transcription_result["text"],
            transcript_segments=transcription_result["segments"],
            email=request.email,
            email_2=request.email_2,
            email_3=request.email_3,
            email_4=request.email_4,
            audio_url=request.audio_url,
            num_speakers=transcription_result.get("num_speakers")
        )
        
        total_time = time.time() - start_time
        logger.info(f"[{request_id}] Request successfully processed in {total_time:.2f}s")
        
        return response
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"[{request_id}] Error during transcription after {total_time:.2f}s: {str(e)}")
        logger.exception(f"[{request_id}] Full traceback:")
        raise HTTPException(status_code=500, detail=str(e)) 