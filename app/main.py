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
from .whisperx_transcription import WhisperXTranscription
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define request and response models
class TranscriptionRequest(BaseModel):
    recording_local: str
    recording_remote: str
    diarize: bool = True
    num_speakers: Optional[int] = None

class TranscriptionResponse(BaseModel):
    transcription_local: str
    transcription_remote: str
    transcription_combined: str
    segments_local: List[Dict]
    segments_remote: List[Dict]
    segments_combined: List[Dict]
    num_speakers: Optional[int] = None

# Initialize FastAPI app
app = FastAPI(
    title="Whisper Transcription API",
    description="API for transcribing audio using OpenAI's Whisper model",
    version="1.0.0"
)

# Initialize WhisperX transcription handler
cuda_available = torch.cuda.is_available()
if cuda_available:
    logger.info("CUDA is available. Initializing WhisperX with GPU acceleration.")
else:
    logger.warning("CUDA is not available! WhisperX will run much slower on CPU.")
    
whisperx_transcriber = WhisperXTranscription(use_gpu=True)  # Always request GPU, will fallback to CPU if not available

# Verify CUDA availability at startup
CUDA_AVAILABLE = cuda_available
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
            transcription_local=transcription_result["text"],
            transcription_remote=transcription_result["text"],
            transcription_combined=transcription_result["text"],
            segments_local=transcription_result["segments"],
            segments_remote=transcription_result["segments"],
            segments_combined=transcription_result["segments"],
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

@app.post("/transcribe/whisperx", response_model=TranscriptionResponse)
async def transcribe_whisperx(request: TranscriptionRequest):
    """
    Transcribe audio using WhisperX, providing improved accuracy and diarization.
    Processes both local and remote recordings, tagging segments with appropriate speakers.
    
    Args:
        request (TranscriptionRequest): Request containing local and remote audio URLs
    
    Returns:
        TranscriptionResponse: Response containing separate and combined transcriptions
    """
    request_id = str(uuid.uuid4())[:8]  # Generate a short request ID for tracking
    logger.info(f"[{request_id}] Received WhisperX transcription request")
    logger.info(f"[{request_id}] Local recording URL: {request.recording_local}")
    logger.info(f"[{request_id}] Remote recording URL: {request.recording_remote}")
    
    start_time = time.time()
    
    try:
        # Download both audio files
        logger.info(f"[{request_id}] Initiating audio downloads")
        download_start = time.time()
        
        # Download local recording
        local_audio_path = await download_audio(request.recording_local)
        logger.info(f"[{request_id}] Local audio downloaded: {local_audio_path}")
        
        # Download remote recording
        remote_audio_path = await download_audio(request.recording_remote)
        logger.info(f"[{request_id}] Remote audio downloaded: {remote_audio_path}")
        
        download_time = time.time() - download_start
        logger.info(f"[{request_id}] Both audio files downloaded in {download_time:.2f}s")
        
        # Transcribe local recording
        logger.info(f"[{request_id}] Starting local recording transcription")
        local_result = await whisperx_transcriber.process_audio(
            local_audio_path,
            diarize=False  # We don't need diarization since we know it's the agent
        )
        
        # Tag all local segments as "agent"
        for segment in local_result["segments"]:
            segment["speaker"] = "agent"
        
        # Transcribe remote recording
        logger.info(f"[{request_id}] Starting remote recording transcription")
        remote_result = await whisperx_transcriber.process_audio(
            remote_audio_path,
            diarize=False  # We don't need diarization since we know it's the client
        )
        
        # Tag all remote segments as "client"
        for segment in remote_result["segments"]:
            segment["speaker"] = "client"
        
        # Combine segments from both transcriptions
        combined_segments = local_result["segments"] + remote_result["segments"]
        
        # Sort combined segments by start time
        combined_segments.sort(key=lambda x: x["start"])
        
        # Create combined text from sorted segments
        combined_text = " ".join(segment["text"] for segment in combined_segments)
        
        # Create response
        logger.info(f"[{request_id}] Assembling response")
        response = TranscriptionResponse(
            transcription_local=local_result["text"],
            transcription_remote=remote_result["text"],
            transcription_combined=combined_text,
            segments_local=local_result["segments"],
            segments_remote=remote_result["segments"],
            segments_combined=combined_segments,
            num_speakers=2  # We always have 2 speakers: agent and client
        )
        
        total_time = time.time() - start_time
        logger.info(f"[{request_id}] WhisperX request successfully processed in {total_time:.2f}s")
        
        # Clean up temporary files
        if os.path.exists(local_audio_path):
            os.remove(local_audio_path)
        if os.path.exists(remote_audio_path):
            os.remove(remote_audio_path)
        
        return response
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"[{request_id}] Error during WhisperX transcription after {total_time:.2f}s: {str(e)}")
        logger.exception(f"[{request_id}] Full traceback:")
        raise HTTPException(status_code=500, detail=str(e)) 