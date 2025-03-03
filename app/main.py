from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import torch
from .utils import download_audio, transcribe_audio
import logging
from pydantic import BaseModel
from typing import List, Optional, Dict

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
    return {"status": "online", "cuda_available": CUDA_AVAILABLE}

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(request: TranscriptionRequest):
    """
    Transcribe audio from the provided request data
    
    Args:
        request (TranscriptionRequest): Request containing lead and audio information
    
    Returns:
        TranscriptionResponse: Response containing transcription and original data
    """
    try:
        # Download the audio file
        audio_path = await download_audio(request.audio_url)
        
        # Transcribe the audio
        transcription_result = await transcribe_audio(
            audio_path, 
            use_gpu=CUDA_AVAILABLE,
            diarize=request.diarize
        )
        
        # Create response
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
        
        return response
        
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 