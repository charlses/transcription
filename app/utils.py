import os
import whisper
import requests
import tempfile
from pathlib import Path
import logging
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)

async def download_audio(url: str) -> str:
    """
    Download audio file from URL to temporary file
    
    Args:
        url (str): URL of the audio file
    
    Returns:
        str: Path to downloaded audio file
    """
    try:
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        temp_path = Path(temp_dir) / f"audio_{hash(url)}.mp3"
        
        # Download file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        logger.info(f"Audio downloaded successfully to {temp_path}")
        return str(temp_path)
        
    except Exception as e:
        logger.error(f"Error downloading audio: {str(e)}")
        raise Exception(f"Failed to download audio: {str(e)}")

async def transcribe_audio(audio_path: str, use_gpu: bool = True) -> dict:
    """
    Transcribe audio file using Whisper model
    
    Args:
        audio_path (str): Path to audio file
        use_gpu (bool): Whether to use GPU for transcription
    
    Returns:
        dict: Dictionary containing full text and segments
    """
    try:
        # Load Whisper model
        device = "cuda" if use_gpu else "cpu"
        model = whisper.load_model("base", device=device)
        
        # Run transcription in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: model.transcribe(audio_path)
        )
        
        # Clean up temporary file
        os.remove(audio_path)
        
        # Process segments to include only necessary information
        processed_segments = [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            }
            for segment in result["segments"]
        ]
        
        return {
            "text": result["text"].strip(),
            "segments": processed_segments
        }
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise Exception(f"Failed to transcribe audio: {str(e)}") 