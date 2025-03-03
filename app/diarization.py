import os
import torch
import logging
from pyannote.audio import Pipeline
from pathlib import Path
import tempfile
from typing import List, Dict, Union, Optional

logger = logging.getLogger(__name__)
# Define model ID for diarization
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

class SpeakerDiarization:
    def __init__(self, use_gpu: bool = True):
        """
        Initialize the speaker diarization system.
        
        Args:
            use_gpu: Whether to use GPU for processing
        """
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.pipeline = None
        
    def load_pipeline(self):
        """
        Load the diarization pipeline.
        """
        try:
            # Get Hugging Face token from environment
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                logger.warning("HF_TOKEN environment variable not set. Model download may fail.")
            
            # Load the diarization pipeline
            logger.info(f"Loading diarization model {DIARIZATION_MODEL}")
            self.pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=hf_token)
            
            # Move pipeline to the appropriate device
            self.pipeline.to(self.device)
            logger.info(f"Diarization pipeline loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            raise
    
    def process_audio(self, audio_path: str, num_speakers: Optional[int] = None) -> Dict:
        """
        Process audio file for speaker diarization.
        
        Args:
            audio_path: Path to the audio file
            num_speakers: Optional number of speakers (if known)
            
        Returns:
            Dictionary with diarization results
        """
        # Load pipeline if not already loaded
        if self.pipeline is None:
            self.load_pipeline()
        
        try:
            # Set parameters if num_speakers is provided
            if num_speakers is not None:
                logger.info(f"Running diarization with {num_speakers} speakers")
                diarization = self.pipeline(audio_path, num_speakers=num_speakers)
            else:
                logger.info("Running diarization with automatic speaker detection")
                diarization = self.pipeline(audio_path)
            
            # Extract speaker segments from diarization results
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            return {
                "segments": segments,
                "num_speakers": len(diarization.labels())
            }
            
        except Exception as e:
            logger.error(f"Error in diarization: {e}")
            raise Exception(f"Failed to perform speaker diarization: {str(e)}")

def align_transcript_with_speakers(
    transcript_segments: List[Dict], 
    speaker_segments: List[Dict]
) -> List[Dict]:
    """
    Align transcript segments with speaker information.
    
    Args:
        transcript_segments: List of transcript segments with start, end, and text
        speaker_segments: List of speaker segments with start, end, and speaker
        
    Returns:
        List of segments with both text and speaker information
    """
    aligned_segments = []
    
    for ts in transcript_segments:
        # Find the speaker with the most overlap for this segment
        segment_start = ts["start"]
        segment_end = ts["end"]
        segment_duration = segment_end - segment_start
        
        best_speaker = None
        max_overlap = 0
        
        for ss in speaker_segments:
            speaker_start = ss["start"]
            speaker_end = ss["end"]
            
            # Calculate overlap
            overlap_start = max(segment_start, speaker_start)
            overlap_end = min(segment_end, speaker_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = ss["speaker"]
        
        # Calculate overlap ratio (to ensure meaningful assignment)
        overlap_ratio = max_overlap / segment_duration if segment_duration > 0 else 0
        
        # Create aligned segment
        aligned_segment = {
            "start": segment_start,
            "end": segment_end,
            "text": ts["text"],
            "speaker": best_speaker if overlap_ratio > 0.3 else "unknown"
        }
        
        aligned_segments.append(aligned_segment)
    
    return aligned_segments 