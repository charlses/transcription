import os
import torch
import logging
from pyannote.audio import Pipeline
from pathlib import Path
import tempfile
from typing import List, Dict, Union, Optional
import warnings
import time
import sys

logger = logging.getLogger(__name__)
# Define model ID for diarization
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# Safely handle torchaudio backend settings to prevent deprecation warnings
def _configure_audio_backend():
    try:
        import torchaudio
        # Check if the deprecated function exists before calling it
        if hasattr(torchaudio, 'set_audio_backend'):
            # Suppress the specific UserWarning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", 
                    message="torchaudio._backend.set_audio_backend has been deprecated", 
                    category=UserWarning)
                torchaudio.set_audio_backend("soundfile")
        logger.debug("torchaudio backend configured")
    except ImportError:
        logger.warning("torchaudio not available")
    except Exception as e:
        logger.warning(f"Failed to configure torchaudio backend: {e}")

# Configure audio backend when module is imported
_configure_audio_backend()

class SpeakerDiarization:
    def __init__(self, use_gpu: bool = True):
        """
        Initialize the speaker diarization system.
        
        Args:
            use_gpu: Whether to use GPU for processing
        """
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.pipeline = None
        
        # Get configuration from environment variables
        self.max_retries = int(os.environ.get("DIARIZATION_MAX_RETRIES", "3"))
        self.default_min_speakers = int(os.environ.get("DIARIZATION_DEFAULT_MIN_SPEAKERS", "1"))
        self.default_max_speakers = int(os.environ.get("DIARIZATION_DEFAULT_MAX_SPEAKERS", "8"))
        
        logger.info(f"Diarization initialized with device={self.device}, "
                   f"max_retries={self.max_retries}, "
                   f"default_min_speakers={self.default_min_speakers}, "
                   f"default_max_speakers={self.default_max_speakers}")
        
    def load_pipeline(self):
        """
        Load the diarization pipeline.
        """
        logger.info("===== Starting diarization pipeline loading =====")
        start_time = time.time()
        
        try:
            # Get Hugging Face token from environment
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                logger.warning("HF_TOKEN environment variable not set. Model download may fail.")
            else:
                logger.info("Using HF_TOKEN from environment variables")
                
            # Check HF_HOME directory
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                logger.info(f"Using HF_HOME directory: {hf_home}")
                if os.path.exists(hf_home):
                    # Log contents of cache directory
                    try:
                        dir_items = os.listdir(hf_home)
                        logger.info(f"HF_HOME contains {len(dir_items)} items")
                        cache_size = 0
                        for item in dir_items:
                            item_path = os.path.join(hf_home, item)
                            if os.path.isdir(item_path):
                                # Get directory size
                                dir_size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                                          for dirpath, _, filenames in os.walk(item_path) 
                                          for filename in filenames)
                                cache_size += dir_size
                                logger.info(f"  - {item}: {dir_size / (1024 * 1024):.2f} MB")
                        logger.info(f"Total cache size: {cache_size / (1024 * 1024):.2f} MB")
                    except Exception as e:
                        logger.warning(f"Failed to analyze HF_HOME contents: {e}")
                else:
                    logger.warning(f"HF_HOME directory does not exist: {hf_home}")
            
            # Clear CUDA cache if using GPU to prevent memory issues
            if self.device == "cuda":
                logger.info("Clearing CUDA cache before loading model")
                torch.cuda.empty_cache()
                
            # Log GPU memory before loading
            if torch.cuda.is_available():
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                allocated_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
                logger.info(f"GPU memory before loading: {allocated_mem:.2f}GB used / {total_mem:.2f}GB total")
            
            # Load the diarization pipeline
            logger.info(f"Loading diarization model {DIARIZATION_MODEL}")
            model_load_start = time.time()
            
            # Log system info
            logger.info(f"Python version: {sys.version}")
            logger.info(f"PyTorch version: {torch.__version__}")
            try:
                import pyannote
                logger.info(f"Pyannote version: {pyannote.__version__}")
            except (ImportError, AttributeError):
                logger.info("Pyannote version information not available")
            
            self.pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=hf_token)
            model_load_time = time.time() - model_load_start
            logger.info(f"Diarization model loaded in {model_load_time:.2f} seconds")
            
            # Move pipeline to the appropriate device
            logger.info(f"Moving pipeline to device: {self.device}")
            device_move_start = time.time()
            self.pipeline.to(self.device)
            device_move_time = time.time() - device_move_start
            logger.info(f"Pipeline moved to {self.device} in {device_move_time:.2f} seconds")
            
            # Log GPU memory after loading
            if torch.cuda.is_available():
                allocated_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
                logger.info(f"GPU memory after loading: {allocated_mem:.2f}GB used / {total_mem:.2f}GB total")
            
            # Log pipeline parameters if available
            try:
                model_parameters = sum(p.numel() for p in self.pipeline.parameters())
                logger.info(f"Pipeline has {model_parameters:,} parameters")
            except Exception as e:
                logger.warning(f"Could not count model parameters: {e}")
                
            total_time = time.time() - start_time
            logger.info(f"===== Diarization pipeline loaded successfully in {total_time:.2f} seconds =====")
            
        except Exception as e:
            load_time = time.time() - start_time
            logger.error(f"Failed to load diarization pipeline after {load_time:.2f} seconds: {e}")
            logger.exception("Full traceback:")
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
        start_time = time.time()
        logger.info(f"===== Starting diarization process for {audio_path} =====")
        
        # Check if audio file exists
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Get file info for logging
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        logger.info(f"Processing audio file: {audio_path} ({file_size_mb:.2f}MB)")
        
        # Load pipeline if not already loaded
        if self.pipeline is None:
            logger.info("Pipeline not loaded yet, loading now...")
            self.load_pipeline()
        else:
            logger.info("Using already loaded pipeline")
        
        try:
            # Clear GPU cache before processing if using GPU
            if self.device == "cuda":
                logger.info("Clearing CUDA cache before diarization")
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    allocated_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
                    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    logger.info(f"GPU memory before diarization: {allocated_mem:.2f}GB used / {total_mem:.2f}GB total")
            
            # Set parameters if num_speakers is provided
            pipeline_start = time.time()
            if num_speakers is not None:
                logger.info(f"Running diarization with {num_speakers} speakers (min={max(1, num_speakers - 1)}, max={num_speakers + 1})")
                diarization = self.pipeline(
                    audio_path, 
                    num_speakers=num_speakers,
                    min_speakers=max(1, num_speakers - 1),  # Provide reasonable range
                    max_speakers=num_speakers + 1
                )
            else:
                logger.info(f"Running diarization with automatic speaker detection (min={self.default_min_speakers}, max={self.default_max_speakers})")
                # Set reasonable defaults for min/max speakers to improve accuracy
                diarization = self.pipeline(
                    audio_path,
                    min_speakers=self.default_min_speakers,
                    max_speakers=self.default_max_speakers
                )
            
            pipeline_time = time.time() - pipeline_start
            logger.info(f"Diarization pipeline execution completed in {pipeline_time:.2f} seconds")
            
            # Log GPU memory after processing if using CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                allocated_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                logger.info(f"GPU memory after diarization: {allocated_mem:.2f}GB used / {total_mem:.2f}GB total")
            
            # Extract speaker segments from diarization results
            logger.info("Extracting speaker segments from diarization results")
            extract_start = time.time()
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            extract_time = time.time() - extract_start
            logger.info(f"Extracted {len(segments)} speaker segments in {extract_time:.2f} seconds")
            
            # Validate results
            if not segments:
                logger.warning("Diarization produced no segments")
                
            # Use actual detected speakers count
            detected_speakers = len(diarization.labels())
            logger.info(f"Total speakers detected: {detected_speakers}")
            
            # Log detailed speaker statistics
            speaker_stats = {}
            total_duration = 0
            for segment in segments:
                speaker = segment["speaker"]
                duration = segment["end"] - segment["start"]
                total_duration += duration
                
                if speaker not in speaker_stats:
                    speaker_stats[speaker] = {"count": 0, "duration": 0}
                
                speaker_stats[speaker]["count"] += 1
                speaker_stats[speaker]["duration"] += duration
            
            for speaker, stats in speaker_stats.items():
                percentage = (stats["duration"] / total_duration * 100) if total_duration > 0 else 0
                logger.info(f"Speaker {speaker}: {stats['count']} segments, {stats['duration']:.2f} seconds ({percentage:.1f}%)")
            
            total_time = time.time() - start_time
            logger.info(f"===== Diarization completed in {total_time:.2f} seconds =====")
            
            return {
                "segments": segments,
                "num_speakers": detected_speakers
            }
            
        except Exception as e:
            logger.error(f"Error in diarization: {e}")
            logger.exception("Full traceback:")
            # Add specific handling for common errors
            if "CUDA out of memory" in str(e):
                logger.error("GPU out of memory error - try processing a smaller file or using CPU")
            elif "No speaker was found" in str(e):
                logger.error("No speakers detected in the audio - file may be silent or corrupted")
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
    logger.info(f"Starting speaker-transcript alignment: {len(transcript_segments)} transcript segments, {len(speaker_segments)} speaker segments")
    start_time = time.time()
    
    aligned_segments = []
    unknown_count = 0
    overlap_stats = []
    
    for i, ts in enumerate(transcript_segments):
        # Find the speaker with the most overlap for this segment
        segment_start = ts["start"]
        segment_end = ts["end"]
        segment_duration = segment_end - segment_start
        segment_text = ts["text"]
        
        logger.debug(f"Aligning segment {i+1}/{len(transcript_segments)}: {segment_start:.2f}s to {segment_end:.2f}s ({segment_duration:.2f}s)")
        
        best_speaker = None
        max_overlap = 0
        best_overlap_ratio = 0
        matching_speaker_segments = []
        
        for ss in speaker_segments:
            speaker_start = ss["start"]
            speaker_end = ss["end"]
            speaker_id = ss["speaker"]
            
            # Calculate overlap
            overlap_start = max(segment_start, speaker_start)
            overlap_end = min(segment_end, speaker_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > 0:
                matching_speaker_segments.append({
                    "speaker": speaker_id,
                    "start": speaker_start,
                    "end": speaker_end,
                    "overlap": overlap
                })
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = speaker_id
                    best_overlap_ratio = overlap / segment_duration if segment_duration > 0 else 0
        
        # Calculate overlap ratio (to ensure meaningful assignment)
        overlap_ratio = max_overlap / segment_duration if segment_duration > 0 else 0
        overlap_stats.append(overlap_ratio)
        
        # If multiple segments match, log them for debugging
        if len(matching_speaker_segments) > 1:
            logger.debug(f"Segment has {len(matching_speaker_segments)} matching speaker segments:")
            for match in matching_speaker_segments:
                logger.debug(f"  - Speaker {match['speaker']}: {match['start']:.2f}s to {match['end']:.2f}s, overlap: {match['overlap']:.2f}s")
        
        # Create aligned segment
        speaker_label = best_speaker if overlap_ratio > 0.3 else "unknown"
        if speaker_label == "unknown":
            unknown_count += 1
            logger.debug(f"Low overlap ratio ({overlap_ratio:.2f}) for segment {i+1}, marking as unknown")
        else:
            logger.debug(f"Assigned speaker {speaker_label} with overlap ratio {overlap_ratio:.2f}")
        
        aligned_segment = {
            "start": segment_start,
            "end": segment_end,
            "text": segment_text,
            "speaker": speaker_label
        }
        
        aligned_segments.append(aligned_segment)
    
    # Log alignment statistics
    total_time = time.time() - start_time
    avg_overlap_ratio = sum(overlap_stats) / len(overlap_stats) if overlap_stats else 0
    
    logger.info(f"Alignment completed in {total_time:.2f} seconds")
    logger.info(f"Created {len(aligned_segments)} aligned segments")
    logger.info(f"Unknown speaker segments: {unknown_count} ({unknown_count / len(aligned_segments) * 100:.1f}% of total)")
    logger.info(f"Average overlap ratio: {avg_overlap_ratio:.2f}")
    
    # Count segments per speaker
    speaker_counts = {}
    for segment in aligned_segments:
        speaker = segment["speaker"]
        if speaker not in speaker_counts:
            speaker_counts[speaker] = 0
        speaker_counts[speaker] += 1
    
    for speaker, count in speaker_counts.items():
        logger.info(f"Speaker {speaker}: {count} segments ({count / len(aligned_segments) * 100:.1f}% of total)")
    
    return aligned_segments 