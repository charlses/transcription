import os
import torch
import logging
import numpy as np
from typing import Dict, List, Optional, Union
import time
import asyncio
import warnings
import whisperx
from pathlib import Path
import traceback
import math

# Configure logging
logger = logging.getLogger(__name__)

class WhisperXTranscription:
    def __init__(self, use_gpu: bool = True):
        """
        Initialize the WhisperX transcription system.
        
        Args:
            use_gpu: Whether to use GPU for processing
        """
        # Store use_gpu preference
        self.use_gpu = use_gpu
        
        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()
        
        # Determine device type based on availability and preference
        if use_gpu and not cuda_available:
            logger.warning("GPU requested but CUDA is not available. Using CPU instead.")
            logger.warning("This will significantly slow down processing.")
        
        # Force CUDA if requested and available
        self.device_type = "cuda" if use_gpu and cuda_available else "cpu"
        self.device = torch.device(self.device_type)
        
        # Log GPU information if using CUDA
        if self.device_type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Using GPU: {gpu_name} with {total_memory:.2f}GB memory")
        else:
            logger.warning("Running on CPU. Processing will be much slower.")
        
        self.whisper_model = None
        self.alignment_model = None
        self.diarization_pipeline = None
        
        # Get configuration from environment variables
        self.model_size = os.environ.get("WHISPERX_MODEL_SIZE", "large-v3")
        self.language = os.environ.get("WHISPERX_LANGUAGE", "de")
        self.compute_type = os.environ.get("WHISPERX_COMPUTE_TYPE", "float16")
        self.hf_token = os.environ.get("HF_TOKEN", None)
        
        if not self.hf_token:
            logger.warning("HF_TOKEN not set in environment. Some model downloads may fail.")
        
        logger.info(f"WhisperX initialized with device={self.device}, "
                   f"model_size={self.model_size}, "
                   f"language={self.language}, "
                   f"compute_type={self.compute_type}")
    
    def load_models(self):
        """Load all required models for transcription and diarization."""
        logger.info("Loading WhisperX models")
        
        if torch.cuda.is_available() and self.use_gpu:
            self.device_type = "cuda"
            self.device = torch.device("cuda")
            self.compute_type = "float16"
            
            # Log GPU information
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
            
            free_memory, total_memory = torch.cuda.mem_get_info(0)
            logger.info(f"Free GPU memory before loading models: {free_memory / (1024**3):.2f} GB / {total_memory / (1024**3):.2f} GB")
        else:
            if self.use_gpu:
                logger.warning("GPU was requested but is not available. Falling back to CPU.")
            self.device_type = "cpu"
            self.device = torch.device("cpu")
            self.compute_type = "int8"
            logger.info("Using CPU for processing")
        
        try:
            # Track memory usage
            if torch.cuda.is_available() and self.use_gpu:
                free_memory_before, _ = torch.cuda.mem_get_info(0)
            
            # 1. Load ASR model using the standard WhisperX approach
            logger.info(f"Loading Whisper model (size={self.model_size}, device={self.device_type}, compute={self.compute_type})")
            
            self.whisper_model = whisperx.load_model(
                self.model_size,
                self.device_type,
                compute_type=self.compute_type,
                language=self.language
            )
            
            logger.info("Successfully loaded ASR model using standard WhisperX approach")
            
            # 2. Log memory usage after loading ASR model
            if torch.cuda.is_available() and self.use_gpu:
                free_memory_after, total_memory = torch.cuda.mem_get_info(0)
                memory_used = (free_memory_before - free_memory_after) / (1024**3)
                logger.info(f"ASR model loaded. GPU memory used: {memory_used:.2f} GB, Free: {free_memory_after / (1024**3):.2f} GB / {total_memory / (1024**3):.2f} GB")
            
            # 3. Load diarization model if token is available
            if self.hf_token and torch.cuda.is_available() and self.use_gpu:
                try:
                    logger.info("Loading diarization pipeline")
                    diarization_start = time.time()
                    
                    self.diarization_pipeline = whisperx.DiarizationPipeline(
                        model_name="pyannote/speaker-diarization-3.1",
                        use_auth_token=self.hf_token,
                        device=self.device_type
                    )
                    
                    diarization_time = time.time() - diarization_start
                    logger.info(f"Diarization pipeline loaded in {diarization_time:.2f} seconds")
                    
                    # Track memory usage after loading diarization
                    if torch.cuda.is_available() and self.use_gpu:
                        free_memory_after_diarize, total_memory = torch.cuda.mem_get_info(0)
                        memory_used = (free_memory_after - free_memory_after_diarize) / (1024**3)
                        logger.info(f"Diarization model loaded. Additional GPU memory used: {memory_used:.2f} GB, Free: {free_memory_after_diarize / (1024**3):.2f} GB / {total_memory / (1024**3):.2f} GB")
                except Exception as e:
                    logger.warning(f"Failed to load diarization model: {str(e)}")
                    logger.warning("Diarization will not be available")
            else:
                logger.info("Skipping diarization model loading (no token or CUDA not available)")
            
            # Note: We'll load alignment models on-demand during transcription
            # since they're language-specific
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            logger.exception("Full traceback:")
            return False
    
    async def process_audio(self, audio_path: str, diarize: bool = True, num_speakers: Optional[int] = None) -> Dict:
        """
        Process audio file to create a transcript with optional diarization.
        
        Args:
            audio_path: Path to the audio file to process
            diarize: Whether to perform speaker diarization
            num_speakers: Number of speakers if known (helps with diarization)
            
        Returns:
            Dictionary containing the transcript with detailed segment information
        """
        start_time = time.time()
        logger.info(f"Starting audio processing with WhisperX (diarize={diarize}, num_speakers={num_speakers})")
        
        # Check if audio file exists
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Get file info for logging
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        logger.info(f"Processing audio file: {audio_path} ({file_size_mb:.2f}MB)")
        
        # Load models if not already loaded
        if self.whisper_model is None:
            logger.info("Models not loaded yet, loading now...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.load_models)
        else:
            logger.info("Using already loaded models")
        
        try:
            # Following the standard WhisperX pipeline:
            # 1. Load audio
            logger.info("Loading audio with WhisperX...")
            loop = asyncio.get_event_loop()
            audio = await loop.run_in_executor(
                None,
                lambda: whisperx.load_audio(audio_path)
            )
            logger.info("Audio loaded successfully")
            
            # 2. Transcribe with Whisper (batched)
            logger.info(f"Starting WhisperX transcription on {self.device_type}")
            transcribe_start = time.time()
            
            result = await loop.run_in_executor(
                None,
                lambda: self.whisper_model.transcribe(
                    audio, 
                    batch_size=16
                )
            )
            
            transcribe_time = time.time() - transcribe_start
            logger.info(f"WhisperX transcription completed in {transcribe_time:.2f} seconds")
            
            if result and "segments" in result:
                logger.info(f"Initial transcription: {len(result['segments'])} segments")
            
            # 3. Perform alignment (more accurate word-level timestamps)
            if result and "segments" in result and len(result["segments"]) > 0:
                try:
                    logger.info(f"Starting WhisperX alignment on {self.device_type}")
                    alignment_start = time.time()
                    
                    # Load alignment model for the detected language
                    language_code = result.get("language", "en")
                    logger.info(f"Loading alignment model for language: {language_code}")
                    
                    align_model, align_metadata = await loop.run_in_executor(
                        None,
                        lambda: whisperx.load_align_model(language_code=language_code, device=self.device_type)
                    )
                    
                    # Align the transcription
                    aligned_result = await loop.run_in_executor(
                        None,
                        lambda: whisperx.align(
                            result["segments"],
                            align_model,
                            align_metadata,
                            audio,
                            self.device_type,
                            return_char_alignments=False
                        )
                    )
                    
                    alignment_time = time.time() - alignment_start
                    logger.info(f"WhisperX alignment completed in {alignment_time:.2f} seconds")
                    
                    # Replace the result with the aligned version
                    if aligned_result and "segments" in aligned_result:
                        result = aligned_result
                        logger.info(f"After alignment: {len(result['segments'])} segments")
                    
                except Exception as e:
                    logger.warning(f"Alignment failed: {str(e)}")
                    logger.warning("Continuing without alignment")
            
            # 4. Perform diarization if requested
            if diarize and self.diarization_pipeline is not None:
                try:
                    logger.info(f"Starting WhisperX diarization on {self.device_type}")
                    diarization_start = time.time()
                    
                    # Configure diarization parameters
                    diarize_kwargs = {}
                    if num_speakers is not None:
                        diarize_kwargs["min_speakers"] = num_speakers
                        diarize_kwargs["max_speakers"] = num_speakers
                    
                    # Run diarization
                    diarize_segments = await loop.run_in_executor(
                        None,
                        lambda: self.diarization_pipeline(audio, **diarize_kwargs)
                    )
                    
                    # Assign speakers to words/segments
                    result = await loop.run_in_executor(
                        None,
                        lambda: whisperx.assign_word_speakers(diarize_segments, result)
                    )
                    
                    diarization_time = time.time() - diarization_start
                    logger.info(f"WhisperX diarization completed in {diarization_time:.2f} seconds")
                    
                    # Get number of speakers
                    unique_speakers = set()
                    if "segments" in result:
                        for segment in result["segments"]:
                            speaker = segment.get("speaker")
                            if speaker:
                                unique_speakers.add(speaker)
                    
                    result["num_speakers"] = len(unique_speakers)
                    logger.info(f"Diarization detected {len(unique_speakers)} speakers")
                    
                except Exception as e:
                    logger.error(f"Diarization failed: {str(e)}")
                    logger.warning("Continuing without diarization")
                    result["num_speakers"] = None
            else:
                logger.info("Skipping diarization based on request or model availability")
                result["num_speakers"] = None
            
            # 5. Post-process segments to create shorter segments
            if "segments" in result:
                logger.info("Post-processing to create very short segments")
                original_segment_count = len(result["segments"])
                processed_segments = []
                
                # Process each segment to ensure max duration of 2 seconds
                MAX_FINAL_SEGMENT_DURATION = 2.0  # Maximum 2 seconds per segment
                
                for segment in result["segments"]:
                    # Get segment attributes
                    segment_start = segment.get("start", 0)
                    segment_end = segment.get("end", 0)
                    segment_duration = segment_end - segment_start
                    text = segment.get("text", "").strip()
                    speaker = segment.get("speaker", "SPEAKER_UNK")
                    words = segment.get("words", [])
                    
                    # If already short enough or has no words, keep as is
                    if segment_duration <= MAX_FINAL_SEGMENT_DURATION or not words:
                        processed_segments.append(segment)
                        continue
                    
                    # If the segment has words, split it into smaller chunks
                    if words:
                        # Calculate how many chunks we need 
                        num_chunks = math.ceil(segment_duration / MAX_FINAL_SEGMENT_DURATION)
                        words_per_chunk = max(1, len(words) // num_chunks)
                        
                        # Split words into chunks
                        word_chunks = []
                        for i in range(0, len(words), words_per_chunk):
                            chunk = words[i:i+words_per_chunk]
                            if chunk:  # Only add non-empty chunks
                                word_chunks.append(chunk)
                        
                        # Create a new segment for each chunk
                        for chunk in word_chunks:
                            chunk_start = chunk[0].get("start", segment_start)
                            chunk_end = chunk[-1].get("end", chunk_start + MAX_FINAL_SEGMENT_DURATION)
                            chunk_text = " ".join(w.get("word", "") for w in chunk)
                            
                            # Create new mini-segment
                            new_segment = {
                                "start": chunk_start,
                                "end": chunk_end,
                                "text": chunk_text,
                                "speaker": speaker,
                                "words": chunk
                            }
                            
                            # Copy any additional fields from original segment
                            for key, value in segment.items():
                                if key not in ["start", "end", "text", "speaker", "words"]:
                                    new_segment[key] = value
                            
                            processed_segments.append(new_segment)
                    else:
                        # Fallback if no word timestamps: split evenly by time
                        words = text.split()
                        if words:
                            chars_per_second = len(text) / segment_duration
                            # Split into chunks of approximately 2 seconds each
                            num_chunks = math.ceil(segment_duration / MAX_FINAL_SEGMENT_DURATION)
                            words_per_chunk = max(1, len(words) // num_chunks)
                            
                            # Create word groups
                            word_chunks = []
                            for i in range(0, len(words), words_per_chunk):
                                chunk = words[i:i+words_per_chunk]
                                if chunk:  # Only add non-empty chunks
                                    word_chunks.append(chunk)
                            
                            # Create a new segment for each chunk with estimated times
                            chunk_duration = segment_duration / len(word_chunks)
                            for i, chunk in enumerate(word_chunks):
                                chunk_start = segment_start + (i * chunk_duration)
                                chunk_end = chunk_start + chunk_duration
                                chunk_text = " ".join(chunk)
                                
                                new_segment = {
                                    "start": chunk_start,
                                    "end": chunk_end,
                                    "text": chunk_text,
                                    "speaker": speaker
                                }
                                
                                # Copy any additional fields
                                for key, value in segment.items():
                                    if key not in ["start", "end", "text", "speaker"]:
                                        new_segment[key] = value
                                
                                processed_segments.append(new_segment)
                        else:
                            # If no words at all, just keep the segment
                            processed_segments.append(segment)
                
                # Replace segments with our processed version
                result["segments"] = processed_segments
                logger.info(f"Final segment count after post-processing: {len(processed_segments)} (from {original_segment_count})")
            
            # 6. Ensure text key exists in the result (for full transcript)
            if "text" not in result and "segments" in result and result["segments"]:
                # Build full text from segments
                result["text"] = " ".join(segment.get("text", "") for segment in result["segments"])
            
            # Track total processing time
            process_time = time.time() - start_time
            logger.info(f"===== WhisperX processing completed in {process_time:.2f} seconds =====")
            
            return result
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"Error during WhisperX processing after {process_time:.2f}s: {str(e)}")
            logger.exception("Full traceback:")
            raise 