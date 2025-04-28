import whisperx
import torch
import gc
import os
import logging
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("configurable_transcription")

class TranscriptionConfig(BaseModel):
    """Configuration parameters for WhisperX transcription."""
    # Basic configuration
    language: Optional[str] = "de"  # ISO language code, None for auto-detection
    compute_type: str = "float16"  # "float16", "float32", or "int8"
    
    # Transcription parameters
    temperature: float = 0.0  # Lower temperature for more precision
    beam_size: int = 5  # Beam size for beam search
    word_timestamps: bool = True  # Enable word-level timestamps
    batch_size: int = 16  # Batch size for parallelization
    condition_on_previous_text: bool = True  # Use context for better accuracy
    
    # Silence handling
    vad_filter: bool = True  # Enable voice activity detection
    no_speech_threshold: float = 0.6  # Higher value = more aggressive with silence
    compression_ratio_threshold: float = 2.4  # Higher value = more compression allowed
    
    # VAD parameters
    vad_onset: float = 0.500  # VAD onset threshold
    vad_offset: float = 0.363  # VAD offset threshold
    
    # Alignment
    align_output: bool = True  # Whether to align output for accurate word-level timestamps

class WhisperXConfigurableTranscription:
    def __init__(self, use_gpu: bool = True):
        """Initialize the configurable WhisperX transcription system.
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
        """
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        # Log GPU information if using CUDA
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Using GPU: {gpu_name} with {total_memory:.2f}GB memory")
        
        # Initialize models
        self.model = None
        self.align_models = {}
    
    def _load_model(self, config: TranscriptionConfig):
        """Initialize WhisperX ASR model with specific configuration
        
        Args:
            config: The transcription configuration
        """
        logger.info(f"Loading ASR model on device: {self.device}, compute type: {config.compute_type}")
        
        try:
            # Build the ASR options dictionary
            asr_options = {
                "temperatures": [config.temperature],
                "beam_size": config.beam_size,
                "word_timestamps": config.word_timestamps,
                "condition_on_previous_text": config.condition_on_previous_text,
                "compression_ratio_threshold": config.compression_ratio_threshold,
                "no_speech_threshold": config.no_speech_threshold
            }
            
            # Build the VAD options dictionary
            vad_options = None
            if config.vad_filter:
                vad_options = {
                    "vad_onset": config.vad_onset,
                    "vad_offset": config.vad_offset,
                }
            
            # Load the model with the specified options
            self.model = whisperx.load_model(
                "large-v3",
                device=self.device,
                compute_type=config.compute_type,
                language=config.language,
                asr_options=asr_options,
                vad_options=vad_options if config.vad_filter else None
            )
            logger.info("ASR model loaded successfully with custom configuration")
            return self.model
            
        except Exception as e:
            logger.error(f"Error loading ASR model: {str(e)}", exc_info=True)
            raise
    
    def _load_align_model(self, language_code: str):
        """Load alignment model for a specific language if not already loaded
        
        Args:
            language_code: ISO language code
            
        Returns:
            Tuple of (alignment model, metadata)
        """
        if language_code not in self.align_models:
            logger.info(f"Loading alignment model for language: {language_code}")
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=language_code,
                    device=self.device
                )
                self.align_models[language_code] = (model_a, metadata)
                logger.info(f"Alignment model for {language_code} loaded successfully")
            except Exception as e:
                logger.error(f"Error loading alignment model for {language_code}: {str(e)}", exc_info=True)
                raise
        
        return self.align_models[language_code]
    
    async def transcribe_audio(self, audio_path: str, speaker: str, config: TranscriptionConfig) -> Dict:
        """
        Transcribe a single audio file using WhisperX with customizable configuration.
        
        Args:
            audio_path: Path to the audio file
            speaker: Speaker identifier (local or remote)
            config: Configuration parameters for transcription
            
        Returns:
            Dictionary containing the transcription with segments
        """
        try:
            # Validate audio file
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                raise ValueError(f"Audio file is empty: {audio_path}")
            
            logger.info(f"Processing {speaker} audio: {audio_path} ({(file_size/1024/1024):.2f}MB)")
            
            # Load audio
            audio = whisperx.load_audio(audio_path)
            
            # Load model with configuration or use existing model
            if self.model is None:
                self._load_model(config)
            
            # Transcribe with WhisperX
            logger.info("Starting transcription with custom configuration...")
            result = self.model.transcribe(
                audio,
                batch_size=config.batch_size,
                language=config.language,
                task="transcribe"
            )
            logger.info("Transcription completed successfully")
            
            # Get detected language
            detected_lang = result.get("language", config.language or "de")
            logger.info(f"Detected language: {detected_lang}")
            
            # Clean up GPU memory if using CUDA
            if self.device == "cuda":
                logger.info("Cleaning up GPU memory")
                gc.collect()
                torch.cuda.empty_cache()
            
            # Align output if requested
            if config.align_output:
                try:
                    logger.info(f"Starting alignment with {detected_lang} model")
                    model_a, metadata = self._load_align_model(detected_lang)
                    alignment = whisperx.align(
                        result["segments"],
                        model_a,
                        metadata,
                        audio,
                        device=self.device,
                        return_char_alignments=False
                    )
                    logger.info("Alignment completed successfully")
                except Exception as e:
                    logger.error(f"Error aligning with detected language ({detected_lang}): {str(e)}")
                    logger.info("Falling back to German alignment model")
                    
                    # Fallback to German alignment
                    model_a, metadata = self._load_align_model("de")
                    alignment = whisperx.align(
                        result["segments"],
                        model_a,
                        metadata,
                        audio,
                        device=self.device,
                        return_char_alignments=False
                    )
                    logger.info("Fallback alignment completed successfully")
                
                # Use aligned segments
                result_segments = alignment["segments"]
            else:
                # Use original segments
                result_segments = result["segments"]
            
            # Add speaker label to segments and combine text
            full_text = ""
            for segment in result_segments:
                # Assign speaker label based on input parameter
                segment["speaker"] = "agent" if speaker == "local" else "client"
                full_text += " " + segment.get("text", "")
            
            return {
                "segments": result_segments,
                "text": full_text.strip()
            }
            
        except Exception as e:
            logger.error(f"Error transcribing {speaker} audio: {str(e)}", exc_info=True)
            raise
    
    async def transcribe_audio_combined(self, local_audio_path: str, remote_audio_path: str, config: TranscriptionConfig) -> Dict:
        """
        Transcribe two audio files and combine the results.
        
        Args:
            local_audio_path: Path to the local (agent) audio file
            remote_audio_path: Path to the remote (client) audio file
            config: Configuration parameters for transcription
            
        Returns:
            Dictionary containing individual transcriptions and combined result
        """
        try:
            # Load model with configuration
            if self.model is None:
                self._load_model(config)
            
            # Transcribe remote recording
            logger.info("Starting remote recording transcription")
            transcript_remote = await self.transcribe_audio(remote_audio_path, "remote", config)
            
            # Transcribe local recording
            logger.info("Starting local recording transcription")
            transcript_local = await self.transcribe_audio(local_audio_path, "local", config)
            
            # Combine and sort segments by start time
            all_segments = []
            
            # Add remote segments with client tag
            for segment in transcript_remote["segments"]:
                segment["speaker"] = "client"  # Remote audio is from client
                all_segments.append(segment)
                
            # Add local segments with agent tag
            for segment in transcript_local["segments"]:
                segment["speaker"] = "agent"  # Local audio is from agent
                all_segments.append(segment)
            
            # Sort segments by start time
            all_segments.sort(key=lambda x: x["start"])
            
            # Create combined transcript text
            combined_text = " ".join(segment["text"] for segment in all_segments)
            
            return {
                "transcript_remote": transcript_remote,
                "transcript_local": transcript_local,
                "transcript_combined": {
                    "segments": all_segments,
                    "text": combined_text
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing audio files: {str(e)}", exc_info=True)
            raise 