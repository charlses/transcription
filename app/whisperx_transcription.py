import whisperx
import torch
import gc
import os
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("transcription")

class WhisperXTranscription:
    def __init__(self, use_gpu: bool = True):
        """Initialize the WhisperX transcription system."""
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "float32"
        
        # Log GPU information if using CUDA
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Using GPU: {gpu_name} with {total_memory:.2f}GB memory")
        
        # Initialize models
        self.model = None
        self.align_models = {}
        
        # Load ASR model
        self._load_model()
    
    def _load_model(self):
        """Initialize WhisperX ASR model if not already loaded"""
        if self.model is None:
            logger.info(f"Loading ASR model on device: {self.device}, compute type: {self.compute_type}")
            try:
                self.model = whisperx.load_model(
                    "large-v3",
                    device=self.device,
                    compute_type=self.compute_type,
                )
                logger.info("ASR model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading ASR model: {str(e)}", exc_info=True)
                raise
    
    def _load_align_model(self, language_code: str):
        """Load alignment model for a specific language if not already loaded"""
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
    
    async def transcribe_audio(self, audio_path: str, speaker: str) -> Dict:
        """
        Transcribe a single audio file using WhisperX.
        
        Args:
            audio_path: Path to the audio file
            speaker: Speaker identifier (local or remote)
            
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
            
            # Transcribe with WhisperX
            logger.info("Starting transcription...")
            result = self.model.transcribe(
                audio,
                batch_size=16,
                language="de",
                task="transcribe",
                # Silence handling parameters
                no_speech_threshold=0.6,  # Higher threshold to be more aggressive with silence
                condition_on_previous_text=True,  # Use context for better accuracy
                compression_ratio_threshold=2.4,  # Higher threshold to allow more compression
                # Precision parameters
                word_timestamps=True,  # Enable word-level timestamps
                temperature=0.0,  # Lower temperature for more precise transcription
                beam_size=5,  # Larger beam size for more thorough search
                # Additional parameters
                vad_filter=True,  # Enable voice activity detection
                vad_parameters=dict(
                    min_silence_duration_ms=500,  # Minimum silence duration
                    speech_pad_ms=100  # Padding around speech segments
                )
            )
            logger.info("Transcription completed successfully")
            
            # Get detected language
            detected_lang = result.get("language", "de")
            logger.info(f"Detected language: {detected_lang}")
            
            # Clean up GPU memory if using CUDA
            if self.device == "cuda":
                logger.info("Cleaning up GPU memory")
                gc.collect()
                torch.cuda.empty_cache()
            
            # Try alignment with detected language first
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
            
            # Add speaker label to segments and combine text
            full_text = ""
            for segment in alignment["segments"]:
                segment["speaker"] = "agent" if speaker == "local" else "client"
                full_text += " " + segment.get("text", "")
            
            return {
                "segments": alignment["segments"],
                "text": full_text.strip()
            }
            
        except Exception as e:
            logger.error(f"Error transcribing {speaker} audio: {str(e)}", exc_info=True)
            raise 