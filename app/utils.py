import os
import whisper
import requests
import tempfile
from pathlib import Path
import logging
import asyncio
from typing import Optional
import mimetypes
from .diarization import SpeakerDiarization, align_transcript_with_speakers
import torch

# Set up logging for debugging and error reporting
logger = logging.getLogger(__name__)

def get_file_extension(url: str) -> str:
    """
    Determines the file extension based on the URL's MIME type.

    Args:
        url (str): URL of the audio file.

    Returns:
        str: Appropriate file extension for the audio file (e.g., '.mp3', '.wav').
    """
    # Guess the MIME type from the URL and map it to an appropriate file extension
    mime_type, _ = mimetypes.guess_type(url)
    if mime_type:
        # If MIME type is found, try to get the corresponding file extension
        extension = mimetypes.guess_extension(mime_type)
        return extension if extension else '.mp3'  # Default to '.mp3' if no extension found
    return '.mp3'  # Default to '.mp3' if MIME type is not determined


async def download_audio(url: str) -> str:
    """
    Downloads an audio file from a URL and saves it to a temporary location.

    Args:
        url (str): URL of the audio file to download.

    Returns:
        str: Path to the downloaded audio file.

    Raises:
        Exception: If an error occurs during download.
    """
    logger.info(f"Starting audio download from URL: {url}")
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Create a temporary file path using the URL's hash and appropriate file extension
        temp_dir = tempfile.gettempdir()  # Get the system's temporary directory
        extension = get_file_extension(url)  # Get the correct file extension for the audio file
        temp_path = Path(temp_dir) / f"audio_{hash(url)}{extension}"  # Create the full temp path
        
        logger.info(f"Will save audio to temporary file: {temp_path}")

        # Download the audio file from the URL
        logger.info(f"Initiating download from {url}")
        download_start = asyncio.get_event_loop().time()
        response = requests.get(url, stream=True)  # Start downloading the file
        
        if response.status_code != 200:  # Check if the request was successful
            logger.error(f"Failed to download, status code: {response.status_code}")
            logger.error(f"Response content: {response.text[:500]}...")  # Log partial response for debugging
            raise Exception(f"Failed to download audio with status code {response.status_code}")

        # Get content length if available
        content_length = response.headers.get('content-length')
        if content_length:
            content_length = int(content_length)
            logger.info(f"Audio file size: {content_length / (1024 * 1024):.2f} MB")
        else:
            logger.info("Audio file size unknown (no content-length header)")

        # Write the downloaded content to the temporary file in chunks to avoid memory issues
        bytes_downloaded = 0
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bytes_downloaded += len(chunk)
                # Log progress for large files every 10MB
                if bytes_downloaded % (10 * 1024 * 1024) < 8192 and content_length:
                    progress = bytes_downloaded / content_length * 100
                    logger.info(f"Download progress: {progress:.1f}% ({bytes_downloaded / (1024 * 1024):.2f}MB / {content_length / (1024 * 1024):.2f}MB)")

        download_time = asyncio.get_event_loop().time() - download_start
        logger.info(f"Download completed in {download_time:.2f} seconds")
        
        # Verify the downloaded file
        if os.path.exists(temp_path):
            file_size = os.path.getsize(temp_path)
            logger.info(f"Downloaded file size: {file_size / (1024 * 1024):.2f} MB")
            if file_size == 0:
                logger.error("Downloaded file is empty (0 bytes)")
                raise Exception("Downloaded audio file is empty")
        else:
            logger.error(f"File not found at expected location: {temp_path}")
            raise Exception("Failed to save downloaded audio")

        total_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"Audio downloaded successfully to {temp_path} in {total_time:.2f} seconds")
        return str(temp_path)  # Return the path to the downloaded audio file

    except Exception as e:
        # Log any errors that occur during the download process
        logger.error(f"Error downloading audio: {str(e)}")
        logger.exception("Full traceback:")
        raise Exception(f"Failed to download audio: {str(e)}")


async def transcribe_audio(audio_path: str, use_gpu: bool = True, diarize: bool = True, num_speakers: Optional[int] = None) -> dict:
    """
    Transcribes the audio file to text using the Whisper model.

    Args:
        audio_path (str): Path to the audio file to transcribe.
        use_gpu (bool): Whether to use GPU for transcription. Defaults to True.
        diarize (bool): Whether to perform speaker diarization. Defaults to True.
        num_speakers (Optional[int]): Number of speakers for diarization, if known. Defaults to None.

    Returns:
        dict: A dictionary containing the full transcribed text and detailed segment information.

    Raises:
        Exception: If an error occurs during transcription.
    """
    logger.info(f"Starting transcription process for {audio_path}")
    logger.info(f"Configuration: use_gpu={use_gpu}, diarize={diarize}, num_speakers={num_speakers}")
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Determine the device (CPU or GPU) and load the appropriate Whisper model
        device = "cuda" if use_gpu else "cpu"  # Select device based on `use_gpu` flag
        logger.info(f"Using device: {device}")
        
        # Log GPU memory if using CUDA
        if device == "cuda" and torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            allocated_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
            logger.info(f"GPU memory before loading Whisper: {allocated_mem:.2f}GB used / {total_mem:.2f}GB total")
        
        logger.info(f"Loading Whisper model ('large') on {device}")
        model_load_start = asyncio.get_event_loop().time()
        model = whisper.load_model("large", device=device)  # Load Whisper model (large version)
        model_load_time = asyncio.get_event_loop().time() - model_load_start
        logger.info(f"Whisper model loaded in {model_load_time:.2f} seconds")

#         # If using GPU, convert the model to half precision to optimize performance
#         if use_gpu:
#             model = model.half()  # Convert model to use 16-bit precision (half precision)

        # Log GPU memory after loading if using CUDA
        if device == "cuda" and torch.cuda.is_available():
            allocated_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
            logger.info(f"GPU memory after loading Whisper: {allocated_mem:.2f}GB used / {total_mem:.2f}GB total")

        # Run the transcription process asynchronously using an executor to avoid blocking
        logger.info(f"Starting transcription of audio file: {audio_path}")
        transcription_start = asyncio.get_event_loop().time()
        loop = asyncio.get_event_loop()  # Get the event loop for async execution
        result = await loop.run_in_executor(
            None,  # Use default executor (thread pool)
            lambda: model.transcribe(audio_path)  # Run the transcription function
        )
        transcription_time = asyncio.get_event_loop().time() - transcription_start
        logger.info(f"Transcription completed in {transcription_time:.2f} seconds")
        logger.info(f"Transcription result: {len(result['segments'])} segments, {len(result['text'])} characters")

        # Process the transcription result to extract and clean segment data
        logger.info("Processing transcription segments")
        processed_segments = [
            {
                "start": segment["start"],  # Start time of the segment
                "end": segment["end"],  # End time of the segment
                "text": segment["text"].strip()  # Cleaned text from the segment (remove leading/trailing whitespace)
            }
            for segment in result["segments"]
        ]
        logger.info(f"Processed {len(processed_segments)} transcript segments")

        # Perform speaker diarization if requested
        if diarize:
            try:
                logger.info(f"Starting speaker diarization for audio: {audio_path}")
                diarization_start = asyncio.get_event_loop().time()
                
                # Create diarization object
                logger.info("Initializing speaker diarization")
                diarizer = SpeakerDiarization(use_gpu=use_gpu)
                
                # Process audio for speaker diarization with retry mechanism
                max_retries = int(os.environ.get("DIARIZATION_MAX_RETRIES", "3"))
                logger.info(f"Diarization will attempt up to {max_retries} retries if needed")
                retry_count = 0
                diarization_successful = False
                diarization_result = None
                
                while retry_count < max_retries and not diarization_successful:
                    try:
                        logger.info(f"Diarization attempt {retry_count + 1}/{max_retries}")
                        diarization_attempt_start = asyncio.get_event_loop().time()
                        diarization_result = await loop.run_in_executor(
                            None,
                            lambda: diarizer.process_audio(audio_path, num_speakers=num_speakers)
                        )
                        diarization_attempt_time = asyncio.get_event_loop().time() - diarization_attempt_start
                        logger.info(f"Diarization attempt {retry_count + 1} succeeded in {diarization_attempt_time:.2f} seconds")
                        diarization_successful = True
                    except Exception as e:
                        retry_count += 1
                        logger.warning(f"Diarization attempt {retry_count} failed: {str(e)}")
                        if retry_count == max_retries:
                            logger.error(f"All {max_retries} diarization attempts failed, giving up")
                            raise
                        # Wait a moment before retrying
                        logger.info(f"Waiting 1 second before retry {retry_count + 1}")
                        await asyncio.sleep(1)
                
                # Align transcript segments with speaker information
                logger.info("Aligning transcript with speaker segments")
                alignment_start = asyncio.get_event_loop().time()
                speaker_segments = diarization_result["segments"]
                logger.info(f"Aligning {len(processed_segments)} transcript segments with {len(speaker_segments)} speaker segments")
                aligned_segments = align_transcript_with_speakers(processed_segments, speaker_segments)
                alignment_time = asyncio.get_event_loop().time() - alignment_start
                logger.info(f"Speaker alignment completed in {alignment_time:.2f} seconds")
                
                diarization_time = asyncio.get_event_loop().time() - diarization_start
                logger.info(f"Diarization completed in {diarization_time:.2f} seconds with {diarization_result['num_speakers']} speakers detected")
                
                # Return the results with diarization
                result = {
                    "text": result["text"].strip(),
                    "segments": aligned_segments,
                    "num_speakers": diarization_result["num_speakers"]
                }
                logger.info(f"Final result: {len(aligned_segments)} segments with speaker information")
                return result
            except Exception as e:
                # Log the error but continue without diarization
                logger.error(f"Diarization failed: {str(e)}")
                logger.info("Continuing without diarization - using generic speaker labels")
                
                # Create basic speaker segments (as a fallback)
                # Assign speakers based on basic heuristics (e.g., alternating speakers)
                logger.info("Creating fallback speaker segments using basic heuristics")
                fallback_start = asyncio.get_event_loop().time()
                basic_segments = []
                current_speaker = "SPEAKER_1"
                
                for i, segment in enumerate(processed_segments):
                    # Simple alternating speakers if we can't diarize
                    # Only switch speaker if there's a gap of more than 1.5 seconds
                    if i > 0 and segment["start"] - processed_segments[i-1]["end"] > 1.5:
                        previous_speaker = current_speaker
                        current_speaker = "SPEAKER_2" if current_speaker == "SPEAKER_1" else "SPEAKER_1"
                        logger.debug(f"Speaker change detected at {segment['start']:.2f}s: {previous_speaker} → {current_speaker}")
                    
                    basic_segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"],
                        "speaker": current_speaker
                    })
                
                fallback_time = asyncio.get_event_loop().time() - fallback_start
                logger.info(f"Created {len(basic_segments)} fallback segments in {fallback_time:.2f} seconds")
                
                return {
                    "text": result["text"].strip(),
                    "segments": basic_segments,
                    "num_speakers": 2  # Default to 2 speakers in fallback mode
                }

        # Return the full transcribed text and the processed segments
        logger.info("Returning transcription without diarization")
        return {
            "text": result["text"].strip(),  # Full transcribed text (with whitespace removed)
            "segments": processed_segments  # List of segments with start, end times, and cleaned text
        }

    except Exception as e:
        # Log any errors that occur during the transcription process
        logger.error(f"Error in transcribe_audio function: {str(e)}")
        logger.exception("Full traceback:")
        raise Exception(f"Failed to transcribe audio: {str(e)}")

    finally:
        # Clean up the temporary audio file after transcription, regardless of success or failure
        if os.path.exists(audio_path):
            logger.info(f"Cleaning up temporary audio file {audio_path}")
            os.remove(audio_path)
            logger.info(f"Temporary audio file {audio_path} has been deleted.")  # Log cleanup
        
        total_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"Total transcription process completed in {total_time:.2f} seconds")


