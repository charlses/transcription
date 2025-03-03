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
    try:
        # Create a temporary file path using the URL's hash and appropriate file extension
        temp_dir = tempfile.gettempdir()  # Get the system's temporary directory
        extension = get_file_extension(url)  # Get the correct file extension for the audio file
        temp_path = Path(temp_dir) / f"audio_{hash(url)}{extension}"  # Create the full temp path

        # Download the audio file from the URL
        response = requests.get(url, stream=True)  # Start downloading the file
        if response.status_code != 200:  # Check if the request was successful
            logger.error(f"Failed to download, status code: {response.status_code}")
            raise Exception(f"Failed to download audio with status code {response.status_code}")

        # Write the downloaded content to the temporary file in chunks to avoid memory issues
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Audio downloaded successfully to {temp_path}")
        return str(temp_path)  # Return the path to the downloaded audio file

    except Exception as e:
        # Log any errors that occur during the download process
        logger.error(f"Error downloading audio: {str(e)}")
        raise Exception(f"Failed to download audio: {str(e)}")


async def transcribe_audio(audio_path: str, use_gpu: bool = True, diarize: bool = True) -> dict:
    """
    Transcribes the audio file to text using the Whisper model.

    Args:
        audio_path (str): Path to the audio file to transcribe.
        use_gpu (bool): Whether to use GPU for transcription. Defaults to True.
        diarize (bool): Whether to perform speaker diarization. Defaults to True.

    Returns:
        dict: A dictionary containing the full transcribed text and detailed segment information.

    Raises:
        Exception: If an error occurs during transcription.
    """
    try:
        # Determine the device (CPU or GPU) and load the appropriate Whisper model
        device = "cuda" if use_gpu else "cpu"  # Select device based on `use_gpu` flag
        model = whisper.load_model("large", device=device)  # Load Whisper model (large version)

#         # If using GPU, convert the model to half precision to optimize performance
#         if use_gpu:
#             model = model.half()  # Convert model to use 16-bit precision (half precision)

        # Run the transcription process asynchronously using an executor to avoid blocking
        loop = asyncio.get_event_loop()  # Get the event loop for async execution
        result = await loop.run_in_executor(
            None,  # Use default executor (thread pool)
            lambda: model.transcribe(audio_path)  # Run the transcription function
        )

        # Process the transcription result to extract and clean segment data
        processed_segments = [
            {
                "start": segment["start"],  # Start time of the segment
                "end": segment["end"],  # End time of the segment
                "text": segment["text"].strip()  # Cleaned text from the segment (remove leading/trailing whitespace)
            }
            for segment in result["segments"]
        ]

        # Perform speaker diarization if requested
        if diarize:
            try:
                logger.info("Performing speaker diarization")
                # Create diarization object
                diarizer = SpeakerDiarization(use_gpu=use_gpu)
                
                # Process audio for speaker diarization
                diarization_result = await loop.run_in_executor(
                    None,
                    lambda: diarizer.process_audio(audio_path)
                )
                
                # Align transcript segments with speaker information
                speaker_segments = diarization_result["segments"]
                aligned_segments = align_transcript_with_speakers(processed_segments, speaker_segments)
                
                # Return the results with diarization
                return {
                    "text": result["text"].strip(),
                    "segments": aligned_segments,
                    "num_speakers": diarization_result["num_speakers"]
                }
            except Exception as e:
                # Log the error but continue without diarization
                logger.error(f"Diarization failed: {str(e)}")
                logger.info("Continuing without diarization")

        # Return the full transcribed text and the processed segments
        return {
            "text": result["text"].strip(),  # Full transcribed text (with whitespace removed)
            "segments": processed_segments  # List of segments with start, end times, and cleaned text
        }

    except Exception as e:
        # Log any errors that occur during the transcription process
        logger.error(f"Error in transcribe_audio function: {str(e)}")
        raise Exception(f"Failed to transcribe audio: {str(e)}")

    finally:
        # Clean up the temporary audio file after transcription, regardless of success or failure
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logger.info(f"Temporary audio file {audio_path} has been deleted.")  # Log cleanup


