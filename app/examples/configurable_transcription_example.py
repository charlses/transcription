#!/usr/bin/env python3
"""
Example script demonstrating how to use the configurable WhisperX transcription endpoint.
"""

import asyncio
import json
import aiohttp
import argparse
from typing import Dict, Any

# Default API URL
DEFAULT_API_URL = "http://localhost:8000"

async def transcribe_audio(
    api_url: str,
    remote_audio_url: str,
    local_audio_url: str,
    config: Dict[str, Any]
) -> Dict:
    """
    Transcribe audio files using the configurable WhisperX transcription API.
    
    Args:
        api_url: Base URL of the transcription API
        remote_audio_url: URL to the remote (client) audio file
        local_audio_url: URL to the local (agent) audio file
        config: Transcription configuration parameters
        
    Returns:
        Transcription results including remote, local and combined transcripts
    """
    # Prepare the request payload
    payload = {
        "recording_remote": remote_audio_url,
        "recording_local": local_audio_url,
        "config": config
    }
    
    # Make the request to the API
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{api_url}/transcribe/configurable", json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API request failed with status {response.status}: {error_text}")
            
            # Parse and return the response
            return await response.json()

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Transcribe audio using configurable WhisperX API")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API base URL")
    parser.add_argument("--remote-audio", required=True, help="URL to remote audio file")
    parser.add_argument("--local-audio", required=True, help="URL to local audio file")
    parser.add_argument("--language", default="de", help="Language code (e.g., 'de', 'en')")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size")
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD filtering")
    parser.add_argument("--no-align", action="store_true", help="Disable output alignment")
    
    args = parser.parse_args()
    
    # Prepare configuration
    config = {
        "language": args.language,
        "temperature": args.temp,
        "beam_size": args.beam_size,
        "vad_filter": not args.no_vad,
        "align_output": not args.no_align
    }
    
    print(f"Transcribing audio with configuration: {json.dumps(config, indent=2)}")
    
    try:
        # Call the transcription API
        results = await transcribe_audio(
            args.api_url,
            args.remote_audio,
            args.local_audio,
            config
        )
        
        # Print combined transcript
        print("\nCombined Transcript:")
        print(results["transcript_combined"]["text"])
        
        # Print segment information
        print(f"\nTotal segments: {len(results['transcript_combined']['segments'])}")
        
        # Save full results to file
        with open("transcription_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("\nFull results saved to transcription_results.json")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 