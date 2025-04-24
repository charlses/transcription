#!/bin/bash
# Example of using the configurable WhisperX transcription endpoint with curl

# Set API endpoint
API_URL="http://localhost:8000/transcribe/configurable"

# Set audio file URLs (replace with your actual audio URLs)
REMOTE_AUDIO_URL="https://example.com/remote-audio.webm"
LOCAL_AUDIO_URL="https://example.com/local-audio.webm"

# Create a JSON file with the request payload
cat > request.json << EOL
{
  "recording_remote": "${REMOTE_AUDIO_URL}",
  "recording_local": "${LOCAL_AUDIO_URL}",
  "config": {
    "language": "de",
    "compute_type": "float16",
    "temperature": 0.0,
    "beam_size": 5,
    "word_timestamps": true,
    "batch_size": 16,
    "condition_on_previous_text": true,
    "vad_filter": true,
    "no_speech_threshold": 0.6,
    "compression_ratio_threshold": 2.4,
    "vad_onset": 0.500,
    "vad_offset": 0.363,
    "min_silence_duration_ms": 500,
    "speech_pad_ms": 100,
    "align_output": true
  }
}
EOL

# Make the API request with curl
echo "Sending request to ${API_URL}..."
curl -X POST ${API_URL} \
  -H "Content-Type: application/json" \
  -d @request.json \
  --output response.json

echo "Response saved to response.json"

# Optional: Format and display a summary of the response
if command -v jq &> /dev/null; then
  echo -e "\nCombined transcript:"
  jq -r '.transcript_combined.text' response.json
  
  echo -e "\nNumber of segments:"
  jq '.transcript_combined.segments | length' response.json
else
  echo -e "\nInstall jq to display a summary of the response: https://stedolan.github.io/jq/"
fi 