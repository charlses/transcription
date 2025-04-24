# WhisperX Transcription API

This API provides endpoints for transcribing audio files using WhisperX, with support for both standard and configurable transcription options.

## Endpoints

### 1. Standard Transcription

```
POST /transcribe
```

Takes two audio file URLs (local and remote recordings) and transcribes them using pre-configured WhisperX settings.

**Request Body:**

```json
{
  "recording_remote": "https://example.com/remote-audio.webm",
  "recording_local": "https://example.com/local-audio.webm"
}
```

### 2. Configurable Transcription

```
POST /transcribe/configurable
```

Takes two audio file URLs plus a configuration object that allows for customizing the WhisperX transcription parameters.

**Request Body:**

```json
{
  "recording_remote": "https://example.com/remote-audio.webm",
  "recording_local": "https://example.com/local-audio.webm",
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
    "vad_onset": 0.5,
    "vad_offset": 0.363,
    "min_silence_duration_ms": 500,
    "speech_pad_ms": 100,
    "align_output": true
  }
}
```

**Configuration Options:**

| Parameter                   | Type    | Default   | Description                                                              |
| --------------------------- | ------- | --------- | ------------------------------------------------------------------------ |
| language                    | string  | "de"      | ISO language code (e.g., "de", "en", "fr"). Use null for auto-detection. |
| compute_type                | string  | "float16" | Computation precision. Options: "float16", "float32", "int8"             |
| temperature                 | float   | 0.0       | Sampling temperature (lower = more deterministic)                        |
| beam_size                   | integer | 5         | Beam size for beam search                                                |
| word_timestamps             | boolean | true      | Enable word-level timestamps                                             |
| batch_size                  | integer | 16        | Batch size for parallelized transcription                                |
| condition_on_previous_text  | boolean | true      | Use context for improved accuracy                                        |
| vad_filter                  | boolean | true      | Enable Voice Activity Detection filtering                                |
| no_speech_threshold         | float   | 0.6       | Threshold for filtering out silent segments (higher = more aggressive)   |
| compression_ratio_threshold | float   | 2.4       | Threshold for compression (higher = more compression allowed)            |
| vad_onset                   | float   | 0.500     | VAD onset threshold                                                      |
| vad_offset                  | float   | 0.363     | VAD offset threshold                                                     |
| min_silence_duration_ms     | integer | 500       | Minimum duration of silence to be considered (milliseconds)              |
| speech_pad_ms               | integer | 100       | Padding around speech segments (milliseconds)                            |
| align_output                | boolean | true      | Align output for accurate word-level timestamps                          |

## Response Format

Both endpoints return the same response format:

```json
{
  "transcript_remote": {
    "segments": [...],
    "text": "Full transcript of remote audio"
  },
  "transcript_local": {
    "segments": [...],
    "text": "Full transcript of local audio"
  },
  "transcript_combined": {
    "segments": [...],
    "text": "Combined transcript from both audio files"
  }
}
```

The `segments` array contains objects with the following structure:

```json
{
  "start": 0.0,
  "end": 2.5,
  "text": "Segment text",
  "words": [
    {"word": "word1", "start": 0.1, "end": 0.5},
    {"word": "word2", "start": 0.6, "end": 0.9}
  ],
  "speaker": "agent" or "client"
}
```

## Health Check

```
GET /health
```

Returns the health status of the API.

## System Information

```
GET /
```

Returns detailed system information, including GPU availability and model versions.
