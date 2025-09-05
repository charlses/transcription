# Transcription Service

This service handles the transcription of call recordings using WhisperX, with support for multiple API endpoints and automatic failover.

## Overview

The service monitors a Strapi API for calls that need transcription, processes them using WhisperX, and updates their status in the database. It supports multiple API endpoints in a round-robin fashion with automatic failover.

## Configuration

### Environment Variables

Create a `.env` file with the following configuration:

```env
# API Configuration (JSON arrays)
API_URLS=["https://api1.example.com", "https://api2.example.com"]
API_TOKENS=["token1", "token2"]

# Transcription API URL
TRANSCRIPTION_API_URL="http://localhost:8000/transcribe/configurable"
```

## Service Flow

1. **Initialization**

   - Loads configuration from environment variables
   - Validates API URLs and tokens
   - Initializes round-robin API selection system
   - Sets up error tracking for each API

2. **Main Loop**

   - Runs continuously, checking for calls that need transcription
   - Uses a configurable page size (default: 1000 calls per page)
   - Implements exponential backoff for API errors

3. **Call Processing Flow**

   ```
   Start
   ↓
   Fetch calls needing transcription
   ↓
   For each call:
   ├─ Set status to 'WAIT'
   ├─ Get recording URLs from call data
   ├─ Send to WhisperX for transcription
   ├─ Update call with transcription results
   └─ Set status to 'DONE' or 'ERROR'
   ```

4. **API Round-Robin System**

   - Rotates through configured API endpoints
   - Tracks errors per API
   - Implements exponential backoff (30s \* error count)
   - Automatically recovers from temporary API issues

5. **Error Handling**
   - Tracks errors per API endpoint
   - Implements exponential backoff
   - Automatically switches to next available API
   - Updates call status to 'ERROR' if processing fails

## Status Codes

Calls can have the following statuses:

- `WAIT`: Currently being processed
- `DONE`: Successfully transcribed
- `ERROR`: Failed to process

## Error Recovery

1. **API Errors**

   - Tracks errors per API endpoint
   - Implements 30-second delay \* error count
   - Automatically switches to next available API
   - Resets error count on successful API calls

2. **Processing Errors**
   - Updates call status to 'ERROR'
   - Logs detailed error information
   - Continues with next call
   - Maintains error tracking for API endpoints

## Performance Considerations

1. **Batch Processing**

   - Configurable page size (default: 1000)
   - Processes calls in batches
   - Implements rate limiting between batches

2. **Resource Management**
   - Manages GPU memory (if available)
   - Implements proper error handling and recovery

## Logging

The service provides detailed logging:

- API selection and rotation
- Call processing status
- Error tracking and recovery
- Performance metrics

## Dependencies

- Node.js
- WhisperX (Python)
- CUDA (optional, for GPU acceleration)

## Usage

1. Configure environment variables
2. Start the service:
   ```bash
   node transcription-service.js
   ```

The service will automatically:

- Monitor for calls needing transcription
- Process them using WhisperX
- Update their status in the database
- Handle errors and recovery
