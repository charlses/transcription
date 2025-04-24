# Call Transcription Processor

This Node.js application processes call recordings by:
1. Transcribing audio using WhisperX
2. Creating 30-second audio segments
3. Uploading segments to AWS S3
4. Updating call records with transcriptions and segment information

## Prerequisites

- Node.js (v14 or higher)
- ffmpeg installed on your system
- AWS credentials with S3 access
- WhisperX transcription service running locally

## Setup

1. Install dependencies:
```bash
npm install
```

2. Set up environment variables:
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
```

## Usage

Run the processor:
```bash
npm start
```

The script will:
1. Fetch untranscribed calls from the API
2. Download local and remote recordings
3. Transcribe the audio
4. Create 30-second segments
5. Upload segments to S3
6. Update call records with transcriptions

## Configuration

The following environment variables can be configured:
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `API_BASE_URL`: Base URL for the calls API (default: http://152.53.23.48:42069/api)
- `TRANSCRIPTION_API_URL`: URL for the WhisperX transcription service (default: http://localhost:8000/transcribe/whisperx)

## Output

For each processed call, the script will:
1. Create transcriptions (local, remote, and combined)
2. Generate 30-second audio segments
3. Upload segments to S3
4. Update the call record with:
   - `transcriptLocal`: Local recording transcription
   - `transcriptRemote`: Remote recording transcription
   - `transcriptCombined`: Combined transcription
   - `segmentsLocal`: Array of local audio segments
   - `segmentsRemote`: Array of remote audio segments 