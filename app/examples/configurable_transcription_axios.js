/**
 * Example script demonstrating how to use the configurable WhisperX transcription endpoint with Axios.
 *
 * To run this script in Node.js, install axios:
 * npm install axios
 */

// Import Axios
// For Node.js environment:
// const axios = require('axios');
// For ES modules:
// import axios from 'axios';

// API configuration
const API_URL = 'http://localhost:8000/transcribe/configurable'

// Audio file URLs
const REMOTE_AUDIO_URL = 'https://example.com/remote-audio.webm'
const LOCAL_AUDIO_URL = 'https://example.com/local-audio.webm'

/**
 * Configure transcription parameters
 * These can be adjusted based on your specific audio characteristics
 */
const transcriptionConfig = {
  // Basic configuration
  language: 'de', // ISO language code, null for auto-detection
  compute_type: 'float16', // "float16", "float32", or "int8"

  // Transcription parameters
  temperature: 0.0, // Lower temperature for more precision
  beam_size: 5, // Beam size for beam search
  word_timestamps: true, // Enable word-level timestamps
  batch_size: 16, // Batch size for parallelization
  condition_on_previous_text: true, // Use context for better accuracy

  // Silence handling
  vad_filter: true, // Enable voice activity detection
  no_speech_threshold: 0.6, // Higher value = more aggressive with silence
  compression_ratio_threshold: 2.4, // Higher value = more compression allowed

  // VAD parameters
  vad_onset: 0.5, // VAD onset threshold
  vad_offset: 0.363, // VAD offset threshold
  min_silence_duration_ms: 500, // Minimum silence duration
  speech_pad_ms: 100, // Padding around speech segments

  // Alignment
  align_output: true // Whether to align output for word-level timestamps
}

/**
 * Sends the transcription request to the API using Axios
 * @returns {Promise} Promise resolving to the transcription result
 */
async function transcribeAudio() {
  console.log('Sending transcription request with Axios...')

  // Prepare the request payload
  const payload = {
    recording_remote: REMOTE_AUDIO_URL,
    recording_local: LOCAL_AUDIO_URL,
    config: transcriptionConfig
  }

  try {
    // Send the POST request using Axios
    const response = await axios.post(API_URL, payload, {
      headers: {
        'Content-Type': 'application/json'
      }
    })

    // Axios automatically throws for non-2xx responses
    // so if we get here, the request was successful

    // Return the response data
    return response.data
  } catch (error) {
    // Axios error handling
    if (error.response) {
      // The request was made and the server responded with a non-2xx status
      console.error(
        `API Error (${error.response.status}):`,
        error.response.data
      )
      throw new Error(`API request failed with status ${error.response.status}`)
    } else if (error.request) {
      // The request was made but no response was received
      console.error('No response received:', error.request)
      throw new Error('No response received from the server')
    } else {
      // Something happened in setting up the request
      console.error('Request setup error:', error.message)
      throw error
    }
  }
}

/**
 * Main function to execute the example
 */
async function main() {
  console.log(`Transcribing audio with configuration:`)
  console.log(JSON.stringify(transcriptionConfig, null, 2))

  try {
    // Call the transcription API
    const result = await transcribeAudio()

    // Display combined transcript
    console.log('\nCombined Transcript:')
    console.log(result.transcript_combined.text)

    // Display segment count
    console.log(
      `\nTotal segments: ${result.transcript_combined.segments.length}`
    )

    // Optional: Show speakers distribution
    const speakerCounts = {}
    result.transcript_combined.segments.forEach((segment) => {
      speakerCounts[segment.speaker] = (speakerCounts[segment.speaker] || 0) + 1
    })
    console.log('\nSpeaker distribution:')
    console.log(speakerCounts)

    // Optional: Save results to a file (Node.js environment)
    // Uncomment if running in Node.js:
    /*
    const fs = require('fs');
    fs.writeFileSync('transcription_results.json', JSON.stringify(result, null, 2));
    console.log('\nFull results saved to transcription_results.json');
    */

    return result
  } catch (error) {
    console.error('Transcription failed:', error.message)
  }
}

// Execute the main function
main().catch((error) => console.error('Unhandled error:', error))

// For browser environment or modules, you can use this to trigger the transcription
// export { transcribeAudio, main };
