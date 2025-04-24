/**
 * Example script demonstrating how to use the configurable WhisperX transcription endpoint with JavaScript.
 *
 * This example uses the Fetch API for modern JavaScript environments.
 * To run this script in Node.js, install node-fetch:
 * npm install node-fetch
 */

// For Node.js environment, uncomment this line:
// const fetch = require('node-fetch');

// API configuration
const API_URL = 'http://localhost:8000/transcribe/configurable'

// Audio file URLs
const REMOTE_AUDIO_URL = 'https://example.com/remote-audio.webm'
const LOCAL_AUDIO_URL = 'https://example.com/local-audio.webm'

/**
 * Custom transcription configuration
 * Adjust these parameters to control the transcription process
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
 * Sends the transcription request to the API
 * @returns {Promise} Promise resolving to the transcription result
 */
async function transcribeAudio() {
  console.log('Sending transcription request...')

  // Prepare the request payload
  const payload = {
    recording_remote: REMOTE_AUDIO_URL,
    recording_local: LOCAL_AUDIO_URL,
    config: transcriptionConfig
  }

  try {
    // Send the POST request
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    })

    // Check if the request was successful
    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(
        `API request failed with status ${response.status}: ${errorText}`
      )
    }

    // Parse and return the response
    const result = await response.json()
    return result
  } catch (error) {
    console.error('Error transcribing audio:', error)
    throw error
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

    // Optional: Display the first few segments
    console.log('\nFirst few segments:')
    const previewSegments = result.transcript_combined.segments.slice(0, 3)
    console.log(JSON.stringify(previewSegments, null, 2))

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

// For browser environment, you can use this to trigger the transcription
// export { transcribeAudio, main };
