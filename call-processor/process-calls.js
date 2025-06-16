require('dotenv').config()
const fetch = require('node-fetch')
const AWS = require('aws-sdk')

// Validate required environment variables
const requiredEnvVars = [
  'AWS_ACCESS_KEY_ID',
  'AWS_SECRET_ACCESS_KEY',
  'AWS_REGION',
  'AWS_S3_BUCKET'
]
const missingEnvVars = requiredEnvVars.filter(
  (varName) => !process.env[varName]
)

if (missingEnvVars.length > 0) {
  console.error(
    'Missing required environment variables:',
    missingEnvVars.join(', ')
  )
  console.error('Please check your .env file')
  process.exit(1)
}

// Configure AWS
AWS.config.update({
  region: process.env.AWS_REGION,
  accessKeyId: process.env.AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
})

// Constants
const API_BASE_URL =
  process.env.API_BASE_URL || 'https://whisper-training-prep.tess-dev.de/api'
const TRANSCRIPTION_API_URL =
  process.env.TRANSCRIPTION_API_URL || 'http://localhost:8000/transcribe/configurable'
const RR_API_URL = process.env.RR_API_URL

// Transcription configuration
const transcriptionConfig = {
  // Basic configuration
  language: 'de', // ISO language code, null for auto-detection
  compute_type: 'float32', // "float16", "float32", or "int8"

  // Transcription parameters
  temperature: 0.0, // Lower temperature for more precision
  beam_size: 5, // Beam size for beam search
  word_timestamps: true, // Enable word-level timestamps
  batch_size: 16, // Batch size for parallelization
  condition_on_previous_text: true, // Use context for better accuracy

  // Silence handling
  vad_filter: true, // Enable voice activity detection
  no_speech_threshold: 0.4, // Lowered from 0.6 to be less aggressive with silence detection
  compression_ratio_threshold: 2.4, // Higher value = more compression allowed

  // VAD parameters
  vad_onset: 0.5, // VAD onset threshold
  vad_offset: 0.363, // VAD offset threshold
  min_silence_duration_ms: 1000, // Increased from 500ms to 1000ms to better handle longer silences
  speech_pad_ms: 200, // Increased from 100ms to 200ms to capture more context around speech

  // Alignment
  align_output: true // Whether to align output for word-level timestamps
}

// Add timestamp to log messages
function log(message, type = 'info') {
  const timestamp = new Date().toISOString()
  const prefix = `[${timestamp}]`

  switch (type) {
    case 'error':
      console.error(`${prefix} ❌ ${message}`)
      break
    case 'success':
      console.log(`${prefix} ✅ ${message}`)
      break
    case 'warn':
      console.warn(`${prefix} ⚠️ ${message}`)
      break
    default:
      console.log(`${prefix} ℹ️ ${message}`)
  }
}

async function processCall(call) {
  const startTime = Date.now()
  try {
    log(`Starting to process call ${call._id}`)
    log(
      `Call details: Duration=${call.duration}s, Type=${call.type}, Number=${call.number}`
    )

    // Transcribe audio
    log('Starting transcription...')
    let transcriptionData
    try {
      const transcriptionResponse = await fetch(TRANSCRIPTION_API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          recording_local: call.recordingLocal.url,
          recording_remote: call.recordingRemote.url,
          config: transcriptionConfig
        })
      })

      if (!transcriptionResponse.ok) {
        const errorText = await transcriptionResponse.text()
        log(
          `Transcription API Response Status: ${transcriptionResponse.status}`,
          'error'
        )
        log(
          `Transcription API Response Headers: ${JSON.stringify(
            Object.fromEntries(transcriptionResponse.headers.entries())
          )}`,
          'error'
        )
        log(`Transcription API Error Response: ${errorText}`, 'error')
        throw new Error(
          `Transcription failed: ${transcriptionResponse.status} ${transcriptionResponse.statusText}\nResponse: ${errorText}`
        )
      }

      transcriptionData = await transcriptionResponse.json()

      // Validate transcription response structure
      if (!transcriptionData || typeof transcriptionData !== 'object') {
        throw new Error('Invalid transcription response structure')
      }

      // Ensure we have the required fields
      if (
        !transcriptionData.transcript_local ||
        !transcriptionData.transcript_remote ||
        !transcriptionData.transcript_combined
      ) {
        throw new Error('Missing required transcription fields')
      }

      // Validate segment structure
      if (
        !Array.isArray(transcriptionData.transcript_local.segments) ||
        !Array.isArray(transcriptionData.transcript_remote.segments) ||
        !Array.isArray(transcriptionData.transcript_combined.segments)
      ) {
        throw new Error('Invalid segment structure in transcription response')
      }

      log('Transcription completed successfully')
    } catch (error) {
      log(`Transcription error details: ${error.message}`, 'error')
      if (error.cause) {
        log(`Error cause: ${error.cause}`, 'error')
      }
      throw error
    }

    // Update call in API
    log('Updating call record in API...')
    const updateUrl = `${API_BASE_URL}/calls/${call._id}/transcripts`
    log(`Update URL: ${updateUrl}`)

    const updateResponse = await fetch(updateUrl, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        transcriptLocal: transcriptionData.transcript_local.text,
        transcriptRemote: transcriptionData.transcript_remote.text,
        transcriptCombined: transcriptionData.transcript_combined.text,
        segmentsLocal: transcriptionData.transcript_local.segments,
        segmentsRemote: transcriptionData.transcript_remote.segments,
        segmentsCombined: transcriptionData.transcript_combined.segments,
        transcribed: true
      })
    })

    if (!updateResponse.ok) {
      const errorText = await updateResponse.text()
      throw new Error(
        `Failed to update call: ${updateResponse.status} ${updateResponse.statusText}\nURL: ${updateUrl}\nResponse: ${errorText}`
      )
    }

    const updateUrl2 = `${RR_API_URL}/calls/${call._id}`
    const updateResponse2 = await fetch(updateUrl2, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        transcriptionLocal: transcriptionData.transcript_local,
        transcriptionRemote: transcriptionData.transcript_remote,
        transcription_status: 'DONE'
      })
    })

    if (!updateResponse2.ok) {
      const errorText2 = await updateResponse2.text()
      log(
        `Failed to update call record in RR API: ${updateResponse2.status} ${updateResponse2.statusText}\nResponse: ${errorText2}`,
        'error'
      )
    }

    const updateResult = await updateResponse.json()
    log('Call record updated successfully', 'success')

    const processingTime = ((Date.now() - startTime) / 1000).toFixed(2)
    log(
      `Successfully processed call ${call._id} in ${processingTime}s`,
      'success'
    )
  } catch (error) {
    log(`Error processing call ${call._id}: ${error.message}`, 'error')
    // Don't throw the error, just log it and continue with the next call
  }
}

// Update rate limiting configuration
const RATE_LIMIT_DELAY = 1000 // 1 second between requests
const MAX_RETRIES = 3
const RETRY_DELAY = 5000 // 5 seconds between retries
const BATCH_SIZE = 100 // Process 100 calls at a time
const BATCH_DELAY = 10000 // 10 seconds between batches

// Add delay helper function
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

// Add retry logic for API calls
async function fetchWithRetry(url, options = {}, retries = MAX_RETRIES) {
  for (let i = 0; i < retries; i++) {
    try {
      const response = await fetch(url, options)
      if (response.ok) {
        return response
      }

      if (response.status === 429) {
        // Too Many Requests
        const waitTime = RETRY_DELAY * Math.pow(2, i) // Exponential backoff
        log(
          `Rate limited. Waiting ${waitTime / 1000}s before retry ${
            i + 1
          }/${retries}...`,
          'warn'
        )
        await delay(waitTime)
        continue
      }

      throw new Error(`HTTP error! status: ${response.status}`)
    } catch (error) {
      if (i === retries - 1) throw error
      const waitTime = RETRY_DELAY * Math.pow(2, i)
      log(
        `Request failed. Waiting ${waitTime / 1000}s before retry ${
          i + 1
        }/${retries}...`,
        'warn'
      )
      await delay(waitTime)
    }
  }
}

async function processPage(page, limit = BATCH_SIZE) {
  try {
    log(`Fetching page ${page} with ${limit} calls...`)
    const response = await fetchWithRetry(
      `${API_BASE_URL}/calls?page=${page}&limit=${limit}`
    )

    const data = await response.json()
    log(`Page ${page} fetched successfully (${data.data.length} calls)`)

    // Process each call in the current page
    for (const call of data.data) {
      if (call.recordingLocal && call.recordingRemote && !call.transcribed) {
        await processCall(call)
        // Add delay between processing calls
        await delay(RATE_LIMIT_DELAY)
      } else {
        log(
          `Skipping call ${call._id} - ${
            !call.recordingLocal
              ? 'missing local recording'
              : !call.recordingRemote
              ? 'missing remote recording'
              : 'already transcribed'
          }`,
          'warn'
        )
      }
    }

    return data.pagination
  } catch (error) {
    log(`Error processing page ${page}: ${error.message}`, 'error')
    return null
  }
}

async function main() {
  const startTime = Date.now()
  try {
    log('Starting call processing script')
    let currentPage = 1
    let hasMorePages = true
    let processedCalls = 0
    let totalCalls = 0

    // First, get the total number of pages
    const initialResponse = await fetchWithRetry(
      `${API_BASE_URL}/calls?page=1&limit=${BATCH_SIZE}`
    )
    const initialData = await initialResponse.json()
    totalCalls = initialData.pagination.total
    log(`Total calls to process: ${totalCalls}`)

    while (hasMorePages) {
      const pagination = await processPage(currentPage, BATCH_SIZE)

      if (!pagination) {
        log('Failed to process page, stopping', 'error')
        break
      }

      processedCalls += pagination.total
      const progress = ((processedCalls / totalCalls) * 100).toFixed(1)
      log(
        `Progress: Page ${currentPage}/${pagination.totalPages} (${processedCalls}/${totalCalls} calls, ${progress}%)`
      )

      // Check if we've reached the last page
      if (currentPage >= pagination.totalPages) {
        hasMorePages = false
      } else {
        currentPage++
      }

      // Add a longer delay between batches to avoid rate limits
      if (hasMorePages) {
        log(`Waiting ${BATCH_DELAY / 1000} seconds before next batch...`)
        await delay(BATCH_DELAY)
      }
    }

    const totalTime = ((Date.now() - startTime) / 1000).toFixed(2)
    log(
      `Finished processing all pages. Total time: ${totalTime}s, Total calls processed: ${processedCalls}`,
      'success'
    )

    main()
  } catch (error) {
    log(`Error in main process: ${error.message}`, 'error')
    main()
  }
}

main()
