require('dotenv').config()
const fetch = require('node-fetch')

// API URLs
const RR_API_URL = process.env.RR_API_URL
const TRANSCRIPTION_API_URL =
  process.env.TRANSCRIPTION_API_URL ||
  'http://localhost:8000/transcribe/configurable'

const STRAPI_API_TOKEN = process.env.STRAPI_API_TOKEN

// Constants for rate limiting and batch processing
const INITIAL_PAGE_SIZE = 1000 // Start with a large page size
const SMALL_PAGE_SIZE = 10 // Smaller page size for frequent checks
const LOOP_INTERVAL = 10000 // 10 seconds between checks
const DELAY_BETWEEN_CALLS = 1000 // 1 second between processing calls

// Tracking variables
let currentPageSize = INITIAL_PAGE_SIZE
let isInitialScan = true // Flag to track if we're in the initial full scan

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

// Add delay helper function
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

/**
 * Fetch calls from the API that need transcription
 * @param {number} page - The page number to fetch
 * @param {number} limit - Number of calls per page
 * @returns {Promise<Object>} Object with calls array and pagination info
 */
async function fetchCallsNeedingTranscription(
  page = 1,
  limit = currentPageSize
) {
  try {
    // Create URL for fetching calls from Strapi that don't have transcription_status "DONE"
    const url = new URL(`${RR_API_URL}/calls`)

    // Use Strapi's filter format: filters[field][operator]=value
    url.searchParams.append(
      'populate',
      'recording,recordingLocal,recordingRemote'
    )
    url.searchParams.append('pagination[limit]', limit.toString())
    url.searchParams.append('pagination[page]', page.toString())

    // Filter calls where transcription_status is not DONE
    url.searchParams.append('filters[transcription_status][$ne]', 'DONE')

    // Add sorting to get newest calls first (changed to sort by id desc)
    url.searchParams.append('sort', 'id:desc')

    log(`Fetching page ${page} with limit ${limit} from ${url}`)

    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${STRAPI_API_TOKEN}`
      },
      method: 'GET'
    })

    if (!response.ok) {
      throw new Error(
        `Failed to fetch calls: ${response.status} ${response.statusText}`
      )
    }

    const data = await response.json()

    // Strapi responds with { data: [...], meta: {...} }
    if (!data || !Array.isArray(data.data)) {
      log(
        `Unexpected API response format: ${JSON.stringify(data).substring(
          0,
          200
        )}...`,
        'error'
      )
      throw new Error('Invalid response format from Strapi API')
    }

    // Filter calls that have recording URLs
    const validCalls = data.data.filter((call) => {
      // Get the actual attributes from Strapi response
      const attributes = call.attributes || call

      return (
        attributes.recordingLocal &&
        attributes.recordingLocal.url &&
        attributes.recordingRemote &&
        attributes.recordingRemote.url
      )
    })

    // If we're in initial scan and got no valid calls from a large page,
    // reduce the page size for future scans
    if (
      isInitialScan &&
      limit > SMALL_PAGE_SIZE &&
      validCalls.length === 0 &&
      data.data.length > 0
    ) {
      log(
        `No calls with recordings found in a page of ${data.data.length} calls. Reducing page size for future scans.`,
        'warn'
      )
      currentPageSize = SMALL_PAGE_SIZE
    }

    log(
      `Found ${validCalls.length} calls that need transcription (out of ${data.data.length} total calls on page ${page})`
    )

    // Return both the valid calls and pagination metadata
    return {
      calls: validCalls,
      pagination: data.meta
        ? data.meta.pagination
        : {
            page,
            pageSize: limit,
            pageCount: Math.ceil(data.data.length / limit),
            total: data.data.length
          }
    }
  } catch (error) {
    log(`Error fetching calls: ${error.message}`, 'error')
    // Return empty result instead of throwing, to keep the continuous loop running
    return { calls: [], pagination: { page, pageCount: 0, total: 0 } }
  }
}

/**
 * Configure WhisperX transcription parameters
 * @returns {Object} Configuration object for WhisperX
 */
function getTranscriptionConfig() {
  return {
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
    // Removed min_silence_duration_ms and speech_pad_ms as per user's edits

    // Alignment
    align_output: true // Whether to align output for word-level timestamps
  }
}

/**
 * Transcribe a call using the WhisperX configurable endpoint
 * @param {Object} call - Call object with recording URLs
 * @returns {Promise<Object>} Transcription results
 */
async function transcribeCall(call) {
  const callId = call.id || call._id

  try {
    log(`Transcribing call ${callId}...`)

    // Handle Strapi's data structure - attributes might be nested
    const attributes = call.attributes || call

    // Prepare the transcription request with configurable parameters
    const transcriptionPayload = {
      recording_local: attributes.recordingLocal.url,
      recording_remote: attributes.recordingRemote.url,
      config: getTranscriptionConfig()
    }

    // Call the transcription API
    const response = await fetch(TRANSCRIPTION_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(transcriptionPayload)
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(
        `Transcription failed: ${response.status} ${response.statusText} - ${errorText}`
      )
    }

    const transcriptionData = await response.json()

    // Validate transcription response
    if (
      !transcriptionData ||
      !transcriptionData.transcript_local ||
      !transcriptionData.transcript_remote
    ) {
      throw new Error('Invalid transcription response format')
    }

    log(`Call ${callId} transcribed successfully`)
    return transcriptionData
  } catch (error) {
    log(`Error transcribing call ${callId}: ${error.message}`, 'error')
    throw error
  }
}

/**
 * Update a call with transcription results
 * @param {Object} call - Call object to update
 * @param {Object} transcriptionData - Transcription data
 * @returns {Promise<Object>} Updated call object
 */
async function updateCallWithTranscription(call, transcriptionData) {
  const callId = call.id || call._id

  try {
    log(`Updating call ${callId} with transcription results...`)

    // Create the update payload for Strapi
    // Changed to save the entire transcript object as per user's edit
    const updatePayload = {
      data: {
        transcriptionLocal: transcriptionData.transcript_local,
        transcriptionRemote: transcriptionData.transcript_remote,
        transcription_status: 'DONE'
      }
    }

    // Update the call in the Strapi API
    const response = await fetch(`${RR_API_URL}/calls/${callId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${STRAPI_API_TOKEN}`
      },
      body: JSON.stringify(updatePayload)
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(
        `Failed to update call: ${response.status} ${response.statusText} - ${errorText}`
      )
    }

    const updatedCall = await response.json()
    log(
      `Call ${callId} updated successfully with transcription_status=DONE`,
      'success'
    )
    return updatedCall
  } catch (error) {
    log(`Error updating call ${callId}: ${error.message}`, 'error')

    // Try to mark as FAILED if update fails
    try {
      const failResponse = await fetch(`${RR_API_URL}/calls/${callId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${STRAPI_API_TOKEN}`
        },
        body: JSON.stringify({
          data: {
            transcription_status: 'FAILED'
          }
        })
      })

      if (failResponse.ok) {
        log(`Call ${callId} marked as FAILED`, 'warn')
      }
    } catch (markFailedError) {
      log(
        `Error marking call ${callId} as FAILED: ${markFailedError.message}`,
        'error'
      )
    }

    throw error
  }
}

/**
 * Process a single call - transcribe and update
 * @param {Object} call - Call object to process
 * @returns {Promise<Object>} Updated call
 */
async function processCall(call) {
  const callId = call.id || call._id

  try {
    const attributes = call.attributes || call

    // Skip calls that already have DONE status
    if (attributes.transcription_status === 'DONE') {
      log(`Skipping call ${callId} - already has DONE status`, 'warn')
      return call
    }

    // Set status to WAIT while processing
    await fetch(`${RR_API_URL}/calls/${callId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        data: {
          transcription_status: 'WAIT'
        }
      })
    })

    // Transcribe the call
    const transcriptionData = await transcribeCall(call)

    // Update the call with transcription results
    const updatedCall = await updateCallWithTranscription(
      call,
      transcriptionData
    )

    return updatedCall
  } catch (error) {
    log(`Error processing call ${callId}: ${error.message}`, 'error')
    throw error
  }
}

/**
 * Process all calls in a batch
 * @param {Array} calls - Array of call objects to process
 */
async function processBatch(calls) {
  if (calls.length === 0) {
    log('No calls to process in this batch')
    return
  }

  log(`Processing batch of ${calls.length} calls...`)

  // Process each call with delay between them
  for (let i = 0; i < calls.length; i++) {
    const call = calls[i]

    try {
      await processCall(call)
    } catch (error) {
      // Log error but continue with next call
      log(
        `Failed to process call ${call.id}, continuing with next call`,
        'error'
      )
    }

    // Add delay between processing calls to avoid rate limiting
    if (i < calls.length - 1) {
      await delay(DELAY_BETWEEN_CALLS)
    }
  }

  log(`Completed processing batch of ${calls.length} calls`, 'success')
}

/**
 * Process all pages of calls
 */
async function processAllPages() {
  try {
    log('Starting full scan of all pages...')

    let currentPage = 1
    let hasMorePages = true
    let totalProcessed = 0

    // Process all pages
    while (hasMorePages) {
      const result = await fetchCallsNeedingTranscription(
        currentPage,
        currentPageSize
      )

      if (result.calls.length > 0) {
        await processBatch(result.calls)
        totalProcessed += result.calls.length
      }

      // Check if there are more pages
      if (
        result.pagination &&
        result.pagination.page < result.pagination.pageCount
      ) {
        currentPage++
        log(
          `Moving to next page: ${currentPage} of ${result.pagination.pageCount}`
        )
      } else {
        hasMorePages = false
        log('Reached the last page')
      }
    }

    log(
      `Completed processing all pages. Total calls processed: ${totalProcessed}`
    )
    return totalProcessed
  } catch (error) {
    log(`Error processing all pages: ${error.message}`, 'error')
    return 0
  }
}

/**
 * Main function that runs continuously
 */
async function runContinuously() {
  try {
    // Start with the initial full scan
    if (isInitialScan) {
      log('Starting initial full scan of all pages...')
      await processAllPages()
      isInitialScan = false
      currentPageSize = SMALL_PAGE_SIZE // Switch to smaller page size for continuous mode
      log(
        'Completed initial scan, switching to continuous mode with 10-second intervals'
      )
    }

    // In continuous mode, just check the first page of recent calls
    else {
      log('Checking recent calls in continuous mode...')
      const result = await fetchCallsNeedingTranscription(1, currentPageSize)
      if (result.calls.length > 0) {
        await processBatch(result.calls)
      }
    }
  } catch (error) {
    log(`Error in continuous processing: ${error.message}`, 'error')
  }

  // Schedule the next run regardless of success or failure
  log(`Scheduling next run in ${LOOP_INTERVAL / 1000} seconds...`)
  setTimeout(runContinuously, LOOP_INTERVAL)
}

// Start the continuous loop
log('Starting transcription service in continuous mode')
runContinuously()
