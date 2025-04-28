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
    url.searchParams.append('populate', 'recording,recordingLocal,recordingRemote')
    url.searchParams.append('pagination[pageSize]', limit.toString())
    url.searchParams.append('pagination[page]', page.toString())
    url.searchParams.append('pagination[withCount]', 'true')
    url.searchParams.append('sort', 'id:desc')

    log(`Fetching page ${page} with limit ${limit} from ${url}`)

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${STRAPI_API_TOKEN}`
      },
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

    log(
      `Found ${data.data.length} calls on page ${page}`
    )

    // Return all calls and pagination 
    console.log(data.meta)
    return {
      calls: data.data,
      meta: data.meta
    }
  } catch (error) {
    log(`Error in fetchCallsNeedingTranscription: ${error.message}`, 'error')
    return { calls: [], meta: { pagination: { pageCount: 0 } } }
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

    // Skip calls without recordings
    if (!attributes.recordingLocal?.data?.[0]?.attributes?.url || 
        !attributes.recordingRemote?.data?.[0]?.attributes?.url) {
      log(`Skipping call ${callId} - missing recordings`, 'warn')
      return call
    }

    // Set status to WAIT while processing
    await fetch(`${RR_API_URL}/calls/${callId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${STRAPI_API_TOKEN}`
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
    const updatedCall = await updateCallWithTranscription(call, transcriptionData)

    return updatedCall
  } catch (error) {
    log(`Error processing call ${callId}: ${error.message}`, 'error')
    throw error
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
      currentPageSize = INITIAL_PAGE_SIZE // Keep the large page size (1000) for initial scan
      
      // Get total records first
      const initialResult = await fetchCallsNeedingTranscription(1, currentPageSize)
      const totalRecords = initialResult.meta.pagination.total
      const totalPages = initialResult.meta.pagination.pageCount
      log(`Total records to scan: ${totalRecords}, will process in ${totalPages} pages with size ${currentPageSize}`)
      
      // Process all pages with large page size
      let currentPage = 1
      while (currentPage <= totalPages) {
        const result = await fetchCallsNeedingTranscription(currentPage, currentPageSize)
        log(`Processing page ${currentPage} of ${totalPages} (${result.calls.length} calls)`)
        
        // Process each call individually
        for (const call of result.calls) {
          try {
            await processCall(call)
          } catch (error) {
            log(`Error processing call ${call.id}: ${error.message}`, 'error')
            // Continue with next call even if this one fails
          }
        }
        
        currentPage++
      }
      
      // Only after processing ALL pages, switch to continuous mode
      log('Completed full database scan')
      isInitialScan = false
      currentPageSize = INITIAL_PAGE_SIZE // Keep using 1000 for continuous mode
      log('Switching to continuous mode with 10-second intervals')
    }
    // In continuous mode, process all pages in sequence
    else {
      log('Starting continuous mode page scan...')
      let currentPage = 1
      let hasMorePages = true
      
      while (hasMorePages) {
        const result = await fetchCallsNeedingTranscription(currentPage, currentPageSize)
        const totalPages = result.meta.pagination.pageCount
        const totalRecords = result.meta.pagination.total
        
        log(`Processing page ${currentPage} of ${totalPages} (${result.calls.length} calls, total records: ${totalRecords})`)
        
        if (result.calls.length === 0) {
          hasMorePages = false
          log('No more calls to process in this cycle')
          break
        }
        
        // Process each call individually
        for (const call of result.calls) {
          try {
            await processCall(call)
          } catch (error) {
            log(`Error processing call ${call.id}: ${error.message}`, 'error')
            // Continue with next call even if this one fails
          }
        }
        
        currentPage++
        
        // Only stop if we've processed all pages
        if (currentPage > totalPages) {
          hasMorePages = false
          log(`Reached the last page (${totalPages}) in this cycle`)
        } else {
          log(`Moving to next page ${currentPage} of ${totalPages}`)
        }
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
      recording_local: attributes.recordingLocal.data[0].attributes.url,
      recording_remote: attributes.recordingRemote.data[0].attributes.url,
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

    // Alignment
    align_output: true // Whether to align output for word-level timestamps
  }
}

// Rest of the file remains unchanged