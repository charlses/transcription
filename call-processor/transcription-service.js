require("dotenv").config();
const fetch = require("node-fetch");

// Parse API URLs and tokens from JSON arrays in .env
const API_URLS = JSON.parse(process.env.API_URLS || "[]");
const API_TOKENS = JSON.parse(process.env.API_TOKENS || "[]");
const TRANSCRIPTION_API_URL =
  process.env.TRANSCRIPTION_API_URL ||
  "http://localhost:8000/transcribe/configurable";

// Validate API configuration
if (API_URLS.length === 0 || API_TOKENS.length === 0) {
  console.error("No API URLs or tokens configured in .env");
  process.exit(1);
}

if (API_URLS.length !== API_TOKENS.length) {
  console.error("Number of API URLs must match number of API tokens");
  process.exit(1);
}

// Constants for rate limiting and batch processing
const INITIAL_PAGE_SIZE = 1000; // Start with a large page size
const LOOP_INTERVAL = 10000; // 10 seconds between checks
const API_ERROR_DELAY = 30000; // 30 seconds delay after API error

// Tracking variables
let currentPageSize = INITIAL_PAGE_SIZE;
let isInitialScan = true; // Flag to track if we're in the initial full scan
let currentApiIndex = 0; // Track current API being used
let apiErrorCounts = new Map(); // Track errors per API
let shouldRestartFromPage1 = false; // Flag to restart from page 1 in next cycle
let page1RetryCount = 0; // Track how many times we've tried page 1
const MAX_PAGE1_RETRIES = 3; // Maximum number of retries for page 1

// Add timestamp to log messages
function log(message, type = "info") {
  const timestamp = new Date().toISOString();
  const prefix = `[${timestamp}]`;

  switch (type) {
    case "error":
      console.error(`${prefix} ❌ ${message}`);
      break;
    case "success":
      console.log(`${prefix} ✅ ${message}`);
      break;
    case "warn":
      console.warn(`${prefix} ⚠️ ${message}`);
      break;
    default:
      console.log(`${prefix} ℹ️ ${message}`);
  }
}

// Add delay helper function
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Get next available API
function getNextApi() {
  if (!API_URLS.length) {
    log("No API URLs configured", "error");
    return null;
  }

  const startIndex = currentApiIndex;
  do {
    // Ensure currentApiIndex is a valid number
    if (isNaN(currentApiIndex) || currentApiIndex < 0) {
      currentApiIndex = 0;
    }

    const apiUrl = API_URLS[currentApiIndex];
    if (!apiUrl) {
      log(`Invalid API URL at index ${currentApiIndex}`, "error");
      currentApiIndex = (currentApiIndex + 1) % API_URLS.length;
      if (currentApiIndex === startIndex) {
        return null;
      }
      continue;
    }

    const errorCount = apiErrorCounts.get(apiUrl) || 0;

    // If this API has had errors, check if it's time to retry
    if (errorCount > 0) {
      const lastErrorTime = apiErrorCounts.get(`${apiUrl}_lastError`) || 0;
      const timeSinceLastError = Date.now() - lastErrorTime;

      if (timeSinceLastError < API_ERROR_DELAY * errorCount) {
        // Move to next API
        currentApiIndex = (currentApiIndex + 1) % API_URLS.length;
        if (currentApiIndex === startIndex) {
          // If we've checked all APIs and none are available, wait
          return null;
        }
        continue;
      }
    }

    // This API is available
    return {
      url: apiUrl,
      token: API_TOKENS[currentApiIndex],
      index: currentApiIndex,
    };
  } while (currentApiIndex !== startIndex);

  return null;
}

// Record API error
function recordApiError(apiUrl) {
  const currentErrors = apiErrorCounts.get(apiUrl) || 0;
  apiErrorCounts.set(apiUrl, currentErrors + 1);
  apiErrorCounts.set(`${apiUrl}_lastError`, Date.now());
}

// Record API success
function recordApiSuccess(apiUrl) {
  apiErrorCounts.set(apiUrl, 0);
  apiErrorCounts.delete(`${apiUrl}_lastError`);
}

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
  const api = getNextApi();
  if (!api) {
    log("No available APIs at the moment, waiting...", "warn");
    await delay(API_ERROR_DELAY);
    return { calls: [], meta: { pagination: { pageCount: 0 } } };
  }

  try {
    // Create URL for fetching calls from Strapi that don't have transcription_status "DONE"
    const url = new URL(`${api.url}/calls`);
    if (!url.hostname) {
      throw new Error(`Invalid API URL: ${api.url}`);
    }

    // Use Strapi's filter format: filters[field][operator]=value
    url.searchParams.append(
      "populate",
      "recording,recordingLocal,recordingRemote"
    );

    // Filter for calls that have both recordings and are not done
    url.searchParams.append("filters[recordingLocal][data][$notNull]", "true");
    url.searchParams.append("filters[recordingRemote][data][$notNull]", "true");
    url.searchParams.append("filters[transcription_status][$ne]", "DONE");
    url.searchParams.append("pagination[pageSize]", limit.toString());
    url.searchParams.append("pagination[page]", page.toString());
    url.searchParams.append("pagination[withCount]", "true");
    url.searchParams.append("sort", "id:desc");

    log(`Fetching page ${page} with limit ${limit} from ${api.url}`);

    const response = await fetch(url, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${api.token}`,
      },
    });

    if (!response.ok) {
      throw new Error(
        `Failed to fetch calls: ${response.status} ${response.statusText}`
      );
    }

    const data = await response.json();
    recordApiSuccess(api.url);

    if (!data || !Array.isArray(data.data)) {
      throw new Error("Invalid response format from API");
    }

    // Safety filter: ensure all calls have recordings (in case Strapi filtering fails)
    const transcribableCalls = data.data.filter(
      (call) =>
        call.attributes.recordingLocal?.data?.[0]?.attributes?.url &&
        call.attributes.recordingRemote?.data?.[0]?.attributes?.url
    );

    log(
      `Found ${transcribableCalls.length} transcribable calls on page ${page}`
    );

    // Add the API info to each call for consistent processing
    const callsWithApi = transcribableCalls.map((call) => ({
      ...call,
      _api: api, // Store the API info with the call
    }));

    return {
      calls: callsWithApi,
      meta: data.meta,
    };
  } catch (error) {
    log(
      `Error in fetchCallsNeedingTranscription for ${api.url}: ${error.message}`,
      "error"
    );
    recordApiError(api.url);
    return { calls: [], meta: { pagination: { pageCount: 0 } } };
  } finally {
    // Move to next API for next request (unless we're restarting from page 1)
    if (!shouldRestartFromPage1) {
      log(
        `DEBUG: fetchCallsNeedingTranscription - Moving to next API, currentApiIndex: ${currentApiIndex} -> ${
          (currentApiIndex + 1) % API_URLS.length
        }`
      );
      currentApiIndex = (currentApiIndex + 1) % API_URLS.length;
    } else {
      log(
        `DEBUG: fetchCallsNeedingTranscription - Not moving to next API due to restart flag, staying at currentApiIndex: ${currentApiIndex}`
      );
    }
  }
}

/**
 * Process a single call - transcribe and update
 * @param {Object} call - Call object to process
 * @returns {Promise<Object>} Updated call
 */
async function processCall(call) {
  // Use the API that was used to fetch this call
  const api = call._api;
  if (!api) {
    log("No API information found for call", "error");
    return;
  }

  const callId = call.id;
  try {
    // Set status to WAIT while processing
    await fetch(`${api.url}/calls/${callId}`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${api.token}`,
      },
      body: JSON.stringify({
        data: {
          transcription_status: "WAIT",
        },
      }),
    });

    // Transcribe the call (we know it has recordings since it passed the API filter)
    const transcriptionData = await transcribeCall(call);

    // Update the call with transcription results
    const updatedCall = await updateCallWithTranscription(
      call,
      transcriptionData,
      api
    );

    recordApiSuccess(api.url);
    return updatedCall;
  } catch (error) {
    log(
      `Error processing call ${callId} with ${api.url}: ${error.message}`,
      "error"
    );
    recordApiError(api.url);

    // Update status to ERROR if processing fails
    try {
      await fetch(`${api.url}/calls/${callId}`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${api.token}`,
        },
        body: JSON.stringify({
          data: {
            transcription_status: "ERROR",
          },
        }),
      });
    } catch (updateError) {
      log(
        `Failed to update error status for call ${callId}: ${updateError.message}`,
        "error"
      );
    }
  }
}

/**
 * Main function that runs continuously
 */
async function runContinuously() {
  try {
    // Start with the initial full scan
    if (isInitialScan) {
      log("Starting initial full scan of all pages...");
      currentPageSize = INITIAL_PAGE_SIZE; // Keep the large page size (1000) for initial scan

      // Get total records first
      const initialResult = await fetchCallsNeedingTranscription(
        1,
        currentPageSize
      );
      const totalRecords = initialResult.meta.pagination.total;
      const totalPages = initialResult.meta.pagination.pageCount;
      log(
        `Total records to scan: ${totalRecords}, will process in ${totalPages} pages with size ${currentPageSize}`
      );

      // Process all pages with large page size
      let currentPage = 1;
      while (currentPage <= totalPages) {
        const result = await fetchCallsNeedingTranscription(
          currentPage,
          currentPageSize
        );
        log(
          `Processing page ${currentPage} of ${totalPages} (${result.calls.length} calls)`
        );

        // If no transcribable calls found, restart from page 1
        if (result.calls.length === 0) {
          log(
            "No transcribable calls found on this page during initial scan, restarting from page 1"
          );
          shouldRestartFromPage1 = true;
          break;
        }

        // Process each call individually
        for (const call of result.calls) {
          try {
            await processCall(call);
          } catch (error) {
            log(`Error processing call ${call.id}: ${error.message}`, "error");
            // Continue with next call even if this one fails
          }
        }

        currentPage++;
      }

      // Only after processing ALL pages, switch to continuous mode
      log("Completed full database scan");
      isInitialScan = false;
      currentPageSize = INITIAL_PAGE_SIZE; // Keep using 1000 for continuous mode
      log("Switching to continuous mode with 10-second intervals");
    }
    // In continuous mode, process all pages in sequence
    else {
      log("Starting continuous mode page scan...");

      // Get the next available API
      const api = getNextApi();
      if (!api) {
        log("No available APIs at the moment, waiting 20 seconds...", "warn");
        await delay(20000); // Wait 20 seconds
        setTimeout(runContinuously, LOOP_INTERVAL);
        return;
      }

      log(`Processing calls from API: ${api.url}`);

      // If we need to restart from page 1, reset the flag and start from page 1
      if (shouldRestartFromPage1) {
        log("Restarting from page 1 as requested");
        shouldRestartFromPage1 = false;
        page1RetryCount++;

        // If we've tried page 1 too many times, move to next API
        if (page1RetryCount >= MAX_PAGE1_RETRIES) {
          log(
            `Page 1 tried ${page1RetryCount} times with no transcribable calls, moving to next API`
          );
          page1RetryCount = 0; // Reset counter
          currentApiIndex = (currentApiIndex + 1) % API_URLS.length;

          // Check if we've tried all APIs
          if (currentApiIndex === 0) {
            log(
              "Tried all APIs, waiting 20 seconds before retrying...",
              "warn"
            );
            await delay(20000);
          }

          setTimeout(runContinuously, LOOP_INTERVAL);
          return;
        }
      } else {
        // Reset retry counter if we're not restarting from page 1
        page1RetryCount = 0;
      }

      let currentPage = 1;
      let hasMorePages = true;

      while (hasMorePages) {
        try {
          // Fetch calls from current API
          const url = new URL(`${api.url}/calls`);
          if (!url.hostname) {
            throw new Error(`Invalid API URL: ${api.url}`);
          }

          // Use Strapi's filter format: filters[field][operator]=value
          url.searchParams.append(
            "populate",
            "recording,recordingLocal,recordingRemote"
          );

          // Filter for calls that have both recordings and are not done
          url.searchParams.append(
            "filters[recordingLocal][data][$notNull]",
            "true"
          );
          url.searchParams.append(
            "filters[recordingRemote][data][$notNull]",
            "true"
          );
          url.searchParams.append("filters[transcription_status][$ne]", "DONE");
          url.searchParams.append(
            "pagination[pageSize]",
            currentPageSize.toString()
          );
          url.searchParams.append("pagination[page]", currentPage.toString());
          url.searchParams.append("pagination[withCount]", "true");
          url.searchParams.append("sort", "id:desc");

          log(
            `Fetching page ${currentPage} with limit ${currentPageSize} from ${api.url}`
          );

          const response = await fetch(url, {
            method: "GET",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${api.token}`,
            },
          });

          if (!response.ok) {
            throw new Error(
              `Failed to fetch calls: ${response.status} ${response.statusText}`
            );
          }

          const data = await response.json();
          if (!data || !Array.isArray(data.data)) {
            throw new Error("Invalid response format from API");
          }

          // Safety filter: ensure all calls have recordings (in case Strapi filtering fails)
          const transcribableCalls = data.data.filter(
            (call) =>
              call.attributes.recordingLocal?.data?.[0]?.attributes?.url &&
              call.attributes.recordingRemote?.data?.[0]?.attributes?.url
          );

          const totalPages = data.meta.pagination.pageCount;
          const totalRecords = data.meta.pagination.total;

          log(
            `Processing page ${currentPage} of ${totalPages} (${transcribableCalls.length} transcribable calls out of ${data.data.length} total, total records: ${totalRecords})`
          );

          if (transcribableCalls.length === 0) {
            hasMorePages = false;
            log(
              "No transcribable calls found on this page, restarting from page 1"
            );
            shouldRestartFromPage1 = true;
            break;
          }

          // Reset retry counter when we find transcribable calls
          page1RetryCount = 0;

          log(
            `Processing ${transcribableCalls.length} transcribable calls on page ${currentPage}`
          );

          // Process each call individually (all calls returned are transcribable)
          for (const call of transcribableCalls) {
            try {
              await processCall({ ...call, _api: api });
            } catch (error) {
              log(
                `Error processing call ${call.id}: ${error.message}`,
                "error"
              );
              // Continue with next call even if this one fails
            }
          }

          currentPage++;

          // Only stop if we've processed all pages
          if (currentPage > totalPages) {
            hasMorePages = false;
            log(`Reached the last page (${totalPages}) in this cycle`);
          } else {
            log(`Moving to next page ${currentPage} of ${totalPages}`);
          }

          recordApiSuccess(api.url);
        } catch (error) {
          log(
            `Error processing page ${currentPage} from ${api.url}: ${error.message}`,
            "error"
          );
          recordApiError(api.url);
          hasMorePages = false;
        }
      }

      // Move to next API for next cycle (unless we're restarting from page 1)
      if (!shouldRestartFromPage1) {
        log(
          `DEBUG: Moving to next API, currentApiIndex: ${currentApiIndex} -> ${
            (currentApiIndex + 1) % API_URLS.length
          }`
        );
        currentApiIndex = (currentApiIndex + 1) % API_URLS.length;
      } else {
        log(
          `DEBUG: Not moving to next API due to restart flag, staying at currentApiIndex: ${currentApiIndex}`
        );
      }
    }
  } catch (error) {
    log(`Error in continuous processing: ${error.message}`, "error");
  }

  // Schedule the next run regardless of success or failure
  log(`Scheduling next run in ${LOOP_INTERVAL / 1000} seconds...`);
  setTimeout(runContinuously, LOOP_INTERVAL);
}

// Start the continuous loop
log("Starting transcription service in continuous mode");
runContinuously();

/**
 * Transcribe a call using the WhisperX configurable endpoint
 * @param {Object} call - Call object with recording URLs
 * @returns {Promise<Object>} Transcription results
 */
async function transcribeCall(call) {
  const callId = call.id || call._id;

  try {
    log(`Transcribing call ${callId}...`);

    // Handle Strapi's data structure - attributes might be nested
    const attributes = call.attributes || call;

    // Prepare the transcription request with configurable parameters
    const transcriptionPayload = {
      recording_local: attributes.recordingLocal.data[0].attributes.url,
      recording_remote: attributes.recordingRemote.data[0].attributes.url,
      config: getTranscriptionConfig(),
    };

    // Call the transcription API
    const response = await fetch(TRANSCRIPTION_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(transcriptionPayload),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `Transcription failed: ${response.status} ${response.statusText} - ${errorText}`
      );
    }

    const transcriptionData = await response.json();

    // Validate transcription response
    if (
      !transcriptionData ||
      !transcriptionData.transcript_local ||
      !transcriptionData.transcript_remote
    ) {
      throw new Error("Invalid transcription response format");
    }

    log(`Call ${callId} transcribed successfully`);
    return transcriptionData;
  } catch (error) {
    log(`Error transcribing call ${callId}: ${error.message}`, "error");
    throw error;
  }
}

/**
 * Update a call with transcription results
 * @param {Object} call - Call object to update
 * @param {Object} transcriptionData - Transcription data
 * @param {Object} api - API information to use
 * @returns {Promise<Object>} Updated call object
 */
async function updateCallWithTranscription(call, transcriptionData, api) {
  const callId = call.id || call._id;

  try {
    log(`Updating call ${callId} with transcription results...`);

    // Create the update payload for Strapi
    const updatePayload = {
      data: {
        transcriptionLocal: transcriptionData.transcript_local,
        transcriptionRemote: transcriptionData.transcript_remote,
        transcription_status: "DONE",
      },
    };

    // Update the call in the Strapi API
    const response = await fetch(`${api.url}/calls/${callId}`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${api.token}`,
      },
      body: JSON.stringify(updatePayload),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `Failed to update call: ${response.status} ${response.statusText} - ${errorText}`
      );
    }

    const updatedCall = await response.json();
    log(
      `Call ${callId} updated successfully with transcription_status=DONE`,
      "success"
    );
    recordApiSuccess(api.url);
    return updatedCall;
  } catch (error) {
    log(`Error updating call ${callId}: ${error.message}`, "error");
    recordApiError(api.url);

    // Try to mark as FAILED if update fails
    try {
      const failResponse = await fetch(`${api.url}/calls/${callId}`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${api.token}`,
        },
        body: JSON.stringify({
          data: {
            transcription_status: "FAILED",
          },
        }),
      });

      if (failResponse.ok) {
        log(`Call ${callId} marked as FAILED`, "warn");
      }
    } catch (markFailedError) {
      log(
        `Error marking call ${callId} as FAILED: ${markFailedError.message}`,
        "error"
      );
    }

    throw error;
  }
}

/**
 * Configure WhisperX transcription parameters
 * @returns {Object} Configuration object for WhisperX
 */
function getTranscriptionConfig() {
  return {
    // Basic configuration
    language: "de", // ISO language code, null for auto-detection
    compute_type: "float16", // "float16", "float32", or "int8"

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
    align_output: true, // Whether to align output for word-level timestamps
  };
}

// Rest of the file remains unchanged
