require('dotenv').config()
const fetch = require('node-fetch')
const fs = require('fs')
const path = require('path')
const { exec } = require('child_process')
const { promisify } = require('util')
const execAsync = promisify(exec)
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

const s3 = new AWS.S3()

// Constants
const API_BASE_URL =
  process.env.API_BASE_URL || 'https://whisper-training-prep.tess-dev.de/api'
const TRANSCRIPTION_API_URL =
  process.env.TRANSCRIPTION_API_URL || 'http://localhost:8000/transcribe'
const S3_BUCKET = process.env.AWS_S3_BUCKET

const RR_API_URL = process.env.RR_API_URL

// Create temp directory if it doesn't exist
const tempDir = path.join(__dirname, 'temp')
if (!fs.existsSync(tempDir)) {
  fs.mkdirSync(tempDir)
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

async function downloadFile(url, outputPath) {
  try {
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error(`Failed to download file: ${response.statusText}`)
    }
    const buffer = await response.buffer()
    fs.writeFileSync(outputPath, buffer)
    return outputPath
  } catch (error) {
    console.error(`Error downloading file from ${url}:`, error)
    throw error
  }
}

async function uploadToS3(filePath, key) {
  try {
    if (!fs.existsSync(filePath)) {
      throw new Error(`File not found: ${filePath}`)
    }

    const fileContent = fs.readFileSync(filePath)
    const params = {
      Bucket: S3_BUCKET,
      Key: key,
      Body: fileContent,
      ContentType: 'audio/webm'
    }

    const result = await s3.upload(params).promise()
    return result.Location
  } catch (error) {
    console.error(`Error uploading file to S3: ${filePath}`, error)
    throw error
  }
}

async function cutAudio(inputPath, startTime, endTime, outputPath) {
  try {
    // Check if input file exists and get its duration
    if (!fs.existsSync(inputPath)) {
      throw new Error(`Input file not found: ${inputPath}`)
    }

    // Get audio duration using ffprobe
    const probeCommand = `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${inputPath}"`
    const { stdout: durationStr } = await execAsync(probeCommand)
    const audioDuration = parseFloat(durationStr)

    // Validate times
    if (startTime < 0) {
      log(`Warning: Start time ${startTime} is negative, setting to 0`, 'warn')
      startTime = 0
    }
    if (endTime > audioDuration) {
      log(
        `Warning: End time ${endTime} exceeds audio duration ${audioDuration}, setting to duration`,
        'warn'
      )
      endTime = audioDuration
    }
    if (startTime >= endTime) {
      throw new Error(
        `Invalid segment times: start (${startTime}) must be less than end (${endTime})`
      )
    }

    // Escape Windows paths and wrap in quotes
    const escapedInputPath = `"${inputPath.replace(/\\/g, '/')}"`
    const escapedOutputPath = `"${outputPath.replace(/\\/g, '/')}"`

    // Use more precise timing control with -ss and -to
    // Add -avoid_negative_ts 1 to prevent negative timestamps
    // Add -copyts to preserve timestamps
    const command = `ffmpeg -y -i ${escapedInputPath} -ss ${startTime} -to ${endTime} -avoid_negative_ts 1 -copyts -c:a copy -c:v copy ${escapedOutputPath}`
    log(`Executing ffmpeg command: ${command}`)

    const { stdout, stderr } = await execAsync(command)

    if (stderr) {
      log(`FFmpeg stderr: ${stderr}`, 'warn')
      // Check if the error is critical
      if (
        stderr.includes('Error') ||
        stderr.includes('Invalid') ||
        stderr.includes('Failed')
      ) {
        throw new Error(`FFmpeg error: ${stderr}`)
      }
    }

    // Verify the output file
    if (!fs.existsSync(outputPath)) {
      throw new Error(`Output file was not created: ${outputPath}`)
    }

    // Verify the output duration
    const { stdout: outputDurationStr } = await execAsync(
      `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${outputPath}"`
    )
    const outputDuration = parseFloat(outputDurationStr)
    const expectedDuration = endTime - startTime
    const durationDiff = Math.abs(outputDuration - expectedDuration)

    if (durationDiff > 0.1) {
      // Allow 100ms tolerance
      log(
        `Warning: Output duration (${outputDuration.toFixed(
          2
        )}s) differs from expected (${expectedDuration.toFixed(
          2
        )}s) by ${durationDiff.toFixed(2)}s`,
        'warn'
      )
    }

    return outputPath
  } catch (error) {
    log(`Error cutting audio: ${error.message}`, 'error')
    // Clean up any partial output file
    try {
      if (fs.existsSync(outputPath)) {
        fs.unlinkSync(outputPath)
      }
    } catch (cleanupError) {
      log(
        `Error cleaning up failed output file: ${cleanupError.message}`,
        'warn'
      )
    }
    throw error
  }
}

async function processSegments(
  audioPath,
  segments,
  outputDir,
  callId,
  isLocal = true
) {
  const segmentFiles = []
  let currentSegments = []
  let segmentIndex = 0
  const PADDING = 0.2 // 200ms padding for audio cutting only
  const segmentType = isLocal ? 'local' : 'remote'

  try {
    log(
      `Starting ${segmentType} segment processing for ${path.basename(
        audioPath
      )} (${segments.length} segments)`
    )

    // Validate segments structure
    if (!Array.isArray(segments)) {
      throw new Error(`Invalid segments structure for ${segmentType} audio`)
    }

    // Sort segments by start time to ensure chronological order
    segments.sort((a, b) => a.start - b.start)

    // Validate segment times
    for (const segment of segments) {
      if (
        typeof segment.start !== 'number' ||
        typeof segment.end !== 'number' ||
        segment.start >= segment.end ||
        segment.start < 0
      ) {
        throw new Error(
          `Invalid segment times: start=${segment.start}, end=${segment.end}`
        )
      }
    }

    for (let i = 0; i < segments.length; i++) {
      const segment = segments[i]

      // Add current segment to the group
      currentSegments.push(segment)

      // Check if we should create a new segment file
      const isLastSegment = i === segments.length - 1
      const nextSegment = !isLastSegment ? segments[i + 1] : null
      const currentGroupDuration =
        currentSegments[currentSegments.length - 1].end -
        currentSegments[0].start

      // Create new segment if:
      // 1. This is the last segment, or
      // 2. Adding the next segment would make the current group too long
      if (
        isLastSegment ||
        (nextSegment && nextSegment.end - currentSegments[0].start > 30)
      ) {
        // Use padding only for audio cutting, not for segment timings
        const audioCutStart = Math.max(0, currentSegments[0].start - PADDING)
        const audioCutEnd =
          currentSegments[currentSegments.length - 1].end + PADDING

        log(
          `Processing ${segmentType} segment group ${
            segmentIndex + 1
          }/${Math.ceil(segments.length / 30)} (${audioCutStart.toFixed(
            2
          )}s to ${audioCutEnd.toFixed(2)}s, duration: ${(
            audioCutEnd - audioCutStart
          ).toFixed(2)}s)`
        )
        log(`Group contains ${currentSegments.length} segments`)

        // Create segment file with call ID and segment type in the name
        const outputPath = path.join(
          outputDir,
          `segment_${segmentIndex}_${segmentType}_call_${callId}.webm`
        )
        log(`Cutting audio segment to ${outputPath}`)

        try {
          await cutAudio(audioPath, audioCutStart, audioCutEnd, outputPath)
        } catch (error) {
          log(
            `Error cutting ${segmentType} segment group ${segmentIndex + 1}: ${
              error.message
            }`,
            'error'
          )
          throw error
        }

        // Upload to S3
        const s3Key = `segments/${path.basename(outputPath)}`
        log(
          `Uploading ${segmentType} segment group ${
            segmentIndex + 1
          } to S3: ${s3Key}`
        )

        try {
          const s3Url = await uploadToS3(outputPath, s3Key)

          // Store cut audio segment information
          segmentFiles.push({
            start: audioCutStart, // Start time with padding
            end: audioCutEnd, // End time with padding
            url: s3Url,
            originalSegments: currentSegments.map((seg) => ({
              start: seg.start,
              end: seg.end,
              text: seg.text,
              speaker: seg.speaker,
              words: seg.words
                ? seg.words.map((word) => ({
                    word: word.word,
                    start: word.start,
                    end: word.end,
                    score: word.score
                  }))
                : []
            }))
          })

          log(
            `${segmentType} segment group ${
              segmentIndex + 1
            } completed and uploaded successfully`
          )
        } catch (error) {
          log(
            `Error uploading ${segmentType} segment group ${
              segmentIndex + 1
            } to S3: ${error.message}`,
            'error'
          )
          throw error
        }

        // Reset for next group
        currentSegments = []
        segmentIndex++
      }
    }

    log(
      `Completed processing all ${segmentType} segments for ${path.basename(
        audioPath
      )} (${segmentFiles.length} segment groups created)`
    )
    return segmentFiles
  } catch (error) {
    log(
      `Error processing ${segmentType} segments for ${audioPath}: ${error.message}`,
      'error'
    )
    throw error
  }
}

async function processCall(call) {
  const startTime = Date.now()
  try {
    log(`Starting to process call ${call._id}`)
    log(
      `Call details: Duration=${call.duration}s, Type=${call.type}, Number=${call.number}`
    )

    // Create temporary directory for this call
    const tempDir = path.join(__dirname, 'temp', call._id)
    fs.mkdirSync(tempDir, { recursive: true })
    log(`Created temporary directory: ${tempDir}`)

    // Download audio files
    const localAudioPath = path.join(tempDir, 'local.webm')
    const remoteAudioPath = path.join(tempDir, 'remote.webm')

    log('Downloading local recording...')
    await downloadFile(call.recordingLocal.url, localAudioPath)

    log('Downloading remote recording...')
    await downloadFile(call.recordingRemote.url, remoteAudioPath)

    // Transcribe audio
    log('Starting transcription...')
    let transcriptionData
    try {
      // Check if files exist and have content before sending to transcription
      const localStats = fs.statSync(localAudioPath)
      const remoteStats = fs.statSync(remoteAudioPath)

      if (localStats.size === 0 || remoteStats.size === 0) {
        throw new Error(
          `Audio file is empty: ${localStats.size === 0 ? 'local' : 'remote'}`
        )
      }

      log(`Local audio size: ${(localStats.size / 1024 / 1024).toFixed(2)} MB`)
      log(
        `Remote audio size: ${(remoteStats.size / 1024 / 1024).toFixed(2)} MB`
      )

      const transcriptionResponse = await fetch(TRANSCRIPTION_API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          recording_local: call.recordingLocal.url,
          recording_remote: call.recordingRemote.url
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

    // Process segments and create audio files only if there are segments
    let cutAudioLocal = []
    let cutAudioRemote = []

    if (transcriptionData.transcript_local.segments.length > 0) {
      log('Processing local segments...')
      cutAudioLocal = await processSegments(
        localAudioPath,
        transcriptionData.transcript_local.segments,
        tempDir,
        call.sql_id,
        true
      )
    } else {
      log('No local segments to process', 'warn')
    }

    if (transcriptionData.transcript_remote.segments.length > 0) {
      log('Processing remote segments...')
      cutAudioRemote = await processSegments(
        remoteAudioPath,
        transcriptionData.transcript_remote.segments,
        tempDir,
        call.sql_id,
        false
      )
    } else {
      log('No remote segments to process', 'warn')
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
        cutAudioLocal: cutAudioLocal,
        cutAudioRemote: cutAudioRemote,
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

    // Clean up
    log('Cleaning up temporary files...')
    fs.rmSync(tempDir, { recursive: true, force: true })
    log('Cleanup completed')
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

    // Clean up temp directory after processing the page
    try {
      if (fs.existsSync(tempDir)) {
        fs.rmSync(tempDir, { recursive: true, force: true })
        log('Cleaned up temp directory after page processing')
      }
    } catch (cleanupError) {
      log(
        `Warning: Failed to clean up temp directory: ${cleanupError.message}`,
        'warn'
      )
    }

    return data.pagination
  } catch (error) {
    log(`Error processing page ${page}: ${error.message}`, 'error')
    // Try to clean up even if there was an error
    try {
      if (fs.existsSync(tempDir)) {
        fs.rmSync(tempDir, { recursive: true, force: true })
        log('Cleaned up temp directory after error')
      }
    } catch (cleanupError) {
      log(
        `Warning: Failed to clean up temp directory after error: ${cleanupError.message}`,
        'warn'
      )
    }
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

    // Final cleanup of temp directory
    try {
      if (fs.existsSync(tempDir)) {
        fs.rmSync(tempDir, { recursive: true, force: true })
        log('Final cleanup of temp directory completed')
      }
    } catch (cleanupError) {
      log(
        `Warning: Failed to clean up temp directory in final cleanup: ${cleanupError.message}`,
        'warn'
      )
    }

    main()
  } catch (error) {
    log(`Error in main process: ${error.message}`, 'error')
    // Try to clean up even if there was an error in main
    try {
      if (fs.existsSync(tempDir)) {
        fs.rmSync(tempDir, { recursive: true, force: true })
        log('Cleaned up temp directory after main process error')
      }
    } catch (cleanupError) {
      log(
        `Warning: Failed to clean up temp directory after main error: ${cleanupError.message}`,
        'warn'
      )
    }
    main()
  }
}

main()
