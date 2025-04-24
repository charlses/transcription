const testTranscription = async () => {
    try {
        const response = await fetch('http://localhost:8000/transcribe/whisperx', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                recording_local: 'https://reputationsritter-media-bucket-2.s3.eu-central-1.amazonaws.com/record_local_for_call_undefined_a9cf96c6c2.webm',
                recording_remote: 'https://reputationsritter-media-bucket-2.s3.eu-central-1.amazonaws.com/record_remote_for_call_undefined_c96b3e73da.webm'
            })
        })
        
        if (!response.ok) {
            const errorData = await response.json()
            throw new Error(`API Error: ${errorData.detail || response.statusText}`)
        }
        
        const data = await response.json()
        console.log('Transcription Response:', data)
    } catch (error) {
        console.error('Error during transcription:', error)
    }
}

testTranscription()
