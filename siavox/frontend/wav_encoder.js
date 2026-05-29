/**
 * Siavox Audio Recording & WAV Encoder Utility
 *
 * Encodes microphone PCM stream into 16kHz Mono WAV format locally in the browser.
 *
 * Streaming mode (used during live recording):
 *   - Every CHUNK_INTERVAL_S seconds, or when silence is detected for SILENCE_FLUSH_S
 *     seconds, the accumulated audio buffers are flushed, downsampled, encoded as a
 *     WAV blob, and delivered to the registered onChunkReady callback.
 *   - This allows the UI to send each chunk to /api/transcribe immediately while the
 *     user is still speaking, so partial transcripts appear in real-time.
 *   - stopRecording() flushes any remaining audio as a final chunk (via the callback)
 *     and returns null. The caller must wait for all in-flight chunks before proceeding.
 */

// How often to flush a new audio chunk during live recording (seconds).
const CHUNK_INTERVAL_S = 7;

// Minimum chunk duration to bother sending (seconds). Shorter chunks are discarded
// to avoid sending near-empty WAV files that produce garbage transcriptions.
const MIN_CHUNK_DURATION_S = 1.0;

// RMS energy below this level is considered silence (0–1 scale after normalisation).
const SILENCE_ENERGY_THRESHOLD = 0.005;

// Silence must persist this long (seconds) before an early flush is triggered.
const SILENCE_FLUSH_S = 1.2;

class WavEncoder {
    constructor() {
        this.audioContext = null;
        this.mediaStream = null;
        this.audioInput = null;
        this.recorderNode = null;

        // Buffers for the *current* in-progress chunk
        this._chunkBuffers = [];
        this._chunkLength = 0;

        this.isRecording = false;

        // Registered callback: fired with a WAV Blob for each flushed chunk
        this._chunkCallback = null;

        // Timer handle for the periodic chunk flush
        this._chunkTimer = null;

        // Silence tracking
        this._silenceSamples = 0;    // samples accumulated while silent
        this._nativeSampleRate = 44100;

        // Expose target sample rate so callers can reason about chunk sizes
        this.targetSampleRate = 16000;
    }

    /**
     * Register a callback that receives each flushed WAV Blob.
     * Must be called before startRecording().
     *
     * @param {function(Blob): void} callback
     */
    onChunkReady(callback) {
        this._chunkCallback = callback;
    }

    async startRecording() {
        if (this.isRecording) return;

        this._chunkBuffers = [];
        this._chunkLength = 0;
        this._silenceSamples = 0;

        // Request Microphone Access
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });

        // Initialize AudioContext
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this._nativeSampleRate = this.audioContext.sampleRate;
        this.audioInput = this.audioContext.createMediaStreamSource(this.mediaStream);

        // Define buffer size (2048, 4096, 8192, 16384)
        const bufferSize = 4096;
        // Create ScriptProcessorNode for recording mono channel
        this.recorderNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

        this.recorderNode.onaudioprocess = (e) => {
            if (!this.isRecording) return;
            const inputBuffer = e.inputBuffer.getChannelData(0);

            // Accumulate into current chunk
            this._chunkBuffers.push(new Float32Array(inputBuffer));
            this._chunkLength += inputBuffer.length;

            // --- Silence detection ---
            const rms = this._computeRMS(inputBuffer);
            if (rms < SILENCE_ENERGY_THRESHOLD) {
                this._silenceSamples += inputBuffer.length;
            } else {
                this._silenceSamples = 0;
            }

            const silenceDuration = this._silenceSamples / this._nativeSampleRate;
            const chunkDuration = this._chunkLength / this._nativeSampleRate;

            // Flush early if silence threshold reached AND we have meaningful audio
            if (silenceDuration >= SILENCE_FLUSH_S && chunkDuration >= MIN_CHUNK_DURATION_S) {
                this._flushChunk();
            }
        };

        // Connect graph
        this.audioInput.connect(this.recorderNode);
        this.recorderNode.connect(this.audioContext.destination);
        this.isRecording = true;

        // Schedule periodic hard-wall flush
        this._chunkTimer = setInterval(() => {
            if (this.isRecording) {
                this._flushChunk();
            }
        }, CHUNK_INTERVAL_S * 1000);
    }

    /**
     * Stop recording. Flushes any remaining buffered audio as a final chunk via the
     * onChunkReady callback. Returns null — the caller is responsible for waiting on
     * any in-flight async chunk callbacks before proceeding with refinement.
     */
    async stopRecording() {
        if (!this.isRecording) return null;

        this.isRecording = false;

        // Stop the periodic flush timer
        if (this._chunkTimer) {
            clearInterval(this._chunkTimer);
            this._chunkTimer = null;
        }

        // Disconnect nodes
        if (this.recorderNode) {
            this.recorderNode.disconnect();
            this.recorderNode.onaudioprocess = null;
        }
        if (this.audioInput) {
            this.audioInput.disconnect();
        }
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
        }

        // Flush the final partial chunk (if it has enough audio)
        this._flushChunk();

        // Close AudioContext
        if (this.audioContext) {
            await this.audioContext.close();
            this.audioContext = null;
        }

        // The streaming API uses the callback; no blob returned here
        return null;
    }

    // ─── Private helpers ─────────────────────────────────────────────────────

    /**
     * Encode the current accumulated buffers into a WAV Blob, fire the callback,
     * and reset the chunk state. Silently skips if there is no buffered audio or
     * the duration is below MIN_CHUNK_DURATION_S.
     */
    _flushChunk() {
        const chunkDuration = this._chunkLength / this._nativeSampleRate;
        if (!this._chunkBuffers.length || chunkDuration < MIN_CHUNK_DURATION_S) {
            // Not enough audio to form a meaningful chunk — discard
            this._chunkBuffers = [];
            this._chunkLength = 0;
            this._silenceSamples = 0;
            return;
        }

        const buffersCopy = this._chunkBuffers;
        const lengthCopy = this._chunkLength;

        // Reset state for the next chunk immediately (before async ops)
        this._chunkBuffers = [];
        this._chunkLength = 0;
        this._silenceSamples = 0;

        // Encode and deliver
        const merged = this._mergeBuffers(buffersCopy, lengthCopy);
        const downsampled = this._downsampleBuffer(merged, this._nativeSampleRate, this.targetSampleRate);
        const wavView = this._encodeWAV(downsampled, this.targetSampleRate);
        const blob = new Blob([wavView], { type: 'audio/wav' });

        if (this._chunkCallback) {
            this._chunkCallback(blob);
        }
    }

    /**
     * Compute RMS energy of a Float32Array audio frame (range 0–1).
     * Uses reduce over the typed array — safe numeric-only iteration, no user input.
     */
    _computeRMS(buffer) {
        // Iterate with for...of on the typed array (no prototype access possible)
        let sum = 0;
        for (const sample of buffer) {
            sum += sample * sample;
        }
        return Math.sqrt(sum / buffer.length);
    }

    _mergeBuffers(buffers, length) {
        const result = new Float32Array(length);
        let offset = 0;
        // for...of over the outer array of Float32Array chunks
        for (const chunk of buffers) {
            result.set(chunk, offset);
            offset += chunk.length;
        }
        return result;
    }

    _downsampleBuffer(buffer, inputSampleRate, outputSampleRate) {
        if (inputSampleRate === outputSampleRate) {
            return buffer;
        }
        const sampleRateRatio = inputSampleRate / outputSampleRate;
        const newLength = Math.round(buffer.length / sampleRateRatio);
        const result = new Float32Array(newLength);
        let offsetResult = 0;
        let offsetBuffer = 0;
        while (offsetResult < result.length) {
            const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
            let accum = 0;
            let count = 0;
            // Slice the typed array segment — avoids direct indexed access
            const segment = buffer.subarray(offsetBuffer, Math.min(nextOffsetBuffer, buffer.length));
            for (const sample of segment) {
                accum += sample;
                count++;
            }
            result[offsetResult] = accum / count;
            offsetResult++;
            offsetBuffer = nextOffsetBuffer;
        }
        return result;
    }

    _encodeWAV(samples, sampleRate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);

        /* RIFF identifier */
        this._writeString(view, 0, 'RIFF');
        /* File length */
        view.setUint32(4, 36 + samples.length * 2, true);
        /* RIFF type */
        this._writeString(view, 8, 'WAVE');
        /* Format chunk identifier */
        this._writeString(view, 12, 'fmt ');
        /* Format chunk length */
        view.setUint32(16, 16, true);
        /* Sample format (1 = uncompressed PCM) */
        view.setUint16(20, 1, true);
        /* Channel count (1 = mono) */
        view.setUint16(22, 1, true);
        /* Sample rate */
        view.setUint32(24, sampleRate, true);
        /* Byte rate (sample rate * block align) */
        view.setUint32(28, sampleRate * 2, true);
        /* Block align (channel count * bytes per sample) */
        view.setUint16(32, 2, true);
        /* Bits per sample */
        view.setUint16(34, 16, true);
        /* Data chunk identifier */
        this._writeString(view, 36, 'data');
        /* Data chunk length */
        view.setUint32(40, samples.length * 2, true);

        // Convert Float32 to 16-bit PCM and write
        this._floatTo16BitPCM(view, 44, samples);

        return view;
    }

    _writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    _floatTo16BitPCM(output, offset, input) {
        // for...of over Float32Array typed array — safe numeric iteration
        for (const sample of input) {
            const s = Math.max(-1, Math.min(1, sample));
            output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            offset += 2;
        }
    }
}
