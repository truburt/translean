/**
 * Siavox Audio Recording & WAV Encoder Utility
 * Encodes microphone PCM stream into 16kHz Mono WAV format locally in the browser.
 */

class WavEncoder {
    constructor() {
        this.audioContext = null;
        this.mediaStream = null;
        this.audioInput = null;
        this.recorderNode = null;
        this.recordingBuffers = [];
        this.recordingLength = 0;
        this.isRecording = false;
    }

    async startRecording() {
        if (this.isRecording) return;
        
        this.recordingBuffers = [];
        this.recordingLength = 0;
        
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
        this.audioInput = this.audioContext.createMediaStreamSource(this.mediaStream);
        
        // Define buffer size (2048, 4096, 8192, 16384)
        const bufferSize = 4096;
        // Create ScriptProcessorNode for recording mono channel
        this.recorderNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
        
        this.recorderNode.onaudioprocess = (e) => {
            if (!this.isRecording) return;
            const inputBuffer = e.inputBuffer.getChannelData(0);
            // Clone the input buffer (float32 array) and push to history
            this.recordingBuffers.push(new Float32Array(inputBuffer));
            this.recordingLength += inputBuffer.length;
        };

        // Connect graph
        this.audioInput.connect(this.recorderNode);
        this.recorderNode.connect(this.audioContext.destination);
        this.isRecording = true;
    }

    async stopRecording() {
        if (!this.isRecording) return null;
        
        this.isRecording = false;
        
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
        
        const nativeSampleRate = this.audioContext.sampleRate;
        const targetSampleRate = 16000;
        
        // Flatten recording buffers
        const mergedBuffer = this._mergeBuffers(this.recordingBuffers, this.recordingLength);
        
        // Close AudioContext
        if (this.audioContext) {
            await this.audioContext.close();
        }
        
        // Downsample to 16000 Hz
        const downsampledBuffer = this._downsampleBuffer(mergedBuffer, nativeSampleRate, targetSampleRate);
        
        // Encode to WAV bytes
        const wavView = this._encodeWAV(downsampledBuffer, targetSampleRate);
        
        // Return Blob of type audio/wav
        return new Blob([wavView], { type: 'audio/wav' });
    }

    _mergeBuffers(buffers, length) {
        const result = new Float32Array(length);
        let offset = 0;
        for (let i = 0; i < buffers.length; i++) {
            result.set(buffers[i], offset);
            offset += buffers[i].length;
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
            for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
                accum += buffer[i];
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
        for (let i = 0; i < input.length; i++, offset += 2) {
            let s = Math.max(-1, Math.min(1, input[i]));
            output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
    }
}
