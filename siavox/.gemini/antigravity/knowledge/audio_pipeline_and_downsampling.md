# Audio Pipeline and Downsampling

To eliminate heavy backend dependencies (such as `ffmpeg` or `libsndfile`) inside the Docker container, Siavox performs all audio capture, resampling, and format encoding directly in the user's web browser.

---

## 1. Browser Audio Capture Constraints

Audio capture is implemented in [wav_encoder.js](file:///wsl.localhost/Ubuntu-24.04/home/truburt/dev/translean/siavox/frontend/wav_encoder.js). It initializes recording by requesting microphone access through `navigator.mediaDevices.getUserMedia` with strict audio processing constraints:

```javascript
this.mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
        channelCount: 1,       // Request single channel (mono) to save bandwidth
        echoCancellation: true, // Reduce room echo
        noiseSuppression: true, // Filter background noise
        autoGainControl: true   // Maintain consistent volume levels
    }
});
```

### Buffer Processing
* **`AudioContext`**: Manages the processing graph.
* **`ScriptProcessorNode`**: Created with a buffer size of `4096` samples, `1` input channel, and `1` output channel.
* **Stream Collection**: On each `onaudioprocess` event, the float PCM data buffer is cloned into a `Float32Array` and appended to a tracking list (`this.recordingBuffers`). The total frame length is accumulated in `this.recordingLength`.

---

## 2. Mathematical Downsampling (Resampling)

Most user devices record audio at a native sample rate of **44.1 kHz** or **48 kHz**. The Gemma 4 model is trained to process audio at exactly **16,000 Hz**. Processing higher sample rates increases network payload size and causes translation degradation.

The browser downsamples the flattened float array to `16000 Hz` using a linear decimation step implemented in `WavEncoder._downsampleBuffer`:

```javascript
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
        result[offsetResult] = accum / count; // Average the samples in the ratio window
        offsetResult++;
        offsetBuffer = nextOffsetBuffer;
    }
    return result;
}
```

---

## 3. RIFF WAV Header Specification

The downsampled float PCM array is packed into an uncompressed **16-bit Signed Integer mono PCM WAV file**. This is done by writing a standard **44-byte RIFF header** followed by the raw data chunk into a Javascript `ArrayBuffer`.

| Byte Offset | Field Name | DataType | Value / Formula | Description |
| :--- | :--- | :--- | :--- | :--- |
| **0 - 3** | `ChunkID` | Char[4] | `"RIFF"` | Resource Interchange File Format ID |
| **4 - 7** | `ChunkSize` | UInt32 | `36 + (SamplesLength * 2)` | Total size of the file minus 8 bytes |
| **8 - 11** | `Format` | Char[4] | `"WAVE"` | Format signature |
| **12 - 15** | `Subchunk1ID`| Char[4] | `"fmt "` | Format sub-chunk header |
| **16 - 19** | `Subchunk1Size`| UInt32 | `16` | Length of format sub-chunk (16 bytes for PCM) |
| **20 - 21** | `AudioFormat` | UInt16 | `1` | Compression code (1 = Linear PCM) |
| **22 - 23** | `NumChannels` | UInt16 | `1` | Number of channels (1 = Mono) |
| **24 - 27** | `SampleRate` | UInt32 | `16000` | Sampling frequency (16 kHz) |
| **28 - 31** | `ByteRate` | UInt32 | `32000` | `SampleRate * NumChannels * (BitsPerSample / 8)` |
| **32 - 33** | `BlockAlign` | UInt16 | `2` | `NumChannels * (BitsPerSample / 8)` |
| **34 - 35** | `BitsPerSample`| UInt16 | `16` | Bits per sample (16 bits) |
| **36 - 39** | `Subchunk2ID`| Char[4] | `"data"` | Data sub-chunk header |
| **40 - 43** | `Subchunk2Size`| UInt32 | `SamplesLength * 2` | Size of the raw audio data payload |

---

## 4. Float-to-PCM Audio Quantization

To convert the internal Float32 audio samples (which span `[-1.0, 1.0]`) into 16-bit Signed Integers (`[-32768, 32767]`), the encoder clamps the floats and scales them to their integer ranges:

```javascript
_floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, input[i])); // Clamping float buffer values
        output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true); // Little-Endian output
    }
}
```

The resulting `DataView` is wrapped in a `Blob` with headers configured to `type: 'audio/wav'` and transmitted directly via HTTP POST.
