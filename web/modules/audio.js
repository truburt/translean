/*
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.
*/

import { CONFIG, RNNOISE_CONSTRAINTS, AUDIO_CONSTRAINTS, RECORDING_SAMPLE_RATE, RECORDING_CHANNELS, getRecordingConfig } from './config.js';
import { state } from './state.js';
import { t } from './utils.js';
import { updateStatusIndicator, updateStreamBlock, updateFabState, updateVADIndicator, renderError } from './ui.js';
import { connectWebSocket } from './websocket.js';

function getChunkSizeBytes(chunk) {
    if (!chunk) return 0;
    if (typeof chunk.size === 'number') return chunk.size;
    if (typeof chunk.byteLength === 'number') return chunk.byteLength;
    return 0;
}

function floatTo16BitPCM(float32Array) {
    const output = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i += 1) {
        let sample = float32Array[i];
        sample = Math.max(-1, Math.min(1, sample));
        output[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    }
    return output;
}

export async function initAudioSession() {
    // Re-enabling RNNoise to test if it breaks S21
    const useRnnoise = CONFIG.rnnoise.enabled;

    if (window.AudioContext && useRnnoise) {
        // ... (RNNoise setup, currently disabled)
    }

    // Fallback: Standard context (allow native sample rate)
    let context = null;
    if (window.AudioContext) {
        // Remove forced sampleRate to allow browser default (usually 48k on mobile)
        // This avoids resampling artifacts or failures
        context = new AudioContext();
    }

    const stream = await navigator.mediaDevices.getUserMedia(AUDIO_CONSTRAINTS);
    return { stream, context, rnnoiseNode: null };
}

export async function setupAudioPipeline(stream, audioContext, rnnoiseNode) {
    if (!audioContext) {
        console.warn('Web Audio API unavailable; streaming raw microphone audio.');
        return { stream, cleanup: null, nodes: {} };
    }

    if (audioContext.state === 'suspended') {
        await audioContext.resume();
        console.info('Resumed suspended audio context to prevent silent output.');
    }

    const source = audioContext.createMediaStreamSource(stream);

    const highPass = audioContext.createBiquadFilter();
    highPass.type = 'highpass';
    highPass.frequency.value = CONFIG.highPassHz;

    const compressor = audioContext.createDynamicsCompressor();
    compressor.threshold.value = CONFIG.compressor.threshold;
    compressor.knee.value = CONFIG.compressor.knee;
    compressor.ratio.value = CONFIG.compressor.ratio;
    compressor.attack.value = CONFIG.compressor.attack;
    compressor.release.value = CONFIG.compressor.release;

    let lastNode = highPass;

    source.connect(highPass);

    if (rnnoiseNode) {
        lastNode.connect(rnnoiseNode);
        lastNode = rnnoiseNode;
        console.info('RNNoise worklet added to processing chain');
    }

    lastNode.connect(compressor);

    // 2. Prepare Output Stream
    // If we are processing at 48000 (for RNNoise), but Whisper wants 16000 (implicitly via our preference),
    // we must downsample. Browser MediaStreamDestination uses the context sample rate.
    // So we pipe the output of the processing context into a NEW context running at 16000.

    let outputStream;
    let outputContext = null;
    let bridgeDest;
    let bridgeSource;
    let finalDest;
    let destination;

    if (audioContext.sampleRate !== 16000) {
        console.log(`Resampling audio from ${audioContext.sampleRate}Hz to 16000Hz`);

        // Create the destination on the processing context
        bridgeDest = audioContext.createMediaStreamDestination();
        bridgeDest.channelCount = 1;
        compressor.connect(bridgeDest);

        // Create a 16kHz context for final output
        outputContext = new AudioContext({ sampleRate: RECORDING_SAMPLE_RATE });
        if (outputContext.state === 'suspended') {
            await outputContext.resume();
            console.info('Resumed output audio context to prevent silent output.');
        }

        // Connect the bridge stream to the new context
        bridgeSource = outputContext.createMediaStreamSource(bridgeDest.stream);
        finalDest = outputContext.createMediaStreamDestination();
        finalDest.channelCount = 1;
        bridgeSource.connect(finalDest);

        outputStream = finalDest.stream;

    } else {
        // Standard path (processing at 16kHz)
        destination = audioContext.createMediaStreamDestination();
        destination.channelCount = 1;
        compressor.connect(destination);
        outputStream = destination.stream;
    }

    const cleanup = async () => {
        source.disconnect();
        highPass.disconnect();
        if (rnnoiseNode) rnnoiseNode.disconnect();
        compressor.disconnect();
        await audioContext.close();
        if (outputContext) await outputContext.close();
    };

    state.audioContext = audioContext;
    return {
        stream: outputStream,
        cleanup,
        nodes: {
            source,
            highPass,
            compressor,
            rnnoiseNode,
            bridgeDest: bridgeDest || null,
            bridgeSource: bridgeSource || null,
            finalDest: finalDest || null,
            outputContext,
            destination: destination || null
        }
    };
}

export function updatePreRollBuffer(blob) {
    const chunkSize = getChunkSizeBytes(blob);
    if (!blob || chunkSize === 0) return;
    state.preRollChunks.push(blob);
    if (state.preRollChunks.length > state.preRollMaxChunks) {
        state.preRollChunks.shift();
    }
}

export function flushPreRollBuffer() {
    if (!state.preRollChunks.length) return;
    state.preRollChunks.forEach((chunk) => state.socket.send(chunk));
    state.preRollChunks = [];
}

export async function toggleRecording() {
    const isRecording = state.isRecording;
    if (isRecording) {
        await stopRecording();
    } else {
        await startRecording();
    }
}

export async function startRecording() {
    if (!state.token) {
        alert(t('alert_sign_in'));
        return;
    }

    // Ensure socket is ready
    if (!state.socket || state.socket.readyState !== WebSocket.OPEN) {
        renderError(t('alert_connection_not_ready'));
        connectWebSocket();
        return;
    }

    const previousStatus = state.uiStatus;

    try {
        state.isStopping = false;
        state.isRecording = true;
        updateFabState(true);
        updateStatusIndicator('ACTIVE');

        const { stream: rawStream, context, rnnoiseNode } = await initAudioSession();
        const audioPipeline = await setupAudioPipeline(rawStream, context, rnnoiseNode);
        const stream = audioPipeline.stream;

        state.recordingConfig = state.recordingConfig || getRecordingConfig();
        state.recordingMode = state.recordingConfig.mode;

        // --- VAD Setup ---
        state.headerChunk = null;
        state.headerSent = false;
        state.isSpeaking = false; // Reset
        state.vadReady = false; // Reset
        state.vadEnabled = false;
        state.isStopping = false;
        state.rawStream = rawStream;
        state.audioPipelineCleanup = audioPipeline.cleanup;
        // Keep pipeline logic alive
        state.audioPipeline = audioPipeline;
        state.preRollChunks = [];
        state.preRollPending = false;
        state.preRollMaxChunks = Math.ceil(CONFIG.preRollMs / CONFIG.timeslice);

        if (window.vad?.MicVAD && window.WebAssembly) {
            try {
                state.vad = await window.vad.MicVAD.new({
                    baseAssetPath: `/vad/`,
                    onnxWASMBasePath: `/ort/`,
                    stream: stream,
                    onSpeechStart: () => {
                        console.log("Speech Started");
                        state.isSpeaking = true;
                        state.preRollPending = true;
                        updateVADIndicator(true);
                        // Force immediate data availability if recorder is active
                        if (state.recordingMode === 'media_recorder' && state.mediaRecorder?.state === 'recording') {
                            state.mediaRecorder.requestData();
                        }
                    },
                    onSpeechEnd: () => {
                        console.log("Speech Ended");
                        state.isSpeaking = false;
                        updateVADIndicator(false);
                    },
                });
                state.vadEnabled = true;
                state.vad.start();
            } catch (err) {
                console.warn('VAD initialization failed, falling back to continuous streaming.', err);
                state.vadEnabled = false;
            }
        } else {
            console.warn('VAD unavailable; falling back to continuous streaming.');
        }

        if (!state.vadEnabled) {
            state.isSpeaking = true;
            updateVADIndicator(false);
        }

        if (state.recordingMode === 'media_recorder') {
            state.mediaRecorder = new MediaRecorder(stream, { mimeType: state.recordingConfig.mimeType });

            state.mediaRecorder.onstop = () => {
                console.log("MediaRecorder stopped");
                if (state.socket && state.socket.readyState === WebSocket.OPEN) {
                    console.log("Sending stop_recording signal...");
                    state.socket.send(JSON.stringify({ type: 'stop_recording' }));
                }
            };

            state.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size === 0) return;

                const blob = event.data;
                // Always capture the first chunk as the container header
                const isFirstChunk = !state.headerChunk;
                if (isFirstChunk) {
                    state.headerChunk = blob;
                    console.log("Captured container header chunk size:", blob.size);
                }
                const isHeaderChunk = blob === state.headerChunk;

                if (state.vadEnabled && !state.isSpeaking && !state.isStopping) {
                    if (!isHeaderChunk) {
                        updatePreRollBuffer(blob);
                    }
                    return;
                }

                const socketOpen = state.socket?.readyState === WebSocket.OPEN;
                if (!socketOpen) return;

                // Ensure the backend receives the header before any audio
                if (!state.headerSent && state.headerChunk) {
                    state.socket.send(state.headerChunk);
                    state.headerSent = true;
                    console.log("Sent container header");

                    if (isHeaderChunk) return;
                }

                if (state.preRollPending) {
                    flushPreRollBuffer();
                    state.preRollPending = false;
                }

                state.socket.send(blob);
            };

            state.mediaRecorder.start(CONFIG.timeslice);
        } else {
            const pcmContext = new AudioContext({ sampleRate: RECORDING_SAMPLE_RATE });
            if (pcmContext.state === 'suspended') {
                await pcmContext.resume();
                console.info('Resumed PCM audio context to prevent silent output.');
            }

            const pcmSource = pcmContext.createMediaStreamSource(stream);
            const pcmProcessor = pcmContext.createScriptProcessor(4096, RECORDING_CHANNELS, RECORDING_CHANNELS);

            pcmProcessor.onaudioprocess = (event) => {
                const input = event.inputBuffer.getChannelData(0);
                const pcm16 = floatTo16BitPCM(input);
                const payload = pcm16.buffer;

                if (state.vadEnabled && !state.isSpeaking && !state.isStopping) {
                    updatePreRollBuffer(payload);
                    return;
                }

                const socketOpen = state.socket?.readyState === WebSocket.OPEN;
                if (!socketOpen) return;

                if (state.preRollPending) {
                    flushPreRollBuffer();
                    state.preRollPending = false;
                }

                state.socket.send(payload);
            };

            pcmSource.connect(pcmProcessor);
            pcmProcessor.connect(pcmContext.destination);

            state.pcmContext = pcmContext;
            state.pcmSource = pcmSource;
            state.pcmProcessor = pcmProcessor;
            state.mediaRecorder = null;
        }

        // Give VAD a moment to settle
        setTimeout(() => {
            state.vadReady = true;
            if (state.isSpeaking) {
                updateVADIndicator(true);
            }
        }, 200);

        if (state.isModelsWarmingUp) {
            updateStreamBlock(t('system_loading_models'), "", true, "placeholder-pending");
        } else {
            updateStreamBlock(t('system_listening'), "", true, "placeholder-pending");
        }

    } catch (err) {
        console.error(err);
        alert(t('alert_mic_denied_prefix') + err.message);
        state.isRecording = false;
        state.isStopping = false;
        updateFabState(false);
        updateStatusIndicator(state.isModelsWarmingUp ? 'LOADING' : previousStatus);
    }
}



export function stopRecording() {
    return new Promise((resolve) => {
        state.isStopping = true;

        if (!state.isRecording) {
            resolve();
            return;
        }

        const finalizeCleanup = () => {
            if (state.rawStream) {
                state.rawStream.getTracks().forEach(track => track.stop());
            }

            state.rawStream = null;

            if (state.vad) {
                state.vad.pause();
            }

            if (state.audioPipelineCleanup) {
                state.audioPipelineCleanup().catch((err) => console.warn("Failed to close audio pipeline", err));
                state.audioPipelineCleanup = null;
            }

            state.isRecording = false;
            updateFabState(false);
            updateStatusIndicator('FINISHING');
        };

        if (state.recordingMode !== 'media_recorder') {
            if (state.socket && state.socket.readyState === WebSocket.OPEN) {
                state.socket.send(JSON.stringify({ type: 'stop_recording' }));
            }
            if (state.pcmProcessor) {
                state.pcmProcessor.disconnect();
                state.pcmProcessor.onaudioprocess = null;
                state.pcmProcessor = null;
            }
            if (state.pcmSource) {
                state.pcmSource.disconnect();
                state.pcmSource = null;
            }
            if (state.pcmContext) {
                state.pcmContext.close().catch((err) => console.warn("Failed to close PCM context", err));
                state.pcmContext = null;
            }
            finalizeCleanup();
            resolve();
            return;
        }

        if (!state.mediaRecorder || state.mediaRecorder.state === 'inactive') {
            finalizeCleanup();
            resolve();
            return;
        }

        const originalOnStop = state.mediaRecorder.onstop;
        state.mediaRecorder.onstop = (e) => {
            if (originalOnStop) originalOnStop(e);
            finalizeCleanup();
            resolve();
        };

        state.mediaRecorder.stop();
    });
}
