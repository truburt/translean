/*
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.
*/

const supportsMediaRecorder = typeof window !== 'undefined' && typeof window.MediaRecorder !== 'undefined';

const supportsWebmOpus = supportsMediaRecorder
    ? MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
    : false;

const supportsMp4Audio = supportsMediaRecorder
    ? MediaRecorder.isTypeSupported('audio/mp4')
    : false;

export const CONFIG = {
    recordingMimeType: supportsWebmOpus
        ? 'audio/webm;codecs=opus'
        : supportsMp4Audio
            ? 'audio/mp4'
            : 'audio/pcm;codecs=pcm',
    timeslice: 250,
    preRollMs: 400,
    highPassHz: 100,
    compressor: {
        threshold: -24,
        knee: 24,
        ratio: 4,
        attack: 0.003,
        release: 0.25,
    },
    rnnoise: {
        enabled: true,
        workletUrl: '/audio/rnnoise-worklet.js',
        moduleUrl: '/rnnoise/rnnoise.js',
        wasmUrl: '/rnnoise/rnnoise.wasm',
        noiseGateThreshold: 0.012,
    },
};

export const AUDIO_CONSTRAINTS = {
    audio: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        voiceIsolation: true,
    },
};

export const RECORDING_SAMPLE_RATE = 16000;
export const RECORDING_CHANNELS = 1;

export function getRecordingConfig() {
    if (supportsWebmOpus) {
        return {
            mode: 'media_recorder',
            mimeType: 'audio/webm;codecs=opus',
            container: 'webm',
            codec: 'opus'
        };
    }

    if (supportsMp4Audio) {
        return {
            mode: 'media_recorder',
            mimeType: 'audio/mp4',
            container: 'mp4',
            codec: 'aac'
        };
    }

    return {
        mode: 'pcm',
        mimeType: 'audio/pcm;codecs=pcm',
        container: 'pcm',
        codec: 'pcm_s16le'
    };
}

// RNNoise constraints are different from standard audio constraints,
// to avoid AGC and noise suppression conflicts.
export const RNNOISE_CONSTRAINTS = {
    audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        voiceIsolation: false,
    },
};

export const SPINNER_SVG = `<svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-accent" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
</svg>`;

export const MIC_ANIMATION_SVG = `<svg class="h-4 w-4 relative" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg" style="top:4px">
<circle cx="12" cy="12" r="8" fill="currentColor" fill-opacity="0.2">
<animate attributeName="r" values="8;11;8" dur="1.5s" repeatCount="indefinite" />
<animate attributeName="fill-opacity" values="0.1;0.3;0.1" dur="1.5s" repeatCount="indefinite" />
</circle>
<path d="M12 14C13.66 14 15 12.66 15 11V5C15 3.34 13.66 2 12 2C10.34 2 9 3.34 9 5V11C9 12.66 10.34 14 12 14Z" />
<path d="M17 11C17 13.76 14.76 16 12 16C9.24 16 7 13.76 7 11H5C5 14.53 7.61 17.43 11 17.92V21H13V17.92C16.39 17.43 19 14.53 19 11H17Z" />
</svg>`;

export const TYPING_INDICATOR_HTML = `
<span class="typing-indicator">
  <span></span>
  <span></span>
  <span></span>
</span>`;

export const MAX_FAVORITE_LANGUAGES = 5;

export const UI_LANGUAGES = [
    { code: 'en', label: 'English' },
    { code: 'ru', label: 'Русский' },
    { code: 'fi', label: 'Suomi' },
];

export const BACKEND_INIT_ERROR_MAP = {
    WHISPER_INIT_FAILED: {
        key: 'error_whisper_init_failed',
        service: 'whisper',
    },
    LLM_INIT_FAILED: {
        key: 'error_llm_init_failed',
        service: 'llm',
    },
};
