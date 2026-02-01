/*
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.
*/

import { state } from './state.js';
import { t, getBackendInitError } from './utils.js';
import { BACKEND_INIT_ERROR_MAP, RECORDING_SAMPLE_RATE, RECORDING_CHANNELS, getRecordingConfig } from './config.js';
import {
    renderSystemMessage, updateStatusIndicator, renderError, updateStreamBlock,
    setRebuildLoading, setRebuildVisibility, updateRefineButtonVisibility, showRebuildError,
    updateTitleUI, applyConversationLanguages, renderSourceOptions, renderTargetOptions, syncLanguageUI
} from './ui.js';
import { fetchLanguages, fetchWithAuth } from './api.js';
import { syncUrlState } from './router.js';
import { stopRecording } from './audio.js';

function hasStableTranscriptPayload(data) {
    const source = (data.source || '').trim();
    const translation = (data.translation || '').trim();
    const isStable = data.is_final || data.type === 'stable';
    return isStable && Boolean(source || translation);
}

function syncConversationUrlIfReady(data) {
    if (state.hasSyncedConversationId || !state.pendingConversationId) return;
    if (!hasStableTranscriptPayload(data)) return;

    // Only persist the conversation ID after the first stable transcription
    // to avoid URLs pointing at server-pruned empty sessions.
    syncUrlState(true, { conversationId: state.pendingConversationId, view: 'main' });
    state.hasSyncedConversationId = true;
    state.pendingConversationId = null;
}

export function connectWebSocket() {
    if (state.socket && (state.socket.readyState === WebSocket.OPEN || state.socket.readyState === WebSocket.CONNECTING)) {
        return;
    }

    const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    let url = `${wsProtocol}://${window.location.host}/ws/stream?token=${encodeURIComponent(state.token)}`;
    if (state.activeConversationId) {
        url += `&conversation_id=${state.activeConversationId}`;
    }

    state.socket = new WebSocket(url);

    state.socket.onopen = () => {
        console.log('WebSocket connected');
        state.isModelsWarmingUp = true;
        state.isWhisperReady = false;
        state.isLlmReady = false;
        updateStatusIndicator('LOADING');
        renderSystemMessage(t('system_loading_transcription_ready'));
        sendConfig();
        if (!state.languagesFetched) {
            console.log("Connection established, retrying language fetch...");
            fetchLanguages().then(() => {
                if (state.languagesFetched) {
                    renderSourceOptions();
                    renderTargetOptions();
                    syncLanguageUI();
                }
            });
        }
    };

    state.socket.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.error || data.type === 'error') {
            const errMsg = data.error || data.message || "Unknown error";
            const backendInitError = getBackendInitError(data.error_code, BACKEND_INIT_ERROR_MAP);

            const displayMessage = backendInitError ? t(backendInitError.key) : errMsg;
            console.error("Backend Error:", errMsg);
            setRebuildLoading(false);

            const btn = document.getElementById('dynamic-rebuild-btn');
            if (btn) {
                showRebuildError();
                if (backendInitError) {
                    state.isModelsWarmingUp = false;
                    if (backendInitError.service === 'llm') state.isLlmReady = false;
                    if (backendInitError.service === 'whisper') state.isWhisperReady = false;
                    updateStatusIndicator('ERROR');
                    renderError(displayMessage);
                }
            } else {
                if (backendInitError) {
                    state.isModelsWarmingUp = false;
                    if (backendInitError.service === 'llm') state.isLlmReady = false;
                    if (backendInitError.service === 'whisper') state.isWhisperReady = false;
                    updateStatusIndicator('ERROR');
                }
                renderError(displayMessage);
            }

            if (state.isRecording) {
                stopRecording();
            }
            return;
        }

        if (data.status) {
            const isRecording = state.isRecording;

            if (data.status === 'warming_up') {
                state.isModelsWarmingUp = true;
                state.isWhisperReady = false;
                state.isLlmReady = false;
                updateStatusIndicator('LOADING');
                renderSystemMessage(t('system_loading_transcription_ready'));
            } else if (data.status === 'ready') {
                state.isModelsWarmingUp = false;
                state.isWhisperReady = true;
                state.isLlmReady = true;
                updateStatusIndicator('READY');
                if (!isRecording) renderSystemMessage(t('system_all_models_ready'), 'success');
            } else if (data.status === 'whisper_ready') {
                state.isWhisperReady = true;
                state.isModelsWarmingUp = false;
                if (!state.isLlmReady) {
                    updateStatusIndicator('LOADING');
                    renderSystemMessage(t('system_loading_translation_ready'));
                } else {
                    updateStatusIndicator('READY');
                    if (!isRecording) renderSystemMessage(t('system_all_models_ready'), 'success');
                }
            } else if (data.status === 'llm_ready') {
                state.isLlmReady = true;
                state.isModelsWarmingUp = false;
                if (state.isWhisperReady) {
                    updateStatusIndicator('READY');
                    if (!isRecording) renderSystemMessage(t('system_all_models_ready'), 'success');
                } else {
                    updateStatusIndicator('LOADING');
                }
            } else if (data.status === 'error') {
                if (data.service === 'llm') {
                    state.isLlmReady = false;
                    updateStatusIndicator('ERROR');
                    renderError(t('error_translation_model_failed'));
                    if (state.isWhisperReady) state.isModelsWarmingUp = false;
                } else if (data.service === 'whisper') {
                    state.isWhisperReady = false;
                    state.isModelsWarmingUp = false;
                    updateStatusIndicator('ERROR');
                    renderError(t('error_transcription_model_failed'));
                    if (isRecording) stopRecording();
                } else {
                    state.isModelsWarmingUp = false;
                    updateStatusIndicator('ERROR');
                    if (data.error) renderError(data.error);
                }
            } else if (data.status === 'processing_complete') {
                if (state.isWhisperReady && state.isLlmReady) updateStatusIndicator('READY');
                if (state.liveBlock) updateStreamBlock(undefined, undefined, false, null, null, 'stable');
                updateRefineButtonVisibility();
            }
            return;
        }

        if (data.conversation_id) {
            if (!state.activeConversationId) {
                state.activeConversationId = data.conversation_id;
            }
            if (!state.hasSyncedConversationId) {
                state.pendingConversationId = data.conversation_id;
            }
        }

        if (data.source_language || data.target_language) {
            applyConversationLanguages(data.source_language, data.target_language);
        }

        if (data.title) {
            updateTitleUI(data.title);
        }

        if (data.type === 'final') {
            setRebuildLoading(false);
            setRebuildVisibility(false);

            if (data.dataset_updates && Array.isArray(data.dataset_updates)) {
                data.dataset_updates.forEach(update => {
                    updateStreamBlock(
                        update.source,
                        update.translation,
                        false,
                        update.paragraph_id,
                        update.detected_language,
                        update.type
                    );
                });
            } else {
                updateStreamBlock(
                    data.source,
                    data.translation,
                    false,
                    data.paragraph_id,
                    data.detected_language,
                    'refined'
                );
            }

            updateRefineButtonVisibility();
            return;
        }

        if (data.source || data.translation || data.unstable_text) {
            const currentParaId = state.liveBlock?.dataset.paragraphId;
            const type = data.type || (data.is_final ? 'stable' : 'active');
            // Normalize rebuild streaming updates to match live rendering expectations.
            const displayType = ['chunk', 'status'].includes(type) ? 'active' : type;

            if (data.paragraph_id && currentParaId) {
                if (currentParaId === 'placeholder-pending') {
                    state.liveBlock.dataset.paragraphId = data.paragraph_id;
                }
            }

            if (data.paragraph_id) {
                if (data.translation_pending === true) {
                    state.pendingTranslations.add(data.paragraph_id);
                } else if (data.translation_pending === false) {
                    state.pendingTranslations.delete(data.paragraph_id);
                }
            }

            updateStreamBlock(
                data.source,
                data.translation,
                !data.is_final,
                data.paragraph_id,
                data.detected_language,
                displayType,
                data.unstable_text || ''
            );

            syncConversationUrlIfReady(data);
        }
    };

    state.socket.onclose = async () => {
        console.log('WebSocket disconnected');
        updateStatusIndicator('DISCONNECTED');

        if (state.isRecording) {
            stopRecording();
        }

        if (state.token) {
            try {
                await fetchWithAuth('/auth/verify');
            } catch (e) {
                if (state.reloginInProgress) return;
                console.warn("Auth check failed, assuming network issue and retrying:", e);
            }

            if (!state.reloginInProgress) {
                setTimeout(connectWebSocket, 3000);
            }
        }
    };

    state.socket.onerror = (err) => {
        console.error('WebSocket error:', err);
        updateStatusIndicator('DISCONNECTED');
    };
}

export function sendConfig() {
    if (state.socket && state.socket.readyState === WebSocket.OPEN) {
        let src = state.selectedSource;
        if (src !== 'auto') {
            src = [src];
        }
        let targetLang = state.targetLanguage;
        const langObj = state.targetLanguages.find(l => l.code === targetLang);
        if (langObj) targetLang = langObj.name;
        const recordingConfig = state.recordingConfig || getRecordingConfig();
        state.recordingConfig = recordingConfig;
        state.recordingMode = recordingConfig.mode;

        state.socket.send(JSON.stringify({
            source_language: src,
            target_language: targetLang,
            recording_format: recordingConfig.container,
            recording_mime_type: recordingConfig.mimeType,
            recording_sample_rate: RECORDING_SAMPLE_RATE,
            recording_channels: RECORDING_CHANNELS
        }));
    }
}
