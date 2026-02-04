/*
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.
*/

import { state } from './state.js';
import { dom } from './dom.js';
import { t } from './utils.js';
import { fetchServerConfig, updateServerConfig } from './api.js';

function setServerSettingsStatus(message, type = 'info') {
    const statusEl = dom.serverSettingsStatus;
    if (!statusEl) return;
    statusEl.textContent = message || '';
    statusEl.className = `text-sm ${type === 'error' ? 'text-red-500' : type === 'success' ? 'text-green-500' : 'text-[var(--text-secondary)]'}`;
}

export async function loadServerSettings() {
    if (!state.isAdmin) return;
    setServerSettingsStatus(t('server_settings_loading'));
    try {
        const data = await fetchServerConfig();
        dom.whisperEndpointInput.value = data.whisper_base_url || '';
        dom.whisperModelInput.value = data.whisper_model || '';
        dom.whisperKeepAliveInput.value = data.whisper_keep_alive_seconds ?? '';
        dom.ollamaEndpointInput.value = data.ollama_base_url || '';
        dom.ollamaModelInput.value = data.llm_model_translation || '';
        dom.ollamaKeepAliveInput.value = data.ollama_keep_alive_seconds ?? '';
        dom.commitTimeoutInput.value = data.commit_timeout_seconds ?? '';
        dom.silenceFinalizeInput.value = data.silence_finalize_seconds ?? '';
        dom.minPreviewBufferInput.value = data.min_preview_buffer_seconds ?? '';
        dom.stableWindowInput.value = data.stable_window_seconds ?? '';
        dom.noSpeechProbSkipInput.value = data.no_speech_prob_skip ?? '';
        dom.noSpeechProbLogprobSkipInput.value = data.no_speech_prob_logprob_skip ?? '';
        dom.avgLogprobSkipInput.value = data.avg_logprob_skip ?? '';
        dom.compressionRatioSkipInput.value = data.compression_ratio_skip ?? '';
        setServerSettingsStatus(t('server_settings_loaded'), 'success');
    } catch (e) {
        console.error('Failed to load server settings', e);
        setServerSettingsStatus(t('server_settings_load_failed'), 'error');
    }
}

export async function handleSaveServerSettings(event) {
    event.preventDefault();
    if (!state.isAdmin) {
        alert(t('admin_access_denied'));
        return;
    }
    const payload = {
        whisper_base_url: dom.whisperEndpointInput.value.trim(),
        whisper_model: dom.whisperModelInput.value.trim(),
        whisper_keep_alive_seconds: Number(dom.whisperKeepAliveInput.value),
        ollama_base_url: dom.ollamaEndpointInput.value.trim(),
        llm_model_translation: dom.ollamaModelInput.value.trim(),
        ollama_keep_alive_seconds: Number(dom.ollamaKeepAliveInput.value),
        commit_timeout_seconds: Number(dom.commitTimeoutInput.value),
        silence_finalize_seconds: Number(dom.silenceFinalizeInput.value),
        min_preview_buffer_seconds: Number(dom.minPreviewBufferInput.value),
        stable_window_seconds: Number(dom.stableWindowInput.value),
        no_speech_prob_skip: Number(dom.noSpeechProbSkipInput.value),
        no_speech_prob_logprob_skip: Number(dom.noSpeechProbLogprobSkipInput.value),
        avg_logprob_skip: Number(dom.avgLogprobSkipInput.value),
        compression_ratio_skip: Number(dom.compressionRatioSkipInput.value),
    };
    setServerSettingsStatus(t('server_settings_saving'));
    try {
        await updateServerConfig(payload);
        setServerSettingsStatus(t('server_settings_saved'), 'success');
    } catch (e) {
        console.error('Failed to save server settings', e);
        setServerSettingsStatus(t('server_settings_save_failed'), 'error');
    }
}
