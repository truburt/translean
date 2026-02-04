/*
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.
*/

import { state } from './state.js';
import { dom } from './dom.js';
import { t } from './utils.js';
import { triggerRelogin } from './auth.js';
// Imported UI functions for flow control.
import { renderConversation, renderError, setRebuildLoading, setRebuildVisibility, refreshConversations, resetSession, showHistoryUndoButton } from './ui.js';
import { showMain, toggleMenu } from './ui.js'; // for loadConversationDetail view switching
import { stopRecording } from './audio.js'; // for loadConversationDetail stopping recording


export async function fetchWithAuth(url, options = {}) {
    if (!state.token) {
        triggerRelogin(t('auth_missing_token'));
        throw new Error(t('auth_missing_token'));
    }

    const headers = {
        ...(options.headers || {}),
        Authorization: `Bearer ${state.token}`,
    };

    const res = await fetch(url, { ...options, headers });
    if (res.status === 401 || res.status === 403) {
        triggerRelogin(t('auth_token_expired'));
        throw new Error('Unauthorized');
    }
    return res;
}

export async function fetchLanguages() {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000); // 5s timeout

        const res = await fetch('/api/languages', { signal: controller.signal });
        clearTimeout(timeoutId);
        if (!res.ok) throw new Error('Failed to fetch languages');
        const data = await res.json();
        state.sourceLanguages = data.source.sort((a, b) => a.name.localeCompare(b.name));
        state.targetLanguages = data.target.sort((a, b) => a.name.localeCompare(b.name));
        state.languagesFetched = true;

        localStorage.setItem('cached_languages', JSON.stringify({
            source: state.sourceLanguages,
            target: state.targetLanguages,
            timestamp: Date.now()
        }));
    } catch (e) {
        console.warn('Error loading languages, trying cache:', e);
        const cached = localStorage.getItem('cached_languages');
        if (cached) {
            try {
                const data = JSON.parse(cached);
                state.sourceLanguages = data.source || [];
                state.targetLanguages = data.target || [];
                console.log("Loaded languages from cache");
            } catch (err) {
                console.error("Failed to parse cached languages", err);
                state.sourceLanguages = [{ code: 'en', name: 'English' }];
                state.targetLanguages = [{ code: 'en', name: 'English' }];
            }
        } else {
            state.sourceLanguages = [{ code: 'en', name: 'English' }];
            state.targetLanguages = [{ code: 'en', name: 'English' }];
        }
        state.languagesFetched = false;
    }
}

export async function fetchAuthStatus() {
    const res = await fetchWithAuth('/auth/verify');
    if (!res.ok) throw new Error('Failed to verify auth');
    return res.json();
}

export async function fetchServerConfig() {
    const res = await fetchWithAuth('/api/admin/config');
    if (!res.ok) throw new Error('Failed to load server configuration');
    return res.json();
}

export async function updateServerConfig(payload) {
    const res = await fetchWithAuth('/api/admin/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    if (!res.ok) {
        const errorText = await res.text();
        throw new Error(errorText || 'Failed to save server configuration');
    }
    return res.json();
}

export async function saveTitle(newTitle) {
    if (!state.activeConversationId || !newTitle) {
        return;
    }
    try {
        const res = await fetchWithAuth(`/api/conversations/${state.activeConversationId}/title`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: newTitle })
        });
        if (!res.ok) throw new Error("Failed to update title");
        const data = await res.json();
        return data;
    } catch (e) {
        if (state.reloginInProgress) return;
        console.error("Error saving title:", e);
        throw e;
    }
}

export async function loadConversationDetail(conversationId, options = {}) {
    const { skipRouting = false, silentOnNotFound = false } = options;
    if (!state.token) {
        triggerRelogin(t('auth_missing_token'));
        return;
    }

    showMain({ skipRouting: true });
    toggleMenu(false, { skipHistory: skipRouting });

    if (state.isRecording) {
        // We should await stop? But synchronous UI update is needed. 
        // stopRecording returns promise.
        stopRecording();
    }

    dom.streamContainer.innerHTML = `<div class="p-4 text-center text-[var(--text-secondary)]">${t('history_loading_conversation')}</div>`;

    try {
        const res = await fetchWithAuth(`/api/conversations/${conversationId}`);
        if (!res.ok) {
            if (res.status === 404) {
                if (silentOnNotFound) {
                    console.warn("Conversation not found (404), silently resetting session.");
                    resetSession();
                    return;
                } else {
                    throw new Error("Conversation not found");
                }
            }
            const errorText = await res.text();
            throw new Error(errorText || t('history_load_conversation_failed'));
        }
        const conversation = await res.json();
        // Render
        renderConversation(conversation, { skipRouting });

    } catch (err) {
        if (state.reloginInProgress) return;
        console.error('Unable to load conversation', err);
        if (!silentOnNotFound) {
            dom.streamContainer.innerHTML = '';
            renderError(err.message || t('history_load_conversation_failed'));
            setTimeout(() => {
                resetSession();
            }, 2000);
        } else {
            resetSession();
        }
    }
}

export async function rebuildTranslation() {
    if (!state.activeConversationId) return;
    setRebuildLoading(true);
    try {
        const existingRefined = dom.streamContainer.querySelectorAll('[data-type="refined"], [data-type="summary"]');
        existingRefined.forEach(el => el.remove());

        const res = await fetchWithAuth(`/api/conversations/${state.activeConversationId}/rebuild`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!res.ok) throw new Error('Rebuild failed to start');
        // Backend sends updates via WebSocket.
    } catch (e) {
        if (state.reloginInProgress) return;
        console.error(e);
        const btn = document.getElementById('dynamic-rebuild-btn');
        if (btn) {
            btn.innerHTML = `<svg class="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg><span>${t('status_error')}</span>`;
            setTimeout(() => setRebuildVisibility(true), 2000);
        }
        setRebuildLoading(false);
    }
}

export async function handleDeleteConversation(id) {
    try {
        const res = await fetchWithAuth(`/api/conversations/${id}/pending-deletion`, { method: 'POST' });
        if (!res.ok) throw new Error('Delete failed');

        // We set pending id in ui usually, but here is fine or ui handles?
        // UI code (original) set global state.
        state.historyPendingDeletionId = id;

        refreshConversations();

        if (state.activeConversationId === id) {
            resetSession();
        }
        showHistoryUndoButton();

    } catch (e) {
        if (state.reloginInProgress) return;
        alert(t('history_delete_failed'));
        console.error(e);
    }
}
