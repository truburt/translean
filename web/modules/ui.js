/*
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.
*/

import { state } from './state.js';
import { dom, buttons, views } from './dom.js';
import { t } from './utils.js';
import { SPINNER_SVG as CFG_SPINNER, MIC_ANIMATION_SVG as CFG_MIC, TYPING_INDICATOR_HTML as CFG_TYPING, MAX_FAVORITE_LANGUAGES, UI_LANGUAGES } from './config.js';
import { fetchWithAuth } from './api.js'; // fetchLanguages unused? verify. fetchWithAuth used in history.

// Cyclic imports for actions
import { connectWebSocket, sendConfig } from './websocket.js';
import { stopRecording } from './audio.js';
import { syncUrlState } from './router.js'; // for view switching

import { loadConversationDetail as apiLoadConversationDetail } from './api.js';

// Re-export constants for local usage if needed (or just use CFG_)
const SPINNER_SVG = CFG_SPINNER;
const MIC_ANIMATION_SVG = CFG_MIC;
const TYPING_INDICATOR_HTML = CFG_TYPING;

// --- Language UI ---

export function applyTranslations() {
    document.documentElement.lang = state.uiLanguage;
    document.querySelectorAll('[data-i18n]').forEach((el) => {
        el.textContent = t(el.dataset.i18n);
    });
    document.querySelectorAll('[data-i18n-html]').forEach((el) => {
        el.innerHTML = t(el.dataset.i18nHtml);
    });
    document.querySelectorAll('[data-i18n-title]').forEach((el) => {
        el.title = t(el.dataset.i18nTitle);
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach((el) => {
        el.placeholder = t(el.dataset.i18nPlaceholder);
    });
    if (dom.uiLanguageSelect) {
        dom.uiLanguageSelect.value = state.uiLanguage;
    }
    if (state.sourceLanguages.length || state.targetLanguages.length) {
        renderSourceOptions();
        renderTargetOptions();
        syncLanguageUI();
    }
    updateStatusIndicator(state.uiStatus);
    updateMenuUserName();
}

export function setUiLanguage(code, persist = true) {
    // logic from main.js
    const candidates = UI_LANGUAGES.map(l => l.code);
    const allowed = new Set(candidates);
    const next = allowed.has(code) ? code : 'en';
    state.uiLanguage = next;
    if (persist) {
        localStorage.setItem('ui_language', next);
    }
    applyTranslations();
    setupUiLanguageSelect(); // re-render select to ensuring value? or just update value.
}

export function setupUiLanguageSelect() {
    if (!dom.uiLanguageSelect) return;
    dom.uiLanguageSelect.innerHTML = '';
    UI_LANGUAGES.forEach((lang) => {
        const option = document.createElement('option');
        option.value = lang.code;
        option.textContent = lang.label;
        dom.uiLanguageSelect.appendChild(option);
    });
    dom.uiLanguageSelect.value = state.uiLanguage;
}

export function updateMenuUserName() {
    if (!dom.menuUserName) return;
    dom.menuUserName.textContent = state.userName || t('auth_signed_in_default');
}

// --- View Management ---

export function hideAllViews() {
    views.login.classList.add('hidden');
    views.main.classList.add('hidden');
    views.onboarding.classList.add('hidden');
    views.history.classList.add('hidden');
    views.about.classList.add('hidden');

    if (state.activeView === 'history') {
        hideHistoryUndoButton();
    }
}

export function showLogin() {
    hideAllViews();
    views.login.classList.remove('hidden');
    state.activeView = 'login';
}

export function showMain(options = {}) {
    const { skipRouting = false } = options;
    hideAllViews();
    views.main.classList.remove('hidden');
    state.activeView = 'main';
    if (!skipRouting) {
        syncUrlState();
    }
}

export function showHistory(options = {}) {
    const { skipRouting = false } = options;
    hideAllViews();
    views.history.classList.remove('hidden');
    state.activeView = 'history';
    if (!skipRouting) {
        syncUrlState();
    }
    state.historyOffset = 0;
    dom.conversationList.innerHTML = '';
    refreshConversations();
}

export function showAbout(options = {}) {
    const { skipRouting = false } = options;
    hideAllViews();
    views.about.classList.remove('hidden');
    state.activeView = 'about';
    state.activeConversationId = null;
    state.hasSyncedConversationId = false;
    state.pendingConversationId = null;
    if (!skipRouting) {
        syncUrlState();
    }
}

export function showOnboarding() {
    hideAllViews();
    views.onboarding.classList.remove('hidden');
    state.activeView = 'onboarding';
}

export function toggleMenu(open, options = {}) {
    const { fromPopstate = false, skipHistory = false } = options;
    state.menuOpen = open;
    if (open) {
        views.backdrop.classList.remove('hidden');
        views.menu.classList.remove('-translate-x-full');
    } else {
        views.backdrop.classList.add('hidden');
        views.menu.classList.add('-translate-x-full');
    }
    if (skipHistory) return;
    if (open) {
        syncUrlState(false, { menuOpen: true });
    } else if (!fromPopstate) {
        syncUrlState(true, { menuOpen: false });
    }
}

// --- Status & Messages ---

export function updateStatusIndicator(status) {
    const label = dom.fabStatusLabel;
    if (!label || !buttons.record) return;

    state.uiStatus = status;

    label.className = 'fab-status-label';
    label.removeAttribute('style');

    const disableFab = status === 'DISCONNECTED' || (status === 'ERROR' && !state.isWhisperReady);
    buttons.record.disabled = disableFab;
    buttons.record.classList.toggle('record-fab--disabled', disableFab);
    buttons.record.classList.toggle('record-fab--enabled', !disableFab);
    buttons.record.classList.toggle('cursor-not-allowed', disableFab);
    buttons.record.setAttribute('aria-disabled', disableFab);

    if (status === 'DISCONNECTED') {
        label.classList.add('bg-gray-200', 'text-gray-500', 'dark:bg-gray-700', 'dark:text-gray-400');
        label.textContent = t('status_disconnected');
    } else if (status === 'LOADING') {
        label.classList.add('bg-yellow-100', 'text-yellow-700', 'animate-pulse', 'dark:bg-yellow-900/30', 'dark:text-yellow-400');
        label.textContent = t('status_loading');
    } else if (status === 'FINISHING') {
        label.classList.add('bg-yellow-100', 'text-yellow-700', 'animate-pulse', 'dark:bg-yellow-900/30', 'dark:text-yellow-400');
        label.textContent = t('status_finishing');
    } else if (status === 'READY') {
        label.classList.add('bg-green-100', 'text-green-700', 'dark:bg-green-900/30', 'dark:text-green-400');
        label.textContent = t('status_ready');
        if (state.activeConversationId) {
            updateRefineButtonVisibility();
        }
    } else if (status === 'ACTIVE') {
        label.classList.add('bg-accent', 'text-white', 'shadow-md');
        label.textContent = t('status_recording');
    } else if (status === 'ERROR') {
        label.classList.add('bg-red-100', 'text-red-700', 'dark:bg-red-900/30', 'dark:text-red-400');
        label.textContent = t('status_error');
    }
}

export function renderError(message) {
    renderSystemMessage(message, 'error');
}

export function stopSystemMessageTimer() {
    if (state.systemMessageTimer) {
        clearInterval(state.systemMessageTimer);
        state.systemMessageTimer = null;
    }
    state.systemMessageStartTime = null;
}

export function startSystemMessageTimer(baseText, blockElement) {
    state.systemMessageStartTime = Date.now();
    let timerSpan = blockElement.querySelector('.system-timer');
    if (!timerSpan) {
        timerSpan = document.createElement('span');
        timerSpan.className = 'system-timer text-xs opacity-75 font-mono ml-4';
        const content = blockElement.querySelector('.translation-text');
        if (content) { // ensure appended correctly
            blockElement.appendChild(timerSpan);
        }
    }

    const updateTimer = () => {
        if (!state.systemMessageStartTime) return;
        const elapsed = Math.floor((Date.now() - state.systemMessageStartTime) / 1000);
        timerSpan.textContent = `${elapsed}s`;
    };
    updateTimer();
    state.systemMessageTimer = setInterval(updateTimer, 1000);
}

export function renderSystemMessage(text, type = 'info') {
    if (!dom.streamContainer) return;

    const scrollState = captureStreamScrollState();
    stopSystemMessageTimer();

    const getLastStreamBlock = () => {
        const blocks = Array.from(dom.streamContainer.children).filter(
            el => !el.classList.contains('stream-bottom-spacer')
        );
        return blocks[blocks.length - 1] || null;
    };

    const lastBlock = getLastStreamBlock();
    let block;
    let content;

    if (lastBlock && lastBlock.dataset.type === 'system') {
        block = lastBlock;
        block.className = `group relative pl-4 py-2 transition-all duration-300 type-system system-${type} flex justify-between items-center`;
        block.dataset.type = 'system';

        content = block.querySelector('.translation-text');
        if (!content) {
            content = document.createElement('p');
            content.className = 'translation-text';
            block.appendChild(content);
        }
        const existingTimer = block.querySelector('.system-timer');
        if (existingTimer && type !== 'info') {
            existingTimer.remove();
        }
        content.textContent = text;
        restoreStreamScrollState(scrollState);
    } else {
        block = document.createElement('div');
        block.className = `group relative pl-4 py-2 transition-all duration-300 type-system system-${type} flex justify-between items-center`;
        block.dataset.type = 'system';
        content = document.createElement('p');
        content.className = 'translation-text';
        content.textContent = text;
        block.appendChild(content);
        const spacer = ensureStreamSpacer();
        if (spacer && spacer.parentElement === dom.streamContainer) {
            dom.streamContainer.insertBefore(block, spacer);
        } else {
            dom.streamContainer.appendChild(block);
        }
        restoreStreamScrollState(scrollState);
    }

    if (type === 'info') {
        startSystemMessageTimer(text, block);
    }
}

export function ensureStreamSpacer() {
    if (!dom.streamContainer) return null;
    let spacer = dom.streamContainer.querySelector('.stream-bottom-spacer');
    if (!spacer) {
        spacer = document.createElement('div');
        spacer.className = 'stream-bottom-spacer';
        dom.streamContainer.appendChild(spacer);
    }
    return spacer;
}

// --- Stream Block ---

function captureStreamScrollState() {
    if (!dom.streamContainer) return null;
    const container = dom.streamContainer;
    const scrollTop = container.scrollTop;
    const scrollHeight = container.scrollHeight;
    const clientHeight = container.clientHeight;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
    return {
        container,
        scrollTop,
        scrollHeight,
        isNearBottom: distanceFromBottom < 120,
    };
}

function restoreStreamScrollState(state) {
    if (!state?.container) return;
    const { container, scrollTop, scrollHeight, isNearBottom } = state;
    const newScrollHeight = container.scrollHeight;
    if (isNearBottom) {
        const bottomOffset = Math.max(0, newScrollHeight - container.clientHeight);
        container.scrollTo({ top: bottomOffset, behavior: 'smooth' });
        return;
    }
    const delta = newScrollHeight - scrollHeight;
    if (delta) {
        container.scrollTop = scrollTop + delta;
    }
}

export function updateStreamBlock(sourceText, translatedText, isLive = true, paragraphId = null, detectedLanguage = null, paragraphType = 'active', unstableText = '') {
    const scrollState = captureStreamScrollState();
    ensureStreamSpacer();
    let block = null;

    if (state.liveBlock && (!paragraphId || String(paragraphId) === state.liveBlock.dataset.paragraphId)) {
        block = state.liveBlock;
    } else if (paragraphId) {
        block = dom.streamContainer.querySelector(`[data-paragraph-id="${paragraphId}"]`);
    }

    if (!block) {
        if (state.liveBlock) {
            state.liveBlock.classList.remove('type-active', 'border-accent', 'animate-pulse-light', 'active-paragraph');
            state.liveBlock.classList.add('type-stable');
            state.liveBlock.dataset.type = 'stable';
        }

        block = document.createElement('div');
        block.className = "group relative pl-4 py-2 transition-all duration-300";
        if (paragraphId) block.dataset.paragraphId = paragraphId;
        block.dataset.type = paragraphType;
        block.classList.add(`type-${paragraphType}`);

        const topLine = document.createElement('div');
        topLine.className = "flex items-baseline gap-2 text-sm text-[var(--text-secondary)] mb-1 font-medium";
        const langTag = document.createElement('span');
        langTag.className = "font-bold text-accent uppercase tracking-wider text-xs hidden";
        langTag.textContent = "";
        const origSpan = document.createElement('span');
        origSpan.className = "original-text";
        topLine.appendChild(langTag);
        topLine.appendChild(origSpan);

        const botLine = document.createElement('p');
        botLine.className = "text-xl md:text-2xl font-bold text-[var(--text-primary)] leading-tight tracking-tight translation-text";

        if (paragraphType === 'refined' || paragraphType === 'summary') {
            const header = document.createElement('div');
            header.className = "text-md font-bold text-[var(--text-primary)] mb-1 block uppercase";
            header.textContent = paragraphType === 'refined'
                ? t('paragraph_full_translation')
                : t('paragraph_conversation_summary');
            block.appendChild(header);
        }

        block.appendChild(topLine);
        block.appendChild(botLine);

        const spacer = ensureStreamSpacer();
        if (spacer && spacer.parentElement === dom.streamContainer) {
            dom.streamContainer.insertBefore(block, spacer);
        } else {
            dom.streamContainer.appendChild(block);
        }
        state.liveBlock = block;
    }

    if (block.dataset.type !== paragraphType) {
        block.classList.remove(`type-${block.dataset.type}`);
        block.dataset.type = paragraphType;
        block.classList.add(`type-${paragraphType}`);
    }

    const origEl = block.querySelector('.original-text');
    const transEl = block.querySelector('.translation-text');

    const shouldUpdateSource = sourceText !== undefined || unstableText;
    const shouldUpdateTranslation = translatedText !== undefined;
    const showOriginal = sourceText && (sourceText !== translatedText || unstableText);
    const showUnstable = paragraphType === 'active' && Boolean(unstableText);
    const hasExistingSource = Boolean(origEl?.textContent?.trim());
    // Ignore empty active-source updates so transient backend payloads don't clear visible text.
    const ignoreEmptySourceUpdate = paragraphType === 'active'
        && (sourceText === '' || sourceText === null)
        && !showUnstable
        && hasExistingSource;

    if (shouldUpdateSource && !ignoreEmptySourceUpdate && (sourceText || showUnstable)) {
        if (paragraphId === 'placeholder-pending') {
            origEl.textContent = sourceText;
        } else if (showOriginal || showUnstable) {
            origEl.textContent = '';
            const stableSpan = document.createElement('span');
            stableSpan.className = "stable-text";
            stableSpan.textContent = sourceText || '';
            origEl.appendChild(stableSpan);

            if (showUnstable) {
                const unstableSpan = document.createElement('span');
                unstableSpan.className = "unstable-text";
                const prefix = unstableText.startsWith(' ') ? '' : ' ';
                unstableSpan.textContent = prefix + unstableText;
                origEl.appendChild(unstableSpan);
            }
        } else {
            origEl.textContent = '';
        }
    } else if (shouldUpdateSource && !ignoreEmptySourceUpdate) {
        origEl.textContent = '';
    }

    const isPending = paragraphId && state.pendingTranslations.has(paragraphId);
    const isTranslationBlank = translatedText === '' || translatedText === null;
    const hasExistingTranslation = Boolean(transEl?.textContent?.trim());
    // Preserve existing translation content on empty interim payloads while streaming.
    const ignoreEmptyTranslationUpdate = isTranslationBlank
        && hasExistingTranslation
        && (isPending || (isLive && paragraphType === 'active'));

    if (shouldUpdateTranslation && translatedText) {
        transEl.textContent = translatedText;
        transEl.classList.remove('text-gray-400', 'italic', 'text-base', 'font-normal');
    } else if (shouldUpdateTranslation && !ignoreEmptyTranslationUpdate) {
        if (paragraphType === 'active' && (sourceText || showUnstable) && !state.isLlmReady && !translatedText) {
            transEl.innerHTML = `
           <span class="inline-flex items-center gap-2">
             ${SPINNER_SVG}
             ${t('system_loading_model')}
           </span>
         `;
            transEl.classList.add('text-gray-400', 'italic', 'text-base', 'font-normal');
        } else {
            transEl.textContent = '';
            transEl.classList.remove('text-gray-400', 'italic', 'text-base', 'font-normal');
        }
    }

    if (isPending) {
        if (!transEl.innerHTML.includes('typing-indicator')) {
            transEl.insertAdjacentHTML('beforeend', TYPING_INDICATOR_HTML);
        }
    }

    const langTag = block.querySelector('.font-bold.text-accent');
    if (langTag) {
        if (paragraphType === 'active') {
            langTag.innerHTML = MIC_ANIMATION_SVG;
            langTag.classList.remove('hidden');
        } else if (detectedLanguage && detectedLanguage !== 'auto') {
            langTag.textContent = detectedLanguage;
            langTag.classList.remove('hidden');
        } else {
            if (detectedLanguage) {
                langTag.classList.add('hidden');
            }
        }
    }

    if (!isLive) {
        if (state.liveBlock === block) {
            state.liveBlock = null;
        }
    }

    restoreStreamScrollState(scrollState);
}

// --- History UI ---

export async function refreshConversations(limit = state.historyLimit, offset = state.historyOffset) {
    if (!state.token) return;

    if (offset === 0) {
        dom.conversationList.innerHTML = `<div class="text-center p-4 text-[var(--text-secondary)]">
      <div class="inline-flex items-center justify-center">
        ${SPINNER_SVG}
        ${t('history_loading')}
      </div>
    </div>`;
    } else {
        const btn = document.getElementById('history-load-more-btn');
        if (btn) {
            btn.innerHTML = `<div class="inline-flex items-center justify-center">
        ${SPINNER_SVG}
        ${t('history_loading')}
      </div>`;
            btn.disabled = true;
        }
    }

    try {
        const res = await fetchWithAuth(`/api/conversations?limit=${limit}&offset=${offset}`);
        if (!res.ok) throw new Error('Failed to load history');
        const items = await res.json();
        renderHistoryList(items, offset > 0);
    } catch { // unused e
        if (state.reloginInProgress) return;
        if (offset === 0) {
            dom.conversationList.innerHTML = `<div class="text-center p-4 text-red-500">${t('history_failed')}</div>`;
        } else {
            alert(t('history_load_more_failed'));
            const btn = document.getElementById('history-load-more-btn');
            if (btn) {
                btn.innerHTML = t('history_load_more');
                btn.disabled = false;
            }
        }
    }
}

export function renderHistoryList(items, append = false) {
    if (!append) {
        dom.conversationList.innerHTML = '';
    }
    const existingBtn = document.getElementById('history-load-more-btn');
    if (existingBtn) existingBtn.remove(); // will re-add at bottom

    if (items.length === 0 && !append) {
        dom.conversationList.innerHTML = `<div class="text-center p-4 text-[var(--text-secondary)]">${t('history_no_conversations')}</div>`;
        return;
    }

    items.forEach(item => {
        const li = document.createElement('li');
        li.className = "flex bg-[var(--bg-app)] border border-[var(--divider)] rounded-lg hover:border-accent transition-colors shadow-sm overflow-hidden group";
        const content = document.createElement('div');
        content.className = "flex-1 p-4 cursor-pointer min-w-0";
        content.addEventListener('click', () => {
            loadConversation(item.id);
        });

        const top = document.createElement('div');
        top.className = "flex justify-between items-start mb-2 gap-2";

        const title = document.createElement('h3');
        title.className = "font-bold text-[var(--text-primary)] break-words leading-tight";
        title.textContent = item.title || t('history_untitled');

        const date = document.createElement('span');
        date.className = "text-xs text-[var(--text-secondary)] shrink-0 pt-0.5";
        date.textContent = new Date(item.updated_at).toLocaleDateString();

        top.appendChild(title);
        top.appendChild(date);
        const bot = document.createElement('div');
        bot.className = "text-xs font-mono text-accent";
        bot.textContent = `${item.source_language} → ${item.target_language}`;
        content.appendChild(top);
        content.appendChild(bot);

        const delBtn = document.createElement('button');
        delBtn.className = "w-12 bg-gray-50 text-[var(--text-secondary)] hover:bg-gray-100 hover:text-red-500 flex items-center justify-center transition-colors border-l border-[var(--divider)] dark:bg-white/5 dark:hover:bg-white/10 dark:text-gray-400 dark:hover:text-red-400 shrink-0";
        delBtn.title = t('history_delete_title');
        delBtn.innerHTML = `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>`;
        delBtn.onclick = (e) => {
            e.stopPropagation();
            handleDeleteConversation(item.id);
        };
        li.appendChild(content);
        li.appendChild(delBtn);
        dom.conversationList.appendChild(li);
    });

    if (items.length === state.historyLimit) {
        const btn = document.createElement('button');
        btn.id = 'history-load-more-btn';
        btn.className = "w-full py-3 mt-4 text-[var(--text-secondary)] hover:text-[var(--text-primary)] text-sm font-medium border border-dashed border-[var(--divider)] rounded-lg hover:bg-[var(--bg-app-secondary)] transition-colors";
        btn.textContent = t('history_load_more');
        btn.onclick = () => {
            state.historyOffset += state.historyLimit;
            refreshConversations();
        };
        const li = document.createElement('li');
        li.className = "list-none";
        li.appendChild(btn);
        dom.conversationList.appendChild(li);
    }
}

function loadConversation(id) {
    if (!id) return;
    apiLoadConversationDetail(id);
}

export function loadLanguageUsage() {
    try {
        const stored = localStorage.getItem('language_usage');
        if (stored) {
            const parsed = JSON.parse(stored);
            state.languageUsage = {
                source: parsed.source || {},
                target: parsed.target || {},
            };
        }
    } catch (e) {
        console.warn('Unable to load language usage', e);
    }
}

export function saveLanguageUsage() {
    localStorage.setItem('language_usage', JSON.stringify(state.languageUsage));
}

export function recordLanguageUsage(kind, code) {
    if (!code || (kind !== 'source' && kind !== 'target')) return;
    const usage = state.languageUsage[kind] || {};
    usage[code] = (usage[code] || 0) + 1;
    state.languageUsage[kind] = usage;
    saveLanguageUsage();
}

export function getTopLanguageCodes(usageMap, availableCodes, includeAuto = false) {
    if (!usageMap) return [];
    const entries = Object.entries(usageMap)
        .filter(([code, count]) => count > 0 && (includeAuto || code !== 'auto'))
        .filter(([code]) => code === 'auto' ? includeAuto : availableCodes.has(code))
        .sort((a, b) => b[1] - a[1]);

    const top = [];
    for (const [code] of entries) {
        if (code === 'auto' && !includeAuto) continue;
        if (code !== 'auto' && !availableCodes.has(code)) continue;
        top.push(code);
        if (top.length >= MAX_FAVORITE_LANGUAGES) break;
    }
    return top;
}

export function renderSourceOptions() {
    const selects = [dom.sourceSelect, dom.onbSourceSelect];
    const availableCodes = new Set(state.sourceLanguages.map(l => l.code));
    const frequentCodes = getTopLanguageCodes(state.languageUsage.source, availableCodes, true);

    selects.forEach(select => {
        if (!select) return;
        select.innerHTML = '';
        const added = new Set();
        const appendOption = (parent, code, label) => {
            if (added.has(code)) return;
            const option = document.createElement('option');
            option.value = code;
            option.textContent = label;
            parent.appendChild(option);
            added.add(code);
        };
        const autoLabel = select === dom.onbSourceSelect ? t('onboarding_auto_detect') : t('language_auto');
        appendOption(select, 'auto', autoLabel);

        if (frequentCodes.length > 0) {
            const favGroup = document.createElement('optgroup');
            favGroup.label = t('language_group_frequent');
            frequentCodes.forEach(code => {
                if (code === 'auto') return;
                const lang = state.sourceLanguages.find(l => l.code === code);
                if (lang) appendOption(favGroup, lang.code, lang.name);
            });
            if (favGroup.children.length > 0) select.appendChild(favGroup);
        }
        const allGroup = document.createElement('optgroup');
        allGroup.label = t('language_group_all');
        state.sourceLanguages.forEach(lang => appendOption(allGroup, lang.code, lang.name));
        select.appendChild(allGroup);

        const desiredValue = added.has(state.selectedSource) ? state.selectedSource : 'auto';
        select.value = desiredValue;
    });
}

export function renderTargetOptions() {
    const selects = [dom.targetSelect, dom.onbTargetSelect];
    const availableCodes = new Set(state.targetLanguages.map(l => l.code));
    const frequentCodes = getTopLanguageCodes(state.languageUsage.target, availableCodes);

    selects.forEach(select => {
        if (!select) return;
        select.innerHTML = '';
        const added = new Set();
        const appendOption = (parent, code, label) => {
            if (added.has(code)) return;
            const option = document.createElement('option');
            option.value = code;
            option.textContent = label;
            parent.appendChild(option);
            added.add(code);
        };
        if (frequentCodes.length > 0) {
            const favGroup = document.createElement('optgroup');
            favGroup.label = t('language_group_frequent');
            frequentCodes.forEach(code => {
                const lang = state.targetLanguages.find(l => l.code === code);
                if (lang) appendOption(favGroup, lang.code, lang.name);
            });
            if (favGroup.children.length > 0) select.appendChild(favGroup);
        }
        const allGroup = document.createElement('optgroup');
        allGroup.label = t('language_group_all');
        state.targetLanguages.forEach(lang => appendOption(allGroup, lang.code, lang.name));
        select.appendChild(allGroup);

        if (state.targetLanguage && Array.from(select.options).some(opt => opt.value === state.targetLanguage)) {
            select.value = state.targetLanguage;
        } else if (state.targetLanguages.length > 0) {
            state.targetLanguage = state.targetLanguages[0].code;
            select.value = state.targetLanguage;
        }
    });
}

export function syncLanguageUI() {
    [dom.sourceSelect, dom.onbSourceSelect].forEach(el => {
        if (el && el.value !== state.selectedSource) el.value = state.selectedSource;
    });
    [dom.targetSelect, dom.onbTargetSelect].forEach(el => {
        if (el && el.value !== state.targetLanguage) el.value = state.targetLanguage;
    });
    if (buttons.swap) {
        buttons.swap.disabled = state.selectedSource === 'auto';
    }
}

export function updateFabState(isRecording) {
    if (!buttons.record) return;
    buttons.record.setAttribute('data-recording', isRecording);
    if (isRecording) {
        buttons.record.classList.add('fab-recording', 'record-fab--recording');
        if (buttons.micIcon) buttons.micIcon.classList.add('hidden');
        if (buttons.stopIcon) buttons.stopIcon.classList.remove('hidden');
    } else {
        buttons.record.classList.remove('fab-recording', 'record-fab--recording', 'animate-pulse-ring');
        if (buttons.micIcon) buttons.micIcon.classList.remove('hidden');
        if (buttons.stopIcon) buttons.stopIcon.classList.add('hidden');
    }
}

export function updateVADIndicator(active) {
    if (!buttons.record) return;
    if (active && state.vadReady) {
        buttons.record.classList.add('animate-pulse-ring');
    } else {
        buttons.record.classList.remove('animate-pulse-ring');
    }
}

export function loadLanguageSettings() {
    const stored = localStorage.getItem('language_settings');
    if (stored) {
        const s = JSON.parse(stored);
        // Legacy support
        if (Array.isArray(s.source)) {
            state.selectedSource = s.source.length > 0 ? s.source[0] : 'auto';
        } else if (s.source) {
            state.selectedSource = s.source;
        }
        if (s.auto === true) state.selectedSource = 'auto';
        if (s.target) state.targetLanguage = s.target;
    } else {
        state.selectedSource = 'auto';
        try {
            const browserLang = navigator.language.split('-')[0];
            const isSupported = state.targetLanguages.some(l => l.code === browserLang);
            if (isSupported) state.targetLanguage = browserLang;
        } catch { /**/ }
    }
    syncLanguageUI();
}

export function saveLanguageSettings() {
    const settings = {
        source: state.selectedSource,
        target: state.targetLanguage,
        auto: state.selectedSource === 'auto'
    };
    localStorage.setItem('language_settings', JSON.stringify(settings));
}

export function handleLanguageSwap() {
    if (state.selectedSource === 'auto') return;

    const oldSource = state.selectedSource;
    const oldTarget = state.targetLanguage;

    applyConversationLanguages(oldTarget, oldSource);

    recordLanguageUsage('source', state.selectedSource);
    recordLanguageUsage('target', state.targetLanguage);

    renderSourceOptions();
    renderTargetOptions();
    syncLanguageUI();

    if (typeof sendConfig === 'function') {
        sendConfig();
    }
}

export function applyConversationLanguages(sourceLanguage, targetLanguage) {
    if (sourceLanguage) {
        let normalizedSource = sourceLanguage;
        if (normalizedSource.includes(',')) {
            const parts = normalizedSource.split(',').map(p => p.trim()).filter(Boolean);
            normalizedSource = parts.includes('auto') ? 'auto' : (parts[0] || 'auto');
        }

        const availableSourceCodes = new Set(state.sourceLanguages.map(l => l.code));
        if (normalizedSource !== 'auto' && !availableSourceCodes.has(normalizedSource)) {
            normalizedSource = 'auto';
        }
        state.selectedSource = normalizedSource;
    }

    if (targetLanguage) {
        const availableTargetCodes = new Set(state.targetLanguages.map(l => l.code));
        if (availableTargetCodes.has(targetLanguage)) {
            state.targetLanguage = targetLanguage;
        }
    }

    syncLanguageUI();
    saveLanguageSettings();
}

export function updateTitleUI(title) {
    if (dom.sessionTitle) {
        if (document.activeElement === dom.sessionTitle) return;

        if (title) {
            dom.sessionTitle.textContent = title;
            dom.sessionTitle.title = title;
            dom.sessionTitle.classList.remove('hidden');
        } else {
            dom.sessionTitle.textContent = '';
            dom.sessionTitle.title = '';
            dom.sessionTitle.classList.add('hidden');
        }
    }
}

// Actions like resetSession / handleDelete
// resetSession calls stopRecording -> audio.js
// so we need import stopRecording.
// and connectWebSocket -> websocket.js

export function resetSession(options = {}) {
    const { keepPendingDeletion = false } = options;

    if (state.isRecording) {
        stopRecording();
    }

    dom.streamContainer.innerHTML = '';
    renderSystemMessage(t('system_loading'));
    ensureStreamSpacer();

    state.activeConversationId = null;
    state.hasSyncedConversationId = false;
    state.pendingConversationId = null;
    state.latestTitle = null;
    updateTitleUI(null);
    setRebuildVisibility(false);

    dismissUndoNotification({ clearState: !keepPendingDeletion });

    showMain({ skipRouting: true });
    syncUrlState(true, { conversationId: null, view: 'main', menuOpen: state.menuOpen });

    if (state.socket) {
        state.socket.onclose = null;
        state.socket.close();
        state.socket = null;
    }
    updateStatusIndicator('LOADING');
    connectWebSocket();
}

export async function handleDeleteConversation(id) {
    try {
        const res = await fetchWithAuth(`/api/conversations/${id}/pending-deletion`, { method: 'POST' });
        if (!res.ok) throw new Error('Delete failed');
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

export async function handleDeleteCurrentConversation() {
    if (!state.activeConversationId) return;
    try {
        const res = await fetchWithAuth(`/api/conversations/${state.activeConversationId}/pending-deletion`, {
            method: 'POST',
        });
        if (!res.ok) throw new Error('Failed');
        state.pendingDeletionId = state.activeConversationId;
        resetSession({ keepPendingDeletion: true });
        showUndoNotification();
    } catch (e) {
        //...
        console.error(e);
        alert('Failed to delete');
    }
}

// Rebuild UI
export function checkRefineEligibility() {
    if (!state.activeConversationId) return false;
    const blocks = Array.from(dom.streamContainer.querySelectorAll('[data-paragraph-id]'));
    const realBlocks = blocks.filter(el => el.dataset.paragraphId !== 'placeholder-pending');
    if (realBlocks.length === 0) return false;
    const lastBlock = realBlocks[realBlocks.length - 1];
    const type = lastBlock.dataset.type;
    if (type === 'refined' || type === 'summary') return false;
    return true;
}

export function updateRefineButtonVisibility() {
    const visible = checkRefineEligibility();
    setRebuildVisibility(visible);
}

export function ensureRebuildButton() {
    let container = document.getElementById('dynamic-actions-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'dynamic-actions-container';
        container.className = 'actions-container fade-in';
        container.innerHTML = `
      <button id="dynamic-delete-btn" class="delete-btn" title="Delete conversation">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
            d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16">
          </path>
        </svg>
      </button>
      <button id="dynamic-rebuild-btn" class="rebuild-btn">
        <span>✨&nbsp;Enchant</span>
      </button>
    `;
        const deleteBtn = container.querySelector('#dynamic-delete-btn');
        const rebuildBtn = container.querySelector('#dynamic-rebuild-btn');
        deleteBtn.addEventListener('click', handleDeleteCurrentConversation);
        rebuildBtn.addEventListener('click', rebuildTranslation);
        views.main.appendChild(container);
    }
    return container;
}

export function setRebuildVisibility(show) {
    const container = document.getElementById('dynamic-actions-container');
    if (show) {
        const c = ensureRebuildButton();
        c.classList.remove('hidden');
    } else {
        if (container) container.remove();
    }
}

export function setRebuildLoading(isLoading) {
    const btn = document.getElementById('dynamic-rebuild-btn');
    if (!btn) return;
    if (isLoading) {
        btn.classList.add('animate-color-pulse', 'pointer-events-none');
        btn.innerHTML = `${SPINNER_SVG} Processing...`;
        btn.disabled = true;
    } else {
        btn.classList.remove('animate-color-pulse', 'pointer-events-none');
        btn.disabled = false;
    }
}

export function showRebuildError() {
    const btn = document.getElementById('dynamic-rebuild-btn');
    if (btn) {
        btn.innerHTML = `<svg class="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg><span>${t('status_error')}</span>`;
        setTimeout(() => setRebuildVisibility(true), 2000);
    }
    setRebuildLoading(false);
}

import { rebuildTranslation } from './api.js'; // Need to impl in api.js

// Undo Notifications

export function showUndoNotification() {
    dismissUndoNotification({ clearState: false });
    const notification = document.createElement('div');
    notification.id = 'undo-notification';
    notification.className = 'undo-notification fade-in';
    notification.innerHTML = `
    <span class="undo-text">${t('undo_pending_delete')}</span>
    <button id="undo-delete-btn" class="undo-btn">${t('history_undo')}</button>
  `;
    const undoBtn = notification.querySelector('#undo-delete-btn');
    undoBtn.dataset.conversationId = state.pendingDeletionId || '';
    undoBtn.addEventListener('click', handleUndoDelete);

    views.main.appendChild(notification);
    setupUndoDismissal({
        notification,
        undoButton: undoBtn,
        onDismiss: dismissUndoNotification,
        controllerKey: 'undoDismissController',
    });
}

function setupUndoDismissal({ notification, undoButton, onDismiss, controllerKey }) {
    if (!notification) return;
    if (state[controllerKey]) state[controllerKey].abort();
    const controller = new AbortController();
    state[controllerKey] = controller;
    const { signal } = controller;

    const shouldIgnoreKeydown = (event) => {
        if (!undoButton) return false;
        if (event.target !== undoButton) return false;
        return event.key === 'Enter' || event.key === ' ' || event.key === 'Spacebar';
    };

    const handleDismiss = (event) => {
        if (!notification.isConnected) {
            controller.abort();
            state[controllerKey] = null;
            return;
        }
        if (undoButton && (event.target === undoButton || undoButton.contains(event.target))) {
            if (event.type === 'keydown' && shouldIgnoreKeydown(event)) return;
            if (event.type !== 'keydown') return;
        }
        if (event.type === 'keydown' && shouldIgnoreKeydown(event)) return;
        onDismiss();
    };
    document.addEventListener('pointerdown', handleDismiss, { capture: true, signal });
    document.addEventListener('keydown', handleDismiss, { capture: true, signal });
}

export function dismissUndoNotification(options = {}) {
    const { clearState = true } = options;
    const notification = document.getElementById('undo-notification');
    if (notification) notification.remove();
    if (clearState) state.pendingDeletionId = null;
    if (state.undoDismissController) {
        state.undoDismissController.abort();
        state.undoDismissController = null;
    }
}

export async function handleUndoDelete(event) {
    const idToRestore = event?.currentTarget?.dataset?.conversationId || state.pendingDeletionId;
    if (!idToRestore) return;
    try {
        const res = await fetchWithAuth(`/api/conversations/${idToRestore}/pending-deletion`, { method: 'DELETE' });
        if (!res.ok) throw new Error('Restoration failed');
        state.pendingDeletionId = null;
        dismissUndoNotification({ clearState: false });
        await apiLoadConversationDetail(idToRestore); // use imported api
    } catch {
        if (state.reloginInProgress) return;
        alert(t('history_restore_failed'));
    }
}

export function showHistoryUndoButton() {
    dismissHistoryUndo({ clearState: false });
    const historyView = document.getElementById('history-view');
    if (!historyView) return;
    const container = document.createElement('div');
    container.id = 'history-undo-container';
    container.className = 'history-undo-container';
    container.innerHTML = `
     <div class="history-undo-content">
       <span class="history-undo-text">${t('history_deleted_toast')}</span>
       <button id="history-undo-btn" class="history-undo-btn">${t('history_undo')}</button>
     </div>
   `;
    historyView.appendChild(container);
    const undoBtn = container.querySelector('#history-undo-btn');
    undoBtn.addEventListener('click', handleHistoryUndoDelete);
    setupUndoDismissal({
        notification: container,
        undoButton: undoBtn,
        onDismiss: dismissHistoryUndo,
        controllerKey: 'historyUndoDismissController',
    });
}

export function dismissHistoryUndo(options = {}) {
    const { clearState = true } = options;
    const existing = document.getElementById('history-undo-container');
    if (existing) existing.remove();
    if (clearState) state.historyPendingDeletionId = null;
    if (state.historyUndoDismissController) {
        state.historyUndoDismissController.abort();
        state.historyUndoDismissController = null;
    }
}

export function hideHistoryUndoButton() { dismissHistoryUndo(); }

export async function handleHistoryUndoDelete() {
    if (!state.historyPendingDeletionId) return;
    try {
        const res = await fetchWithAuth(`/api/conversations/${state.historyPendingDeletionId}/pending-deletion`, { method: 'DELETE' });
        if (!res.ok) throw new Error('Failed');
        state.historyPendingDeletionId = null;
        hideHistoryUndoButton();
        refreshConversations();
    } catch {
        if (state.reloginInProgress) return;
        alert(t('history_restore_failed'));
    }
}

export function renderConversation(conversation, options = {}) {
    const { skipRouting = false } = options;
    dom.streamContainer.innerHTML = '';
    if (dom.emptyState) {
        dom.streamContainer.appendChild(dom.emptyState);
        dom.emptyState.classList.add('hidden');
    }

    state.liveBlock = null;
    ensureStreamSpacer();

    applyConversationLanguages(conversation.source_language, conversation.target_language);
    state.activeConversationId = conversation.id;
    state.hasSyncedConversationId = true;
    state.pendingConversationId = null;
    state.latestTitle = conversation.title || state.latestTitle;
    updateTitleUI(state.latestTitle);

    if (state.socket) {
        state.socket.onclose = null;
        state.socket.close();
        state.socket = null;
    }
    connectWebSocket(); // imported

    const paragraphs = Array.isArray(conversation.paragraphs) ? conversation.paragraphs.slice() : [];
    paragraphs.sort((a, b) => a.paragraph_index - b.paragraph_index);

    if (paragraphs.length === 0 && dom.emptyState) {
        dom.emptyState.classList.remove('hidden');
    }

    let lastRefinedIndex = -1;
    let lastSummaryIndex = -1;
    paragraphs.forEach((p, idx) => {
        if (p.type === 'refined') lastRefinedIndex = idx;
        if (p.type === 'summary') lastSummaryIndex = idx;
    });

    paragraphs.forEach((p, idx) => {
        if (p.type === 'refined' && idx !== lastRefinedIndex) return;
        if (p.type === 'summary' && idx !== lastSummaryIndex) return;
        const src = p.source_text || p.source || "";
        const trans = p.translated_text || p.translation || "";
        if (!src && !trans) return;
        updateStreamBlock(src, trans, false, p.id, p.detected_language, p.type);
    });

    updateRefineButtonVisibility();

    if (!skipRouting) {
        syncUrlState(false, { conversationId: conversation.id, view: 'main' });
    }
}
