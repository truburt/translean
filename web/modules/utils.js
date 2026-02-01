/*
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.
*/

import { I18N } from './i18n.js';
import { state } from './state.js';
import { UI_LANGUAGES } from './config.js';

export function t(key) {
    return I18N[state.uiLanguage]?.[key] ?? I18N.en[key] ?? key;
}

export function getUiLanguageCodeCandidates() {
    return UI_LANGUAGES.map((lang) => lang.code);
}

export function matchUiLanguage(locale) {
    if (!locale) return null;
    const normalized = String(locale).replace(/_/g, '-');
    const [language, region] = normalized.toLowerCase().split('-');

    if (language === 'ru' || region === 'ru') return 'ru';
    if (language === 'fi' || region === 'fi') return 'fi';
    if (language === 'en' || region === 'us' || region === 'gb') return 'en';
    return null;
}

export function getInitialUiLanguage() {
    const stored = localStorage.getItem('ui_language');
    const allowed = new Set(getUiLanguageCodeCandidates());
    if (stored && allowed.has(stored)) return stored;

    const localeCandidates = [
        ...(navigator.languages || []),
        navigator.language,
        Intl.DateTimeFormat().resolvedOptions().locale,
    ].filter(Boolean);

    for (const locale of localeCandidates) {
        const match = matchUiLanguage(locale);
        if (match && allowed.has(match)) return match;
    }

    return 'en';
}

export function getBackendInitError(errorCode, errorMap) {
    if (!errorCode) return null;
    return errorMap[errorCode] ?? null;
}
