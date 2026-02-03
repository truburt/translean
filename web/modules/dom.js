/*
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.
*/

export const views = {
    login: document.getElementById('login-view'),
    main: document.getElementById('main-view'),
    history: document.getElementById('history-view'),
    about: document.getElementById('about-view'),
    serverSettings: document.getElementById('server-settings-view'),
    menu: document.getElementById('menu-view'),
    onboarding: document.getElementById('onboarding-view'),
    backdrop: document.getElementById('sidebar-backdrop'),
};

export const dom = {
    streamContainer: document.getElementById('stream-container'),
    emptyState: document.getElementById('empty-state'),

    // Main Controls (Footer)
    sourceSelect: document.getElementById('source-language'),
    targetSelect: document.getElementById('target-language'),

    // Onboarding inputs
    onbSourceSelect: document.getElementById('onb-source-language'),
    onbTargetSelect: document.getElementById('onb-target-language'),

    // History
    conversationList: document.getElementById('conversation-list'),
    menuUserName: document.getElementById('menu-user-name'),
    uiLanguageSelect: document.getElementById('ui-language-select'),

    // Status
    fabStatusLabel: document.getElementById('fab-status-label'),
    sessionTitle: document.getElementById('session-title'),

    // Server settings
    serverSettingsForm: document.getElementById('server-settings-form'),
    serverSettingsStatus: document.getElementById('server-settings-status'),
    whisperEndpointInput: document.getElementById('whisper-endpoint'),
    whisperModelInput: document.getElementById('whisper-model'),
    whisperKeepAliveInput: document.getElementById('whisper-keep-alive'),
    ollamaEndpointInput: document.getElementById('ollama-endpoint'),
    ollamaModelInput: document.getElementById('ollama-model'),
    ollamaKeepAliveInput: document.getElementById('ollama-keep-alive'),
    commitTimeoutInput: document.getElementById('commit-timeout'),
    silenceFinalizeInput: document.getElementById('silence-finalize'),
    minPreviewBufferInput: document.getElementById('min-preview-buffer'),
    stableWindowInput: document.getElementById('stable-window'),
    noSpeechProbSkipInput: document.getElementById('no-speech-prob-skip'),
    noSpeechProbLogprobSkipInput: document.getElementById('no-speech-prob-logprob-skip'),
    avgLogprobSkipInput: document.getElementById('avg-logprob-skip'),
    compressionRatioSkipInput: document.getElementById('compression-ratio-skip'),
};

export const buttons = {
    loginStart: document.getElementById('login-start-button'),
    onboardingStart: document.getElementById('onboarding-start-btn'),
    record: document.getElementById('record-button'),
    burger: document.getElementById('burger-menu-btn'),
    menuClose: document.getElementById('menu-close-btn'),
    menuNewSession: document.getElementById('menu-new-session-btn'),
    newSession: document.getElementById('new-session-btn'),
    swap: document.getElementById('swap-languages-btn'),
    historyDeleteAll: document.getElementById('history-delete-all-btn'),

    // Nav
    navHistory: document.getElementById('nav-history'),
    navAbout: document.getElementById('nav-about'),
    navServerSettings: document.getElementById('nav-server-settings'),
    navLogout: document.getElementById('nav-logout'),
    historyBack: document.getElementById('history-back-btn'),
    aboutBack: document.getElementById('about-back-btn'),
    serverSettingsBack: document.getElementById('server-settings-back-btn'),

    // Icons
    micIcon: document.getElementById('mic-icon'),
    stopIcon: document.getElementById('stop-icon'),
};
