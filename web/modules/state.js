/*
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.
*/

export const state = {
    mediaRecorder: null,
    isRecording: false,
    recordingMode: 'media_recorder',
    recordingConfig: null,
    socket: null,
    token: null,
    activeConversationId: null,
    activeView: 'login',
    liveBlock: null, // The current DOM block being updated

    // Audio State
    headerChunk: null, // First chunk (WebM header)
    headerSent: false, // Whether header has been sent to backend
    rawStream: null,
    audioContext: null,
    audioPipelineCleanup: null,
    preRollChunks: [],
    preRollPending: false,
    preRollMaxChunks: 0,
    isStopping: false,
    pcmContext: null,
    pcmSource: null,
    pcmProcessor: null,
    pcmRemainder: null,

    // Language State
    sourceLanguages: [], // Fetched from backend
    targetLanguages: [], // Fetched from backend
    selectedSource: 'auto', // Default to Auto (Single select string)
    targetLanguage: 'es', // Default target
    languageUsage: { source: {}, target: {} },

    // VAD
    vad: null,
    isSpeaking: false,
    vadReady: false,
    vadEnabled: false,

    // Connectivity
    languagesFetched: false,
    isModelsWarmingUp: false,
    isWhisperReady: false,
    isLlmReady: false,

    // History Pagination
    historyLimit: 20,
    historyOffset: 0,
    historyTotalLoaded: 0, // Track loaded count to decide if load more is needed

    // UI
    menuOpen: false,
    userName: '',
    reloginInProgress: false,
    uiLanguage: 'en',
    uiStatus: 'DISCONNECTED',
    latestTitle: null,

    // Conversation routing
    pendingConversationId: null, // Server-assigned ID awaiting first stable transcript
    hasSyncedConversationId: false, // Whether the URL already reflects the active conversation

    // Pending deletion tracking
    pendingDeletionId: null, // Track conversation marked for deletion (for active conversation)
    historyPendingDeletionId: null, // Track conversation marked for deletion from history list
    pendingTranslations: new Set(), // Track which paragraphs are waiting for translation
    undoDismissController: null,
    historyUndoDismissController: null,

    // System Message Timer
    systemMessageTimer: null,
    systemMessageStartTime: null,

    // Title editing
    isTitleManual: false,
};
