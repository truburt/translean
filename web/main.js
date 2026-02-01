
/*
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.
*/
import { state } from './modules/state.js';
import { dom, buttons, views } from './modules/dom.js';

import { t, getInitialUiLanguage } from './modules/utils.js';
import { loadTokens, handleLogin, handleLogout } from './modules/auth.js';
import { fetchLanguages, fetchWithAuth, saveTitle } from './modules/api.js';
import {
  applyTranslations, updateStatusIndicator, renderSourceOptions, renderTargetOptions,
  loadLanguageUsage, loadLanguageSettings, saveLanguageSettings,
  renderSystemMessage, syncLanguageUI, setUiLanguage, setupUiLanguageSelect,
  showHistory, showAbout, showMain, toggleMenu, handleLanguageSwap, resetSession,
  handleUndoDelete
} from './modules/ui.js';
import { toggleRecording } from './modules/audio.js'; // startRecording if needed
import { connectWebSocket, sendConfig } from './modules/websocket.js';
import { applyRouteFromLocation, handlePopState } from './modules/router.js';

// Re-map a few things for listeners
import { recordLanguageUsage } from './modules/ui.js';

async function init() {
  setupEventListeners(); // Setup listeners immediately so buttons work

  const storedUiLanguage = localStorage.getItem('ui_language');
  state.uiLanguage = getInitialUiLanguage();
  if (!storedUiLanguage) {
    localStorage.setItem('ui_language', state.uiLanguage);
  }
  setupUiLanguageSelect();
  loadTokens(); // Load tokens first to show Login/Main UI immediately

  applyTranslations();
  updateStatusIndicator('DISCONNECTED');

  // Non-blocking language fetch (or handled by timeout in api.js, but we await it here)
  // If we await, we must ensure it doesn't hang forever.
  await fetchLanguages();

  loadLanguageUsage();
  loadLanguageSettings();
  renderSourceOptions();
  renderTargetOptions();

  // init() loads tokens which calls showMain/Login.
  // We need to setup listeners too.


  startKeepAlive();
  await applyRouteFromLocation({ replaceHistory: true });

  if (state.token) {
    connectWebSocket();
  }

  // Initial Empty State Message if on main view without conversation
  if (state.activeView === 'main' && !state.activeConversationId) {
    renderSystemMessage(t('system_loading'));
  }
}

function startKeepAlive() {
  setInterval(() => {
    // Only if socket is open
    if (!state.socket || state.socket.readyState !== WebSocket.OPEN) return;

    // Only if not recording
    if (state.isRecording) return;

    // Send keep-alive
    console.log("Sending keep-alive");
    try {
      state.socket.send(JSON.stringify({ type: 'keep_alive' }));
    } catch (e) {
      console.warn("Failed to send keep-alive", e);
    }
  }, 10 * 60 * 1000); // 10 minutes
}

function setupEventListeners() {
  window.addEventListener('popstate', handlePopState);
  buttons.loginStart.addEventListener('click', handleLogin);
  buttons.burger.addEventListener('click', () => toggleMenu(true));
  buttons.menuClose.addEventListener('click', () => toggleMenu(false));
  views.backdrop.addEventListener('click', () => toggleMenu(false));

  // Nav
  buttons.navHistory.addEventListener('click', () => {
    toggleMenu(false);
    showHistory();
  });
  buttons.navAbout.addEventListener('click', () => {
    toggleMenu(false);
    showAbout();
  });
  buttons.navLogout.addEventListener('click', () => {
    toggleMenu(false);
    handleLogout();
  });

  // History Back
  buttons.historyBack.addEventListener('click', async () => {
    if (state.pendingDeletionId) {
      await handleUndoDelete();
    }
    window.history.back();
  });
  buttons.aboutBack.addEventListener('click', () => window.history.back());

  if (buttons.menuNewSession) {
    buttons.menuNewSession.addEventListener('click', () => {
      resetSession();
      toggleMenu(false);
    });
  }
  buttons.newSession.addEventListener('click', resetSession);

  if (buttons.historyDeleteAll) {
    buttons.historyDeleteAll.addEventListener('click', handleDeleteAllHistory);
  }

  // Swap Languages
  if (buttons.swap) {
    buttons.swap.addEventListener('click', handleLanguageSwap);
  }

  // Record FAB
  buttons.record.addEventListener('click', toggleRecording);

  // Settings
  [dom.sourceSelect, dom.onbSourceSelect].forEach(el => {
    if (!el) return;
    el.addEventListener('change', (e) => {
      state.selectedSource = e.target.value;
      recordLanguageUsage('source', state.selectedSource);
      renderSourceOptions();
      syncLanguageUI();
      saveLanguageSettings();
      sendConfig();
    });
  });

  [dom.targetSelect, dom.onbTargetSelect].forEach(el => {
    if (!el) return;
    el.addEventListener('change', (e) => {
      state.targetLanguage = e.target.value;
      recordLanguageUsage('target', state.targetLanguage);
      renderTargetOptions();
      syncLanguageUI();
      saveLanguageSettings();
      sendConfig();
    });
  });

  // Onboarding listeners
  // Needs handleOnboardingComplete
  buttons.onboardingStart.addEventListener('click', handleOnboardingComplete);

  if (dom.uiLanguageSelect) {
    dom.uiLanguageSelect.addEventListener('change', (event) => {
      setUiLanguage(event.target.value);
    });
  }

  // Title Editing
  if (dom.sessionTitle) {
    dom.sessionTitle.addEventListener('blur', () => {
      const newTitle = dom.sessionTitle.textContent.trim();
      saveTitle(newTitle); // Imported from api.js
    });

    dom.sessionTitle.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        dom.sessionTitle.blur();
      }
    });

    dom.sessionTitle.addEventListener('focus', () => {
      // Optional
    });
  }
}

function handleOnboardingComplete() {
  localStorage.setItem('onboarding_complete', 'true');
  showMain();
}

async function handleDeleteAllHistory() {
  if (!confirm(t('history_delete_all_confirm'))) return;

  try {
    const res = await fetchWithAuth(`/api/conversations`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Delete all failed');
    if (state.activeConversationId) {
      resetSession();
    }
    window.location.reload();
  } catch (e) {
    if (state.reloginInProgress) return;
    alert(t('history_delete_all_failed'));
    console.error(e);
  }
}

init();
