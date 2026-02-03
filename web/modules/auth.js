/*
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.
*/

import { state } from './state.js';

import { t } from './utils.js';
import { updateMenuUserName, showLogin, showMain, showOnboarding, updateAdminMenuVisibility } from './ui.js';
import { syncUrlState } from './router.js';

export function loadTokens() {
    state.token = localStorage.getItem('access_token');
    state.userName = localStorage.getItem('user_name') || '';
    updateMenuUserName();
    if (state.token) {
        if (localStorage.getItem('onboarding_complete')) {
            showMain({ skipRouting: true });
        } else {
            showOnboarding();
        }
    } else {
        showLogin();
    }
}

export async function handleLogin() {
    // Clear existing error
    const existingError = document.getElementById('login-error-msg');
    if (existingError) existingError.remove();

    const btn = document.getElementById('login-start-button');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Connecting...`;
    }

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 2000); // 2s timeout

        // We use a simple fetch to check if backend is reachable.
        // /auth/login is the endpoint, we can HEAD or GET it. 
        // Since it's likely a redirect or HTML page, fetch might return 200 or 401 or similar.
        // We just care if it fails with network error.
        await fetch('/auth/login', {
            method: 'HEAD',
            signal: controller.signal,
            cache: 'no-store'
        });
        clearTimeout(timeoutId);
        window.location.href = '/auth/login';
    } catch (e) {
        console.error("Login check failed:", e);
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = `<span data-i18n="login_sign_in">${t('login_sign_in')}</span>`;

            // Show error below button
            const errorDiv = document.createElement('div');
            errorDiv.id = 'login-error-msg';
            errorDiv.className = 'mt-4 p-3 bg-red-100 text-red-700 rounded-lg text-sm dark:bg-red-900/30 dark:text-red-400 fade-in';
            errorDiv.textContent = t('error_backend_offline') || "Backend server is unreachable. Please make sure it is running.";
            btn.parentNode.insertBefore(errorDiv, btn.nextSibling);
        }
    }
}

export function handleLogout() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user_name');
    state.token = null;
    state.userName = '';
    state.isAdmin = false;
    state.activeConversationId = null;
    state.hasSyncedConversationId = false;
    state.pendingConversationId = null;
    updateMenuUserName();
    updateAdminMenuVisibility();
    loadTokens();
    syncUrlState(true);
}

export function triggerRelogin(reason = 'Session expired') {
    if (state.reloginInProgress) return;
    state.reloginInProgress = true;
    console.warn(`${reason}. Redirecting to login.`);
    if (state.socket && state.socket.readyState !== WebSocket.CLOSED) {
        state.socket.close();
    }
    localStorage.removeItem('access_token');
    localStorage.removeItem('user_name');
    state.isAdmin = false;
    updateAdminMenuVisibility();

    // Ideally we should just redirect
    window.location.href = '/auth/login';
}
