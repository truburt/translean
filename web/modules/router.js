/*
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.
*/

import { state } from './state.js';
import { toggleMenu, showHistory, showAbout, showMain } from './ui.js';
import { loadConversationDetail } from './api.js';

export function getRouteFromLocation() {
    const params = new URLSearchParams(window.location.search);
    const viewParam = params.get('view');
    const view = viewParam === 'history' ? 'history' : viewParam === 'about' ? 'about' : 'main';
    return {
        view,
        conversationId: view === 'main' ? params.get('conversation_id') : null,
        menuOpen: params.get('menu') === 'open',
    };
}

export function buildUrlFromRoute(route) {
    const params = new URLSearchParams();
    if (route.view === 'history' || route.view === 'about') params.set('view', route.view);
    if (route.conversationId) params.set('conversation_id', route.conversationId);
    if (route.menuOpen) params.set('menu', 'open');

    const query = params.toString();
    return `${window.location.pathname}${query ? `?${query}` : ''}`;
}

export function deriveRouteFromState(overrides = {}) {
    const desiredView = overrides.view ?? state.activeView;
    const view = desiredView === 'history' ? 'history' : desiredView === 'about' ? 'about' : 'main';
    return {
        view,
        conversationId: view === 'main' ? (overrides.conversationId ?? state.activeConversationId) : null,
        menuOpen: overrides.menuOpen ?? state.menuOpen,
    };
}

export function syncUrlState(replace = false, overrides = {}) {
    const route = deriveRouteFromState(overrides);
    const url = buildUrlFromRoute(route);
    const method = replace ? 'replaceState' : 'pushState';
    window.history[method](route, '', url);
}

export async function applyRouteFromLocation(options = {}) {
    const { replaceHistory = false } = options;
    const route = getRouteFromLocation();

    if (route.menuOpen !== state.menuOpen) {
        toggleMenu(route.menuOpen, { fromPopstate: true, skipHistory: true });
    }

    if (!state.token) {
        return;
    }

    if (route.view === 'history') {
        showHistory({ skipRouting: true });
    } else if (route.view === 'about') {
        showAbout({ skipRouting: true });
    } else {
        showMain({ skipRouting: true });
    }

    if (route.view === 'main' && route.conversationId && route.conversationId !== state.activeConversationId) {
        state.activeConversationId = route.conversationId;
        state.hasSyncedConversationId = true;
        state.pendingConversationId = null;
        await loadConversationDetail(route.conversationId, {
            skipRouting: true,
            silentOnNotFound: replaceHistory
        });
    } else if (route.view === 'main' && route.conversationId) {
        state.hasSyncedConversationId = true;
        state.pendingConversationId = null;
    } else if (route.view === 'main' && !route.conversationId && state.activeConversationId) {
        state.activeConversationId = null;
        state.hasSyncedConversationId = false;
        state.pendingConversationId = null;
    } else if (route.view !== 'main') {
        state.activeConversationId = null;
        state.hasSyncedConversationId = false;
        state.pendingConversationId = null;
    }

    if (replaceHistory) {
        syncUrlState(true);
    }
}

export function handlePopState() {
    applyRouteFromLocation();
}
