/**
 * Siavox Application Logic Orchestrator
 *
 * Processing pipeline (streaming mode):
 *   1. User presses Record → WavEncoder starts capturing audio.
 *   2. Every 7 s (or on 1.2 s silence), WavEncoder fires onChunkReady with a WAV blob.
 *   3. Each chunk is queued and dispatched sequentially to POST /api/transcribe.
 *      Partial transcripts appear in the live-transcript panel as they arrive.
 *   4. User presses Stop → WavEncoder flushes the last chunk via the callback.
 *      The app waits for all in-flight transcriptions to finish.
 *   5. POST /api/refine is called with the full accumulated transcript.
 *      The result panel updates with the final refined output.
 */

document.addEventListener("DOMContentLoaded", () => {
    // ── Config and State ──────────────────────────────────────────────────────
    let currentUser = null;
    let selectedFormat = "note";
    let selectedLanguage = "auto";
    let isRecording = false;
    let recordingTimerInterval = null;
    let recordingSeconds = 0;

    // Streaming transcription state
    let liveTranscriptChunks = [];    // accumulated partial transcripts (one per chunk)
    let firstChunkAudioPath = null;   // audio_path from the first chunk (used for DB record)
    let chunkQueue = [];              // queue of WAV Blobs waiting to be sent
    let isChunkInFlight = false;      // true while a transcribe request is active
    let stopRequested = false;        // true once user pressed Stop (pending drain)
    let chunkFlushResolve = null;     // resolves when the chunk queue is fully drained

    // Wave Recorder instance
    const encoder = new WavEncoder();

    // ── DOM Elements ──────────────────────────────────────────────────────────
    const authScreen = document.getElementById("auth-screen");
    const mainDashboard = document.getElementById("main-dashboard");
    const mockLoginContainer = document.getElementById("mock-login-container");
    const mockLoginBtn = document.getElementById("mock-login-btn");

    // Header
    const userAvatarInitials = document.getElementById("user-avatar-initials");
    const logoutBtn = document.getElementById("logout-btn");

    // Recorder
    const recorderPanel = document.getElementById("recorder-panel");
    const recorderStatus = document.getElementById("recorder-status");
    const recordBtn = document.getElementById("record-btn");
    const micIcon = document.getElementById("mic-icon");
    const stopIcon = document.getElementById("stop-icon");
    const recordingTimer = document.getElementById("recording-timer");

    // Controls
    const formatNote = document.getElementById("format-note");
    const formatMessage = document.getElementById("format-message");
    const targetLanguageSelect = document.getElementById("target-language-select");
    const instructionsAccordionBtn = document.getElementById("instructions-accordion-btn");
    const instructionsAccordionContent = document.getElementById("instructions-accordion-content");
    const customInstructionsInput = document.getElementById("custom-instructions-input");

    // Loader and Result Cards
    const processingLoader = document.getElementById("processing-loader");
    const resultPanel = document.getElementById("result-panel");
    const tagFormat = document.getElementById("tag-format");
    const tagLang = document.getElementById("tag-lang");
    const copyResultBtn = document.getElementById("copy-result-btn");
    const transformedTextArea = document.getElementById("transformed-text-area");
    const transcriptAccordionBtn = document.getElementById("transcript-accordion-btn");
    const transcriptAccordionBody = document.getElementById("transcript-accordion-body");

    // Live transcript display (injected during recording; see ensureLiveTranscriptPanel)
    let liveTranscriptPanel = null;

    // History Bottom Drawer
    const historyDrawer = document.getElementById("history-drawer");
    const historyDrawerHandle = document.getElementById("history-drawer-handle");
    const historyOverlay = document.getElementById("history-overlay");
    const historyListContainer = document.getElementById("history-list-container");

    // Toast
    const toast = document.getElementById("toast");

    // Initialize application
    initApp();

    // ── Initialization ────────────────────────────────────────────────────────

    async function initApp() {
        // 1. Fetch public configurations
        try {
            const configRes = await fetch("/api/auth/config");
            const config = await configRes.json();

            // If the client ID is placeholder, enable the developer mock bypass
            if (config.google_client_id.startsWith("your-google-client-id")) {
                mockLoginContainer.classList.remove("hidden");
            }

            // Initialize Google Identity Services if client ID is valid
            if (config.google_client_id && !config.google_client_id.startsWith("your-google-client-id")) {
                google.accounts.id.initialize({
                    client_id: config.google_client_id,
                    callback: handleGoogleCredentialResponse
                });
                google.accounts.id.renderButton(
                    document.getElementById("google-signin-btn"),
                    { theme: "outline", size: "large", shape: "pill", width: "280" }
                );
            }
        } catch (e) {
            console.error("Config fetch failed, enabling fallback mock login options", e);
            mockLoginContainer.classList.remove("hidden");
        }

        // 2. Check current session status
        try {
            const sessionRes = await fetch("/api/auth/me");
            const session = await sessionRes.json();
            if (session.authenticated) {
                loginSuccess(session.user);
            } else {
                showAuthScreen();
            }
        } catch (e) {
            console.error("Session check failed", e);
            showAuthScreen();
        }
    }

    // ── Auth ──────────────────────────────────────────────────────────────────

    // Google JWT Credential callback
    async function handleGoogleCredentialResponse(response) {
        try {
            const authRes = await fetch("/api/auth/google", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ credential: response.credential })
            });
            const data = await authRes.json();
            if (data.success) {
                loginSuccess(data.user);
            } else {
                showToast("Auth failed: " + data.detail);
            }
        } catch (e) {
            console.error("Google auth request failed", e);
            showToast("Connection to server failed");
        }
    }

    // Login success state transition
    function loginSuccess(user) {
        currentUser = user;

        // Setup avatar initials
        if (user.name) {
            const initials = user.name.split(" ").map(n => n[0]).join("").substring(0, 2).toUpperCase();
            userAvatarInitials.textContent = initials || user.email[0].toUpperCase();
        } else {
            userAvatarInitials.textContent = user.email[0].toUpperCase();
        }

        authScreen.classList.add("hidden");
        mainDashboard.classList.remove("hidden");

        // Fetch User history
        fetchHistory();
    }

    function showAuthScreen() {
        currentUser = null;
        mainDashboard.classList.add("hidden");
        authScreen.classList.remove("hidden");
    }

    // Logout
    logoutBtn.addEventListener("click", async () => {
        try {
            await fetch("/api/auth/logout", { method: "POST" });
            showAuthScreen();
        } catch (e) {
            console.error("Logout request failed", e);
            showAuthScreen(); // fallback transition
        }
    });

    // Mock Login Trigger
    mockLoginBtn.addEventListener("click", async () => {
        const mockToken = "mock_token_sandbox_user";
        await handleGoogleCredentialResponse({ credential: mockToken });
    });

    // ── Format / Language Controls ────────────────────────────────────────────

    formatNote.addEventListener("click", () => selectFormatType("note"));
    formatMessage.addEventListener("click", () => selectFormatType("message"));

    function selectFormatType(format) {
        selectedFormat = format;
        if (format === "note") {
            formatNote.classList.add("active");
            formatMessage.classList.remove("active");
        } else {
            formatMessage.classList.add("active");
            formatNote.classList.remove("active");
        }
    }

    targetLanguageSelect.addEventListener("change", (e) => {
        selectedLanguage = e.target.value;
    });

    // Custom instructions accordion toggle
    instructionsAccordionBtn.addEventListener("click", () => {
        instructionsAccordionBtn.classList.toggle("active");
        instructionsAccordionContent.classList.toggle("expanded");
    });

    // ── Recording ─────────────────────────────────────────────────────────────

    recordBtn.addEventListener("click", toggleRecording);

    async function toggleRecording() {
        if (!isRecording) {
            await startRecording();
        } else {
            await stopRecording();
        }
    }

    async function startRecording() {
        try {
            // Reset pipeline state
            liveTranscriptChunks = [];
            firstChunkAudioPath = null;
            chunkQueue = [];
            isChunkInFlight = false;
            stopRequested = false;
            chunkFlushResolve = null;

            resultPanel.classList.add("hidden");
            removeLiveTranscriptPanel();

            // Register the streaming chunk callback before starting
            encoder.onChunkReady((blob) => onChunkReady(blob));

            // Start hardware stream recording
            await encoder.startRecording();

            isRecording = true;
            recorderPanel.classList.add("recording");
            recorderStatus.textContent = "Listening...";
            micIcon.classList.add("hidden");
            stopIcon.classList.remove("hidden");

            // Timer
            recordingSeconds = 0;
            recordingTimer.textContent = "00:00";
            recordingTimerInterval = setInterval(() => {
                recordingSeconds++;
                const minutes = Math.floor(recordingSeconds / 60).toString().padStart(2, '0');
                const seconds = (recordingSeconds % 60).toString().padStart(2, '0');
                recordingTimer.textContent = `${minutes}:${seconds}`;
            }, 1000);

        } catch (e) {
            console.error("Recording error", e);
            showToast("Could not access microphone");
        }
    }

    async function stopRecording() {
        // Stop the recording timer
        clearInterval(recordingTimerInterval);

        stopRequested = true;
        recorderStatus.textContent = "Finalizing...";

        // Stop hardware capture; last chunk fires via the onChunkReady callback
        await encoder.stopRecording();

        isRecording = false;
        recorderPanel.classList.remove("recording");
        micIcon.classList.remove("hidden");
        stopIcon.classList.add("hidden");
        recordingTimer.textContent = "00:00";

        // Wait for the chunk queue to drain (all in-flight transcriptions to complete)
        if (chunkQueue.length > 0 || isChunkInFlight) {
            recorderStatus.textContent = "Transcribing final chunks...";
            await new Promise((resolve) => {
                chunkFlushResolve = resolve;
            });
        }

        // All chunks transcribed — proceed to refinement
        const fullTranscript = liveTranscriptChunks.join(" ").trim();
        if (!fullTranscript) {
            recorderStatus.textContent = "Ready to Record";
            removeLiveTranscriptPanel();
            showToast("No speech detected");
            return;
        }

        await runRefinement(fullTranscript);
    }

    // ── Chunk Streaming ───────────────────────────────────────────────────────

    /**
     * Called by WavEncoder each time a chunk is ready.
     * Pushes the blob into the queue and starts processing if idle.
     */
    function onChunkReady(blob) {
        chunkQueue.push(blob);
        if (!isChunkInFlight) {
            processNextChunk();
        }
    }

    /**
     * Dequeues the next chunk blob and transcribes it.
     * Chunks are processed strictly sequentially to avoid GPU contention.
     */
    async function processNextChunk() {
        if (chunkQueue.length === 0) {
            isChunkInFlight = false;
            // If stop was requested and the queue is now empty, signal drain complete
            if (stopRequested && chunkFlushResolve) {
                chunkFlushResolve();
                chunkFlushResolve = null;
            }
            return;
        }

        isChunkInFlight = true;
        const blob = chunkQueue.shift();
        const chunkIndex = liveTranscriptChunks.length + 1;

        // Update status
        recorderStatus.textContent = `Transcribing chunk ${chunkIndex}...`;

        // Show the live transcript panel if this is the first chunk
        ensureLiveTranscriptPanel();

        // Show loading indicator for this chunk in the live transcript panel
        const loadingDot = appendLiveTranscriptLoading(chunkIndex);

        try {
            const formData = new FormData();
            formData.append("file", blob, `chunk_${chunkIndex}.wav`);

            const res = await fetch("/api/transcribe", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Transcription failed");
            }

            const data = await res.json();
            const partialText = data.raw_transcript || "";

            // Save audio_path from the first chunk for the DB record
            if (chunkIndex === 1 && data.audio_path) {
                firstChunkAudioPath = data.audio_path;
            }

            // Replace loading dot with actual text
            replaceLiveTranscriptLoading(loadingDot, partialText);

            if (partialText) {
                liveTranscriptChunks.push(partialText);
            }

        } catch (e) {
            console.error(`Chunk ${chunkIndex} transcription failed:`, e);
            replaceLiveTranscriptLoading(loadingDot, null); // mark as failed
        }

        // Resume with the next chunk in queue
        if (!isRecording) {
            // We're draining after stop — update status
            if (chunkQueue.length > 0) {
                recorderStatus.textContent = `Transcribing final chunks... (${chunkQueue.length} remaining)`;
            }
        } else {
            recorderStatus.textContent = "Listening...";
        }

        // Recurse to process next
        await processNextChunk();
    }

    // ── Refinement ────────────────────────────────────────────────────────────

    /**
     * Sends the full accumulated transcript to /api/refine and displays the result.
     *
     * @param {string} fullTranscript - All chunk transcripts joined with spaces.
     */
    async function runRefinement(fullTranscript) {
        const loadingText = document.getElementById("loading-text");
        const loadingWarning = document.getElementById("loading-warning");

        recorderStatus.textContent = "Refining transcript...";

        // Check if model is loaded to show/hide the warm-up warning
        if (loadingWarning) {
            try {
                const healthRes = await fetch("/api/health");
                if (healthRes.ok) {
                    const healthData = await healthRes.json();
                    if (healthData.ollama_connected && !healthData.ollama_model_loaded) {
                        loadingWarning.classList.remove("hidden");
                    } else {
                        loadingWarning.classList.add("hidden");
                    }
                } else {
                    loadingWarning.classList.add("hidden");
                }
            } catch (e) {
                console.error("Health check failed before refinement", e);
                loadingWarning.classList.add("hidden");
            }
        }

        if (loadingText) {
            loadingText.textContent = "Refining and formatting full transcript...";
        }
        processingLoader.classList.remove("hidden");

        // Pre-populate the result panel with the raw transcript before refinement completes
        transcriptAccordionBody.textContent = fullTranscript;
        transformedTextArea.textContent = "Refining transcription...";
        transformedTextArea.classList.add("loading-placeholder");
        tagFormat.textContent = selectedFormat === "note" ? "Structured Note" : "Refined Message";
        tagLang.textContent = "DETECTING...";
        resultPanel.classList.remove("hidden");
        resultPanel.scrollIntoView({ behavior: 'smooth' });

        // Send refinement request
        const audioPath = firstChunkAudioPath || "";
        const refineForm = new FormData();
        refineForm.append("raw_transcript", fullTranscript);
        refineForm.append("audio_path", audioPath);
        refineForm.append("format", selectedFormat);
        refineForm.append("target_language", selectedLanguage);
        if (customInstructionsInput.value.trim()) {
            refineForm.append("custom_instructions", customInstructionsInput.value.trim());
        }

        try {
            const refineRes = await fetch("/api/refine", {
                method: "POST",
                body: refineForm
            });

            if (!refineRes.ok) {
                const errData = await refineRes.json();
                throw new Error(errData.detail || "Refinement failed");
            }

            const refineData = await refineRes.json();
            transformedTextArea.classList.remove("loading-placeholder");
            displayResult(refineData);
            fetchHistory(); // refresh logs drawer
        } catch (e) {
            console.error("Refinement error", e);
            showToast(e.message || "Failed to refine speech input");
            transformedTextArea.textContent = "Refinement failed. Raw transcription is available below.";
            transformedTextArea.classList.remove("loading-placeholder");
        } finally {
            processingLoader.classList.add("hidden");
            recorderStatus.textContent = "Ready to Record";
            removeLiveTranscriptPanel();
        }
    }

    // ── Live Transcript Panel ─────────────────────────────────────────────────

    /**
     * Create and insert the live-transcript panel into the DOM (once, above the loader).
     */
    function ensureLiveTranscriptPanel() {
        if (liveTranscriptPanel) return;

        liveTranscriptPanel = document.createElement("div");
        liveTranscriptPanel.id = "live-transcript-panel";
        liveTranscriptPanel.className = "glass-panel live-transcript-panel";
        liveTranscriptPanel.innerHTML = `
            <div class="live-transcript-header">
                <span class="live-badge">
                    <span class="live-dot"></span>LIVE
                </span>
                <span class="live-transcript-label">Speech Transcript</span>
            </div>
            <div class="live-transcript-body" id="live-transcript-body"></div>
        `;

        // Insert before the processing loader
        processingLoader.parentNode.insertBefore(liveTranscriptPanel, processingLoader);
    }

    function removeLiveTranscriptPanel() {
        if (liveTranscriptPanel) {
            liveTranscriptPanel.remove();
            liveTranscriptPanel = null;
        }
    }

    /**
     * Append a loading placeholder for a chunk being transcribed.
     * Returns the placeholder element so it can be replaced later.
     */
    function appendLiveTranscriptLoading(chunkIndex) {
        const body = document.getElementById("live-transcript-body");
        if (!body) return null;

        const placeholder = document.createElement("span");
        placeholder.className = "live-chunk-loading";
        placeholder.dataset.chunkIndex = chunkIndex;
        placeholder.setAttribute("aria-label", `Transcribing chunk ${chunkIndex}`);
        // Three bouncing dots
        placeholder.innerHTML = `<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>`;

        body.appendChild(placeholder);
        body.scrollTop = body.scrollHeight;
        return placeholder;
    }

    /**
     * Replace a loading placeholder with the transcribed text (or error indicator).
     *
     * @param {Element|null} placeholder - The element returned by appendLiveTranscriptLoading.
     * @param {string|null} text - Transcribed text, or null to show an error.
     */
    function replaceLiveTranscriptLoading(placeholder, text) {
        if (!placeholder) return;

        if (text === null || text === "") {
            // Failed or empty — remove silently
            placeholder.remove();
            return;
        }

        const span = document.createElement("span");
        span.className = "live-chunk-text";
        // Separate chunks with a space when appending
        if (placeholder.previousSibling) {
            span.textContent = " " + text;
        } else {
            span.textContent = text;
        }

        placeholder.replaceWith(span);

        const body = document.getElementById("live-transcript-body");
        if (body) body.scrollTop = body.scrollHeight;
    }

    // ── Result Display ────────────────────────────────────────────────────────

    // Render results in card
    function displayResult(data) {
        transformedTextArea.textContent = data.transformed_text;

        // Setup tags
        tagFormat.textContent = data.format === "note" ? "Structured Note" : "Refined Message";
        tagLang.textContent = (data.target_language || data.source_language || "auto").toUpperCase();

        // Raw transcription
        transcriptAccordionBody.textContent = data.raw_transcript || "Verbatim transcription unavailable.";
        // Reset raw transcript collapse state
        transcriptAccordionBtn.classList.remove("active");
        transcriptAccordionBody.classList.remove("expanded");

        resultPanel.classList.remove("hidden");

        // Auto scroll to result panel for better mobile experience
        resultPanel.scrollIntoView({ behavior: 'smooth' });
    }

    // Verbatim transcript accordion toggle
    transcriptAccordionBtn.addEventListener("click", () => {
        transcriptAccordionBtn.classList.toggle("active");
        transcriptAccordionBody.classList.toggle("expanded");
    });

    // Copy result text to clipboard
    copyResultBtn.addEventListener("click", () => {
        const textToCopy = transformedTextArea.textContent;
        if (!textToCopy) return;

        navigator.clipboard.writeText(textToCopy).then(() => {
            copyResultBtn.classList.add("copied");
            showToast("Copied to clipboard!");
            setTimeout(() => {
                copyResultBtn.classList.remove("copied");
            }, 2000);
        }).catch(err => {
            console.error("Clipboard copy failed", err);
            showToast("Failed to copy text");
        });
    });

    // ── History Drawer ────────────────────────────────────────────────────────

    // Fetch historical records
    async function fetchHistory() {
        try {
            const res = await fetch("/api/history");
            const history = await res.json();
            renderHistory(history);
        } catch (e) {
            console.error("Failed to load interaction logs history", e);
        }
    }

    function renderHistory(history) {
        historyListContainer.innerHTML = "";

        if (history.length === 0) {
            historyListContainer.innerHTML = '<div class="empty-history-text">Speak to record your first interaction</div>';
            return;
        }

        history.forEach(item => {
            const date = new Date(item.created_at);
            const dateString = date.toLocaleDateString(undefined, {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });

            const card = document.createElement("div");
            card.className = "history-card";

            const header = document.createElement("div");
            header.className = "history-card-header";

            const metaTag = document.createElement("span");
            metaTag.className = "meta-tag";
            metaTag.style.fontSize = "0.65rem";
            metaTag.style.padding = "0.15rem 0.4rem";

            const formatLabel = item.format === 'note' ? 'Note' : 'Message';
            const langLabel = (item.target_language || 'auto').substring(0, 2).toUpperCase();
            metaTag.textContent = `${formatLabel} (${langLabel})`;

            const dateSpan = document.createElement("span");
            dateSpan.className = "history-card-date";
            dateSpan.textContent = dateString;

            header.appendChild(metaTag);
            header.appendChild(dateSpan);

            const preview = document.createElement("div");
            preview.className = "history-card-preview";
            preview.textContent = item.transformed_text;

            card.appendChild(header);
            card.appendChild(preview);

            card.addEventListener("click", () => {
                // Populate result panel and close drawer
                displayResult(item);
                closeHistoryDrawer();
            });

            historyListContainer.appendChild(card);
        });
    }

    // History Bottom Drawer Actions
    historyDrawerHandle.addEventListener("click", toggleHistoryDrawer);
    historyOverlay.addEventListener("click", closeHistoryDrawer);

    function toggleHistoryDrawer() {
        historyDrawer.classList.toggle("open");
        historyOverlay.classList.toggle("active");
    }

    function closeHistoryDrawer() {
        historyDrawer.classList.remove("open");
        historyOverlay.classList.remove("active");
    }

    // ── Toast ─────────────────────────────────────────────────────────────────

    // Display feedback toast
    function showToast(message) {
        toast.textContent = message;
        toast.classList.add("show");
        setTimeout(() => {
            toast.classList.remove("show");
        }, 2500);
    }
});
