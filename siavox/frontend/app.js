/**
 * Siavox Application Logic Orchestrator
 */

document.addEventListener("DOMContentLoaded", () => {
    // Config and State Variables
    let currentUser = null;
    let selectedFormat = "note";
    let selectedLanguage = "auto";
    let isRecording = false;
    let recordingTimerInterval = null;
    let recordingSeconds = 0;
    
    // Wave Recorder instance
    const encoder = new WavEncoder();
    
    // DOM Elements
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
    
    // History Bottom Drawer
    const historyDrawer = document.getElementById("history-drawer");
    const historyDrawerHandle = document.getElementById("history-drawer-handle");
    const historyOverlay = document.getElementById("history-overlay");
    const historyListContainer = document.getElementById("history-list-container");
    
    // Toast
    const toast = document.getElementById("toast");

    // Initialize application
    initApp();

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

    // Formatting format selector
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

    // Recording action listeners
    recordBtn.addEventListener("click", toggleRecording);

    async function toggleRecording() {
        if (!isRecording) {
            try {
                // Reset states
                resultPanel.classList.add("hidden");
                
                // Start hardware stream recording
                await encoder.startRecording();
                
                isRecording = true;
                recorderPanel.classList.add("recording");
                recorderStatus.textContent = "Recording Audio...";
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
        } else {
            // Stop recording
            clearInterval(recordingTimerInterval);
            recorderStatus.textContent = "Finalizing Wave Record...";
            
            const audioBlob = await encoder.stopRecording();
            
            isRecording = false;
            recorderPanel.classList.remove("recording");
            micIcon.classList.remove("hidden");
            stopIcon.classList.add("hidden");
            recordingTimer.textContent = "00:00";
            
            if (audioBlob) {
                processAudioPayload(audioBlob);
            } else {
                recorderStatus.textContent = "Ready to Record";
                showToast("Failed to record audio");
            }
        }
    }

    // Audio transmission pipeline
    async function processAudioPayload(audioBlob) {
        recorderStatus.textContent = "Processing audio...";
        processingLoader.classList.remove("hidden");
        resultPanel.classList.add("hidden");
        
        const formData = new FormData();
        formData.append("file", audioBlob, "microphone_clip.wav");
        formData.append("format", selectedFormat);
        formData.append("target_language", selectedLanguage);
        if (customInstructionsInput.value.trim()) {
            formData.append("custom_instructions", customInstructionsInput.value.trim());
        }

        try {
            const res = await fetch("/api/process", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Server processing failed");
            }

            const data = await res.json();
            displayResult(data);
            fetchHistory(); // refresh logs drawer
        } catch (e) {
            console.error("Processing audio error", e);
            showToast(e.message || "Failed to analyze speech input");
        } finally {
            processingLoader.classList.add("hidden");
            recorderStatus.textContent = "Ready to Record";
        }
    }

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

    // Display feedback toast
    function showToast(message) {
        toast.textContent = message;
        toast.classList.add("show");
        setTimeout(() => {
            toast.classList.remove("show");
        }, 2500);
    }
});
