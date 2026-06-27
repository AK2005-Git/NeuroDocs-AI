// =====================================================
// ADVANCED RAG FRONTEND
// =====================================================

const API_URL = "http://127.0.0.1:8000";

// =====================================================
// ELEMENTS
// =====================================================

const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const messages = document.getElementById("messages");

const uploadBtn = document.getElementById("uploadBtn");
const pdfFile = document.getElementById("pdfFile");

const statusBadge = document.getElementById("api-status");

const liveTotalQueries = document.getElementById("liveTotalQueries");
const liveAvgTime = document.getElementById("liveAvgTime");
const liveDocCount = document.getElementById("liveDocCount");
const liveLastConfidence = document.getElementById("liveLastConfidence");
const docLibraryList = document.getElementById("docLibraryList");

const themeToggle = document.getElementById("themeToggle");
const exportChatBtn = document.getElementById("exportChatBtn");
const clearChatBtn = document.getElementById("clearChatBtn");

const runEvalBtn = document.getElementById("runEvalBtn");
const evalResults = document.getElementById("evalResults");

// Running stats kept client-side (resets on page reload —
// only the chat HISTORY itself persists, not these counters)
const sessionStats = {
    totalQueries: 0,
    totalTime: 0,
    documents: []   // {name, size}
};

// =====================================================
// PERSISTENT CHAT HISTORY (localStorage)
// =====================================================
// Stores: [{ sender: "user"|"bot", text, sources, retrieval_time,
//            generation_time, response_time }]
// Stored as plain text/JSON — safe for localStorage (this is a
// real deployed app, not a sandboxed artifact).

const HISTORY_KEY = "neurodocs_chat_history";
const THEME_KEY = "neurodocs_theme";

function loadHistory() {

    try {

        const raw = localStorage.getItem(HISTORY_KEY);

        return raw ? JSON.parse(raw) : [];

    } catch (e) {

        console.error("Failed to load chat history:", e);

        return [];
    }
}

function saveHistory(history) {

    try {

        localStorage.setItem(
            HISTORY_KEY,
            JSON.stringify(history)
        );

    } catch (e) {

        console.error("Failed to save chat history:", e);
    }
}

function appendToHistory(entry) {

    const history = loadHistory();

    history.push(entry);

    saveHistory(history);
}

// =====================================================
// API STATUS CHECK
// =====================================================

async function checkAPI() {

try {

    const response =
        await fetch(
            `${API_URL}/health`
        );

    if (!response.ok)
        throw new Error();

    statusBadge.innerHTML =
        "🟢 Online";

} catch (error) {

    statusBadge.innerHTML =
        "🔴 Offline";
}


}

checkAPI();

setInterval(
checkAPI,
15000
);

loadDocumentLibrary();

// =====================================================
// ADD MESSAGE
// =====================================================

function addMessage(
text,
sender = "bot"
) {


const div =
    document.createElement("div");

div.className =
    sender === "user"
        ? "user-message"
        : "bot-message";

div.innerHTML = text;

messages.appendChild(div);

scrollMessages();

return div;


}

// =====================================================
// SCROLL
// =====================================================

function scrollMessages() {


messages.scrollTop =
    messages.scrollHeight;


}

// =====================================================
// THINKING ANIMATION
// =====================================================

function addThinking() {


const div =
    document.createElement("div");

div.className =
    "bot-message";

div.id = "thinking";

div.innerHTML =
    "🤖 Thinking...";

messages.appendChild(div);

scrollMessages();


}

function removeThinking() {


const thinking =
    document.getElementById(
        "thinking"
    );

if (thinking)
    thinking.remove();


}

// =====================================================
// TYPEWRITER EFFECT
// =====================================================

function typeWriter(
element,
text,
speed = 10,
onComplete = null
) {


let index = 0;

element.innerHTML = "";

function type() {

    if (
        index < text.length
    ) {

        element.innerHTML +=
            text.charAt(index);

        index++;

        scrollMessages();

        setTimeout(
            type,
            speed
        );

    } else if (onComplete) {

        onComplete();
    }
}

type();


}

// =====================================================
// SEND QUESTION
// =====================================================

async function sendQuestion() {


const question =
    questionInput.value.trim();

if (!question)
    return;

addMessage(
    question,
    "user"
);

appendToHistory({
    sender: "user",
    text: question
});

questionInput.value = "";

addThinking();

try {

    const response =
        await fetch(
            `${API_URL}/chat`,
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body:
                    JSON.stringify({
                        question
                    })
            }
        );

    const data =
        await response.json();

    removeThinking();

    // Query rewrite indicator — show what was actually searched
    if (data.was_rewritten && data.search_query) {

        renderRewriteNotice(data.search_query);
    }

    const botDiv =
        addMessage(
            "",
            "bot"
        );

    typeWriter(
        botDiv,
        data.answer,
        10,
        () => {

            linkifyCitations(
                botDiv,
                data.sources || []
            );

            // Confidence-aware routing banner
            // (shown for low_confidence / no_results / unsupported)
            renderRoutingBanner(data);

            // Retrieval panel (chunks + scores + timing)

            if (
                data.sources &&
                data.sources.length > 0
            ) {

                renderRetrievalPanel(data);

                updateLiveStats(data);
            }

            // Persist the final rendered answer (with citation
            // badges already applied) plus all metadata needed
            // to rebuild the retrieval panel on next page load.
            appendToHistory({
                sender: "bot",
                text: botDiv.innerHTML,
                sources: data.sources || [],
                retrieval_time: data.retrieval_time,
                generation_time: data.generation_time,
                response_time: data.response_time,
                routing: data.routing,
                suggestions: data.suggestions,
                was_rewritten: data.was_rewritten,
                search_query: data.search_query
            });
        }
    );

} catch (error) {

    removeThinking();

    addMessage(
        "❌ Failed to contact API server.",
        "bot"
    );

    console.error(error);
}


}

// =====================================================
// REPLAY SAVED HISTORY ON PAGE LOAD
// =====================================================

function replayHistory() {

    const history = loadHistory();

    if (history.length === 0)
        return;

    history.forEach(entry => {

        if (entry.sender === "user") {

            addMessage(entry.text, "user");

        } else {

            if (entry.was_rewritten && entry.search_query) {

                renderRewriteNotice(entry.search_query);
            }

            const botDiv = addMessage(entry.text, "bot");

            if (entry.routing && entry.routing !== "normal") {

                renderRoutingBanner({
                    routing: entry.routing,
                    suggestions: entry.suggestions || []
                });
            }

            if (entry.sources && entry.sources.length > 0) {

                renderRetrievalPanel({
                    sources: entry.sources,
                    retrieval_time: entry.retrieval_time,
                    generation_time: entry.generation_time,
                    response_time: entry.response_time
                });

                // Re-wire citation badge clicks (innerHTML was
                // saved as static markup, so the click listeners
                // from the original session are gone — rebuild
                // them against the freshly rendered chunk cards)
                botDiv.querySelectorAll(".citation-badge").forEach(badge => {

                    badge.addEventListener("click", () => {

                        const rank = badge.dataset.rank;

                        const targetCard = messages.querySelector(
                            `.chunk-card[data-rank="${rank}"]`
                        );

                        if (targetCard) {

                            targetCard.scrollIntoView({
                                behavior: "smooth",
                                block: "center"
                            });

                            targetCard.classList.add("chunk-highlight");

                            setTimeout(() => {
                                targetCard.classList.remove("chunk-highlight");
                            }, 1500);
                        }
                    });
                });
            }
        }
    });

    scrollMessages();
}

// =====================================================
// CLEAR / EXPORT CHAT
// =====================================================

function clearChat() {

    const confirmed = confirm(
        "Clear the entire conversation? This cannot be undone."
    );

    if (!confirmed)
        return;

    localStorage.removeItem(HISTORY_KEY);

    messages.innerHTML = `
        <div class="bot-message">
            👋 Welcome to NeuroDocs Assistant
        </div>
    `;
}

function exportChat() {

    const history = loadHistory();

    if (history.length === 0) {

        alert("No conversation to export yet.");

        return;
    }

    let markdown = "# NeuroDocs AI — Conversation Export\n\n";

    markdown += `Exported: ${new Date().toLocaleString()}\n\n---\n\n`;

    history.forEach(entry => {

        if (entry.sender === "user") {

            markdown += `### 🧑 You\n\n${entry.text}\n\n`;

        } else {

            // Strip HTML tags (citation badges etc) for clean text export
            const plainText = entry.text.replace(/<[^>]*>/g, "");

            markdown += `### 🤖 Assistant\n\n${plainText}\n\n`;

            if (entry.sources && entry.sources.length > 0) {

                markdown += `**Sources:**\n\n`;

                entry.sources.forEach(s => {

                    markdown += `- [${s.rank}] ${s.source_file} (chunk ${s.chunk_id}, score ${s.score}, ${s.confidence} confidence)\n`;
                });

                markdown += "\n";
            }

            if (entry.response_time) {

                markdown += `_Response time: ${entry.response_time}s_\n\n`;
            }
        }

        markdown += "---\n\n";
    });

    const blob = new Blob([markdown], { type: "text/markdown" });

    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");

    a.href = url;

    a.download = `neurodocs-chat-${Date.now()}.md`;

    document.body.appendChild(a);

    a.click();

    document.body.removeChild(a);

    URL.revokeObjectURL(url);
}

if (clearChatBtn) {

    clearChatBtn.addEventListener("click", clearChat);
}

if (exportChatBtn) {

    exportChatBtn.addEventListener("click", exportChat);
}

// Replay history once the DOM is ready
replayHistory();

// =====================================================
// THEME TOGGLE (light / dark)
// =====================================================

function applyTheme(theme) {

    if (theme === "light") {

        document.body.classList.add("light-theme");

        if (themeToggle) {
            themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
        }

    } else {

        document.body.classList.remove("light-theme");

        if (themeToggle) {
            themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
        }
    }
}

function initTheme() {

    const saved = localStorage.getItem(THEME_KEY) || "dark";

    applyTheme(saved);
}

function toggleTheme() {

    const isLight = document.body.classList.contains("light-theme");

    const newTheme = isLight ? "dark" : "light";

    localStorage.setItem(THEME_KEY, newTheme);

    applyTheme(newTheme);
}

if (themeToggle) {

    themeToggle.addEventListener("click", toggleTheme);
}

initTheme();

// =====================================================
// EVALUATION DASHBOARD
// =====================================================

function renderEvalResults(data) {

    if (!evalResults)
        return;

    if (data.status === "no_results") {

        evalResults.innerHTML = `
            <div class="eval-empty">
                ${escapeHTML(data.message)}
            </div>
        `;

        return;
    }

    if (data.status === "error") {

        evalResults.innerHTML = `
            <div class="eval-empty">
                ❌ ${escapeHTML(data.message || "Evaluation failed.")}
            </div>
        `;

        return;
    }

    const summary = data.summary;

    let html = `
        <div class="eval-summary-grid">

            <div class="eval-metric">
                <span class="eval-metric-value">${(summary.recall_at_k * 100).toFixed(0)}%</span>
                <span class="eval-metric-label">Recall@${summary.top_k}</span>
            </div>

            <div class="eval-metric">
                <span class="eval-metric-value">${summary.mrr.toFixed(3)}</span>
                <span class="eval-metric-label">MRR</span>
            </div>

            <div class="eval-metric">
                <span class="eval-metric-value">${summary.hits}/${summary.total_questions}</span>
                <span class="eval-metric-label">Hits</span>
            </div>

        </div>

        <div class="eval-question-list">
    `;

    (data.results || []).forEach(r => {

        html += `
            <div class="eval-question-row ${r.hit ? 'eval-hit' : 'eval-miss'}">

                <div class="eval-question-text">
                    <i class="fas ${r.hit ? 'fa-check' : 'fa-xmark'}"></i>
                    ${escapeHTML(r.question)}
                </div>

                <div class="eval-question-meta">
                    Expected: ${escapeHTML((r.expected_sources || []).join(", "))}
                    ${r.hit ? `· Found at rank ${r.rank_of_first_hit}` : "· Not found in results"}
                </div>

            </div>
        `;
    });

    html += `</div>`;

    evalResults.innerHTML = html;
}

async function loadLastEvalResults() {

    if (!evalResults)
        return;

    try {

        const response = await fetch(`${API_URL}/evaluation/results`);

        const data = await response.json();

        renderEvalResults(data);

    } catch (error) {

        console.error("Failed to load evaluation results:", error);
    }
}

async function runEvaluation() {

    if (!runEvalBtn || !evalResults)
        return;

    runEvalBtn.disabled = true;

    runEvalBtn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Running...`;

    evalResults.innerHTML = `
        <div class="eval-empty">
            Running evaluation against labeled test questions...
        </div>
    `;

    try {

        const response = await fetch(`${API_URL}/evaluation/run`, {
            method: "POST"
        });

        const data = await response.json();

        renderEvalResults(data);

    } catch (error) {

        console.error("Evaluation run failed:", error);

        evalResults.innerHTML = `
            <div class="eval-empty">
                ❌ Failed to run evaluation. Check the server console.
            </div>
        `;

    } finally {

        runEvalBtn.disabled = false;

        runEvalBtn.innerHTML = `<i class="fas fa-flask"></i> Run Evaluation`;
    }
}

if (runEvalBtn) {

    runEvalBtn.addEventListener("click", runEvaluation);
}

loadLastEvalResults();

function updateLiveStats(data) {

    sessionStats.totalQueries += 1;

    sessionStats.totalTime +=
        data.response_time || 0;

    const avg =
        sessionStats.totalTime /
        sessionStats.totalQueries;

    if (liveTotalQueries) {
        liveTotalQueries.innerText =
            sessionStats.totalQueries;
    }

    if (liveAvgTime) {
        liveAvgTime.innerText =
            avg.toFixed(1) + "s";
    }

    if (liveLastConfidence && data.sources?.length) {

        const conf = data.sources[0].confidence;

        liveLastConfidence.innerText = conf;

        liveLastConfidence.className = "";

        liveLastConfidence.style.color =
            conf === "High" ? "#62ff9f" :
            conf === "Medium" ? "#ffc857" :
            "#ff6363";
    }
}

// =====================================================
// CITATIONS — turn [1] [2] into clickable badges
// =====================================================

function linkifyCitations(element, sources) {

    if (!sources || sources.length === 0)
        return;

    const validRanks = new Set(
        sources.map(s => String(s.rank))
    );

    // Matches [1], [2], [12] etc — only numbers, so it won't
    // touch normal bracket text like "[note]"
    const citationPattern = /\[(\d+)\]/g;

    element.innerHTML = element.innerHTML.replace(
        citationPattern,
        (match, num) => {

            if (!validRanks.has(num)) {
                // Not a real source number — leave the text as-is
                return match;
            }

            return `<button type="button" class="citation-badge" data-rank="${num}">${num}</button>`;
        }
    );

    // Wire up click → scroll to matching chunk card in the panel
    // that will be appended right after this (panel is appended
    // to the same `messages` container as a sibling)
    element.querySelectorAll(".citation-badge").forEach(badge => {

        badge.addEventListener("click", () => {

            const rank = badge.dataset.rank;

            const targetCard = messages.querySelector(
                `.chunk-card[data-rank="${rank}"]`
            );

            if (targetCard) {

                targetCard.scrollIntoView({
                    behavior: "smooth",
                    block: "center"
                });

                targetCard.classList.add("chunk-highlight");

                setTimeout(() => {
                    targetCard.classList.remove("chunk-highlight");
                }, 1500);
            }
        });
    });
}

// =====================================================
// QUERY REWRITE NOTICE
// =====================================================

function renderRewriteNotice(rewrittenQuery) {

    const notice = document.createElement("div");

    notice.className = "rewrite-notice";

    notice.innerHTML = `
        <i class="fas fa-wand-magic-sparkles"></i>
        Searched as: <em>"${escapeHTML(rewrittenQuery)}"</em>
    `;

    messages.appendChild(notice);

    scrollMessages();
}

// =====================================================
// CONFIDENCE-AWARE ROUTING BANNER
// =====================================================

function renderRoutingBanner(data) {

    if (!data.routing || data.routing === "normal")
        return;

    const banner = document.createElement("div");

    banner.className = "routing-banner routing-" + data.routing;

    let icon = "fa-circle-info";
    let title = "Heads up";

    if (data.routing === "low_confidence") {

        icon = "fa-triangle-exclamation";

        title = "Low confidence answer";

    } else if (data.routing === "no_results") {

        icon = "fa-magnifying-glass";

        title = "Nothing relevant found";

    } else if (data.routing === "unsupported") {

        icon = "fa-flask-vial";

        title = "Possibly unsupported claim";

    } else if (data.routing === "error") {

        icon = "fa-circle-exclamation";

        title = "Something went wrong";
    }

    let html = `
        <div class="routing-banner-title">
            <i class="fas ${icon}"></i> ${title}
        </div>
    `;

    if (data.suggestions && data.suggestions.length > 0) {

        html += `<ul class="routing-suggestions">`;

        data.suggestions.forEach(s => {

            html += `<li>${escapeHTML(s)}</li>`;
        });

        html += `</ul>`;
    }

    banner.innerHTML = html;

    messages.appendChild(banner);

    scrollMessages();
}

// =====================================================
// RETRIEVAL PANEL (chunks, scores, confidence, timing)
// =====================================================

function confidenceClass(label) {

    if (label === "High")
        return "conf-high";

    if (label === "Medium")
        return "conf-medium";

    return "conf-low";
}

function escapeHTML(text) {

    const div = document.createElement("div");

    div.innerText = text;

    return div.innerHTML;
}

function renderRetrievalPanel(data) {

    const panel =
        document.createElement("div");

    panel.className =
        "retrieval-panel";

    const topConfidence =
        data.sources[0].confidence;

    let html = `
        <div class="retrieval-header">
            <span><i class="fas fa-layer-group"></i> Retrieved Chunks (${data.sources.length})</span>
            <span class="conf-badge ${confidenceClass(topConfidence)}">
                ${topConfidence} Confidence
            </span>
        </div>

        <div class="timing-row">
            <span><i class="fas fa-magnifying-glass"></i> Retrieval: ${data.retrieval_time ?? "—"}s</span>
            <span><i class="fas fa-robot"></i> Generation: ${data.generation_time ?? "—"}s</span>
            <span><i class="fas fa-stopwatch"></i> Total: ${data.response_time ?? "—"}s</span>
        </div>

        <div class="chunk-list">
    `;

    data.sources.forEach(source => {

        html += `
            <div class="chunk-card" data-rank="${source.rank}">

                <div class="chunk-card-top">
                    <span class="chunk-rank">#${source.rank}</span>
                    <span class="chunk-file">
                        <i class="fas fa-file-pdf"></i>
                        ${escapeHTML(source.source_file)}
                        <span class="chunk-id">· chunk ${source.chunk_id}</span>
                    </span>
                    <span class="chunk-score ${confidenceClass(source.confidence)}">
                        ${source.score}
                    </span>
                </div>

                <div class="chunk-preview">
                    ${escapeHTML(source.chunk_preview || "")}
                </div>

            </div>
        `;
    });

    html += `</div>`;

    panel.innerHTML = html;

    messages.appendChild(panel);

    scrollMessages();
}

// =====================================================
// FORM SUBMIT
// =====================================================

if (chatForm) {

chatForm.addEventListener(
    "submit",
    function(e) {

        e.preventDefault();

        sendQuestion();
    }
);

}

// =====================================================
// ENTER KEY
// =====================================================

if (questionInput) {


questionInput.addEventListener(
    "keypress",
    function(e) {

        if (
            e.key === "Enter"
        ) {

            e.preventDefault();

            sendQuestion();
        }
    }
);


}

// =====================================================
// PDF UPLOAD
// =====================================================

async function uploadPDF() {


if (
    !pdfFile.files.length
) {

    alert(
        "Select a PDF file."
    );

    return;
}

const formData =
    new FormData();

formData.append(
    "file",
    pdfFile.files[0]
);

uploadBtn.innerHTML =
    "Uploading...";

uploadBtn.disabled =
    true;

try {

    const response =
        await fetch(
            `${API_URL}/upload`,
            {
                method: "POST",
                body: formData
            }
        );

    uploadBtn.innerHTML =
        "Rebuilding index...";

    const data =
        await response.json();

    uploadBtn.innerHTML =
        "Upload PDF";

    uploadBtn.disabled =
        false;

    if (data.status === "success") {

        alert("✅ " + data.message);

    } else {

        alert("❌ " + (data.message || "Upload failed"));
    }

    await loadDocumentLibrary();

    pdfFile.value = "";

} catch (error) {

    uploadBtn.innerHTML =
        "Upload PDF";

    uploadBtn.disabled =
        false;

    alert(
        "❌ Upload Failed"
    );

    console.error(error);
}

}

// =====================================================
// DOCUMENT LIBRARY SIDEBAR (synced with real backend)
// =====================================================

function formatFileSize(bytes) {

    if (bytes < 1024)
        return bytes + " B";

    if (bytes < 1024 * 1024)
        return (bytes / 1024).toFixed(1) + " KB";

    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

async function loadDocumentLibrary() {

    if (!docLibraryList)
        return;

    try {

        const response = await fetch(`${API_URL}/documents`);

        const data = await response.json();

        renderDocumentLibrary(data.documents || []);

    } catch (error) {

        console.error("Failed to load document library:", error);

        docLibraryList.innerHTML = `
            <div class="doc-sidebar-empty">
                Could not load document list.
            </div>
        `;
    }
}

function renderDocumentLibrary(documents) {

    if (!docLibraryList)
        return;

    if (liveDocCount) {

        liveDocCount.innerText = documents.length;
    }

    if (documents.length === 0) {

        docLibraryList.innerHTML = `
            <div class="doc-sidebar-empty">
                No documents uploaded yet.
                Upload a PDF to get started.
            </div>
        `;

        return;
    }

    docLibraryList.innerHTML = "";

    documents.forEach(doc => {

        const card = document.createElement("div");

        card.className = "doc-card";

        card.innerHTML = `
            <div class="doc-card-name">
                <i class="fas fa-file-pdf"></i>
                <span>${escapeHTML(doc.filename)}</span>
            </div>
            <div class="doc-card-meta">
                <span>${formatFileSize(doc.size_bytes)} · indexed</span>
                <button type="button" class="doc-delete-btn" data-filename="${escapeHTML(doc.filename)}" title="Delete this document">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `;

        docLibraryList.appendChild(card);
    });

    docLibraryList.querySelectorAll(".doc-delete-btn").forEach(btn => {

        btn.addEventListener("click", () => {

            deleteDocument(btn.dataset.filename);
        });
    });
}

async function deleteDocument(filename) {

    const confirmed = confirm(
        `Delete "${filename}"? The entire index will be rebuilt ` +
        `without it — this may take a moment.`
    );

    if (!confirmed)
        return;

    const card = docLibraryList.querySelector(
        `.doc-delete-btn[data-filename="${filename}"]`
    )?.closest(".doc-card");

    if (card) {

        card.style.opacity = "0.4";

        card.innerHTML = `<div class="doc-card-meta">🗑️ Deleting and rebuilding index...</div>`;
    }

    try {

        const response = await fetch(
            `${API_URL}/documents/${encodeURIComponent(filename)}`,
            { method: "DELETE" }
        );

        const data = await response.json();

        if (data.status === "success") {

            await loadDocumentLibrary();

        } else {

            alert("❌ " + (data.message || "Delete failed"));

            await loadDocumentLibrary();
        }

    } catch (error) {

        console.error("Delete failed:", error);

        alert("❌ Failed to delete document.");

        await loadDocumentLibrary();
    }
}

// =====================================================
// UPLOAD BUTTON
// =====================================================

if (uploadBtn) {


uploadBtn.addEventListener(
    "click",
    uploadPDF
);


}

// =====================================================
// HERO COUNTER ANIMATION
// =====================================================

function animateNumbers() {


const statCards =
    document.querySelectorAll(
        ".stat-card h3"
    );

statCards.forEach(card => {

    const value =
        card.innerText;

    const number =
        parseInt(value);

    if (
        isNaN(number)
    )
        return;

    let count = 0;

    const step =
        Math.ceil(
            number / 50
        );

    const timer =
        setInterval(() => {

            count += step;

            if (
                count >= number
            ) {

                card.innerText =
                    value;

                clearInterval(
                    timer
                );

            } else {

                card.innerText =
                    count;
            }

        }, 25);
});


}

window.addEventListener(
"load",
animateNumbers
);

// =====================================================
// SMOOTH FADE IN
// =====================================================

const observer =
new IntersectionObserver(
entries => {


        entries.forEach(
            entry => {

                if (
                    entry.isIntersecting
                ) {

                    entry.target.style.opacity = 1;

                    entry.target.style.transform =
                        "translateY(0)";
                }
            }
        );

    },
    {
        threshold: 0.15
    }
);


document
.querySelectorAll(
".feature-card, .stat-card, .upload-card"
)
.forEach(item => {


item.style.opacity = 0;

item.style.transform =
    "translateY(30px)";

item.style.transition =
    "all 0.8s ease";

observer.observe(item);


});

console.log(
"Advanced RAG Frontend Loaded"
);