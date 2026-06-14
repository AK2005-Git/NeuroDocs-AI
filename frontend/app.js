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
speed = 10
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

    const botDiv =
        addMessage(
            "",
            "bot"
        );

    typeWriter(
        botDiv,
        data.answer
    );

    // Sources

    if (
        data.sources &&
        data.sources.length > 0
    ) {

        let sourceHTML =
            "<br><b>Sources Used:</b><ul>";

        data.sources.forEach(
            source => {

                sourceHTML += `
                <li>
                ${source.source_file}
                (Chunk ${source.chunk_id})
                </li>
                `;
            }
        );

        sourceHTML += "</ul>";

        setTimeout(
            () => {

                addMessage(
                    sourceHTML,
                    "bot"
                );

            },
            1200
        );
    }

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

    const data =
        await response.json();

    uploadBtn.innerHTML =
        "Upload PDF";

    uploadBtn.disabled =
        false;

    alert(
        "✅ " +
        data.message
    );

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
