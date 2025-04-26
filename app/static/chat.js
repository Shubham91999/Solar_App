const chatContainer = document.getElementById("chat-container");
const inputEl       = document.getElementById("chat-input");
const sendBtn       = document.getElementById("send-btn");

sendBtn.addEventListener("click", sendMessage);
inputEl.addEventListener("keypress", e => {
  if (e.key === "Enter") sendMessage();
});

async function sendMessage() {
  const prompt = inputEl.value.trim();
  if (!prompt) return;

  appendBubble("user", prompt);

  inputEl.value    = "";
  inputEl.disabled = true;
  sendBtn.disabled = true;

  // show typing indicator: small spinner + text
  const typingBubble = document.createElement("div");
  typingBubble.classList.add("bubble", "assistant", "typing");
  typingBubble.innerHTML = `
    <div class="spinner-border spinner-border-sm text-primary" role="status"></div>
    <span class="ms-2">Assistant is typing…</span>
  `;
  chatContainer.appendChild(typingBubble);
  autoScroll();

  // fetch assistant response
  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt })
  });
  const { response } = await res.json();

  // remove spinner bubble
  typingBubble.remove();

  appendBubble("assistant", response);

  inputEl.disabled = false;
  sendBtn.disabled = false;
  inputEl.focus();
}

function appendBubble(role, text) {
  const bubble = document.createElement("div");
  bubble.classList.add("bubble", role);

  // preserve line breaks
  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  bubble.innerHTML = escaped.replace(/\n/g, "<br>");

  chatContainer.appendChild(bubble);
  autoScroll();
}

function autoScroll() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}
