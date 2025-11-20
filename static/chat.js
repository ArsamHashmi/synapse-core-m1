const messagesEl = document.getElementById("messages");
const chatBodyEl = document.getElementById("chatBody");
const inputEl = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");

const debugBtn = document.getElementById("debugBtn");
const debugOverlay = document.getElementById("debugOverlay");
const debugCloseBtn = document.getElementById("debugCloseBtn");
const debugBody = document.getElementById("debugBody");

let typingBubbleEl = null;

// story buffering on frontend
let pendingChunks = [];        // chunks you already "sent" (pressed Enter)
let pendingTimer = null;       // inactivity timer
const INACTIVITY_MS = 4000;    // 4 seconds of no typing

function createMessageBubble(msg) {
  const row = document.createElement("div");
  row.classList.add("message-row", msg.sender);

  const bubble = document.createElement("div");
  bubble.classList.add("bubble");
  bubble.textContent = msg.text;

  const time = document.createElement("div");
  time.classList.add("time");
  time.textContent = msg.time || "";

  bubble.appendChild(time);
  row.appendChild(bubble);
  return row;
}

function appendMessage(msg) {
  // Don't render empty messages
  if (!msg || !msg.text || msg.text.trim() === "") return;

  const bubble = createMessageBubble(msg);
  messagesEl.appendChild(bubble);
  scrollToBottom();
}

function scrollToBottom() {
  chatBodyEl.scrollTop = chatBodyEl.scrollHeight;
}

async function loadHistory() {
  try {
    const res = await fetch("/messages");
    const data = await res.json();
    messagesEl.innerHTML = "";
    data.forEach(appendMessage);
    scrollToBottom();
  } catch (err) {
    console.error("Error loading history:", err);
  }
}

function showTypingIndicator() {
  if (typingBubbleEl) return;

  const row = document.createElement("div");
  row.classList.add("message-row", "bot");

  const bubble = document.createElement("div");
  bubble.classList.add("bubble");

  const wrapper = document.createElement("div");
  wrapper.classList.add("typing-bubble");

  for (let i = 0; i < 3; i++) {
    const dot = document.createElement("div");
    dot.classList.add("typing-dot");
    wrapper.appendChild(dot);
  }

  bubble.appendChild(wrapper);
  row.appendChild(bubble);

  typingBubbleEl = row;
  messagesEl.appendChild(row);
  scrollToBottom();
}

function hideTypingIndicator() {
  if (typingBubbleEl) {
    typingBubbleEl.remove();
    typingBubbleEl = null;
  }
}

/**
 * Bump "activity" timer.
 * If we have pending chunks, reset the 4s inactivity timer.
 */
function bumpActivityTimer() {
  if (!pendingChunks.length) {
    // nothing buffered yet, nothing to flush
    return;
  }
  if (pendingTimer) {
    clearTimeout(pendingTimer);
  }
  pendingTimer = setTimeout(flushPendingChunks, INACTIVITY_MS);
}

/**
 * Called when user hits Enter:
 *  - shows bubble immediately
 *  - adds text to pendingChunks
 *  - bumps activity timer (4s)
 */
function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;

  const now = new Date();
  const localTime = now.toTimeString().slice(0, 5);

  // show user bubble right away
  const tempUserMsg = {
    sender: "user",
    text,
    time: localTime,
  };
  appendMessage(tempUserMsg);
  inputEl.value = "";

  // add to local buffer (to be merged later)
  pendingChunks.push(text);

  // user is still "active" -> reset inactivity timer
  bumpActivityTimer();
}

/**
 * Actually send merged message to backend
 * once we've had 4s of no typing.
 */
async function flushPendingChunks() {
  if (!pendingChunks.length) return;

  const mergedText = pendingChunks.join(" ").trim();
  pendingChunks = [];
  pendingTimer = null;

  if (!mergedText) return;

  try {
    setSending(true);
    showTypingIndicator();

    const res = await fetch("/send", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: mergedText }),
    });

    hideTypingIndicator();

    if (!res.ok) {
      console.error("Failed to send");
      setSending(false);
      return;
    }

    const data = await res.json();
    if (data.bot && data.bot.text && data.bot.text.trim() !== "") {
      appendMessage(data.bot);
    }
  } catch (err) {
    hideTypingIndicator();
    console.error("Error sending message:", err);
  } finally {
    setSending(false);
  }
}

function setSending(isSending) {
  if (isSending) {
    sendBtn.classList.add("disabled");
    sendBtn.disabled = true;
  } else {
    sendBtn.classList.remove("disabled");
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", sendMessage);

// Key listener:
//  - Enter => sendMessage (as before)
//  - ANY key while there are pendingChunks => reset inactivity timer
inputEl.addEventListener("keydown", (e) => {
  // any key counts as "activity", so reset timer if chunks exist
  bumpActivityTimer();

  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ----- Debug / Memory popup logic -----

async function openDebugPanel() {
  debugOverlay.classList.remove("hidden");
  debugBody.innerHTML = `
    <div class="debug-section">
      <h3>Loading…</h3>
      <div class="debug-empty">Fetching current memory and state…</div>
    </div>
  `;

  try {
    const res = await fetch("/user_state");
    if (!res.ok) {
      throw new Error("Failed to fetch /user_state");
    }
    const data = await res.json();
    renderDebugContent(data);
  } catch (err) {
    console.error("Error loading user_state:", err);
    debugBody.innerHTML = `
      <div class="debug-section">
        <h3>Error</h3>
        <div class="debug-empty">Couldn't load state. Check console / backend.</div>
      </div>
    `;
  }
}

function closeDebugPanel() {
  debugOverlay.classList.add("hidden");
}

function renderDebugContent(data) {
  const persona = data.persona || {};
  const state = data.state || {};
  const notes = data.memory_notes || [];
  const items = data.memory_items || [];
  const summary = data.memory_summary || "";
  const concerns = state.concerns || [];

  const personalityTagSet = new Set();
  items.forEach((item) => {
    const t = (item.type || "").toLowerCase();
    const imp = item.importance ?? 1;
    if (imp < 2) return;

    if (
      t === "trait" ||
      t === "mood_pattern" ||
      t === "relationship" ||
      t === "preference" ||
      t === "goal" ||
      t === "concern"
    ) {
      if (Array.isArray(item.tags)) {
        item.tags.forEach((tag) => {
          const clean = String(tag || "").trim();
          if (clean) {
            personalityTagSet.add(clean);
          }
        });
      }
    }
  });
  const personalityTags = Array.from(personalityTagSet);

  debugBody.innerHTML = `
    <div class="debug-section">
      <h3>Persona (AI side)</h3>
      <div class="debug-kv">
        <span class="key">Name</span>
        <span class="value">${persona.name || "—"}</span>
      </div>
      <div class="debug-kv">
        <span class="key">Age</span>
        <span class="value">${persona.age || "—"}</span>
      </div>
      <div class="debug-kv">
        <span class="key">Location</span>
        <span class="value">${persona.location || "—"}</span>
      </div>
      <div class="debug-kv">
        <span class="key">Bio</span>
        <span class="value">${persona.bio || "—"}</span>
      </div>
    </div>

    <div class="debug-section">
      <h3>Conversation State</h3>
      <div class="debug-kv">
        <span class="key">Relationship</span>
        <span class="value">${state.relationship_stage || "—"}</span>
      </div>
      <div class="debug-kv">
        <span class="key">User mood</span>
        <span class="value">${state.user_mood || "—"}</span>
      </div>
      <div class="debug-kv">
        <span class="key">AI mood</span>
        <span class="value">${state.ai_mood || "—"}</span>
      </div>
      <div class="debug-kv">
        <span class="key">AI energy</span>
        <span class="value">${state.ai_energy ?? "—"}</span>
      </div>
      <div class="debug-kv">
        <span class="key">Trust level</span>
        <span class="value">${state.ai_trust_level ?? "—"}</span>
      </div>
      <div class="debug-kv">
        <span class="key">Messages</span>
        <span class="value">${state.message_count ?? "0"}</span>
      </div>
      <div class="debug-kv">
        <span class="key">Privacy refusal</span>
        <span class="value">${state.privacy_refusal ? "true" : "false"}</span>
      </div>
      <div style="margin-top:4px;">
        <span class="key" style="color:#9ca3af;font-size:11px;">Open loops</span>
        <div class="debug-tags" style="margin-top:2px;">
          ${
            state.open_loops && state.open_loops.length
              ? state.open_loops.map(t => `<span class="debug-tag">${t}</span>`).join("")
              : `<span class="debug-empty">none</span>`
          }
        </div>
      </div>
    </div>

    <div class="debug-section">
      <h3>User Profile Summary</h3>
      ${
        summary
          ? `<div class="debug-summary">${summary}</div>`
          : `<div class="debug-empty">No summary yet.</div>`
      }

      <div style="margin-top:8px;">
        <span class="key" style="color:#9ca3af;font-size:11px;">Personality tags</span>
        <div class="debug-tags" style="margin-top:4px;">
          ${
            personalityTags.length
              ? personalityTags.map(tag => `<span class="debug-tag">${tag}</span>`).join("")
              : `<span class="debug-empty">no personality tags inferred yet.</span>`
          }
        </div>
      </div>
    </div>

    <div class="debug-section">
      <h3>Personality Notes & Memory</h3>
      ${
        items.length
          ? `
            <div class="debug-table-wrapper">
              <table class="debug-table">
                <thead>
                  <tr>
                    <th>Note</th>
                    <th>Type</th>
                    <th>Importance</th>
                    <th>Tags</th>
                    <th>Created at</th>
                  </tr>
                </thead>
                <tbody>
                  ${
                    items.map(item => {
                      const safeText = item.text || "";
                      const safeType = item.type || "other";
                      const safeImportance = item.importance ?? 1;
                      const safeTags = (item.tags && item.tags.length)
                        ? item.tags.map(t => `<span class="debug-tag">${t}</span>`).join(" ")
                        : `<span class="debug-empty">none</span>`;
                      const created = (item.created_at !== null && item.created_at !== undefined)
                        ? item.created_at
                        : "—";

                      return `
                        <tr>
                          <td><div class="mem-text">${safeText}</div></td>
                          <td><span class="debug-badge">${safeType}</span></td>
                          <td>${safeImportance}</td>
                          <td>${safeTags}</td>
                          <td>${created}</td>
                        </tr>
                      `;
                    }).join("")
                  }
                </tbody>
              </table>
            </div>
          `
          : (
              notes.length
                ? `<ul class="debug-list">
                     ${notes.map(n => `<li>${n}</li>`).join("")}
                   </ul>`
                : `<div class="debug-empty">No personality notes stored yet.</div>`
            )
      }
    </div>

    <div class="debug-section">
      <h3>Concerns to Follow Up</h3>
      ${
        (concerns && concerns.length)
          ? `<ul class="debug-list">
               ${concerns.map(c => `
                 <li>
                   <div class="concern-type">${c.type || "unknown"}</div>
                   <div class="concern-text">${c.text || ""}</div>
                   <div class="debug-kv" style="margin-top:2px;">
                     <span class="key">created_at</span>
                     <span class="value">${c.created_at ?? "—"}</span>
                   </div>
                   <div class="debug-kv">
                     <span class="key">last_asked_at</span>
                     <span class="value">${c.last_asked_at ?? "never"}</span>
                   </div>
                   <div class="debug-kv">
                     <span class="key">resolved</span>
                     <span class="value">${c.resolved ? "true" : "false"}</span>
                   </div>
                 </li>
               `).join("")}
             </ul>`
          : `<div class="debug-empty">No tracked concerns yet (health, exam, interview, etc.).</div>`
      }
    </div>
  `;
}

debugBtn.addEventListener("click", openDebugPanel);
debugCloseBtn.addEventListener("click", closeDebugPanel);
debugOverlay.addEventListener("click", (e) => {
  if (e.target === debugOverlay) {
    closeDebugPanel();
  }
});

// Initial load
loadHistory();