def _index_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>nano-vllm-diy Web Chat</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b1020;
      --bg-2: #101a33;
      --panel: rgba(11, 16, 32, 0.78);
      --panel-border: rgba(255, 255, 255, 0.08);
      --text: #ecf2ff;
      --muted: #9cb0d9;
      --accent: #6ee7ff;
      --accent-2: #8b5cf6;
      --user: #1f6feb;
      --assistant: #13213d;
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Trebuchet MS", "Verdana", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(110, 231, 255, 0.22), transparent 28%),
        radial-gradient(circle at 85% 15%, rgba(139, 92, 246, 0.22), transparent 24%),
        linear-gradient(135deg, var(--bg), var(--bg-2));
      display: grid;
      place-items: center;
      padding: 24px;
    }

    .shell {
      width: min(1100px, 100%);
      min-height: calc(100vh - 48px);
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 16px;
      padding: 20px;
      border: 1px solid var(--panel-border);
      border-radius: 28px;
      background: var(--panel);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }

    header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      padding-bottom: 8px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    .brand {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .badge {
      display: inline-flex;
      width: fit-content;
      align-items: center;
      gap: 8px;
      padding: 6px 12px;
      border-radius: 999px;
      background: rgba(110, 231, 255, 0.12);
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    h1 {
      margin: 0;
      font-size: clamp(28px, 4vw, 44px);
      line-height: 1.02;
    }

    .subtitle {
      margin: 0;
      color: var(--muted);
      max-width: 64ch;
      line-height: 1.6;
    }

    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 14px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.04);
      color: var(--muted);
      font-size: 14px;
      white-space: nowrap;
    }

    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #34d399;
      box-shadow: 0 0 0 6px rgba(52, 211, 153, 0.16);
    }

    .chat {
      overflow: auto;
      padding: 10px 4px 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .message {
      max-width: min(78ch, 88%);
      padding: 14px 16px;
      border-radius: 20px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      line-height: 1.65;
      white-space: pre-wrap;
      word-break: break-word;
      animation: rise 180ms ease-out;
    }

    .message.user {
      align-self: flex-end;
      background: linear-gradient(135deg, rgba(31, 111, 235, 0.92), rgba(139, 92, 246, 0.82));
      box-shadow: 0 16px 36px rgba(31, 111, 235, 0.22);
    }

    .message.assistant {
      align-self: flex-start;
      background: rgba(255, 255, 255, 0.05);
    }

    .message.assistant-think {
      align-self: flex-start;
      color: #d7e3ff;
      background: rgba(56, 189, 248, 0.11);
      border-style: dashed;
      font-family: "Consolas", "Monaco", monospace;
      font-size: 13px;
      line-height: 1.55;
      opacity: 0.92;
    }

    .message.system {
      align-self: center;
      max-width: 100%;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.03);
    }

    .message.assistant-error {
      align-self: flex-start;
      color: #fecaca;
      background: rgba(248, 113, 113, 0.1);
    }

    @keyframes rise {
      from { transform: translateY(10px); opacity: 0; }
      to { transform: translateY(0); opacity: 1; }
    }

    .composer {
      display: grid;
      gap: 12px;
      grid-template-columns: 1fr auto auto auto;
      align-items: end;
      padding-top: 8px;
      border-top: 1px solid rgba(255, 255, 255, 0.08);
    }

    .length-select {
      height: 50px;
      padding: 0 12px;
      border-radius: 14px;
      border: 1px solid rgba(255, 255, 255, 0.14);
      background: rgba(3, 8, 20, 0.82);
      color: var(--text);
      font: inherit;
      outline: none;
      min-width: 140px;
    }

    .length-select:focus {
      border-color: rgba(110, 231, 255, 0.6);
      box-shadow: 0 0 0 4px rgba(110, 231, 255, 0.12);
    }

    textarea {
      width: 100%;
      min-height: 86px;
      resize: vertical;
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid rgba(255, 255, 255, 0.14);
      background: rgba(3, 8, 20, 0.82);
      color: var(--text);
      outline: none;
      line-height: 1.5;
      font: inherit;
    }

    textarea:focus {
      border-color: rgba(110, 231, 255, 0.6);
      box-shadow: 0 0 0 4px rgba(110, 231, 255, 0.12);
    }

    .btn {
      border: 0;
      border-radius: 16px;
      padding: 14px 18px;
      color: white;
      font: inherit;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease, box-shadow 120ms ease;
    }

    .btn:hover { transform: translateY(-1px); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

    .btn.primary {
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      box-shadow: 0 16px 36px rgba(110, 231, 255, 0.18);
    }

    .btn.secondary {
      background: rgba(255, 255, 255, 0.08);
    }

    .footer {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
    }

    .error {
      color: #fca5a5;
    }

    @media (max-width: 760px) {
      .shell {
        min-height: calc(100vh - 24px);
        padding: 16px;
        border-radius: 22px;
      }

      header {
        flex-direction: column;
      }

      .composer {
        grid-template-columns: 1fr;
      }

      .message {
        max-width: 100%;
      }

      .footer {
        flex-direction: column;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div class="brand">
        <div class="badge">nano-vllm-diy</div>
        <h1>Web Chat</h1>
        <p class="subtitle">直接在浏览器里和本地模型对话。页面会把消息发送到服务端，由当前启动的模型生成回复。</p>
      </div>
      <div class="status"><span class="dot"></span><span id="statusText">Ready</span></div>
    </header>

    <section id="chat" class="chat" aria-live="polite"></section>

    <section class="composer">
      <textarea id="prompt" placeholder="输入消息，按 Enter 发送，Shift+Enter 换行"></textarea>
      <select id="lengthSelect" class="length-select" aria-label="生成长度">
        <option value="256">短</option>
        <option value="1024" selected>中</option>
        <option value="4096">长</option>
      </select>
      <button id="clearBtn" class="btn secondary" type="button">Clear</button>
      <button id="sendBtn" class="btn primary" type="button">Send</button>
    </section>

    <div class="footer">
      <div>Model: <span id="modelName"></span></div>
      <div id="hint">Enter 发送，Shift+Enter 换行</div>
    </div>
  </main>

  <script>
    const chat = document.getElementById('chat');
    const prompt = document.getElementById('prompt');
    const sendBtn = document.getElementById('sendBtn');
    const clearBtn = document.getElementById('clearBtn');
    const lengthSelect = document.getElementById('lengthSelect');
    const statusText = document.getElementById('statusText');
    const modelName = document.getElementById('modelName');

    const defaultSystemPrompt = 'You are a helpful assistant. Conversation roles follow OpenAI Chat format: system sets rules, user is the human, assistant is you. When asked about role or identity, answer directly and briefly based on message roles.';
    const defaultTemperature = 1;

    const state = {
      messages: [
        { role: 'system', content: defaultSystemPrompt }
      ]
    };

    function scrollToBottom() {
      chat.scrollTop = chat.scrollHeight;
    }

    function escapeHtml(text) {
      return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function addMessage(role, content) {
      const item = document.createElement('div');
      item.className = `message ${role}`;
      item.innerHTML = escapeHtml(content);
      chat.appendChild(item);
      scrollToBottom();
      return item;
    }

    function splitThinkSections(answer, thinkTags) {
      if (!Array.isArray(thinkTags) || thinkTags.length !== 2) {
        return { answer, thinkSections: [] };
      }

      const [startTag, endTag] = thinkTags;
      const thinkSections = [];

      let cursor = 0;
      let cleanedAnswer = '';
      while (cursor < answer.length) {
        const startIndex = answer.indexOf(startTag, cursor);
        if (startIndex === -1) {
          cleanedAnswer += answer.slice(cursor);
          break;
        }

        cleanedAnswer += answer.slice(cursor, startIndex);
        const contentStart = startIndex + startTag.length;
        const endIndex = answer.indexOf(endTag, contentStart);
        if (endIndex === -1) {
          cleanedAnswer += answer.slice(startIndex);
          break;
        }

        const chunk = answer.slice(contentStart, endIndex).trim();
        if (chunk) {
          thinkSections.push(chunk);
        }

        cursor = endIndex + endTag.length;
      }

      const tripleNewline = String.fromCharCode(10) + String.fromCharCode(10) + String.fromCharCode(10);
      const doubleNewline = String.fromCharCode(10) + String.fromCharCode(10);
      while (cleanedAnswer.includes(tripleNewline)) {
        cleanedAnswer = cleanedAnswer.replace(tripleNewline, doubleNewline);
      }

      return { answer: cleanedAnswer.trim(), thinkSections };
    }

    function addAssistantResponse(answer, thinkSections) {
      for (const section of thinkSections) {
        addMessage('assistant-think', section);
      }
      addMessage('assistant', answer || '(empty response)');
    }

    function setBusy(busy) {
      sendBtn.disabled = busy;
      clearBtn.disabled = busy;
      lengthSelect.disabled = busy;
      prompt.disabled = busy;
      statusText.textContent = busy ? 'Generating...' : 'Ready';
    }

    function selectedMaxTokens() {
      const value = Number.parseInt(lengthSelect.value, 10);
      return Number.isFinite(value) && value > 0 ? value : 256;
    }

    function resetChat() {
      chat.innerHTML = '';
      state.messages = [{ role: 'system', content: defaultSystemPrompt }];
      addMessage('system', 'Chat cleared. Start a new conversation.');
    }

    async function sendMessage() {
      const text = prompt.value.trim();
      if (!text) {
        return;
      }

      state.messages.push({ role: 'user', content: text });
      addMessage('user', text);
      prompt.value = '';

      setBusy(true);
      const loading = addMessage('assistant', 'Thinking...');

      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: state.messages,
            temperature: defaultTemperature,
            max_tokens: selectedMaxTokens(),
            ignore_eos: false,
          }),
        });

        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || 'Request failed');
        }

        loading.remove();
        const tags = Array.isArray(payload.think_tags) && payload.think_tags.length === 2
          ? payload.think_tags
          : ['<think>', '</think>'];
        let answer = payload.answer || '';
        let thinkSections = Array.isArray(payload.think) ? payload.think.filter(Boolean) : [];
        if (thinkSections.length === 0) {
          const parsed = splitThinkSections(answer, tags);
          answer = parsed.answer;
          thinkSections = parsed.thinkSections;
        }

        state.messages.push({ role: 'assistant', content: answer });
        addAssistantResponse(answer, thinkSections);
      } catch (error) {
        loading.remove();
        const message = error instanceof Error ? error.message : String(error);
        addMessage('assistant-error', message);
        statusText.innerHTML = '<span class="error">Request failed</span>';
      } finally {
        setBusy(false);
        prompt.focus();
      }
    }

    sendBtn.addEventListener('click', sendMessage);
    clearBtn.addEventListener('click', resetChat);
    prompt.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
      }
    });

    async function loadInfo() {
      try {
        const response = await fetch('/api/meta');
        const payload = await response.json();
        modelName.textContent = payload.model || 'unknown';
      } catch {
        modelName.textContent = 'unknown';
      }
    }

    resetChat();
    loadInfo();
    prompt.focus();
  </script>
</body>
</html>
"""