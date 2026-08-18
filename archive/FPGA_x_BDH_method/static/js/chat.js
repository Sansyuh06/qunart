const transcript = document.getElementById("chat-transcript");
const input = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");
const statusBadge = document.getElementById("fpga-status-badge");
const statusText = document.getElementById("fpga-status-text");
const ledRail = document.getElementById("led-rail");

let activeContent = null;
let activeTelemetry = null;
let fpgaDetected = false;
let processing = false;

function setDetectedStatus(payload) {
  fpgaDetected = Boolean(payload.connected);
  statusBadge.classList.toggle("connected", fpgaDetected);
  statusBadge.classList.toggle("disconnected", !fpgaDetected);
  ledRail.classList.toggle("detected", fpgaDetected);
  if (fpgaDetected) {
    const port = payload.port || "UNKNOWN COM";
    const baud = payload.baudrate || 115200;
    statusText.textContent = (
      `DEVICE CONNECTED: ${port} | ${baud} BAUD | ` +
      "ZERO-SKIP ON"
    );
    const initTel = document.getElementById("init-telemetry");
    if (initTel) {
      initTel.textContent = (
        `[MODEL: BDH 64D | FPGA: CONNECTED ${port} @ ${baud} | LED RAIL: IDLE]`
      );
    }
  } else {
    const reason = payload.local_reason || "DEVICE DISCONNECTED";
    statusText.textContent = reason;
    const initTel = document.getElementById("init-telemetry");
    if (initTel) {
      initTel.textContent = (
        `[MODEL: BDH 64D | FPGA: ${reason} | LED RAIL: IDLE]`
      );
    }
  }
}


class VirtualLEDChaser {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.leds = this.container ? this.container.querySelectorAll(".led") : [];
    this.pos = 0;
    this.fade = 0.0;
    this.speed = 0.015;
    this.active = false;
    this.targetSpeed = 0.015;
    this.rafId = null;
    this.start();
  }

  setProcessing(isProcessing) {
    this.active = isProcessing;
    this.targetSpeed = isProcessing ? 0.055 : 0.015;
  }

  step() {
    this.speed += (this.targetSpeed - this.speed) * 0.15;
    this.fade += this.speed;
    if (this.fade >= 1.0) {
      this.fade -= 1.0;
      this.pos = (this.pos + 1) % 6;
    }

    const nextPos = (this.pos + 1) % 6;
    // Gamma-corrected squared fade math matching Verilog:
    // fade_sq = fade * fade; inv_sq = (1023 - fade) * (1023 - fade)
    const brIn = Math.pow(this.fade, 2);
    const brOut = Math.pow(1.0 - this.fade, 2);

    for (let i = 0; i < 6; i++) {
      let br = 0.0;
      if (i === this.pos) {
        br = brOut;
      } else if (i === nextPos) {
        br = brIn;
      }

      const ledEl = this.leds[i];
      if (ledEl) {
        if (br > 0.005) {
          const r = this.active ? 25 : 255;
          const g = this.active ? 214 : 255;
          const b = this.active ? 107 : 255;
          const alpha = Math.min(br * 1.1, 1.0).toFixed(3);
          const glowPx = (br * 16).toFixed(1);
          const borderAlpha = Math.max(br * 0.9, 0.25).toFixed(2);

          ledEl.style.backgroundColor = `rgba(${r}, ${g}, ${b}, ${alpha})`;
          ledEl.style.borderColor = `rgba(${r}, ${g}, ${b}, ${borderAlpha})`;
          ledEl.style.boxShadow = `0 0 ${glowPx}px rgba(${r}, ${g}, ${b}, ${alpha}), 0 0 ${(br * 4).toFixed(1)}px rgba(${r}, ${g}, ${b}, 1)`;
          ledEl.style.transform = `scale(${(0.95 + br * 0.18).toFixed(3)})`;
        } else {
          ledEl.style.backgroundColor = "#080808";
          ledEl.style.borderColor = "#222222";
          ledEl.style.boxShadow = "inset 0 0 3px #000000";
          ledEl.style.transform = "scale(0.95)";
        }
      }
    }

    this.rafId = requestAnimationFrame(() => this.step());
  }

  start() {
    if (!this.rafId) {
      this.step();
    }
  }
}

const virtualLedChaser = new VirtualLEDChaser("led-rail");

function setProcessing(value) {
  processing = value;
  virtualLedChaser.setProcessing(value);
  sendButton.disabled = value;
}

function appendMessage(kind, sender, text) {
  const article = document.createElement("article");
  article.className = `message ${kind}`;
  article.innerHTML = `
    <span class="msg-sender"></span>
    <div class="msg-content"></div>
  `;
  article.querySelector(".msg-sender").textContent = sender;
  article.querySelector(".msg-content").textContent = text;
  transcript.appendChild(article);
  transcript.scrollTop = transcript.scrollHeight;
  return article;
}

function createAiMessage() {
  const article = document.createElement("article");
  article.className = "message ai";
  article.innerHTML = `
    <span class="msg-sender">BDH x TANG NANO 9K</span>
    <div class="msg-content"></div>
    <div class="msg-telemetry">[WAITING FOR FPGA TOKEN FRAMES...]</div>
  `;
  transcript.appendChild(article);
  transcript.scrollTop = transcript.scrollHeight;
  activeContent = article.querySelector(".msg-content");
  activeTelemetry = article.querySelector(".msg-telemetry");
}

function sendPrompt(prompt) {
  const text = prompt.trim();
  if (!text || processing) return;
  appendMessage("user", "USER PROMPT", text);
  input.value = "";
  createAiMessage();
  setProcessing(true);

  const offloadModeSelect = document.getElementById("offload-mode");
  const hardwareRequiredCheck = document.getElementById("hardware-required");

  const options = {
    offload: offloadModeSelect ? offloadModeSelect.value : "full",
    hardware_required: hardwareRequiredCheck ? hardwareRequiredCheck.checked : false,
    temperature: 0.7,
    top_k: 3,
    top_p: 0.9,
  };

  window.bdhSocket.sendPrompt(text, options);
}

function applyTokenFrame(frame) {
  if (!activeContent || !activeTelemetry) return;
  activeContent.textContent += frame.token || "";
  const execMode = frame.execution_mode || "CPU";
  activeTelemetry.textContent = (
    `[MODE: ${execMode} | ` +
    `${Number(frame.latency_ms || 0).toFixed(1)} ms/token | ` +
    `${Number(frame.zero_sparsity_pct || 0).toFixed(1)}% ZERO-SKIPPED | ` +
    `${Number(frame.fpga_speedup || 0).toFixed(2)}x SPEEDUP | ` +
    `DSP: ${frame.dsp_ops || 0} | ` +
    `UART: ${frame.uart_bytes_sent || 0} B]`
  );

  // Update sidebar visualization bars
  const activePct = Number(frame.active_pct || 0);
  const skippedPct = Number(frame.zero_sparsity_pct || 0);
  const activeBar = document.getElementById("active-bar");
  const skippedBar = document.getElementById("skipped-bar");
  const speedupTag = document.getElementById("speedup-tag");
  if (activeBar) {
    activeBar.style.width = `${Math.min(activePct, 100)}%`;
    activeBar.textContent = `${activePct.toFixed(1)}%`;
  }
  if (skippedBar) {
    skippedBar.style.width = `${Math.min(skippedPct, 100)}%`;
    skippedBar.textContent = `${skippedPct.toFixed(1)}%`;
  }
  if (speedupTag) {
    speedupTag.textContent = `${Number(frame.fpga_speedup || 0).toFixed(2)}x`;
  }

  window.graphVisualizer.update(frame);
  transcript.scrollTop = transcript.scrollHeight;
}

function applyError(payload) {
  if (!activeContent || !activeTelemetry) {
    createAiMessage();
  }
  activeContent.textContent = payload.message;
  activeTelemetry.textContent = "[HARDWARE REQUIRED ERROR | EXECUTION ABORTED]";
  setProcessing(false);
}

window.bdhSocket.onMessage((payload) => {
  if (payload.type === "status") {
    setDetectedStatus(payload);
  } else if (payload.type === "inference_start") {
    setProcessing(true);
  } else if (payload.type === "token_frame") {
    applyTokenFrame(payload);
  } else if (payload.type === "error") {
    applyError(payload);
  } else if (payload.type === "inference_end") {
    setProcessing(false);
  }
});

sendButton.addEventListener("click", () => sendPrompt(input.value));
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    sendPrompt(input.value);
  }
});

document.querySelectorAll(".quick-btn").forEach((button) => {
  button.addEventListener("click", () => {
    sendPrompt(button.dataset.prompt || "");
  });
});
