class BDHSocket {
  constructor() {
    this.socket = null;
    this.listeners = new Set();
    this.reconnectDelay = 900;
    this.connect();
  }

  connect() {
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    this.socket = new WebSocket(`${scheme}://${window.location.host}/ws/inference`);

    this.socket.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      this.listeners.forEach((listener) => listener(payload));
    };

    this.socket.onclose = () => {
      this.emitLocal({
        type: "status",
        connected: false,
        local_reason: "WEBSOCKET DISCONNECTED",
      });
      window.setTimeout(() => this.connect(), this.reconnectDelay);
    };

    this.socket.onerror = () => {
      this.emitLocal({
        type: "status",
        connected: false,
        local_reason: "SERVER LINK ERROR",
      });
    };
  }

  onMessage(listener) {
    this.listeners.add(listener);
  }

  emitLocal(payload) {
    this.listeners.forEach((listener) => listener(payload));
  }

  sendPrompt(prompt, options = {}) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      this.emitLocal({
        type: "error",
        message: "Backend WebSocket is not connected.",
      });
      return;
    }
    const payload = {
      prompt,
      offload: options.offload || "full",
      hardware_required: Boolean(options.hardware_required),
      temperature: options.temperature || 0.7,
      top_k: options.top_k || 3,
      top_p: options.top_p || 0.9,
    };
    this.socket.send(JSON.stringify(payload));
  }
}

window.bdhSocket = new BDHSocket();
