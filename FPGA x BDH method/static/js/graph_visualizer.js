class GraphVisualizer {
  constructor() {
    this.canvas = document.getElementById("graph-canvas");
    this.ctx = this.canvas.getContext("2d");
    this.nodes = [];
    this.edges = [];
    this.matrix = new Array(256).fill(0);
    this.activePct = 0;
    this.skippedPct = 0;
    this.initGrid();
    this.seedGraph();
    this.resize();
    window.addEventListener("resize", () => this.resize());
    requestAnimationFrame(() => this.draw());
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const scale = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.floor(rect.width * scale));
    this.canvas.height = Math.max(1, Math.floor(rect.height * scale));
    this.ctx.setTransform(scale, 0, 0, scale, 0, 0);
    this.draw();
  }

  initGrid() {
    const grid = document.getElementById("hebbian-grid");
    grid.innerHTML = "";
    for (let i = 0; i < 256; i += 1) {
      const cell = document.createElement("div");
      cell.className = "hebbian-cell";
      grid.appendChild(cell);
    }
  }

  seedGraph() {
    this.nodes = [];
    this.edges = [];
    for (let i = 0; i < 12; i += 1) {
      this.nodes.push({
        id: i,
        type: i < 6 ? "source" : "target",
        x: 0.14 + ((i % 6) * 0.145),
        y: i < 6 ? 0.32 : 0.68,
        act: 0,
        hub_score: 0.1,
      });
    }
    for (let i = 0; i < 6; i += 1) {
      this.edges.push({ source: i, target: i + 6, intensity: 0.15 });
    }
  }

  update(frame) {
    if (Array.isArray(frame.graph_nodes)) {
      this.nodes = frame.graph_nodes.map((node, index) => ({
        ...node,
        x: 0.13 + ((index % 6) * 0.15),
        y: node.type === "source" ? 0.3 : 0.7,
      }));
    }
    if (Array.isArray(frame.graph_edges)) {
      this.edges = frame.graph_edges;
    }
    if (Array.isArray(frame.synaptic_matrix_16x16)) {
      this.matrix = frame.synaptic_matrix_16x16.slice(0, 256);
      this.paintMatrix();
    }
    this.skippedPct = Number(frame.zero_sparsity_pct || 0);
    this.activePct = Number(frame.active_pct || (100 - this.skippedPct));
    this.paintBars(Number(frame.fpga_speedup || 0));
    this.draw();
  }

  paintBars(speedup) {
    const active = document.getElementById("active-bar");
    const skipped = document.getElementById("skipped-bar");
    const tag = document.getElementById("speedup-tag");
    active.style.width = `${Math.max(0, Math.min(100, this.activePct))}%`;
    skipped.style.width = `${Math.max(0, Math.min(100, this.skippedPct))}%`;
    active.textContent = `${this.activePct.toFixed(1)}%`;
    skipped.textContent = `${this.skippedPct.toFixed(1)}%`;
    tag.textContent = `${speedup.toFixed(2)}x`;
  }

  paintMatrix() {
    const cells = document.querySelectorAll(".hebbian-cell");
    cells.forEach((cell, index) => {
      const value = Math.max(0, Math.min(1, Number(this.matrix[index] || 0)));
      const shade = Math.round(value * 255);
      cell.style.background = `rgb(${shade}, ${shade}, ${shade})`;
    });
  }

  draw() {
    const rect = this.canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    this.ctx.clearRect(0, 0, width, height);
    this.ctx.lineCap = "round";

    this.edges.forEach((edge) => {
      const source = this.nodes.find((node) => node.id === edge.source);
      const target = this.nodes.find((node) => node.id === edge.target);
      if (!source || !target) return;
      const intensity = Math.max(0.08, Math.min(1, edge.intensity || 0));
      this.ctx.beginPath();
      this.ctx.moveTo(source.x * width, source.y * height);
      this.ctx.lineTo(target.x * width, target.y * height);
      this.ctx.strokeStyle = `rgba(255, 255, 255, ${intensity})`;
      this.ctx.lineWidth = 1 + intensity * 3;
      this.ctx.shadowColor = "#ffffff";
      this.ctx.shadowBlur = intensity > 0.7 ? 12 : 0;
      this.ctx.stroke();
      this.ctx.shadowBlur = 0;
    });

    this.nodes.forEach((node) => {
      const radius = 5 + Math.min(8, Number(node.hub_score || 0) * 9);
      const x = node.x * width;
      const y = node.y * height;
      this.ctx.beginPath();
      this.ctx.arc(x, y, radius, 0, Math.PI * 2);
      if (node.type === "source") {
        this.ctx.strokeStyle = "#888888";
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
      } else {
        this.ctx.fillStyle = "#ffffff";
        this.ctx.fill();
      }
    });
  }
}

window.graphVisualizer = new GraphVisualizer();
