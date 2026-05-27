const canvas = document.getElementById('geometry-canvas');
const ctx = canvas.getContext('2d');

let width, height, centerX, centerY;
let mouse = { x: null, y: null };
let edges = [];
let particles = [];

const INTERACT_RADIUS = 320;

function resizeCanvas() {
    width = window.innerWidth;
    height = window.innerHeight;
    centerX = width / 2;
    centerY = height / 2;
    canvas.width = width;
    canvas.height = height;
    initEdges();
}

// =============================================
// SPARK PARTICLE — burst from joints on activation
// =============================================
class Spark {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        const angle = Math.random() * Math.PI * 2;
        const speed = 1 + Math.random() * 3;
        this.vx = Math.cos(angle) * speed;
        this.vy = Math.sin(angle) * speed;
        this.life = 1;
        this.decay = 0.02 + Math.random() * 0.03;
        this.size = 1 + Math.random() * 1.5;
    }

    update() {
        this.x += this.vx;
        this.y += this.vy;
        this.vx *= 0.96;
        this.vy *= 0.96;
        this.life -= this.decay;
    }

    draw() {
        if (this.life <= 0) return;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size * this.life, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 30, 30, ${this.life * 0.8})`;
        ctx.fill();
    }
}

// =============================================
// EDGE — independent mechanical limb segment
// =============================================
class Edge {
    constructor(ax, ay, bx, by, polyIndex) {
        this.baseAX = ax; this.baseAY = ay;
        this.baseBX = bx; this.baseBY = by;
        this.polyIndex = polyIndex;

        // Midpoint
        this.midX = (ax + bx) / 2;
        this.midY = (ay + by) / 2;

        // Edge angle and length
        this.baseAngle = Math.atan2(by - ay, bx - ax);
        this.length = Math.sqrt((bx - ax) ** 2 + (by - ay) ** 2);

        // Mechanical state
        this.rotation = 0;
        this.offsetX = 0;
        this.offsetY = 0;
        this.spread = 0;
        this.thickness = 1;

        // Targets
        this.targetRotation = 0;
        this.targetOffsetX = 0;
        this.targetOffsetY = 0;
        this.targetSpread = 0;
        this.targetThickness = 1;

        // Spring velocity
        this.velRotation = 0;
        this.velOffsetX = 0;
        this.velOffsetY = 0;
        this.velSpread = 0;
        this.velThickness = 0;

        // Activation tracking
        this.proximity = 0;
        this.wasActive = false;
        this.activationTime = 0;
        this.tremor = 0;
    }

    update(time) {
        let isActive = false;

        if (mouse.x !== null && mouse.y !== null) {
            const dx = this.midX - mouse.x;
            const dy = this.midY - mouse.y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            if (dist < INTERACT_RADIUS && dist > 0) {
                const factor = Math.pow(1 - dist / INTERACT_RADIUS, 2);
                const angle = Math.atan2(dy, dx);
                isActive = factor > 0.05;

                // Push outward
                this.targetOffsetX = Math.cos(angle) * 35 * factor;
                this.targetOffsetY = Math.sin(angle) * 35 * factor;

                // Tilt — cross product for direction
                const cross = dx * (mouse.y - this.midY) - dy * (mouse.x - this.midX);
                this.targetRotation = Math.sign(cross) * 0.2 * factor;

                // Spread apart perpendicular
                this.targetSpread = factor * 10;

                // Thicken on interaction
                this.targetThickness = 1 + factor * 1.5;

                this.proximity = factor;

                // Tremor — high-frequency micro-vibration on first contact
                if (!this.wasActive && isActive) {
                    this.tremor = 1.0;
                    this.activationTime = time;

                    // Spawn sparks at joints
                    const cosR = Math.cos(this.baseAngle + this.rotation);
                    const sinR = Math.sin(this.baseAngle + this.rotation);
                    const halfLen = this.length / 2;
                    const jAX = this.midX + this.offsetX - cosR * halfLen;
                    const jAY = this.midY + this.offsetY - sinR * halfLen;
                    const jBX = this.midX + this.offsetX + cosR * halfLen;
                    const jBY = this.midY + this.offsetY + sinR * halfLen;

                    for (let i = 0; i < 4; i++) {
                        particles.push(new Spark(jAX, jAY));
                        particles.push(new Spark(jBX, jBY));
                    }
                }
            } else {
                this.targetOffsetX = 0;
                this.targetOffsetY = 0;
                this.targetRotation = 0;
                this.targetSpread = 0;
                this.targetThickness = 1;
                this.proximity = Math.max(0, this.proximity - 0.025);
            }
        } else {
            this.targetOffsetX = 0;
            this.targetOffsetY = 0;
            this.targetRotation = 0;
            this.targetSpread = 0;
            this.targetThickness = 1;
            this.proximity = Math.max(0, this.proximity - 0.025);
        }

        this.wasActive = isActive;

        // Tremor decay
        this.tremor *= 0.9;

        // Spring physics — snappy with heavy damping
        const SPRING = 0.14;
        const DAMP = 0.6;

        this.velOffsetX += (this.targetOffsetX - this.offsetX) * SPRING;
        this.velOffsetY += (this.targetOffsetY - this.offsetY) * SPRING;
        this.velRotation += (this.targetRotation - this.rotation) * SPRING;
        this.velSpread += (this.targetSpread - this.spread) * SPRING;
        this.velThickness += (this.targetThickness - this.thickness) * SPRING;

        this.velOffsetX *= DAMP;
        this.velOffsetY *= DAMP;
        this.velRotation *= DAMP;
        this.velSpread *= DAMP;
        this.velThickness *= DAMP;

        this.offsetX += this.velOffsetX;
        this.offsetY += this.velOffsetY;
        this.rotation += this.velRotation;
        this.spread += this.velSpread;
        this.thickness += this.velThickness;
    }

    draw() {
        // Perpendicular direction for spread
        const edgeDX = this.baseBX - this.baseAX;
        const edgeDY = this.baseBY - this.baseAY;
        const len = Math.sqrt(edgeDX * edgeDX + edgeDY * edgeDY);
        const perpX = -edgeDY / len;
        const perpY = edgeDX / len;

        const sx = perpX * this.spread;
        const sy = perpY * this.spread;

        // Relative to midpoint
        const relAX = this.baseAX - this.midX;
        const relAY = this.baseAY - this.midY;
        const relBX = this.baseBX - this.midX;
        const relBY = this.baseBY - this.midY;

        // Apply rotation + tremor
        const tremorAngle = this.tremor * (Math.sin(Date.now() * 0.08) * 0.04);
        const totalRot = this.rotation + tremorAngle;
        const cos = Math.cos(totalRot);
        const sin = Math.sin(totalRot);

        const rAX = relAX * cos - relAY * sin;
        const rAY = relAX * sin + relAY * cos;
        const rBX = relBX * cos - relBY * sin;
        const rBY = relBX * sin + relBY * cos;

        const fAX = this.midX + this.offsetX + sx + rAX;
        const fAY = this.midY + this.offsetY + sy + rAY;
        const fBX = this.midX + this.offsetX + sx + rBX;
        const fBY = this.midY + this.offsetY + sy + rBY;

        // Base line
        ctx.beginPath();
        ctx.moveTo(fAX, fAY);
        ctx.lineTo(fBX, fBY);
        ctx.strokeStyle = `rgba(90, 90, 90, ${0.18 + this.proximity * 0.12})`;
        ctx.lineWidth = this.thickness;
        ctx.stroke();

        // Red spotlight glow
        if (this.proximity > 0.01 && mouse.x !== null) {
            const spotGrad = ctx.createRadialGradient(
                mouse.x, mouse.y, 0,
                mouse.x, mouse.y, INTERACT_RADIUS
            );
            spotGrad.addColorStop(0, `rgba(255, 0, 0, ${this.proximity})`);
            spotGrad.addColorStop(0.35, `rgba(255, 0, 0, ${this.proximity * 0.4})`);
            spotGrad.addColorStop(1, 'rgba(255, 0, 0, 0)');

            ctx.beginPath();
            ctx.moveTo(fAX, fAY);
            ctx.lineTo(fBX, fBY);
            ctx.strokeStyle = spotGrad;
            ctx.lineWidth = this.thickness + 0.5;

            ctx.save();
            ctx.shadowColor = `rgba(255, 0, 0, ${this.proximity * 0.6})`;
            ctx.shadowBlur = 12 * this.proximity;
            ctx.stroke();
            ctx.restore();
        }

        // Joint dots — glowing articulation points
        if (this.proximity > 0.05) {
            const dotSize = 1.5 + this.proximity * 2.5;
            const dotAlpha = this.proximity * 0.7;

            // Outer glow
            ctx.beginPath();
            ctx.arc(fAX, fAY, dotSize + 3, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 0, 0, ${dotAlpha * 0.15})`;
            ctx.fill();

            ctx.beginPath();
            ctx.arc(fBX, fBY, dotSize + 3, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 0, 0, ${dotAlpha * 0.15})`;
            ctx.fill();

            // Core dot
            ctx.beginPath();
            ctx.arc(fAX, fAY, dotSize, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 40, 40, ${dotAlpha})`;
            ctx.fill();

            ctx.beginPath();
            ctx.arc(fBX, fBY, dotSize, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 40, 40, ${dotAlpha})`;
            ctx.fill();
        }
    }
}

// =============================================
// Build edges from polygon vertex lists
// =============================================
function createEdgesFromPolygon(vertices, polyIndex) {
    const edgeList = [];
    for (let i = 0; i < vertices.length; i++) {
        const a = vertices[i];
        const b = vertices[(i + 1) % vertices.length];
        edgeList.push(new Edge(a.x, a.y, b.x, b.y, polyIndex));
    }
    return edgeList;
}

// =============================================
// INIT
// =============================================
function initEdges() {
    edges = [];
    particles = [];
    const w = width;
    const h = height;

    const polygons = [
        [{ x: w*0.05, y: h*0.15 }, { x: w*0.20, y: h*0.05 }, { x: w*0.42, y: h*0.08 }, { x: w*0.50, y: h*0.25 }, { x: w*0.30, y: h*0.22 }],
        [{ x: w*0.02, y: h*0.40 }, { x: w*0.12, y: h*0.18 }, { x: w*0.35, y: h*0.15 }, { x: w*0.55, y: h*0.30 }, { x: w*0.40, y: h*0.50 }, { x: w*0.15, y: h*0.48 }],
        [{ x: w*0.03, y: h*0.55 }, { x: w*0.18, y: h*0.30 }, { x: w*0.45, y: h*0.38 }, { x: w*0.38, y: h*0.65 }, { x: w*0.10, y: h*0.68 }],
        [{ x: w*0.06, y: h*0.72 }, { x: w*0.01, y: h*0.35 }, { x: w*0.15, y: h*0.12 }, { x: w*0.28, y: h*0.28 }, { x: w*0.22, y: h*0.60 }],
        [{ x: w*0.30, y: h*0.02 }, { x: w*0.48, y: h*0.06 }, { x: w*0.38, y: h*0.18 }],
        [{ x: w*0.35, y: h*0.10 }, { x: w*0.60, y: h*0.04 }, { x: w*0.75, y: h*0.18 }, { x: w*0.55, y: h*0.35 }, { x: w*0.40, y: h*0.25 }],
        [{ x: w*0.08, y: h*0.80 }, { x: w*0.25, y: h*0.55 }, { x: w*0.50, y: h*0.48 }, { x: w*0.55, y: h*0.70 }, { x: w*0.30, y: h*0.78 }],
        [{ x: w*0.10, y: h*0.08 }, { x: w*0.50, y: h*0.15 }, { x: w*0.70, y: h*0.10 }, { x: w*0.65, y: h*0.30 }, { x: w*0.25, y: h*0.35 }],
    ];

    polygons.forEach((poly, idx) => {
        edges.push(...createEdgesFromPolygon(poly, idx));
    });
}

// =============================================
// ANIMATION
// =============================================
function animate(t) {
    ctx.clearRect(0, 0, width, height);

    // Update & draw edges
    for (const edge of edges) {
        edge.update(t);
        edge.draw();
    }

    // Update & draw spark particles
    for (let i = particles.length - 1; i >= 0; i--) {
        particles[i].update();
        particles[i].draw();
        if (particles[i].life <= 0) {
            particles.splice(i, 1);
        }
    }

    requestAnimationFrame(animate);
}

// Mouse
window.addEventListener('mousemove', (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
});
window.addEventListener('mouseout', () => {
    mouse.x = null;
    mouse.y = null;
});

window.addEventListener('resize', resizeCanvas);
resizeCanvas();
requestAnimationFrame(animate);
