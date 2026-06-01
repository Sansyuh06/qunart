import { useRef, useEffect } from 'react';

// Ported from the existing website/script.js polygon animation
// Renders at 8% opacity as a subtle background texture behind all content

export default function GeometryCanvas() {
  const canvasRef = useRef(null);
  const mouseRef = useRef({ x: null, y: null });
  const edgesRef = useRef([]);
  const particlesRef = useRef([]);
  const rafRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const INTERACT_RADIUS = 320;

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

    class Edge {
      constructor(ax, ay, bx, by) {
        this.baseAX = ax; this.baseAY = ay;
        this.baseBX = bx; this.baseBY = by;
        this.midX = (ax + bx) / 2;
        this.midY = (ay + by) / 2;
        this.baseAngle = Math.atan2(by - ay, bx - ax);
        this.length = Math.sqrt((bx - ax) ** 2 + (by - ay) ** 2);
        this.rotation = 0; this.offsetX = 0; this.offsetY = 0;
        this.spread = 0; this.thickness = 1;
        this.targetRotation = 0; this.targetOffsetX = 0; this.targetOffsetY = 0;
        this.targetSpread = 0; this.targetThickness = 1;
        this.velRotation = 0; this.velOffsetX = 0; this.velOffsetY = 0;
        this.velSpread = 0; this.velThickness = 0;
        this.proximity = 0; this.wasActive = false; this.tremor = 0;
      }

      update() {
        const mouse = mouseRef.current;
        let isActive = false;
        if (mouse.x !== null && mouse.y !== null) {
          const dx = this.midX - mouse.x;
          const dy = this.midY - mouse.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < INTERACT_RADIUS && dist > 0) {
            const factor = Math.pow(1 - dist / INTERACT_RADIUS, 2);
            const angle = Math.atan2(dy, dx);
            isActive = factor > 0.05;
            this.targetOffsetX = Math.cos(angle) * 35 * factor;
            this.targetOffsetY = Math.sin(angle) * 35 * factor;
            const cross = dx * (mouse.y - this.midY) - dy * (mouse.x - this.midX);
            this.targetRotation = Math.sign(cross) * 0.2 * factor;
            this.targetSpread = factor * 10;
            this.targetThickness = 1 + factor * 1.5;
            this.proximity = factor;
            if (!this.wasActive && isActive) {
              this.tremor = 1.0;
              const cosR = Math.cos(this.baseAngle + this.rotation);
              const sinR = Math.sin(this.baseAngle + this.rotation);
              const halfLen = this.length / 2;
              const jAX = this.midX + this.offsetX - cosR * halfLen;
              const jAY = this.midY + this.offsetY - sinR * halfLen;
              const jBX = this.midX + this.offsetX + cosR * halfLen;
              const jBY = this.midY + this.offsetY + sinR * halfLen;
              for (let i = 0; i < 4; i++) {
                particlesRef.current.push(new Spark(jAX, jAY));
                particlesRef.current.push(new Spark(jBX, jBY));
              }
            }
          } else {
            this.targetOffsetX = 0; this.targetOffsetY = 0;
            this.targetRotation = 0; this.targetSpread = 0;
            this.targetThickness = 1;
            this.proximity = Math.max(0, this.proximity - 0.025);
          }
        } else {
          this.targetOffsetX = 0; this.targetOffsetY = 0;
          this.targetRotation = 0; this.targetSpread = 0;
          this.targetThickness = 1;
          this.proximity = Math.max(0, this.proximity - 0.025);
        }
        this.wasActive = isActive;
        this.tremor *= 0.9;
        const S = 0.14, D = 0.6;
        this.velOffsetX += (this.targetOffsetX - this.offsetX) * S;
        this.velOffsetY += (this.targetOffsetY - this.offsetY) * S;
        this.velRotation += (this.targetRotation - this.rotation) * S;
        this.velSpread += (this.targetSpread - this.spread) * S;
        this.velThickness += (this.targetThickness - this.thickness) * S;
        this.velOffsetX *= D; this.velOffsetY *= D;
        this.velRotation *= D; this.velSpread *= D; this.velThickness *= D;
        this.offsetX += this.velOffsetX; this.offsetY += this.velOffsetY;
        this.rotation += this.velRotation; this.spread += this.velSpread;
        this.thickness += this.velThickness;
      }

      draw() {
        const edgeDX = this.baseBX - this.baseAX;
        const edgeDY = this.baseBY - this.baseAY;
        const len = Math.sqrt(edgeDX * edgeDX + edgeDY * edgeDY);
        if (len === 0) return;
        const perpX = -edgeDY / len;
        const perpY = edgeDX / len;
        const sx = perpX * this.spread;
        const sy = perpY * this.spread;
        const relAX = this.baseAX - this.midX;
        const relAY = this.baseAY - this.midY;
        const relBX = this.baseBX - this.midX;
        const relBY = this.baseBY - this.midY;
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

        ctx.beginPath();
        ctx.moveTo(fAX, fAY);
        ctx.lineTo(fBX, fBY);
        ctx.strokeStyle = `rgba(90, 90, 90, ${0.18 + this.proximity * 0.12})`;
        ctx.lineWidth = this.thickness;
        ctx.stroke();

        const mouse = mouseRef.current;
        if (this.proximity > 0.01 && mouse.x !== null) {
          const spotGrad = ctx.createRadialGradient(mouse.x, mouse.y, 0, mouse.x, mouse.y, INTERACT_RADIUS);
          spotGrad.addColorStop(0, `rgba(255, 0, 0, ${this.proximity})`);
          spotGrad.addColorStop(0.35, `rgba(255, 0, 0, ${this.proximity * 0.4})`);
          spotGrad.addColorStop(1, 'rgba(255, 0, 0, 0)');
          ctx.beginPath();
          ctx.moveTo(fAX, fAY); ctx.lineTo(fBX, fBY);
          ctx.strokeStyle = spotGrad;
          ctx.lineWidth = this.thickness + 0.5;
          ctx.save();
          ctx.shadowColor = `rgba(255, 0, 0, ${this.proximity * 0.6})`;
          ctx.shadowBlur = 12 * this.proximity;
          ctx.stroke();
          ctx.restore();
        }

        if (this.proximity > 0.05) {
          const dotSize = 1.5 + this.proximity * 2.5;
          const dotAlpha = this.proximity * 0.7;
          ctx.beginPath();
          ctx.arc(fAX, fAY, dotSize + 3, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255, 0, 0, ${dotAlpha * 0.15})`;
          ctx.fill();
          ctx.beginPath();
          ctx.arc(fBX, fBY, dotSize + 3, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255, 0, 0, ${dotAlpha * 0.15})`;
          ctx.fill();
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

    function createEdgesFromPolygon(vertices) {
      const edgeList = [];
      for (let i = 0; i < vertices.length; i++) {
        const a = vertices[i];
        const b = vertices[(i + 1) % vertices.length];
        edgeList.push(new Edge(a.x, a.y, b.x, b.y));
      }
      return edgeList;
    }

    function initEdges() {
      const w = canvas.width;
      const h = canvas.height;
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
      const allEdges = [];
      polygons.forEach(poly => {
        allEdges.push(...createEdgesFromPolygon(poly));
      });
      edgesRef.current = allEdges;
      particlesRef.current = [];
    }

    function resize() {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      initEdges();
    }

    function animate() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (const edge of edgesRef.current) {
        edge.update();
        edge.draw();
      }
      const particles = particlesRef.current;
      for (let i = particles.length - 1; i >= 0; i--) {
        particles[i].update();
        particles[i].draw();
        if (particles[i].life <= 0) particles.splice(i, 1);
      }
      rafRef.current = requestAnimationFrame(animate);
    }

    const onMouseMove = (e) => {
      mouseRef.current = { x: e.clientX, y: e.clientY };
    };
    const onMouseOut = () => {
      mouseRef.current = { x: null, y: null };
    };

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseout', onMouseOut);
    window.addEventListener('resize', resize);
    resize();
    rafRef.current = requestAnimationFrame(animate);

    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseout', onMouseOut);
      window.removeEventListener('resize', resize);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 0,
        pointerEvents: 'none',
        opacity: 0.08,
      }}
    />
  );
}
