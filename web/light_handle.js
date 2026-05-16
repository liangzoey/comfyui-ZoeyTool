import { app } from "../../../../scripts/app.js";

app.registerExtension({
    name: "zoey.lightHandle",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ZoeyLightHandle") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                const node = this;

                // ── Canvas setup ──
                const canvas = document.createElement("canvas");
                canvas.style.width = "100%";
                canvas.style.height = "320px";
                canvas.style.border = "1px solid rgba(255,255,255,0.1)";
                canvas.style.borderRadius = "8px";
                canvas.style.backgroundColor = "#0a0a0f";
                canvas.style.cursor = "crosshair";
                canvas.style.display = "block";

                const ctx = canvas.getContext("2d");
                const dpr = window.devicePixelRatio || 1;

                let cachedImage = null;
                let imageDataUrl = null;

                const getW = (name) => node.widgets?.find((w) => w.name === name);

                // ── Draw ──
                const draw = () => {
                    const rect = canvas.getBoundingClientRect();
                    const w = rect.width;
                    const h = rect.height;
                    if (w === 0 || h === 0) return;

                    canvas.width = w * dpr;
                    canvas.height = h * dpr;
                    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

                    // Background
                    ctx.fillStyle = "#0a0a0f";
                    ctx.fillRect(0, 0, w, h);

                    if (cachedImage) {
                        const scale = Math.min(w / cachedImage.width, h / cachedImage.height);
                        const iw = cachedImage.width * scale;
                        const ih = cachedImage.height * scale;
                        const ix = (w - iw) / 2;
                        const iy = (h - ih) / 2;
                        ctx.drawImage(cachedImage, ix, iy, iw, ih);
                    } else {
                        // Grid placeholder
                        ctx.strokeStyle = "rgba(255,255,255,0.05)";
                        ctx.lineWidth = 1;
                        for (let x = 0; x < w; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
                        for (let y = 0; y < h; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
                        ctx.fillStyle = "rgba(255,255,255,0.3)";
                        ctx.font = "14px sans-serif";
                        ctx.textAlign = "center";
                        ctx.textBaseline = "middle";
                        ctx.fillText("Connect an image and execute to preview", w / 2, h / 2);
                    }

                    // ── Handle ball ──
                    const hx = getW("handle_x")?.value ?? 0.5;
                    const hy = getW("handle_y")?.value ?? 0.5;
                    const bs = getW("ball_size")?.value ?? 0.15;

                    const px = hx * w;
                    const py = hy * h;
                    const br = Math.max(6, bs * Math.max(w, h));

                    // Glow
                    const grad = ctx.createRadialGradient(px, py, 0, px, py, br * 2.5);
                    grad.addColorStop(0, "rgba(255, 215, 0, 0.2)");
                    grad.addColorStop(1, "rgba(255, 215, 0, 0)");
                    ctx.fillStyle = grad;
                    ctx.beginPath();
                    ctx.arc(px, py, br * 2.5, 0, Math.PI * 2);
                    ctx.fill();

                    // Outer ring
                    ctx.beginPath();
                    ctx.arc(px, py, br, 0, Math.PI * 2);
                    ctx.strokeStyle = "rgba(255, 215, 0, 0.7)";
                    ctx.lineWidth = 2;
                    ctx.stroke();

                    // Inner fill
                    ctx.beginPath();
                    ctx.arc(px, py, br - 2, 0, Math.PI * 2);
                    ctx.fillStyle = "rgba(255, 215, 0, 0.15)";
                    ctx.fill();

                    // Crosshair
                    const ch = Math.max(8, br * 0.35);
                    ctx.strokeStyle = "rgba(255,255,255,0.85)";
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    ctx.moveTo(px - ch, py); ctx.lineTo(px + ch, py);
                    ctx.moveTo(px, py - ch); ctx.lineTo(px, py + ch);
                    ctx.stroke();

                    // Center dot
                    ctx.beginPath();
                    ctx.arc(px, py, 3, 0, Math.PI * 2);
                    ctx.fillStyle = "#FFD700";
                    ctx.fill();
                };

                // ── Image loader ──
                const loadImage = (dataUrl) => {
                    imageDataUrl = dataUrl;
                    const img = new Image();
                    img.onload = () => { cachedImage = img; draw(); };
                    img.onerror = () => { cachedImage = null; draw(); };
                    img.src = dataUrl;
                };

                // ── Interaction ──
                const posFromEvent = (clientX, clientY) => {
                    const rect = canvas.getBoundingClientRect();
                    return {
                        x: Math.max(0, Math.min(1, (clientX - rect.left) / rect.width)),
                        y: Math.max(0, Math.min(1, (clientY - rect.top) / rect.height)),
                    };
                };

                const setPos = (pos) => {
                    const hxW = getW("handle_x");
                    const hyW = getW("handle_y");
                    if (hxW) hxW.value = Math.round(pos.x * 100) / 100;
                    if (hyW) hyW.value = Math.round(pos.y * 100) / 100;
                    draw();
                    app.graph.setDirtyCanvas(true, true);
                };

                let dragging = false;

                canvas.addEventListener("mousedown", (e) => {
                    dragging = true;
                    setPos(posFromEvent(e.clientX, e.clientY));
                });

                canvas.addEventListener("mousemove", (e) => {
                    if (!dragging) return;
                    setPos(posFromEvent(e.clientX, e.clientY));
                });

                const onMouseUp = () => { dragging = false; };
                window.addEventListener("mouseup", onMouseUp);

                // Touch
                canvas.addEventListener("touchstart", (e) => {
                    e.preventDefault();
                    dragging = true;
                    const t = e.touches[0];
                    setPos(posFromEvent(t.clientX, t.clientY));
                }, { passive: false });

                canvas.addEventListener("touchmove", (e) => {
                    e.preventDefault();
                    if (!dragging) return;
                    const t = e.touches[0];
                    setPos(posFromEvent(t.clientX, t.clientY));
                }, { passive: false });

                canvas.addEventListener("touchend", (e) => {
                    e.preventDefault();
                    dragging = false;
                }, { passive: false });

                // ── DOM widget ──
                const widget = this.addDOMWidget("light_handle_viewer", "LIGHT_HANDLE", canvas, {
                    getValue() { return ""; },
                    setValue() {},
                });

                widget.computeSize = function (width) {
                    return [Math.max(width || 350, 350), 360];
                };

                // ── Receive executed image ──
                const onExecuted = this.onExecuted;
                this.onExecuted = function (message) {
                    if (onExecuted) onExecuted.apply(this, arguments);
                    if (message?.image_base64?.[0]) {
                        loadImage(message.image_base64[0]);
                    }
                };

                // ── React to widget slider changes ──
                const origOnWidgetChanged = this.onWidgetChanged;
                this.onWidgetChanged = function (name, value, old_val, w) {
                    if (origOnWidgetChanged) origOnWidgetChanged.apply(this, arguments);
                    if (["handle_x", "handle_y", "ball_size"].includes(name)) {
                        draw();
                    }
                };

                // ── Resize observer ──
                const resizeObserver = new ResizeObserver(() => draw());
                resizeObserver.observe(canvas);

                // ── Cleanup ──
                const origOnRemoved = this.onRemoved;
                this.onRemoved = function () {
                    resizeObserver.disconnect();
                    window.removeEventListener("mouseup", onMouseUp);
                    cachedImage = null;
                    if (origOnRemoved) origOnRemoved.apply(this, arguments);
                };

                this.setSize([350, 430]);

                // Initial draw after DOM settles
                setTimeout(draw, 100);

                return r;
            };
        }
    },
});
