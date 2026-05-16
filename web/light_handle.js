import { app } from "../../../../scripts/app.js";

app.registerExtension({
    name: "zoey.lightHandle",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ZoeyLightHandle") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                const node = this;

                const getW = (name) => node.widgets?.find((w) => w.name === name);

                // ── Container ──
                const container = document.createElement("div");
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.width = "100%";
                container.style.gap = "4px";

                // ── Canvas ──
                const canvas = document.createElement("canvas");
                canvas.style.width = "100%";
                canvas.style.height = "280px";
                canvas.style.border = "1px solid rgba(255,255,255,0.1)";
                canvas.style.borderRadius = "8px";
                canvas.style.backgroundColor = "#0a0a0f";
                canvas.style.cursor = "crosshair";
                canvas.style.display = "block";

                const ctx = canvas.getContext("2d");
                const dpr = window.devicePixelRatio || 1;
                let cachedImage = null;

                // ── Color bar ──
                const colorBar = document.createElement("div");
                colorBar.style.display = "flex";
                colorBar.style.alignItems = "center";
                colorBar.style.gap = "6px";
                colorBar.style.padding = "2px 4px";
                colorBar.style.cursor = "pointer";
                colorBar.style.borderRadius = "4px";
                colorBar.style.border = "1px solid rgba(255,255,255,0.1)";
                colorBar.style.backgroundColor = "rgba(10,10,15,0.6)";

                const swatch = document.createElement("div");
                swatch.style.width = "20px";
                swatch.style.height = "20px";
                swatch.style.borderRadius = "3px";
                swatch.style.border = "1px solid rgba(255,255,255,0.2)";
                swatch.style.flexShrink = "0";

                const hexLabel = document.createElement("span");
                hexLabel.style.fontSize = "11px";
                hexLabel.style.color = "rgba(255,255,255,0.6)";
                hexLabel.style.fontFamily = "monospace";

                const colorInput = document.createElement("input");
                colorInput.type = "color";
                colorInput.style.display = "none";

                const updateSwatch = () => {
                    const cw = getW("light_color");
                    const val = cw?.value || "#FFFFFF";
                    swatch.style.backgroundColor = val;
                    hexLabel.textContent = val.toUpperCase();
                };

                colorInput.addEventListener("input", () => {
                    const val = colorInput.value.toUpperCase();
                    swatch.style.backgroundColor = val;
                    hexLabel.textContent = val;
                    const cw = getW("light_color");
                    if (cw) {
                        cw.value = val;
                        draw();
                    }
                    app.graph.setDirtyCanvas(true, true);
                });

                colorBar.addEventListener("click", () => colorInput.click());
                colorBar.appendChild(swatch);
                colorBar.appendChild(hexLabel);
                colorBar.appendChild(colorInput);

                container.appendChild(canvas);
                container.appendChild(colorBar);

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
                    const lc = getW("light_color")?.value || "#FFFFFF";

                    const px = hx * w;
                    const py = hy * h;
                    const br = Math.max(6, bs * Math.max(w, h));

                    // Parse color
                    const hex = lc.replace("#", "");
                    const lr = parseInt(hex.substring(0, 2), 16) || 255;
                    const lg = parseInt(hex.substring(2, 4), 16) || 255;
                    const lb = parseInt(hex.substring(4, 6), 16) || 255;

                    // Glow
                    const grad = ctx.createRadialGradient(px, py, 0, px, py, br * 2.5);
                    grad.addColorStop(0, `rgba(${lr},${lg},${lb},0.2)`);
                    grad.addColorStop(1, `rgba(${lr},${lg},${lb},0)`);
                    ctx.fillStyle = grad;
                    ctx.beginPath();
                    ctx.arc(px, py, br * 2.5, 0, Math.PI * 2);
                    ctx.fill();

                    // Outer glow ring
                    ctx.beginPath();
                    ctx.arc(px, py, br * 1.8, 0, Math.PI * 2);
                    ctx.strokeStyle = `rgba(${lr},${lg},${lb},0.2)`;
                    ctx.lineWidth = 1;
                    ctx.stroke();

                    // Ball ring
                    ctx.beginPath();
                    ctx.arc(px, py, br, 0, Math.PI * 2);
                    ctx.strokeStyle = `rgba(${lr},${lg},${lb},0.8)`;
                    ctx.lineWidth = 2;
                    ctx.stroke();

                    // Ball fill
                    ctx.beginPath();
                    ctx.arc(px, py, br - 2, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(${lr},${lg},${lb},0.2)`;
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
                    ctx.fillStyle = `rgb(${lr},${lg},${lb})`;
                    ctx.fill();

                    updateSwatch();
                };

                // ── Image loader ──
                const loadImage = (dataUrl) => {
                    const img = new Image();
                    img.onload = () => { cachedImage = img; draw(); };
                    img.onerror = () => { cachedImage = null; draw(); };
                    img.src = dataUrl;
                };

                // ── Mouse interaction ──
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

                // ── Scroll wheel → ball_size ──
                canvas.addEventListener("wheel", (e) => {
                    e.preventDefault();
                    const bs = getW("ball_size");
                    if (!bs) return;
                    const delta = e.deltaY > 0 ? -0.01 : 0.01;
                    bs.value = Math.max(0.02, Math.min(0.5, bs.value + delta));
                    draw();
                    app.graph.setDirtyCanvas(true, true);
                }, { passive: false });

                // ── Touch ──
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
                const widget = this.addDOMWidget("light_handle_viewer", "LIGHT_HANDLE", container, {
                    getValue() { return ""; },
                    setValue() {},
                });

                widget.computeSize = function (width) {
                    return [Math.max(width || 350, 350), 320];
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
                    if (["handle_x", "handle_y", "ball_size", "light_color"].includes(name)) {
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

                this.setSize([350, 400]);

                setTimeout(draw, 100);

                return r;
            };
        }
    },
});
