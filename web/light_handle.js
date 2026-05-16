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

                // ── Hide light_color text widget (merged into canvas swatch) ──
                const hideTextWidget = (name) => {
                    const w = node.widgets?.find(w => w.name === name);
                    if (!w) return;
                    w.computeSize = () => [0, -4];
                    setTimeout(() => { if (w.element) w.element.style.display = "none"; }, 50);
                };
                hideTextWidget("light_color");

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
                let imgNaturalW = 0;
                let imgNaturalH = 0;

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
                    if (cw) { cw.value = val; draw(); }
                    app.graph.setDirtyCanvas(true, true);
                });

                colorBar.addEventListener("click", () => colorInput.click());
                colorBar.appendChild(swatch);
                colorBar.appendChild(hexLabel);
                colorBar.appendChild(colorInput);

                container.appendChild(canvas);
                container.appendChild(colorBar);

                // ── Draw shape helper ──
                const drawShape = (cx, cy, br, lr, lg, lb, shape, cw, ch) => {
                    ctx.beginPath();
                    if (shape === "方形") {
                        ctx.rect(cx - br + 1, cy - br + 1, (br - 1) * 2, (br - 1) * 2);
                    } else if (shape === "菱形") {
                        ctx.moveTo(cx, cy - br + 1);
                        ctx.lineTo(cx + br - 1, cy);
                        ctx.lineTo(cx, cy + br - 1);
                        ctx.lineTo(cx - br + 1, cy);
                        ctx.closePath();
                    } else if (shape === "三角形") {
                        const angle = Math.atan2(cy - ch / 2, cx - cw / 2);
                        const r2 = br - 2;
                        ctx.moveTo(cx + r2 * Math.cos(angle), cy + r2 * Math.sin(angle));
                        ctx.lineTo(cx + r2 * 0.5 * Math.cos(angle + 2.094), cy + r2 * 0.5 * Math.sin(angle + 2.094));
                        ctx.lineTo(cx + r2 * 0.5 * Math.cos(angle - 2.094), cy + r2 * 0.5 * Math.sin(angle - 2.094));
                        ctx.closePath();
                    } else { // 圆形
                        ctx.arc(cx, cy, br - 1, 0, Math.PI * 2);
                    }
                    ctx.fillStyle = `rgba(${lr},${lg},${lb},0.2)`;
                    ctx.fill();
                    ctx.strokeStyle = `rgba(${lr},${lg},${lb},0.8)`;
                    ctx.lineWidth = 2;
                    ctx.stroke();
                };

                // ── Draw (image-aware: accounts for letterbox and scale) ──
                const draw = () => {
                    const rect = canvas.getBoundingClientRect();
                    const cw = rect.width;
                    const ch = rect.height;
                    if (cw === 0 || ch === 0) return;

                    canvas.width = cw * dpr;
                    canvas.height = ch * dpr;
                    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

                    // Fill background
                    ctx.fillStyle = "#0a0a0f";
                    ctx.fillRect(0, 0, cw, ch);

                    // Image rendering state (with letterbox)
                    let imgOffX = 0, imgOffY = 0;
                    let imgDispW = cw, imgDispH = ch;
                    const hasImage = cachedImage && imgNaturalW > 0;

                    if (hasImage) {
                        const scale = Math.min(cw / imgNaturalW, ch / imgNaturalH);
                        imgDispW = imgNaturalW * scale;
                        imgDispH = imgNaturalH * scale;
                        imgOffX = (cw - imgDispW) / 2;
                        imgOffY = (ch - imgDispH) / 2;
                        ctx.drawImage(cachedImage, imgOffX, imgOffY, imgDispW, imgDispH);
                    } else {
                        // Grid placeholder
                        ctx.strokeStyle = "rgba(255,255,255,0.05)";
                        ctx.lineWidth = 1;
                        for (let x = 0; x < cw; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, ch); ctx.stroke(); }
                        for (let y = 0; y < ch; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cw, y); ctx.stroke(); }
                        ctx.fillStyle = "rgba(255,255,255,0.3)";
                        ctx.font = "14px sans-serif";
                        ctx.textAlign = "center";
                        ctx.textBaseline = "middle";
                        ctx.fillText("Connect an image and execute to preview", cw / 2, ch / 2);
                    }

                    // ── Handle ──
                    const hx = getW("handle_x")?.value ?? 0.5;
                    const hy = getW("handle_y")?.value ?? 0.5;
                    const bs = getW("ball_size")?.value ?? 0.15;
                    const lc = getW("light_color")?.value || "#FFFFFF";
                    const shape = getW("handle_shape")?.value || "圆形";

                    // Position and radius relative to displayed image (matches Python)
                    const px = imgOffX + hx * imgDispW;
                    const py = imgOffY + hy * imgDispH;
                    const br = Math.max(6, bs * Math.max(imgDispW, imgDispH));

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

                    // Shape
                    drawShape(px, py, br, lr, lg, lb, shape, imgDispW, imgDispH);

                    // Crosshair
                    const ch2 = Math.max(8, br * 0.35);
                    ctx.strokeStyle = "rgba(255,255,255,0.85)";
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    ctx.moveTo(px - ch2, py); ctx.lineTo(px + ch2, py);
                    ctx.moveTo(px, py - ch2); ctx.lineTo(px, py + ch2);
                    ctx.stroke();

                    // Center dot
                    ctx.beginPath();
                    ctx.arc(px, py, 3, 0, Math.PI * 2);
                    ctx.fillStyle = `rgb(${lr},${lg},${lb})`;
                    ctx.fill();

                    updateSwatch();
                };

                // ── Load image (from URL or data-URI) ──
                const loadImage = (url) => {
                    const img = new Image();
                    img.onload = () => {
                        cachedImage = img;
                        imgNaturalW = img.naturalWidth || img.width;
                        imgNaturalH = img.naturalHeight || img.height;
                        draw();
                    };
                    img.onerror = () => { cachedImage = null; imgNaturalW = 0; imgNaturalH = 0; draw(); };
                    img.src = url;
                };

                // ── Try to load image from connected input node ──
                const tryLoadFromInput = () => {
                    const imgInput = node.inputs?.find(inp => inp.name === "image");
                    if (!imgInput || imgInput.link == null) return;

                    const link = app.graph.links[imgInput.link];
                    if (!link) return;

                    const srcNode = app.graph.getNodeById(link[0]);
                    if (!srcNode) return;

                    // LoadImage node → stream directly from ComfyUI server
                    if (srcNode.type === "LoadImage" || srcNode.type === "Zoey True Size Image Loader" || srcNode.computeSize) {
                        const imgWidget = srcNode.widgets?.find(w => w.name === "image");
                        if (imgWidget && imgWidget.value) {
                            const parts = imgWidget.value.split("/");
                            const fn = parts.pop();
                            const sub = parts.join("/");
                            loadImage(`/view?filename=${encodeURIComponent(fn)}&type=input&subfolder=${encodeURIComponent(sub)}&rand=${Date.now()}`);
                            return;
                        }
                    }

                    // Generic fallback: check if node has an image output with cached data
                    if (srcNode.imgs && srcNode.imgs.length > 0) {
                        loadImage(srcNode.imgs[0].src);
                    }
                };

                // ── Handle connection changes ──
                const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
                nodeType.prototype.onConnectionsChange = function (slotType, slot, isConnected, link, outputSlot) {
                    if (origOnConnectionsChange) origOnConnectionsChange.apply(this, arguments);
                    if (slotType === 1 && node.inputs[slot]?.name === "image") {
                        if (isConnected) {
                            setTimeout(tryLoadFromInput, 50);
                        } else {
                            cachedImage = null;
                            imgNaturalW = 0;
                            imgNaturalH = 0;
                            draw();
                        }
                    }
                };

                // ── Interaction: map canvas coords → image-normalized coords ──
                const posFromEvent = (clientX, clientY) => {
                    const rect = canvas.getBoundingClientRect();
                    const cw = rect.width;
                    const ch = rect.height;

                    // Image display rect
                    let imgOffX = 0, imgOffY = 0;
                    let imgDispW = cw, imgDispH = ch;
                    if (imgNaturalW > 0 && imgNaturalH > 0) {
                        const scale = Math.min(cw / imgNaturalW, ch / imgNaturalH);
                        imgDispW = imgNaturalW * scale;
                        imgDispH = imgNaturalH * scale;
                        imgOffX = (cw - imgDispW) / 2;
                        imgOffY = (ch - imgDispH) / 2;
                    }

                    const relX = (clientX - rect.left - imgOffX) / imgDispW;
                    const relY = (clientY - rect.top - imgOffY) / imgDispH;
                    return {
                        x: Math.max(0, Math.min(1, relX)),
                        y: Math.max(0, Math.min(1, relY)),
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

                canvas.addEventListener("mousedown", (e) => { dragging = true; setPos(posFromEvent(e.clientX, e.clientY)); });
                canvas.addEventListener("mousemove", (e) => { if (!dragging) return; setPos(posFromEvent(e.clientX, e.clientY)); });
                const onMouseUp = () => { dragging = false; };
                window.addEventListener("mouseup", onMouseUp);

                // ── Scroll wheel → ball_size ──
                canvas.addEventListener("wheel", (e) => {
                    e.preventDefault();
                    const bs = getW("ball_size");
                    if (!bs) return;
                    bs.value = Math.max(0.02, Math.min(0.5, bs.value + (e.deltaY > 0 ? -0.01 : 0.01)));
                    draw();
                    app.graph.setDirtyCanvas(true, true);
                }, { passive: false });

                // ── Touch ──
                canvas.addEventListener("touchstart", (e) => { e.preventDefault(); dragging = true; const t = e.touches[0]; setPos(posFromEvent(t.clientX, t.clientY)); }, { passive: false });
                canvas.addEventListener("touchmove", (e) => { e.preventDefault(); if (!dragging) return; const t = e.touches[0]; setPos(posFromEvent(t.clientX, t.clientY)); }, { passive: false });
                canvas.addEventListener("touchend", (e) => { e.preventDefault(); dragging = false; }, { passive: false });

                // ── DOM widget ──
                const widget = this.addDOMWidget("light_handle_viewer", "LIGHT_HANDLE", container, {
                    getValue() { return ""; },
                    setValue() {},
                });
                widget.computeSize = function (width) {
                    return [Math.max(width || 350, 350), 320];
                };

                // ── Receive executed image (fallback, or on re-execute) ──
                const onExecuted = this.onExecuted;
                this.onExecuted = function (message) {
                    if (onExecuted) onExecuted.apply(this, arguments);
                    if (message?.image_base64?.[0]) {
                        loadImage(message.image_base64[0]);
                    }
                };

                // ── React to widget changes ──
                const origOnWidgetChanged = this.onWidgetChanged;
                this.onWidgetChanged = function (name, value, old_val, w) {
                    if (origOnWidgetChanged) origOnWidgetChanged.apply(this, arguments);
                    if (["handle_x", "handle_y", "ball_size", "light_color", "handle_shape"].includes(name)) {
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

                // Initial: try to load from connected input
                setTimeout(() => {
                    tryLoadFromInput();
                    draw();
                }, 150);

                return r;
            };
        }
    },
});
