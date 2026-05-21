import { app } from "/scripts/app.js";

app.registerExtension({
    name: "zoey.multiCanvas",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ZoeyMultiCanvas") return;

        const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (slotType, slot, isConnected, link, outputSlot) {
            if (origOnConnectionsChange) origOnConnectionsChange.apply(this, arguments);
            const st = this._mcState;
            if (!st) return;
            if (slotType === 1) {
                const inp = this.inputs[slot];
                if (!inp || !inp.name?.startsWith("image")) return;
                const idx = parseInt(inp.name.replace("image", "")) - 1;
                if (isConnected) {
                    st._loadTimeout = setTimeout(() => loadOneLayer(st, idx), 100);
                } else {
                    if (st.layers[idx]) {
                        st.layers[idx].img = null;
                        st.layers[idx].loaded = false;
                    }
                    if (st.selectedIdx === idx) st.selectedIdx = -1;
                    rebuildLayerList(st);
                    draw(st);
                }
            }
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const rv = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;
            const getW = (name) => node.widgets?.find(w => w.name === name);

            // ── Hide raw widgets ──
            const hideW = (name) => {
                const w = getW(name);
                if (w) {
                    w.computeSize = () => [0, 0];
                    w.draw = () => {};
                    w.mouse = () => false;
                    setTimeout(() => { if (w.element) w.element.style.display = "none"; }, 100);
                }
            };
            ["layer_config", ...Array.from({length:5},(_,i)=>`image${i+1}`)].forEach(hideW);

            // ── State ──
            const s = {
                node,
                layers: Array.from({length:5}, () => ({
                    img: null, loaded: false, loadError: null,
                    ox: 0, oy: 0, scale: 1, opacity: 1, visible: true,
                    rotation: 0, flipH: false, flipV: false,
                })),
                selectedIdx: -1,
                lastInfo: null,
                mode: null, mx0: 0, my0: 0, layer0: null,
                _rotHandlePos: null,
                _loadTimeout: null,
            };
            node._mcState = s;

            // ── Layer colors ──
            const COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373", "#ce93d8"];
            const NAMES = ["图层 1 (底)", "图层 2", "图层 3", "图层 4", "图层 5 (顶)"];

            // ── DOM ──
            const root = document.createElement("div");
            root.style.cssText = "width:100%;display:flex;flex-direction:column;gap:4px;padding:2px 0;user-select:none;";

            const wrap = document.createElement("div");
            wrap.style.cssText = "width:100%;position:relative;background:#1a1a2e;border-radius:6px;overflow:hidden;border:1px solid #333;min-height:180px;";

            const cv = document.createElement("canvas");
            cv.style.cssText = "width:100%;height:100%;display:block;cursor:grab;";
            wrap.appendChild(cv);
            root.appendChild(wrap);

            // ── Controls bar ──
            const bar = document.createElement("div");
            bar.style.cssText = "display:flex;align-items:center;gap:6px;padding:2px 0;flex-wrap:wrap;";

            const reloadBtn = document.createElement("button");
            reloadBtn.textContent = "⟳";
            reloadBtn.title = "重新加载所有图层";
            reloadBtn.style.cssText = "font-size:14px;padding:0 8px;border:1px solid #555;border-radius:4px;background:#2a2a3e;color:#ccc;cursor:pointer;height:26px;line-height:26px;";

            const fitBtn = document.createElement("button");
            fitBtn.textContent = "⊞";
            fitBtn.title = "重置所有图层位置/旋转/翻转";
            fitBtn.style.cssText = "font-size:14px;padding:0 8px;border:1px solid #4caf50;border-radius:4px;background:#1b5e20;color:#a5d6a7;cursor:pointer;height:26px;line-height:26px;";

            const infoLbl = document.createElement("span");
            infoLbl.style.cssText = "font-size:9px;font-family:monospace;color:#666;white-space:nowrap;flex:1;text-align:right;";

            bar.appendChild(reloadBtn);
            bar.appendChild(fitBtn);
            bar.appendChild(infoLbl);
            root.appendChild(bar);

            // ── Layer controls container ──
            const layerPanel = document.createElement("div");
            layerPanel.style.cssText = "display:flex;flex-direction:column;gap:2px;padding:2px 0;";
            root.appendChild(layerPanel);

            function rebuildLayerList() {
                layerPanel.innerHTML = "";
                const hasAny = s.layers.some(l => l.loaded && l.img);
                if (!hasAny) {
                    const h = document.createElement("div");
                    h.style.cssText = "font-size:10px;color:#555;padding:4px;text-align:center;";
                    h.textContent = "连接图像源后显示图层控制";
                    layerPanel.appendChild(h);
                    return;
                }
                for (let i = 0; i < 5; i++) {
                    const l = s.layers[i];
                    if (!l.loaded || !l.img) continue;
                    const row = document.createElement("div");
                    const sel = s.selectedIdx === i;
                    row.style.cssText = `display:flex;align-items:center;gap:4px;padding:3px 6px;border-radius:4px;cursor:pointer;background:${sel ? "rgba(79,195,247,0.15)" : "transparent"};border:1px solid ${sel ? "rgba(79,195,247,0.3)" : "transparent"};`;

                    const dot = document.createElement("span");
                    dot.textContent = "●";
                    dot.style.cssText = `font-size:12px;color:${COLORS[i]};flex:none;`;

                    const name = document.createElement("span");
                    name.textContent = NAMES[i];
                    name.style.cssText = "font-size:10px;color:#aaa;flex:none;overflow:hidden;text-overflow:ellipsis;max-width:72px;";

                    const eye = document.createElement("span");
                    eye.textContent = l.visible ? "👁" : " ";
                    eye.style.cssText = `font-size:11px;cursor:pointer;padding:0 3px;color:${l.visible ? "#aaa" : "#444"};flex:none;`;

                    const opSlider = document.createElement("input");
                    opSlider.type = "range";
                    opSlider.min = 0; opSlider.max = 100; opSlider.value = Math.round(l.opacity * 100);
                    opSlider.style.cssText = "flex:1;height:14px;cursor:pointer;min-width:30px;max-width:60px;";

                    const opLbl = document.createElement("span");
                    opLbl.textContent = `${Math.round(l.opacity * 100)}%`;
                    opLbl.style.cssText = "font-size:9px;font-family:monospace;color:#666;width:26px;text-align:right;flex:none;";

                    const scaleLbl = document.createElement("span");
                    scaleLbl.textContent = `${l.scale.toFixed(2)}x`;
                    scaleLbl.style.cssText = "font-size:9px;font-family:monospace;color:#666;width:36px;text-align:right;flex:none;";

                    // Flip buttons
                    const flipHBtn = document.createElement("span");
                    flipHBtn.textContent = "↔";
                    flipHBtn.title = "水平翻转";
                    flipHBtn.style.cssText = `font-size:12px;cursor:pointer;padding:0 2px;flex:none;color:${l.flipH ? COLORS[i] : "#555"};`;

                    const flipVBtn = document.createElement("span");
                    flipVBtn.textContent = "↕";
                    flipVBtn.title = "垂直翻转";
                    flipVBtn.style.cssText = `font-size:12px;cursor:pointer;padding:0 2px;flex:none;color:${l.flipV ? COLORS[i] : "#555"};`;

                    // Rotation label
                    const rotLbl = document.createElement("span");
                    rotLbl.textContent = `${Math.round(l.rotation)}°`;
                    rotLbl.style.cssText = "font-size:9px;font-family:monospace;color:#888;width:28px;text-align:right;flex:none;";

                    row.appendChild(dot);
                    row.appendChild(name);
                    row.appendChild(eye);
                    row.appendChild(opSlider);
                    row.appendChild(opLbl);
                    row.appendChild(scaleLbl);
                    row.appendChild(flipHBtn);
                    row.appendChild(flipVBtn);
                    row.appendChild(rotLbl);

                    row.addEventListener("click", (e) => {
                        if (e.target === eye || e.target === opSlider || e.target === flipHBtn || e.target === flipVBtn) return;
                        s.selectedIdx = s.selectedIdx === i ? -1 : i;
                        rebuildLayerList();
                        draw(s);
                    });

                    eye.addEventListener("click", (e) => {
                        e.stopPropagation();
                        l.visible = !l.visible;
                        syncConfig();
                        rebuildLayerList();
                        draw(s);
                    });

                    opSlider.addEventListener("input", () => {
                        l.opacity = opSlider.value / 100;
                        opLbl.textContent = `${opSlider.value}%`;
                        syncConfig();
                        draw(s);
                    });

                    flipHBtn.addEventListener("click", (e) => {
                        e.stopPropagation();
                        l.flipH = !l.flipH;
                        flipHBtn.style.color = l.flipH ? COLORS[i] : "#555";
                        syncConfig();
                        draw(s);
                    });

                    flipVBtn.addEventListener("click", (e) => {
                        e.stopPropagation();
                        l.flipV = !l.flipV;
                        flipVBtn.style.color = l.flipV ? COLORS[i] : "#555";
                        syncConfig();
                        draw(s);
                    });

                    layerPanel.appendChild(row);
                }
            }

            // ── Coordinate helpers ──
            function vp() {
                let minX = -0.1, maxX = 1.1, minY = -0.1, maxY = 1.1;
                let hasLayer = false;
                for (const l of s.layers) {
                    if (!l.loaded || !l.img || !l.visible) continue;
                    hasLayer = true;
                    const cx = 0.5 + l.ox;
                    const cy = 0.5 + l.oy;
                    const { halfW, halfH } = ldim(l);
                    const rad = l.rotation * Math.PI / 180;
                    const c = Math.abs(Math.cos(rad));
                    const si = Math.abs(Math.sin(rad));
                    const rHalfW = halfW * c + halfH * si;
                    const rHalfH = halfW * si + halfH * c;
                    minX = Math.min(minX, cx - rHalfW);
                    maxX = Math.max(maxX, cx + rHalfW);
                    minY = Math.min(minY, cy - rHalfH);
                    maxY = Math.max(maxY, cy + rHalfH);
                }
                if (!hasLayer) return { vl: -0.1, vt: -0.1, vr: 1.1, vb: 1.1 };
                const pad = 0.15;
                minX -= pad; minY -= pad; maxX += pad; maxY += pad;
                return { vl: minX, vt: minY, vr: maxX, vb: maxY, vw: maxX - minX, vh: maxY - minY };
            }

            function n2c(nx, ny, info) {
                return { x: info.ox + (nx - info.vl) * info.scX, y: info.oy + (ny - info.vt) * info.scY };
            }
            function c2n(cx, cy, info) {
                return { nx: (cx - info.ox) / info.scX + info.vl, ny: (cy - info.oy) / info.scY + info.vt };
            }
            function ldim(l) {
                // Layer normalized half-dimensions matching backend pixel-space math
                const lW = l._origW || (l.img ? l.img.naturalWidth : 1);
                const lH = l._origH || (l.img ? l.img.naturalHeight : 1);
                const bW = s._baseW || 1;
                const bH = s._baseH || 1;
                return { halfW: (l.scale * lW / bW) / 2, halfH: (l.scale * lH / bH) / 2 };
            }

            // ── Drawing ──
            function draw() {
                const rect = wrap.getBoundingClientRect();
                if (rect.width < 10 || rect.height < 10) return;
                const dpr = window.devicePixelRatio || 1;
                cv.width = rect.width * dpr;
                cv.height = rect.height * dpr;
                const ctx = cv.getContext("2d");
                ctx.scale(dpr, dpr);
                const cw = rect.width, ch = rect.height;

                ctx.fillStyle = "#1a1a2e";
                ctx.fillRect(0, 0, cw, ch);
                s._rotHandlePos = null;

                const hasAny = s.layers.some(l => l.loaded && l.img);
                if (!hasAny) {
                    ctx.fillStyle = "#555"; ctx.font = "13px sans-serif";
                    ctx.textAlign = "center"; ctx.textBaseline = "middle";
                    ctx.fillText("连接图像源后实时预览", cw / 2, ch / 2 - 8);
                    ctx.font = "11px sans-serif"; ctx.fillStyle = "#444";
                    ctx.fillText("拖拽位置 · 滚轮缩放 · 旋转手柄 · 翻转", cw / 2, ch / 2 + 14);
                    s.lastInfo = null;
                    updateInfo();
                    return;
                }

                const { vl, vt, vw, vh } = vp();
                const baseAspect = s._baseAspect || 1;
                let scX, scY;
                if (baseAspect >= 1) {
                    scX = cw / vw;
                    scY = scX / baseAspect;
                    if (scY * vh > ch) { scY = ch / vh; scX = scY * baseAspect; }
                } else {
                    scY = ch / vh;
                    scX = scY * baseAspect;
                    if (scX * vw > cw) { scX = cw / vw; scY = scX / baseAspect; }
                }
                const ox = (cw - vw * scX) / 2;
                const oy = (ch - vh * scY) / 2;
                const info = { vl, vt, vw, vh, scX, scY, ox, oy };
                s.lastInfo = info;
                updateInfo();

                // Draw layers bottom-up with transforms
                for (let i = 0; i < 5; i++) {
                    const l = s.layers[i];
                    if (!l.loaded || !l.img || !l.visible) continue;

                    const cx = 0.5 + l.ox;
                    const cy = 0.5 + l.oy;
                    const { halfW, halfH } = ldim(l);
                    const cs = n2c(cx, cy, info);
                    const c1 = n2c(cx - halfW, cy - halfH, info);
                    const c2 = n2c(cx + halfW, cy + halfH, info);
                    const pixelW = c2.x - c1.x;
                    const pixelH = c2.y - c1.y;
                    if (pixelW < 1 || pixelH < 1) continue;

                    ctx.save();
                    ctx.translate(cs.x, cs.y);
                    ctx.rotate(l.rotation * Math.PI / 180);
                    ctx.scale(l.flipH ? -1 : 1, l.flipV ? -1 : 1);
                    ctx.globalAlpha = l.opacity;
                    ctx.drawImage(l.img, -pixelW / 2, -pixelH / 2, pixelW, pixelH);
                    ctx.globalAlpha = 1;

                    if (s.selectedIdx === i) {
                        ctx.strokeStyle = COLORS[i];
                        ctx.lineWidth = 2;
                        ctx.setLineDash([5, 3]);
                        ctx.strokeRect(-pixelW / 2, -pixelH / 2, pixelW, pixelH);
                        ctx.setLineDash([]);

                        // Corner handles
                        const hs = 7;
                        [[-pixelW/2,-pixelH/2],[pixelW/2,-pixelH/2],[-pixelW/2,pixelH/2],[pixelW/2,pixelH/2]].forEach(([hx, hy]) => {
                            ctx.fillStyle = "#fff";
                            ctx.strokeStyle = COLORS[i];
                            ctx.lineWidth = 1.5;
                            ctx.beginPath();
                            ctx.rect(hx - hs/2, hy - hs/2, hs, hs);
                            ctx.fill();
                            ctx.stroke();
                        });
                    }
                    ctx.restore();
                }

                // ── Rotation handle (drawn outside layer transform) ──
                if (s.selectedIdx >= 0) {
                    const l = s.layers[s.selectedIdx];
                    if (l.loaded && l.img && l.visible) {
                        const cx = 0.5 + l.ox;
                        const cy = 0.5 + l.oy;
                        const { halfW, halfH } = ldim(l);
                        const cs = n2c(cx, cy, info);
                        const c1 = n2c(cx - halfW, cy - halfH, info);
                        const c2 = n2c(cx + halfW, cy + halfH, info);
                        const pixelW = c2.x - c1.x;
                        const pixelH = c2.y - c1.y;

                        // Four corners of unrotated box
                        const rad = l.rotation * Math.PI / 180;
                        const cosR = Math.cos(rad), sinR = Math.sin(rad);
                        const corners = [
                            [-pixelW / 2, -pixelH / 2],
                            [pixelW / 2, -pixelH / 2],
                            [pixelW / 2, pixelH / 2],
                            [-pixelW / 2, pixelH / 2],
                        ].map(([lx, ly]) => ({
                            x: cs.x + lx * cosR - ly * sinR,
                            y: cs.y + lx * sinR + ly * cosR,
                        }));

                        // Edge midpoints → pick topmost (smallest y)
                        const edges = [
                            [(corners[0].x + corners[1].x) / 2, (corners[0].y + corners[1].y) / 2],
                            [(corners[1].x + corners[2].x) / 2, (corners[1].y + corners[2].y) / 2],
                            [(corners[2].x + corners[3].x) / 2, (corners[2].y + corners[3].y) / 2],
                            [(corners[3].x + corners[0].x) / 2, (corners[3].y + corners[0].y) / 2],
                        ];
                        const topEdge = edges.reduce((a, b) => a[1] < b[1] ? a : b);
                        const hx = topEdge[0], hy = topEdge[1];
                        const handleOffY = 22;

                        ctx.save();
                        ctx.strokeStyle = COLORS[s.selectedIdx];
                        ctx.lineWidth = 1.5;
                        ctx.beginPath();
                        ctx.moveTo(hx, hy);
                        ctx.lineTo(hx, hy - handleOffY);
                        ctx.stroke();

                        ctx.fillStyle = COLORS[s.selectedIdx];
                        ctx.beginPath();
                        ctx.arc(hx, hy - handleOffY, 7, 0, Math.PI * 2);
                        ctx.fill();
                        ctx.strokeStyle = "#fff";
                        ctx.lineWidth = 1.5;
                        ctx.stroke();

                        s._rotHandlePos = { x: hx, y: hy - handleOffY };

                        // Rotation angle text
                        ctx.fillStyle = "rgba(255,255,255,0.6)";
                        ctx.font = "10px sans-serif";
                        ctx.textAlign = "center";
                        ctx.textBaseline = "bottom";
                        ctx.fillText(`${Math.round(l.rotation)}°`, hx, hy - handleOffY - 10);
                        ctx.restore();
                    }
                }

                // Layer count badge
                const cnt = s.layers.filter(l => l.loaded && l.img).length;
                ctx.fillStyle = "rgba(255,255,255,0.5)";
                ctx.font = "10px sans-serif";
                ctx.textAlign = "left"; ctx.textBaseline = "top";
                ctx.fillText(`${cnt} 图层`, 6, 4);
            }

            function updateInfo() {
                const cnt = s.layers.filter(l => l.loaded && l.img).length;
                infoLbl.textContent = `${cnt}/5 图层`;
            }

            // ── Widget sync ──
            function syncConfig() {
                const config = {};
                for (let i = 0; i < 5; i++) {
                    const l = s.layers[i];
                    config[`layer${i+1}`] = {
                        x: l.ox, y: l.oy,
                        s: l.scale, o: l.opacity, v: l.visible,
                        r: l.rotation, fh: l.flipH, fv: l.flipV,
                    };
                }
                const w = getW("layer_config");
                if (w) {
                    w.value = JSON.stringify(config);
                    if (w.callback) w.callback(w.value);
                }
                app.graph.setDirtyCanvas(true, true);
            }

            // ── Hit testing ──
            function pointInRotatedRect(px, py, cx, cy, halfW, halfH, angleDeg, flipH, flipV) {
                const dx = px - cx;
                const dy = py - cy;
                const rad = -angleDeg * Math.PI / 180;
                const cosR = Math.cos(rad), sinR = Math.sin(rad);
                let rx = dx * cosR - dy * sinR;
                let ry = dx * sinR + dy * cosR;
                if (flipH) rx = -rx;
                if (flipV) ry = -ry;
                return Math.abs(rx) <= halfW && Math.abs(ry) <= halfH;
            }

            function getLayerAt(mx, my) {
                if (!s.lastInfo) return -1;
                const p = c2n(mx, my, s.lastInfo);
                for (let i = 4; i >= 0; i--) {
                    const l = s.layers[i];
                    if (!l.loaded || !l.img || !l.visible) continue;
                    const cx = 0.5 + l.ox;
                    const cy = 0.5 + l.oy;
                    const { halfW, halfH } = ldim(l);
                    if (pointInRotatedRect(p.nx, p.ny, cx, cy, halfW, halfH, l.rotation, l.flipH, l.flipV)) {
                        return i;
                    }
                }
                return -1;
            }

            function hitRotHandle(mx, my) {
                if (!s._rotHandlePos) return false;
                const dx = mx - s._rotHandlePos.x;
                const dy = my - s._rotHandlePos.y;
                return Math.sqrt(dx * dx + dy * dy) < 14;
            }

            // ── Interaction ──
            cv.addEventListener("mousedown", (e) => {
                const rect = cv.getBoundingClientRect();
                const mx = e.clientX - rect.left, my = e.clientY - rect.top;

                if (hitRotHandle(mx, my) && s.selectedIdx >= 0) {
                    s.mode = "rotate";
                    s.mx0 = mx; s.my0 = my;
                    const l = s.layers[s.selectedIdx];
                    s.layer0 = { rotation: l.rotation };
                    cv.style.cursor = "crosshair";
                    e.preventDefault();
                    return;
                }

                const hitIdx = getLayerAt(mx, my);
                if (hitIdx >= 0) {
                    s.selectedIdx = hitIdx;
                    s.mode = "move";
                    s.mx0 = mx; s.my0 = my;
                    s.layer0 = { ox: s.layers[hitIdx].ox, oy: s.layers[hitIdx].oy };
                    cv.style.cursor = "grabbing";
                    rebuildLayerList();
                } else {
                    s.selectedIdx = -1;
                    s.mode = null;
                    rebuildLayerList();
                }
                draw(s);
            });

            const onMM = (e) => {
                const rect = cv.getBoundingClientRect();
                const mx = e.clientX - rect.left, my = e.clientY - rect.top;

                if (!s.mode) {
                    const onRot = hitRotHandle(mx, my) && s.selectedIdx >= 0;
                    cv.style.cursor = onRot ? "crosshair" : getLayerAt(mx, my) >= 0 ? "grab" : "default";
                    return;
                }
                if (!s.lastInfo || s.selectedIdx < 0) return;

                const l = s.layers[s.selectedIdx];

                if (s.mode === "rotate") {
                    const cs = n2c(0.5 + l.ox, 0.5 + l.oy, s.lastInfo);
                    const curAngle = Math.atan2(my - cs.y, mx - cs.x);
                    const startAngle = Math.atan2(s.my0 - cs.y, s.mx0 - cs.x);
                    let deltaDeg = (curAngle - startAngle) * 180 / Math.PI;
                    if (e.shiftKey) {
                        const raw = (s.layer0?.rotation ?? 0) + deltaDeg;
                        l.rotation = ((Math.round(raw / 15) * 15) % 360 + 360) % 360;
                    } else {
                        l.rotation = (((s.layer0?.rotation ?? 0) + deltaDeg) % 360 + 360) % 360;
                    }
                    syncConfig();
                    draw(s);
                    rebuildLayerList();
                    e.preventDefault();
                    return;
                }

                const p = c2n(mx, my, s.lastInfo);
                const p0 = c2n(s.mx0, s.my0, s.lastInfo);
                l.ox = (s.layer0?.ox ?? l.ox) + p.nx - p0.nx;
                l.oy = (s.layer0?.oy ?? l.oy) + p.ny - p0.ny;
                syncConfig();
                draw(s);
                e.preventDefault();
            };

            const onMU = () => {
                if (s.mode) {
                    s.mode = null;
                    cv.style.cursor = "default";
                }
            };

            // ── Scroll to scale ──
            cv.addEventListener("wheel", (e) => {
                if (s.selectedIdx < 0) return;
                const l = s.layers[s.selectedIdx];
                if (!l.loaded || !l.img) return;
                const delta = -e.deltaY * 0.001;
                l.scale = Math.max(0.05, Math.min(10, l.scale * (1 + delta)));
                syncConfig();
                draw(s);
                e.preventDefault();
            }, { passive: false });

            window.addEventListener("mousemove", onMM);
            window.addEventListener("mouseup", onMU);

            // ── Button events ──
            reloadBtn.addEventListener("click", () => {
                for (let i = 0; i < 5; i++) loadOneLayer(s, i);
            });

            fitBtn.addEventListener("click", () => {
                for (const l of s.layers) {
                    l.ox = 0; l.oy = 0; l.scale = 1;
                    l.rotation = 0; l.flipH = false; l.flipV = false;
                }
                syncConfig();
                draw(s);
            });

            // ── Image loading ──
            function loadOneLayer(st, idx) {
                const l = st.layers[idx];
                const src = findSourceNode(st, `image${idx+1}`);
                if (!src) { l.loaded = false; l.img = null; rebuildLayerList(); draw(st); return; }

                let url = null;
                if (src.imgs?.length > 0) {
                    const el = src.imgs[0];
                    if (typeof el === "string") url = el;
                    else if (el?.src) url = el.src;
                    else if (el?._src) url = el._src;
                    else if (el?.tagName === "CANVAS") url = el.toDataURL?.("image/png");
                }
                if (!url && src.image) {
                    url = typeof src.image === "string" ? src.image : src.image?.src || src.image?._src;
                }
                if (!url) {
                    const imgW = src.widgets?.find(w => w.name === "image");
                    if (imgW) {
                        let fn = "", sub = "", tp = "input";
                        if (typeof imgW.value === "string") {
                            const parts = imgW.value.split("/"); fn = parts.pop() || ""; sub = parts.join("/");
                        } else if (imgW.value && typeof imgW.value === "object") {
                            fn = imgW.value.filename || ""; sub = imgW.value.subfolder || ""; tp = imgW.value.type || "input";
                        }
                        if (fn) {
                            url = `${location.origin}/view?filename=${encodeURIComponent(fn)}&type=${tp}${sub ? "&subfolder="+encodeURIComponent(sub) : ""}&rand=${Date.now()}`;
                        }
                    }
                }
                if (!url) {
                    for (const w of (src.widgets || [])) {
                        let v = w.value;
                        if (typeof v === "object" && v) v = v.filename || v.name || "";
                        if (typeof v === "string" && /\.(png|jpg|jpeg|webp|bmp)$/i.test(v)) {
                            const parts = v.split("/"); const fn = parts.pop();
                            const sub = parts.join("/");
                            url = `${location.origin}/view?filename=${encodeURIComponent(fn)}&type=input${sub ? "&subfolder="+encodeURIComponent(sub) : ""}&rand=${Date.now()}`;
                            break;
                        }
                    }
                }
                if (!url) return;

                l.loadError = null;
                const img = new Image();
                if (url.startsWith("blob:") || url.startsWith("data:")) img.crossOrigin = "anonymous";
                img.onload = () => {
                    l.img = img; l.loaded = true; l.loadError = null;
                    l._origW = img.naturalWidth;
                    l._origH = img.naturalHeight;
                    l.imgAspect = img.naturalWidth / img.naturalHeight;
                    if (idx === 0) { st._baseW = img.naturalWidth; st._baseH = img.naturalHeight; st._baseAspect = l.imgAspect; }
                    rebuildLayerList();
                    draw(st);
                };
                img.onerror = () => { l.loadError = true; l.loaded = false; draw(st); };
                img.src = url;
            }

            function findSourceNode(st, inputName) {
                const inp = st.node.inputs?.find(i => i.name === inputName);
                if (!inp || inp.link == null) return null;
                const tid = inp.link;
                for (const n of (app.graph._nodes || [])) {
                    if (n === st.node || !n.outputs) continue;
                    for (const o of n.outputs) {
                        if (!o.links) continue;
                        for (const lid of o.links) if (lid != null && lid === tid) return n;
                    }
                }
                try {
                    const ld = app.graph.links?.[tid];
                    if (ld) {
                        const id = Array.isArray(ld) ? ld[0] : ld?.origin_id ?? ld?.[0];
                        return app.graph.getNodeById(id);
                    }
                } catch (e) {}
                try {
                    const links = app.graph.links;
                    if (links) for (const [lid, ld] of Object.entries(links)) {
                        if (Number(lid) === tid) {
                            const id = Array.isArray(ld) ? ld[0] : ld?.origin_id ?? ld?.[0];
                            return app.graph.getNodeById(id);
                        }
                    }
                } catch (e) {}
                return null;
            }

            function loadAllLayers(st) {
                for (let i = 0; i < 5; i++) loadOneLayer(st, i);
            }

            // ── Configure (workflow restore) ──
            const origCfg = this.configure;
            this.configure = function (data) {
                if (origCfg) origCfg.apply(this, arguments);
                const st = this._mcState;
                if (!st) return;
                try {
                    const raw = getW("layer_config")?.value || "{}";
                    const cfg = JSON.parse(raw);
                    for (let i = 0; i < 5; i++) {
                        const k = `layer${i+1}`;
                        const c = cfg[k];
                        if (c) {
                            const l = st.layers[i];
                            l.ox = c.x ?? 0;
                            l.oy = c.y ?? 0;
                            l.scale = c.s ?? 1;
                            l.opacity = c.o ?? 1;
                            l.visible = c.v ?? true;
                            l.rotation = (c.r ?? 0) % 360;
                            l.flipH = c.fh ?? false;
                            l.flipV = c.fv ?? false;
                        }
                    }
                } catch (e) {}
                rebuildLayerList();
                setTimeout(() => loadAllLayers(st), 500);
            };

            // ── On executed ──
            const origExec = this.onExecuted;
            this.onExecuted = function (msg) {
                if (origExec) origExec.apply(this, arguments);
                const st = this._mcState;
                if (!st) return;
                setTimeout(() => loadAllLayers(st), 200);
            };

            // ── DOM widget ──
            const widget = this.addDOMWidget("multi_canvas_preview", "MULTI_CANVAS_PREVIEW", root, {
                getValue() { return ""; }, setValue() {},
            });
            widget.computeSize = (width) => {
                const w = Math.max(width || 350, 280);
                wrap.style.height = Math.max(180, Math.floor(w * 0.68)) + "px";
                const layerCount = this._mcState?.layers?.filter(l => l.loaded && l.img).length || 0;
                const extraH = layerCount > 0 ? 30 * layerCount + 4 : 24;
                return [w, Math.floor(w * 0.68) + 38 + extraH];
            };
            this.setSize([350, 310]);

            // ── ResizeObserver ──
            let rt;
            const ro = new ResizeObserver(() => {
                if (rt) cancelAnimationFrame(rt);
                rt = requestAnimationFrame(() => { draw(); rt = null; });
            });
            ro.observe(wrap);

            // ── Cleanup ──
            const origRM = this.onRemoved;
            this.onRemoved = function () {
                ro.disconnect();
                window.removeEventListener("mousemove", onMM);
                window.removeEventListener("mouseup", onMU);
                if (rt) cancelAnimationFrame(rt);
                this._mcState = null;
                if (origRM) origRM.apply(this, arguments);
            };

            // ── Initial load ──
            this.setSize([350, 310]);
            setTimeout(() => loadAllLayers(s), 400);

            s._draw = draw;
            s._rebuildLayerList = rebuildLayerList;

            return rv;
        };
    },
});
